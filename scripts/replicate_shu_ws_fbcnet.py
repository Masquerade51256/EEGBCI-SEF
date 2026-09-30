"""复现官方 WS FBCNet（目标 68.85% ± 14.64%）。
模型用项目框架 models/FBCNet.py 的 Net；训练协议对齐官方：
  - 输入：9 个 4Hz 子带（cheby2，同 FBCSP）× 32ch × 0.5-4s（876 点）
  - batch 16, lr 1e-3, Adam, NLLLoss
  - 10 折 CV，每折 8:1:1（train/val/test），val 准确率早停（patience 30）
  - 上限 500 epochs（官方 1500 但早停远先到；为控时长收敛在此）
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from scipy import signal
from sklearn.model_selection import StratifiedKFold, train_test_split
import mne
import sys

sys.path.insert(0, ".")
from models.FBCNet import Net

FS = 250
ROOT = Path("src/datasets/SHU")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS = 500
PATIENCE = 30
mne.set_log_level("ERROR")
ev_names = ["onset", "duration", "trial_type", "response_time", "sample", "value"]

def get_filter_coeff():
    nyq = FS / 2
    coeff = []
    for f_low in np.arange(4, 40, 4):
        f_pass = np.array([f_low, f_low + 4])
        f_stop = np.array([f_low - 2, f_low + 6])
        order, wn = signal.cheb2ord(f_pass / nyq, f_stop / nyq, 3, 30)
        b, a = signal.cheby2(order, 30, wn, btype="bandpass")
        coeff.append((b, a))
    return coeff

COEFF = get_filter_coeff()

def load_session_fb(sid, ses):
    """返回 (n, 9, 32, 876) 多频带输入与标签"""
    sub = f"sub-{sid:03d}"
    edf = ROOT / "edf_raw" / "edf" / f"{sub}_ses-{ses:02d}_task_motorimagery_eeg.edf"
    evp = ROOT / "events_raw" / "events" / f"{sub}_ses-{ses:02d}_task_motorimagery_events.tsv"
    if not edf.exists():
        return None
    raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
    sig = raw.get_data() * 1e6
    ev = pd.read_csv(evp, sep="\t")
    if "sample" not in ev.columns:
        ev = pd.read_csv(evp, sep="\t", header=None, names=ev_names)
    T, L = [], []
    for _, row in ev.iterrows():
        s = int(row["sample"])
        seg = sig[:, s:s + 1000]
        if seg.shape[1] < 1000:
            continue
        T.append(seg)
        L.append(0 if str(row["trial_type"]).strip() == "left" else 1)
    T = np.asarray(T)
    t0, t1 = int(0.5 * FS), int(4 * FS)
    bands = [np.asarray([signal.lfilter(b, a, t) for t in T])[:, :, t0:t1] for b, a in COEFF]
    X = np.stack(bands, axis=1).astype(np.float32)  # (n, 9, 32, 876)
    return X, np.asarray(L)

def train_fold(Xtr, ytr, Xva, yva, Xte, yte, seed=42):
    torch.manual_seed(seed)
    model = Net(num_classes=2, num_channels=32, sampling_rate=FS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.NLLLoss()
    tr = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr).long()),
        batch_size=16, shuffle=True, drop_last=True)  # drop_last 防止 batch=1 被 squeeze 掉维度
    Xva_t = torch.from_numpy(Xva).to(DEVICE)
    Xte_t = torch.from_numpy(Xte).to(DEVICE)
    best_va, best_state, bad = -1, None, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va = (model(Xva_t).argmax(1).cpu().numpy() == yva).mean()
        if va > best_va:
            best_va, bad = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te = (model(Xte_t).argmax(1).cpu().numpy() == yte).mean()
    return float(te)

accs = {}
for sid in range(1, 26):
    for ses in range(1, 6):
        d = load_session_fb(sid, ses)
        if d is None:
            continue
        X, y = d
        fold_scores = []
        for tr_te, te in StratifiedKFold(10, shuffle=True, random_state=42).split(X, y):
            X_rest, y_rest = X[tr_te], y[tr_te]
            tr_idx, va_idx = train_test_split(np.arange(len(y_rest)), test_size=1/9,
                                              stratify=y_rest, random_state=42)
            fold_scores.append(train_fold(X_rest[tr_idx], y_rest[tr_idx],
                                          X_rest[va_idx], y_rest[va_idx],
                                          X[te], y[te]))
        accs[f"sub-{sid:03d}_ses-{ses:02d}"] = float(np.mean(fold_scores))
        print(f"  sub-{sid:03d}_ses-{ses:02d}: {accs[f'sub-{sid:03d}_ses-{ses:02d}']*100:.1f}%", flush=True)
    a = np.array(list(accs.values()))
    print(f"sub-{sid:03d} done, 累计均值 {a.mean()*100:.2f}% (n={len(a)})", flush=True)

a = np.array(list(accs.values()))
print("\n===== 复现官方 WS FBCNet =====")
print(f"我方复现: {a.mean()*100:.2f}% ± {a.std()*100:.2f}% (n={len(a)})")
print("官方报告: 68.85% ± 14.64% (n=125)")
