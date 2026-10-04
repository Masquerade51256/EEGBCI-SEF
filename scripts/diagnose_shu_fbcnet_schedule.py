"""诊断 SHU WS FBCNet 复现差距：官方训练调度 vs 我方早停调度。

官方 sub-001_ses-01 FBCNet = 78%，我方此前复现 = 54.4%。
差异候选：
  A. 训练调度：官方 maxEpochs=1500 + NoDecrease patience=200(valInacc) + loadBestModel
     我方此前 maxEpochs=500 + patience=30(val acc)
  B. 时间窗：官方 .mat 为 0-4s(1000 点)，我方此前裁成 0.5-4s(876 点)

本脚本仅在 sub-001_ses-01 上跑 10 折，对照三种配置：
  cfg1 = 我方原配置 (0.5-4s, 500ep, patience30)
  cfg2 = 官方调度 + 0.5-4s 窗
  cfg3 = 官方调度 + 0-4s 全窗（完整对齐官方）
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
import time

sys.path.insert(0, ".")
from models.FBCNet import Net

FS = 250
ROOT = Path("src/datasets/SHU")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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


def load_session_fb(sid, ses, t0, t1):
    """返回 (n, 9, 32, t1-t0) 多频带输入与标签"""
    sub = f"sub-{sid:03d}"
    edf = ROOT / "edf_raw" / "edf" / f"{sub}_ses-{ses:02d}_task_motorimagery_eeg.edf"
    evp = ROOT / "events_raw" / "events" / f"{sub}_ses-{ses:02d}_task_motorimagery_events.tsv"
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
    bands = [np.asarray([signal.lfilter(b, a, t) for t in T])[:, :, t0:t1] for b, a in COEFF]
    X = np.stack(bands, axis=1).astype(np.float32)
    return X, np.asarray(L)


def train_fold(Xtr, ytr, Xva, yva, Xte, yte, max_epochs, patience, seed=42):
    torch.manual_seed(seed)
    model = Net(num_classes=2, num_channels=32, sampling_rate=FS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.NLLLoss()
    tr = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr).long()),
        batch_size=16, shuffle=True, drop_last=True)
    Xva_t = torch.from_numpy(Xva).to(DEVICE)
    Xte_t = torch.from_numpy(Xte).to(DEVICE)
    best_va, best_state, bad = -1, None, 0
    ep_run = 0
    for ep in range(max_epochs):
        ep_run = ep + 1
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
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te = (model(Xte_t).argmax(1).cpu().numpy() == yte).mean()
    return float(te), float(best_va), ep_run


CONFIGS = [
    ("cfg1_ours_0.5-4s_500ep_p30", int(0.5 * FS), int(4 * FS), 500, 30),
    ("cfg2_offsched_0.5-4s_1500ep_p200", int(0.5 * FS), int(4 * FS), 1500, 200),
    ("cfg3_offsched_0-4s_1500ep_p200", 0, int(4 * FS), 1500, 200),
]

SID, SES = 1, 1
print(f"诊断 sub-{SID:03d}_ses-{SES:02d}，官方 FBCNet 参考值 = 78.0%", flush=True)

for name, t0, t1, max_ep, pat in CONFIGS:
    X, y = load_session_fb(SID, SES, t0, t1)
    print(f"\n===== {name} =====  X={X.shape}  trials={len(y)}", flush=True)
    scores = []
    t_start = time.time()
    for fi, (tr_te, te) in enumerate(StratifiedKFold(10, shuffle=True, random_state=42).split(X, y)):
        X_rest, y_rest = X[tr_te], y[tr_te]
        tr_idx, va_idx = train_test_split(np.arange(len(y_rest)), test_size=1 / 9,
                                          stratify=y_rest, random_state=42)
        acc, bva, ep_run = train_fold(X_rest[tr_idx], y_rest[tr_idx],
                                      X_rest[va_idx], y_rest[va_idx],
                                      X[te], y[te], max_ep, pat)
        scores.append(acc)
        print(f"  fold{fi}: test={acc*100:.1f}%  best_val={bva*100:.1f}%  epochs={ep_run}", flush=True)
    dt = time.time() - t_start
    print(f"  >>> {name}: mean={np.mean(scores)*100:.2f}% ± {np.std(scores)*100:.2f}%  用时 {dt/60:.1f} min", flush=True)

print("\n===== 对照 =====")
print("官方 sub-001_ses-01 FBCNet: 78.0%")
print("我方此前全量复现该 session: 54.4%")
