"""诊断 SHU WS FBCNet：直接套官方网络实现（m=4, VarLayer 全时间轴, ELU）。

前一轮诊断（diagnose_shu_fbcnet_schedule.py, sub-001_ses-01, 官方 78.0%）：
  cfg1 我方网络+我方调度(0.5-4s, 500ep/p30) : 54.44%
  cfg2 我方网络+官方调度(0.5-4s, 1500ep/p200): 60.44%
  cfg3 我方网络+官方调度+0-4s: 崩溃（我方 FBCNet 硬编码 F.pad(0,1)+reshape/4，T=1000 不整除）

代码对照发现的架构差异（官方 ws/FBCNet/codes/centralRepo/networks.py vs models/FBCNet.py）：
  官方: m=4, VarLayer 对全时间轴取方差 → 特征 4×9=36 维, ELU
  我方: m=32, 时间轴分 4 段 log-var → 特征 32×9×4=1152 维, Swish
  → 我方模型容量对 ~80 训练样本/折 而言过大，可能是差距主因之一。

本轮：直接 import 官方 networks.py 的 FBCNet 跑 10 折：
  cfg4 = 官方网络 + 官方调度(1500ep/p200) + 0-4s 全窗（完整对齐官方）
  cfg5 = 官方网络 + 官方调度 + 0.5-4s 窗（隔离窗口影响）
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
sys.path.insert(0, "src/datasets/SHU/code_raw/code/ws/FBCNet/codes/centralRepo")
import networks as offnet  # 官方 FBCNet 实现

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
    X = np.stack(bands, axis=1).astype(np.float32)  # (n, 9, 32, T)
    return X, np.asarray(L)


def train_fold(Xtr, ytr, Xva, yva, Xte, yte, max_epochs, patience, seed=42):
    torch.manual_seed(seed)
    model = offnet.FBCNet(nChan=32, nTime=1000, nClass=2, nBands=9, m=4,
                          temporalLayer='VarLayer', doWeightNorm=True).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.NLLLoss()

    def to_official_fmt(X):
        # (n, 9, 32, T) -> (n, 1, 32, T, 9)，官方 forward 内部再 permute 回 (n, 9, 32, T)
        return torch.from_numpy(X).permute(0, 2, 3, 1).unsqueeze(1)

    tr = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(to_official_fmt(Xtr), torch.from_numpy(ytr).long()),
        batch_size=16, shuffle=True, drop_last=True)
    Xva_t = to_official_fmt(Xva).to(DEVICE)
    Xte_t = to_official_fmt(Xte).to(DEVICE)
    best_va, best_state, bad = -1, None, 0
    ep_run = 0
    for ep in range(max_epochs):
        ep_run = ep + 1
        model.train()
        for xb, yb in tr:
            xb, yb = xb.float().to(DEVICE), yb.to(DEVICE)
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
    ("cfg4_offnet_offsched_0-4s", 0, int(4 * FS)),
    ("cfg5_offnet_offsched_0.5-4s", int(0.5 * FS), int(4 * FS)),
]

SID, SES = 1, 1
print(f"诊断(官方网络) sub-{SID:03d}_ses-{SES:02d}，官方 FBCNet 参考值 = 78.0%", flush=True)

for name, t0, t1 in CONFIGS:
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
                                      X[te], y[te], 1500, 200)
        scores.append(acc)
        print(f"  fold{fi}: test={acc*100:.1f}%  best_val={bva*100:.1f}%  epochs={ep_run}", flush=True)
    dt = time.time() - t_start
    print(f"  >>> {name}: mean={np.mean(scores)*100:.2f}% ± {np.std(scores)*100:.2f}%  用时 {dt/60:.1f} min", flush=True)

print("\n===== 对照 =====")
print("官方 sub-001_ses-01 FBCNet: 78.0%")
print("前轮: cfg1(我方全部) 54.44% / cfg2(我方网络+官方调度) 60.44%")
