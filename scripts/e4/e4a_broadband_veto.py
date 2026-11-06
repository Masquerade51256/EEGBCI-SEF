"""E4 第 0 步：宽带输入否决性预实验（broadband veto）。

设计文档：E4_实验设计_草案_20261105.md §3 第 0 步 / §7 训练方案。

问题：E2 证明急性卒中患者的判别信息出线在 0–4 Hz 慢电位轴（手工 SCP 族
54.7%，q=0.028），而标准深度管线在预处理阶段（4–40 Hz 滤波 + 2 s 短窗 +
逐窗 zscore）结构性删除该信息。本实验用同一 EEGNet 小网络、同一训练口径，
只换输入表示，检验"仅换输入能否达到手工 SCP 水平"：

  std   臂：4–40 Hz 带通 → 2 s 滑窗（0.5 s 步长）→ 逐窗逐通道 zscore（复刻 E2a）
  broad 臂：0.1–40 Hz 带通 → 整段 4 s/3 s trial → 基线校正（减前 1/4 段均值），
            不做 zscore（保留慢波幅值信息）

判读线（事先锁定）：
  broad ≥ 手工 SCP 54.7% → 瓶颈在输入表示，结构改造必要性存疑；
  broad ≈ std ≈ 机会水平 → 输入不是充分条件，进入第 1 步结构改造；
  中间值              → 输入贡献部分成立，结构改造仍有空间。

健全性锚点：XW std 应 ≈ 50.6%（E2a EEGNet 同协议结果）；BCIC std 应 ≥ ~75%
（健康人标准管线应有表现，达不到则判实现错误）。

协议（与 E2a/E2k 完全相同的折划分与训练口径）：
  - GroupKFold 5 折按 trial 分组（std 臂同 trial 的重叠窗不跨折）
  - 3 seeds（42/123/2026），种子仅影响折划分
  - 固定 100 epochs，Adam lr=1e-3，CE，无增强（第 0 步只隔离输入变量）
  - batch 32（std，窗级）/ 16（broad，trial 级，每被试仅 40 trial）

输出：experiments/e4_step0_broadband/{dataset}/{arm}_fold_results.csv
  列：subject, seed, fold, acc, n_train, n_test, channels（=臂名）

用法：
  python scripts/e4/e4a_broadband_veto.py --dataset xw --arm std
  python scripts/e4/e4a_broadband_veto.py --dataset xw --arm broad
  python scripts/e4/e4a_broadband_veto.py --dataset bcic --arm std --subjects 1
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.e2.e2e_slowpot import load_xw_trials, load_bcic_trials
from scripts.e2.e2a_spatial import group_kfold_indices

OUT = "experiments/e4_step0_broadband"

LOADERS = {
    "xw":   dict(loader=load_xw_trials,   sr=500, subjects=list(range(1, 51))),
    "bcic": dict(loader=load_bcic_trials, sr=250, subjects=list(range(1, 10))),
}


def bandpass_np(X, lo, hi, fs, order=4):
    """X: (n, ch, T) -> 滤波后同形状。butter4 + filtfilt 零相位，逐 trial 独立。"""
    from scipy.signal import butter, filtfilt
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, X, axis=-1)


def to_250hz(X, sr):
    """XW 500 Hz -> 250 Hz（先滤波后抽取，40 Hz << 125 Hz Nyquist，无混叠）。"""
    if sr == 500:
        return X[:, :, ::2]
    return X


def make_std(X, sr):
    """std 臂：4–40 Hz -> 250 Hz -> 2 s 窗 / 0.5 s 步长 -> 逐窗逐通道 zscore。
    返回 Xw (n_win, ch, 500) float32，tidx (n_win,) 窗所属的 trial 索引。"""
    Xf = to_250hz(bandpass_np(X, 4.0, 40.0, sr), sr)
    win, stride = 500, 125                       # 2.0 s / 0.5 s @ 250 Hz
    T = Xf.shape[-1]
    starts = list(range(0, T - win + 1, stride))
    n, ch = Xf.shape[:2]
    Xw = np.stack([Xf[:, :, s:s + win] for s in starts], axis=1)
    Xw = Xw.reshape(n * len(starts), ch, win)
    mu = Xw.mean(-1, keepdims=True)
    sd = Xw.std(-1, keepdims=True) + 1e-12
    Xw = ((Xw - mu) / sd).astype(np.float32)
    tidx = np.repeat(np.arange(n), len(starts))
    return Xw, tidx


def make_broad(X, sr):
    """broad 臂：0.1–40 Hz -> 250 Hz -> 整段 trial -> 基线校正（减前 1/4 段均值）。
    不做逐窗 zscore（保留慢波幅值）。返回 Xb (n, ch, T) float32，tidx = 每 trial 一组。"""
    Xf = to_250hz(bandpass_np(X, 0.1, 40.0, sr), sr)
    q = Xf.shape[-1] // 4
    Xf = Xf - Xf[..., :q].mean(axis=-1, keepdims=True)
    Xb = np.ascontiguousarray(Xf).astype(np.float32)
    tidx = np.arange(Xb.shape[0])
    return Xb, tidx


def run_arm(Xs, y, tidx, sid, seeds, arm, epochs=100, batch=None):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from models.EEGNet import EEGNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if batch is None:
        batch = 16 if arm == "broad" else 32
    y_all = y[tidx]                              # 窗级 / trial 级标签
    rows = []
    for seed in seeds:
        for fi, (tr, te) in enumerate(group_kfold_indices(tidx, 5, seed + sid)):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = EEGNet(num_classes=2, num_channels=Xs.shape[1],
                           num_bands=1, input_length=Xs.shape[-1]).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            lossf = nn.CrossEntropyLoss()
            tds = TensorDataset(torch.from_numpy(Xs[tr]),
                                torch.from_numpy(y_all[tr]).long())
            loader = DataLoader(tds, batch_size=batch, shuffle=True, num_workers=0)
            Xte = torch.from_numpy(Xs[te]).to(device)
            yte = torch.from_numpy(y_all[te]).long().to(device)
            for _ in range(epochs):
                model.train()
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = lossf(model(xb.unsqueeze(1)), yb)
                    loss.backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                acc = float((model(Xte.unsqueeze(1)).argmax(1) == yte)
                            .float().mean().item())
            rows.append((sid, seed, fi, acc, len(tr), len(te), arm))
            print(f"    sub-{sid:02d} seed={seed} fold{fi}: {acc*100:.1f}%",
                  flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["std", "broad"])
    ap.add_argument("--dataset", default="xw", choices=list(LOADERS.keys()))
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--subjects", default=None, help="如 '1,2' 限定被试（冒烟测试用）")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    dspec = LOADERS[args.dataset]
    subjects = dspec["subjects"] if args.subjects is None else \
        [int(s) for s in args.subjects.split(",")]

    out_dir = os.path.join(OUT, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{args.arm}_fold_results.csv")
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(zip(prev.subject, prev.seed))
        print(f"[resume] {len(done)} (subject, seed) 已完成")

    for sid in subjects:
        remain = [s for s in seeds if (sid, s) not in done]
        if not remain:
            continue
        t0 = time.time()
        try:
            X, y = dspec["loader"](sid)
        except Exception as e:
            print(f"sub-{sid:02d}: LOAD ERROR {e}", flush=True)
            continue
        if args.arm == "std":
            Xs, tidx = make_std(X, dspec["sr"])
        else:
            Xs, tidx = make_broad(X, dspec["sr"])
        rows = run_arm(Xs, y, tidx, sid, remain, args.arm, args.epochs)
        m = np.mean([r[3] for r in rows])
        print(f"[{args.dataset}/{args.arm}] sub-{sid:02d} done: {m*100:.2f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows, columns=["subject", "seed", "fold", "acc",
                                         "n_train", "n_test", "channels"])
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    print(f"\n===== {args.dataset}/{args.arm} 完成 =====")


if __name__ == "__main__":
    main()
