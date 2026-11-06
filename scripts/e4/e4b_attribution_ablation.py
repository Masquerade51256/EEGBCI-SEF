"""E4 第 0.5 步：broad 臂增益的归因消融（XW 主战场，OFAT 单因子翻转）。

设计文档：E4_实验设计_草案_20261105.md；结果记录：E4_实验结果记录.md §1。

动机：第 0 步 broad（0.1–40 Hz + 整段 trial + 基线校正）56.48% vs std
（4–40 Hz + 2 s 窗 + 逐窗 zscore）50.47%，+6.01pp（p=0.0012）。但两臂差异是
频带 / 时间粒度 / 归一化三件事的捆绑。本消融从 broad 角点出发每次只翻一个
变量，定位增益的必要条件：

| 臂 | 频带 | 时间粒度 | 归一化 | 隔离变量 |
|---|---|---|---|---|
| broad（已跑，参照） | 0.1–40 | trial | 基线校正 | — |
| band4_trial_base | 4–40 | trial | 基线校正 | 频带：慢波是否必要 |
| broad_win_base | 0.1–40 | 2 s 窗 | 窗内基线校正（前 0.5 s） | 时间粒度：长程结构是否必要 |
| broad_trial_z | 0.1–40 | trial | 逐 trial 逐通道 zscore | 归一化：幅值信息是否必要 |

判读（事先锁定）：某臂掉回机会水平（≈std 的 50.5%）→ 该变量是 broad 增益的
必要条件；多臂都掉 → 增益为多因素协同；都不掉 → 增益来自三者的交互。

协议与第 0 步完全相同：同一 EEGNet、GroupKFold 5 折 × 3 seeds（42/123/2026，
种子仅影响折划分）、fixed 100 epochs、Adam lr=1e-3、CE、无增强；
batch 16（trial 级）/ 32（窗级）。

输出：experiments/e4_step0_ablation/xw/{arm}_fold_results.csv
  列：subject, seed, fold, acc, n_train, n_test, channels（=臂名）

用法：
  python scripts/e4/e4b_attribution_ablation.py --arm band4_trial_base
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.e2.e2e_slowpot import load_xw_trials
from scripts.e2.e2a_spatial import group_kfold_indices
from scripts.e4.e4a_broadband_veto import bandpass_np, to_250hz, run_arm

OUT = "experiments/e4_step0_ablation"

ARMS = ["band4_trial_base", "broad_win_base", "broad_trial_z"]
SR = 500
SUBJECTS = list(range(1, 51))


def make_arm(X, sr, arm):
    """X: (n, ch, T) 未滤波 4 s @500 Hz -> (Xs float32, tidx)"""
    if arm == "band4_trial_base":            # 翻频带：4–40 Hz，trial 级，基线校正
        Xf = to_250hz(bandpass_np(X, 4.0, 40.0, sr), sr)
        q = Xf.shape[-1] // 4
        Xf = Xf - Xf[..., :q].mean(axis=-1, keepdims=True)
        return np.ascontiguousarray(Xf).astype(np.float32), np.arange(len(X)), 16
    if arm == "broad_win_base":              # 翻时间粒度：0.1–40 Hz，2 s 窗，窗内基线校正
        Xf = to_250hz(bandpass_np(X, 0.1, 40.0, sr), sr)
        win, stride = 500, 125               # 2.0 s / 0.5 s @ 250 Hz
        starts = list(range(0, Xf.shape[-1] - win + 1, stride))
        n, ch = Xf.shape[:2]
        Xw = np.stack([Xf[:, :, s:s + win] for s in starts], axis=1)
        Xw = Xw.reshape(n * len(starts), ch, win)
        q = win // 4                          # 窗内前 0.5 s 为基线
        Xw = Xw - Xw[..., :q].mean(axis=-1, keepdims=True)
        tidx = np.repeat(np.arange(n), len(starts))
        return np.ascontiguousarray(Xw).astype(np.float32), tidx, 32
    if arm == "broad_trial_z":               # 翻归一化：0.1–40 Hz，trial 级，逐 trial zscore
        Xf = to_250hz(bandpass_np(X, 0.1, 40.0, sr), sr)
        mu = Xf.mean(-1, keepdims=True)
        sd = Xf.std(-1, keepdims=True) + 1e-12
        return ((Xf - mu) / sd).astype(np.float32), np.arange(len(X)), 16
    raise ValueError(arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--subjects", default=None, help="如 '1,2' 限定被试（冒烟测试用）")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    subjects = SUBJECTS if args.subjects is None else \
        [int(s) for s in args.subjects.split(",")]

    out_dir = os.path.join(OUT, "xw")
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
            X, y = load_xw_trials(sid)
        except Exception as e:
            print(f"sub-{sid:02d}: LOAD ERROR {e}", flush=True)
            continue
        Xs, tidx, batch = make_arm(X, SR, args.arm)
        rows = run_arm(Xs, y, tidx, sid, remain, args.arm, args.epochs, batch=batch)
        m = np.mean([r[3] for r in rows])
        print(f"[xw/{args.arm}] sub-{sid:02d} done: {m*100:.2f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows, columns=["subject", "seed", "fold", "acc",
                                         "n_train", "n_test", "channels"])
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    print(f"\n===== xw/{args.arm} 完成 =====")


if __name__ == "__main__":
    main()
