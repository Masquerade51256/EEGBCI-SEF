"""E2h：相位-幅值耦合（PAC）特征的 within-subject 检验（XWStroke 主战场 + BCIC 阳性对照 + HS 阶段对照）。

设计文档：E2_实验设计_20261026.md §6 E2h。
假设族：判别信息存在于慢波（delta/theta）相位对快波（mu/beta/gamma）幅值的
跨频耦合（PAC）中——独立于功率（E2a）与相位同步（E2b）的信息轴，
且天然连接慢电位（E2e）与功率（E2a）两族。

**与慢电位族共用数据口径**：直接读未滤波原始 trial epoch（复用 e2e 的 LOADERS），
避免标准管线的 4–40 Hz 滤波破坏慢波相位。

特征（每 trial 一个样本，不做滑窗）：
  每通道 × 相位带{delta 2-4, theta 4-8} × 幅值带{mu 8-13, beta 13-30, gamma 30-40}
  的 Tort 调制指数（MI，KL 散度，18 个相位 bin）。
  相位：butter4 带通 filtfilt → hilbert 相位；幅值：带通 → hilbert 包络 → 再按
  对应相位带低通包络（标准 Tort 流程的简化：包络不再二次滤波，直接分 bin 取均值）。

条件（arms）：
  pac_full   全特征（n_ch × 2 × 3 维）+ 收缩 LDA
  pac_sel20  折内嵌套特征选择（训练折 |点二列相关| top-20）+ LDA

协议：GroupKFold 5 折（每 trial 即一组），seeds 42/123/2026；折内标准化。
标签口径与 E2e 完全一致（XW affected 映射 + 右瘫翻转对齐；BCIC 左右手；HS 左右手）。

输出：
  experiments/e2_hypothesis_families/e2h_pac/{dataset}/{arm}_fold_results.csv

用法：
  python scripts/e2/e2h_pac.py --dataset xw --arm pac_full
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.e2.e2a_spatial import DATASETS, group_kfold_indices
from scripts.e2.e2e_slowpot import LOADERS

OUT = "experiments/e2_hypothesis_families/e2h_pac"

PHASE_BANDS = [(2.0, 4.0), (4.0, 8.0)]          # delta, theta
AMP_BANDS = [(8.0, 13.0), (13.0, 30.0), (30.0, 40.0)]  # mu, beta, gamma
N_BINS = 18


def pac_features(X, sr):
    """X: (n, ch, T) -> (n, ch × n_phase × n_amp) Tort 调制指数。"""
    from scipy.signal import butter, filtfilt, hilbert
    n, n_ch, T = X.shape
    # 预计算所有带的解析信号
    phase_signals = []   # list of (n, ch, T) 相位
    for lo, hi in PHASE_BANDS:
        b, a = butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band")
        phase_signals.append(np.angle(hilbert(filtfilt(b, a, X, axis=-1), axis=-1)))
    amp_signals = []
    for lo, hi in AMP_BANDS:
        b, a = butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band")
        amp_signals.append(np.abs(hilbert(filtfilt(b, a, X, axis=-1), axis=-1)))

    edges = np.linspace(-np.pi, np.pi, N_BINS + 1)
    feats = np.zeros((n, n_ch, len(PHASE_BANDS), len(AMP_BANDS)))
    for ip, ph in enumerate(phase_signals):
        bin_idx = np.digitize(ph, edges) - 1          # (n, ch, T) in [0, N_BINS)
        bin_idx = np.clip(bin_idx, 0, N_BINS - 1)
        for ia, am in enumerate(amp_signals):
            # 每 bin 的平均幅值
            mean_amp = np.zeros((n, n_ch, N_BINS))
            cnt = np.zeros(N_BINS)
            for bn in range(N_BINS):
                m = bin_idx == bn
                cnt_bn = m.sum(axis=-1)               # (n, ch)
                s = (am * m).sum(axis=-1)
                mean_amp[:, :, bn] = np.where(cnt_bn > 0, s / np.maximum(cnt_bn, 1), 0.0)
            pa = mean_amp / mean_amp.sum(axis=-1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                kl = np.where(pa > 0, pa * np.log(pa * N_BINS), 0.0).sum(axis=-1)
            feats[:, :, ip, ia] = kl / np.log(N_BINS)  # 归一化 MI
    return feats.reshape(n, -1)


def run_pac(X, y, sid, seeds, arm, sr):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    F = pac_features(X, sr)
    g = np.arange(len(y))
    rows = []
    for seed in seeds:
        for fi, (tr, te) in enumerate(group_kfold_indices(g, 5, seed + sid)):
            Ftr, Fte = F[tr], F[te]
            mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-12
            Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
            if arm == "pac_sel20":
                yz = (y[tr] - y[tr].mean()) / (y[tr].std() + 1e-12)
                sc = np.abs((Ftr * yz[:, None]).mean(0))
                idx = np.argsort(-sc)[:20]
                Ftr, Fte = Ftr[:, idx], Fte[:, idx]
                clf = LinearDiscriminantAnalysis()
            else:
                clf = LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")
            clf.fit(Ftr, y[tr])
            acc = float(clf.score(Fte, y[te]))
            rows.append((sid, seed, fi, acc, len(tr), len(te), F.shape[1]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["pac_full", "pac_sel20"])
    ap.add_argument("--dataset", default="xw", choices=list(LOADERS.keys()))
    ap.add_argument("--seeds", default="42,123,2026")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    loader, sr, epoch_s, _ = LOADERS[args.dataset]

    out_dir = os.path.join(OUT, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{args.arm}_fold_results.csv")
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(zip(prev.subject, prev.seed))
        print(f"[resume] {len(done)} (subject, seed) 已完成")

    for sid in DATASETS[args.dataset]["subjects"]:
        remain = [s for s in seeds if (sid, s) not in done]
        if not remain:
            continue
        t0 = time.time()
        try:
            X, y = loader(sid)
        except Exception as e:
            print(f"sub-{sid:02d}: LOAD ERROR {e}", flush=True)
            continue
        rows = run_pac(X.astype(np.float64), y, sid, remain, args.arm, sr)
        m = np.mean([r[3] for r in rows])
        print(f"[{args.dataset}/{args.arm}] sub-{sid:02d} done: {m*100:.2f}%  ({time.time()-t0:.0f}s)",
              flush=True)
        df = pd.DataFrame(rows, columns=["subject", "seed", "fold", "acc",
                                         "n_train", "n_test", "n_feat"])
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    print(f"\n===== {args.dataset}/{args.arm} 完成 =====")


if __name__ == "__main__":
    main()
