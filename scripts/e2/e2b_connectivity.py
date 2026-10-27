"""E2b：稀疏功能连接特征的 within-subject 检验（XWStroke 主战场 + BCIC 阳性对照）。

设计文档：E2_实验设计_20261026.md §3 E2b。
假设族：判别信息存在于通道间耦合关系（功能连接），而非任何单通道功率。
文献依据：Ben20/K23（卒中相邻任务 63-74%）、Jan26（手指级 MI LOSO 超机会）。

特征：每窗 × 每频带（μ 8-13Hz, β 13-30Hz）计算两种连接度量的全导两两边：
  - wPLI（加权相位滞后指数）：对容积传导稳健的相位同步度量
  - iCoh（相干性虚部）：幅值无关的耦合度量
  30 导 → 435 边/度量/带，共 435×2×2=1740 维（BCIC 22 导 → 231×2×2=924 维）

条件（arms）：
  conn_full   全边 + 收缩 LDA（shrinkage='auto'），无边筛选
  conn_sel50  嵌套边筛选：训练折内按 |点二列相关| 选 top-50 边 + 普通 LDA
  conn_sel20  同上，top-20 边（更稀疏）

协议（继承 E1a/E2a 干净协议，与 E2a csp_full 同折可配对）：
  - GroupKFold 5 折，按 trial 分组；seeds 42/123/2026（与 E2a 相同，折完全一致）
  - 边筛选、特征 zscore 均只在训练折内拟合（审计铁律 2）
  - 输入数据与 E2a 完全相同（4-40Hz 滤波 + zscore），配对基线直接复用 E2a csp_full

输出：
  experiments/e2_hypothesis_families/e2b_connectivity/{dataset}/{arm}_fold_results.csv

用法：
  python scripts/e2/e2b_connectivity.py --dataset xw --arm conn_full
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.e2.e2a_spatial import (DATASETS, load_subject, group_kfold_indices)

OUT = "experiments/e2_hypothesis_families/e2b_connectivity"
BANDS = [(8, 13), (13, 30)]  # μ, β


def band_hilbert(X, lo, hi, fs=250):
    """X: (n, ch, T) -> 解析信号 (n, ch, T)。数据已 4-40Hz 滤波，此处再做带通。"""
    from scipy.signal import butter, filtfilt, hilbert
    b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    Xf = filtfilt(b, a, X, axis=-1)
    return hilbert(Xf, axis=-1)


def connectivity_features(X, fs=250):
    """X: (n, ch, T) -> (n, n_edges*2*2) [wPLI μ, wPLI β, iCoh μ, iCoh β]"""
    n, n_ch, _ = X.shape
    iu = np.triu_indices(n_ch, k=1)
    feats = []
    for lo, hi in BANDS:
        Z = band_hilbert(X, lo, hi, fs)          # (n, ch, T) 解析信号
        # 两两交叉谱虚部：向量化
        Zi = Z[:, iu[0], :]                       # (n, E, T)
        Zj = Z[:, iu[1], :]
        Sij = Zi * np.conj(Zj)                    # 交叉谱
        im = Sij.imag                             # (n, E, T)
        # wPLI = |mean(Im)| / mean(|Im|)
        wpli = np.abs(im.mean(-1)) / (np.abs(im).mean(-1) + 1e-12)
        # iCoh = |mean(Im(Sij))| / sqrt(mean(Sii) * mean(Sjj))
        power = (Z * np.conj(Z)).real.mean(-1)    # (n, ch)
        denom = np.sqrt(power[:, iu[0]] * power[:, iu[1]]) + 1e-12
        icoh = np.abs(im.mean(-1)) / denom
        feats.extend([wpli, icoh])
    return np.concatenate(feats, axis=1)          # (n, E*4)


def edge_scores(F_tr, y_tr):
    """训练折内各边的 |点二列相关|"""
    y = y_tr.astype(float)
    y = (y - y.mean()) / (y.std() + 1e-12)
    Fz = (F_tr - F_tr.mean(0)) / (F_tr.std(0) + 1e-12)
    return np.abs((Fz * y[:, None]).mean(0))


def run_conn(X, y, g, sid, seeds, arm):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    F = connectivity_features(X)                  # 全窗连接特征（无泄漏：逐窗独立）
    rows = []
    topk = {"conn_sel50": 50, "conn_sel20": 20}.get(arm)
    for seed in seeds:
        for fi, (tr, te) in enumerate(group_kfold_indices(g, 5, seed + sid)):
            Ftr, Fte = F[tr], F[te]
            # 归一化：仅训练折拟合
            mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-12
            Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
            if topk is not None:                  # 边筛选：仅训练折
                edges = np.argsort(-edge_scores(Ftr, y[tr]))[:topk]
                Ftr, Fte = Ftr[:, edges], Fte[:, edges]
                clf = LinearDiscriminantAnalysis()
            else:
                edges = None
                clf = LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")
            clf.fit(Ftr, y[tr])
            acc = float(clf.score(Fte, y[te]))
            rows.append((sid, seed, fi, acc, len(tr), len(te),
                         "" if edges is None else str(len(edges))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["conn_full", "conn_sel50", "conn_sel20"])
    ap.add_argument("--dataset", default="xw", choices=list(DATASETS.keys()))
    ap.add_argument("--seeds", default="42,123,2026")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    dspec = DATASETS[args.dataset]

    out_dir = os.path.join(OUT, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{args.arm}_fold_results.csv")
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(zip(prev.subject, prev.seed))
        print(f"[resume] {len(done)} (subject, seed) 已完成")

    for sid in dspec["subjects"]:
        remain = [s for s in seeds if (sid, s) not in done]
        if not remain:
            continue
        t0 = time.time()
        try:
            ds = load_subject(args.dataset, sid)
        except Exception as e:
            print(f"sub-{sid:02d}: LOAD ERROR {e}", flush=True)
            continue
        X = ds.data[:, 0, :, :]                    # (n_win, n_ch, T)
        rows = run_conn(X, ds.labels, ds.group_ids, sid, remain, args.arm)
        m = np.mean([r[3] for r in rows])
        print(f"[{args.dataset}/{args.arm}] sub-{sid:02d} done: {m*100:.2f}%  ({time.time()-t0:.0f}s)",
              flush=True)
        df = pd.DataFrame(rows, columns=["subject", "seed", "fold", "acc",
                                         "n_train", "n_test", "edges"])
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    print(f"\n===== {args.dataset}/{args.arm} 完成 =====")


if __name__ == "__main__":
    main()
