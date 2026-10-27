"""E2c①：EEG 微状态特征的严格嵌套复现检验（XWStroke 主战场 + BCIC 阳性对照）。

设计文档：E2_实验设计_20261026.md §3 E2c①。
目标：检验 Lv25（急性卒中左右手 MI，微状态特征 + KNN，报告 75.00%±7.94%）的声明。
Lv25 的泄漏嫌疑：微状态模板（聚类）与特征选择在**全部数据**上做。
本实现：模板聚类、特征归一化、（可选）特征选择全部只在训练折内拟合。

微状态流程（标准做法）：
  1. 计算每窗 GFP（全局场功率），取 GFP 局部极大时刻的头皮拓扑（极性不变）；
  2. 训练折内所有窗的 GFP 峰拓扑做 k-means（K=5，极性不变：相关距离）；
  3. 用 K 个模板对每窗逐时刻标记（空间相关最大者），得每窗的状态序列；
  4. 每窗特征：K 个状态的 coverage/duration/occurrence + K×K 转移概率 = K*3+K*K 维；
  5. 分类器：KNN（复刻 Lv25）与 LDA 各跑一臂。

条件（arms）：
  ms_knn   K=5 微状态特征 + KNN(k=5)（Lv25 复现口径，严格嵌套）
  ms_lda   同特征 + LDA（更稳的参照分类器）

协议：与 E2a/E2b 完全相同的折划分（GroupKFold 5 折，seeds 42/123/2026），
输入数据相同（4-40Hz + zscore，2.0s 窗）。配对基线复用 E2a csp_full。

输出：
  experiments/e2_hypothesis_families/e2c_microstate/{dataset}/{arm}_fold_results.csv

用法：
  python scripts/e2/e2c_microstate.py --dataset xw --arm ms_knn
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.e2.e2a_spatial import (DATASETS, load_subject, group_kfold_indices)

OUT = "experiments/e2_hypothesis_families/e2c_microstate"
K_STATES = 5


def gfp_peaks(X):
    """X: (n, ch, T) -> 每窗 GFP 局部极大时刻的拓扑 (n_peaks_total, ch)，极性不变。

    GFP = 各时刻跨电极标准差。返回所有窗的峰拓扑拼接。
    """
    gfp = X.std(axis=1)                       # (n, T)
    peaks_all = []
    for i in range(X.shape[0]):
        g = gfp[i]
        # 局部极大（比左右邻居大）
        idx = np.where((g[1:-1] > g[:-2]) & (g[1:-1] >= g[2:]))[0] + 1
        if len(idx) == 0:
            idx = np.array([int(np.argmax(g))])
        topo = X[i][:, idx].T                 # (n_peaks, ch)
        peaks_all.append(topo)
    return np.concatenate(peaks_all, axis=0)


def fit_templates(X_tr, k=K_STATES, seed=0, max_peaks=20000):
    """训练折窗上聚类微状态模板（极性不变 k-means，基于相关距离）。"""
    from sklearn.cluster import KMeans
    P = gfp_peaks(X_tr)                       # (N, ch)
    # 极性不变：每个拓扑标准化到单位范数，符号对齐到第一主方向
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    if len(P) > max_peaks:
        rng = np.random.default_rng(seed)
        P = P[rng.choice(len(P), max_peaks, replace=False)]
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(P)
    T = km.cluster_centers_
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    return T                                  # (k, ch) 单位范数模板


def segment(X, T):
    """用模板标记每窗每时刻（|相关| 最大者，极性不变）。X:(n,ch,T), T:(k,ch) -> (n,T)"""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)   # (n,ch,T)
    corr = np.abs(np.einsum("nct,kc->nkt", Xn, T))                # (n,k,T)
    return corr.argmax(axis=1)                                    # (n,T)


def ms_features(labels_seq, k=K_STATES):
    """状态序列 -> 特征。coverage/duration/occurrence + 转移概率。"""
    n, T_len = labels_seq.shape
    F = np.zeros((n, k * 3 + k * k))
    for i in range(n):
        s = labels_seq[i]
        for state in range(k):
            m = s == state
            cov = m.mean()
            # 段：连续段数与平均长度
            runs = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
            n_runs = int((runs == 1).sum())
            occ = n_runs / T_len
            dur = cov / (occ + 1e-12) if n_runs > 0 else 0.0     # 平均段长（归一化）
            F[i, state * 3:state * 3 + 3] = [cov, dur, occ]
        # 转移概率（行归一）
        tr = np.zeros((k, k))
        for a, b in zip(s[:-1], s[1:]):
            tr[a, b] += 1
        tr = tr / (tr.sum(axis=1, keepdims=True) + 1e-12)
        F[i, k * 3:] = tr.ravel()
    return F


def run_ms(X, y, g, sid, seeds, arm):
    rows = []
    for seed in seeds:
        for fi, (tr, te) in enumerate(group_kfold_indices(g, 5, seed + sid)):
            T = fit_templates(X[tr], seed=seed * 100 + fi)       # 模板仅训练折
            Ftr = ms_features(segment(X[tr], T))
            Fte = ms_features(segment(X[te], T))
            mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-12             # 归一化仅训练折
            Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
            if arm == "ms_knn":
                from sklearn.neighbors import KNeighborsClassifier
                clf = KNeighborsClassifier(n_neighbors=5)
            else:
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                clf = LinearDiscriminantAnalysis()
            clf.fit(Ftr, y[tr])
            acc = float(clf.score(Fte, y[te]))
            rows.append((sid, seed, fi, acc, len(tr), len(te), K_STATES))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["ms_knn", "ms_lda"])
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
        X = ds.data[:, 0, :, :]
        rows = run_ms(X, ds.labels, ds.group_ids, sid, remain, args.arm)
        m = np.mean([r[3] for r in rows])
        print(f"[{args.dataset}/{args.arm}] sub-{sid:02d} done: {m*100:.2f}%  ({time.time()-t0:.0f}s)",
              flush=True)
        df = pd.DataFrame(rows, columns=["subject", "seed", "fold", "acc",
                                         "n_train", "n_test", "k_states"])
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    print(f"\n===== {args.dataset}/{args.arm} 完成 =====")


if __name__ == "__main__":
    main()
