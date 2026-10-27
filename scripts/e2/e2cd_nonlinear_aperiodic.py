"""E2c② + E2d：非线性特征与非周期 1/f 的 within-subject 检验（附带条件，一次收关）。

设计文档：E2_实验设计_20261026.md §3 E2c② / E2d。

条件（arms）：
  pe        排列熵（每导，m=3, τ=1）→ 30/22 维 → 收缩 LDA
            （E2c② 非线性族；对单调变换不变，zscore 无影响）
  bispec    双谱摘要（每导 μ×μ、β×β 双频区平均 |双谱|）→ 60/44 维 → 收缩 LDA
            （E2c②；文献 Sha10 慢性小样本 ~69% 的对应特征）
  aperiodic 非周期 1/f（每导 4-40Hz log-log 谱线性拟合的斜率+截距）→ 60/44 维 → 收缩 LDA
            （E2d；specparam 的简化代理，承认不含峰移除步骤）

协议：与 E2a/E2b 完全相同的折划分与输入数据（4-40Hz + zscore，2.0s 窗）。
配对基线复用 E2a csp_full。所有特征逐窗独立计算，无泄漏；归一化仅训练折拟合。

输出：
  experiments/e2_hypothesis_families/e2cd_nonlinear_aperiodic/{dataset}/{arm}_fold_results.csv

用法：
  python scripts/e2/e2cd_nonlinear_aperiodic.py --dataset xw --arm aperiodic
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.e2.e2a_spatial import (DATASETS, load_subject, group_kfold_indices)

OUT = "experiments/e2_hypothesis_families/e2cd_nonlinear_aperiodic"


def feat_pe(X, m=3, tau=1):
    """排列熵，每导一值。X: (n, ch, T) -> (n, ch)"""
    n, ch, T = X.shape
    out = np.zeros((n, ch))
    perms = None
    from itertools import permutations
    perms = np.array(list(permutations(range(m))))       # (m!, m)
    n_patterns = len(perms)
    for c in range(ch):
        x = X[:, c, :]
        # 滑动窗口嵌入
        idx = np.arange(m)[None, :] * tau + np.arange(T - (m - 1) * tau)[:, None]
        emb = x[:, idx]                                   # (n, L, m)
        order = emb.argsort(axis=2)                       # (n, L, m)
        # 映射到模式索引
        keys = (order[:, :, None, :] == perms[None, None, :, :]).all(-1)
        pat_id = keys.argmax(axis=2)                      # (n, L)
        cnt = np.zeros((n, n_patterns))
        for p in range(n_patterns):
            cnt[:, p] = (pat_id == p).sum(axis=1)
        prob = cnt / cnt.sum(axis=1, keepdims=True)
        prob = np.clip(prob, 1e-12, None)
        out[:, c] = -(prob * np.log(prob)).sum(axis=1) / np.log(n_patterns)
    return out


def feat_bispec(X, fs=250, bands=((8, 13), (13, 30))):
    """双谱摘要：每导每频带对平均 |B(f1,f2)|。X: (n, ch, T) -> (n, ch*len(bands))"""
    Xf = np.fft.rfft(X, axis=-1)                          # (n, ch, F)
    freqs = np.fft.rfftfreq(X.shape[-1], 1 / fs)
    feats = []
    for lo, hi in bands:
        idx = np.where((freqs >= lo) & (freqs <= hi))[0]
        B_mag = np.zeros((X.shape[0], X.shape[1]))
        cnt = 0
        for i1, f1 in enumerate(idx):
            for f2 in idx[i1:]:
                f3 = f1 + f2
                if f3 >= len(freqs):
                    continue
                B = np.abs(Xf[:, :, f1] * Xf[:, :, f2] * np.conj(Xf[:, :, f3]))
                B_mag += B.mean(-1) if B.ndim == 3 else B
                cnt += 1
        feats.append(B_mag / max(cnt, 1))
    return np.concatenate(feats, axis=1)


def feat_aperiodic(X, fs=250):
    """非周期代理：每导 log-log 谱（4-40Hz）线性拟合的斜率+截距。X: (n,ch,T) -> (n, ch*2)"""
    from scipy.signal import welch
    f, psd = welch(X, fs=fs, nperseg=min(256, X.shape[-1]), axis=-1)
    m = (f >= 4) & (f <= 40)
    lf = np.log10(f[m])
    lf_c = lf - lf.mean()
    lp = np.log10(psd[..., m] + 1e-20)                    # (n, ch, F')
    slope = (lp * lf_c).sum(-1) / (lf_c**2).sum()
    offset = lp.mean(-1) - slope * lf.mean()
    return np.concatenate([slope, offset], axis=1)


FEATS = {"pe": feat_pe, "bispec": feat_bispec, "aperiodic": feat_aperiodic}


def run(X, y, g, sid, seeds, arm):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    F = FEATS[arm](X)                                     # 逐窗独立
    F = np.log10(F + 1e-20) if arm == "bispec" else F     # 双谱幅值取 log
    rows = []
    for seed in seeds:
        for fi, (tr, te) in enumerate(group_kfold_indices(g, 5, seed + sid)):
            mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-12  # 归一化仅训练折
            clf = LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")
            clf.fit((F[tr] - mu) / sd, y[tr])
            acc = float(clf.score((F[te] - mu) / sd, y[te]))
            rows.append((sid, seed, fi, acc, len(tr), len(te), F.shape[1]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(FEATS.keys()))
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
        rows = run(X, ds.labels, ds.group_ids, sid, remain, args.arm)
        m = np.mean([r[3] for r in rows])
        print(f"[{args.dataset}/{args.arm}] sub-{sid:02d} done: {m*100:.2f}%  ({time.time()-t0:.0f}s)",
              flush=True)
        df = pd.DataFrame(rows, columns=["subject", "seed", "fold", "acc",
                                         "n_train", "n_test", "n_feat"])
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    print(f"\n===== {args.dataset}/{args.arm} 完成 =====")


if __name__ == "__main__":
    main()
