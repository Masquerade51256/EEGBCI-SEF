"""E2i：时域波形形态特征的 within-subject 检验（XWStroke 主战场 + BCIC 阳性对照 + HS 阶段对照）。

设计文档：E2_实验设计_20261026.md §6 E2i。
假设族：判别信息存在于事件相关波形的一般化形态中（不限于慢电位的幅值/斜率）——
波形偏度/峰度、Hjorth 参数、过零率、峰谷结构，以及 trial 与类模板的形态相似度。

数据口径：复用 E2e 的未滤波原始 trial loader；特征提取前做 0.5–40 Hz 带通
（去掉慢漂移——<4 Hz 慢电位已由 E2e 族专门覆盖，本族聚焦形态而非慢波幅值）。

特征（每 trial 一个样本，不做滑窗），每通道 8 个形状描述子：
  偏度、峰度、Hjorth activity/mobility/complexity、过零率、峰数、平均峰谷深度。
另设模板臂：trial 波形与训练折类模板（两类均值波形）的逐通道相关 = 2×n_ch 特征
（模板仅在训练折内构建，无泄漏）。

条件（arms）：
  wav_full   全描述子（n_ch × 8 维）+ 收缩 LDA
  wav_sel20  折内嵌套特征选择（训练折 |点二列相关| top-20）+ LDA
  wav_tpl    模板相关特征（2×n_ch 维，折内建模）+ LDA

协议：GroupKFold 5 折（每 trial 即一组），seeds 42/123/2026；折内标准化。

输出：
  experiments/e2_hypothesis_families/e2i_waveform/{dataset}/{arm}_fold_results.csv

用法：
  python scripts/e2/e2i_waveform.py --dataset xw --arm wav_full
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

OUT = "experiments/e2_hypothesis_families/e2i_waveform"


def _bandpass(X, sr, lo=0.5, hi=40.0):
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return filtfilt(b, a, X, axis=-1)


def shape_features(X, sr):
    """X: (n, ch, T) -> (n, ch*8) 波形形状描述子。"""
    from scipy.stats import skew, kurtosis
    Xf = _bandpass(X, sr)
    n, ch, T = Xf.shape
    d1 = np.diff(Xf, axis=-1)
    d2 = np.diff(d1, axis=-1)
    var0 = Xf.var(axis=-1)
    var1 = d1.var(axis=-1)
    var2 = d2.var(axis=-1)
    activity = var0
    mobility = np.sqrt(var1 / (var0 + 1e-12))
    complexity = np.sqrt(var2 / (var1 + 1e-12)) / (mobility + 1e-12)
    zcr = ((Xf[..., :-1] * Xf[..., 1:]) < 0).sum(axis=-1) / (T - 1)
    # 峰谷结构
    n_peaks = np.zeros((n, ch))
    depth = np.zeros((n, ch))
    for i in range(n):
        for c in range(ch):
            x = Xf[i, c]
            pk = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
            tr = (x[1:-1] < x[:-2]) & (x[1:-1] < x[2:])
            n_peaks[i, c] = pk.sum()
            if pk.sum() > 0 and tr.sum() > 0:
                depth[i, c] = x[1:-1][pk].mean() - x[1:-1][tr].mean()
    feats = [skew(Xf, axis=-1), kurtosis(Xf, axis=-1), activity, mobility,
             complexity, zcr, n_peaks / (T / sr), depth]
    return np.concatenate(feats, axis=1)   # (n, ch*8)


def template_features(Xtr, ytr, Xte, sr):
    """训练折内构建两类均值模板，输出逐通道相关特征。"""
    Xtr_f, Xte_f = _bandpass(Xtr, sr), _bandpass(Xte, sr)
    tpls = {}
    for cls in np.unique(ytr):
        tpls[cls] = Xtr_f[ytr == cls].mean(axis=0)        # (ch, T)
    def corr_feats(Xf):
        n, ch, T = Xf.shape
        out = np.zeros((n, ch * len(tpls)))
        for j, cls in enumerate(sorted(tpls)):
            t = tpls[cls]
            t = (t - t.mean(-1, keepdims=True)) / (t.std(-1, keepdims=True) + 1e-12)
            xz = (Xf - Xf.mean(-1, keepdims=True)) / (Xf.std(-1, keepdims=True) + 1e-12)
            out[:, j * ch:(j + 1) * ch] = (xz * t).mean(-1)
        return out
    return corr_feats(Xtr_f), corr_feats(Xte_f)


def run_wav(X, y, sid, seeds, arm, sr):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    g = np.arange(len(y))
    rows = []
    F = None
    if arm in ("wav_full", "wav_sel20"):
        F = shape_features(X, sr)                        # 逐 trial 独立
    for seed in seeds:
        for fi, (tr, te) in enumerate(group_kfold_indices(g, 5, seed + sid)):
            if arm == "wav_tpl":
                Ftr, Fte = template_features(X[tr], y[tr], X[te], sr)
            else:
                Ftr, Fte = F[tr], F[te]
            mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-12
            Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
            if arm == "wav_sel20":
                yz = (y[tr] - y[tr].mean()) / (y[tr].std() + 1e-12)
                sc = np.abs((Ftr * yz[:, None]).mean(0))
                idx = np.argsort(-sc)[:20]
                Ftr, Fte = Ftr[:, idx], Fte[:, idx]
                clf = LinearDiscriminantAnalysis()
            elif arm == "wav_tpl":
                clf = LinearDiscriminantAnalysis()
            else:
                clf = LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")
            clf.fit(Ftr, y[tr])
            acc = float(clf.score(Fte, y[te]))
            rows.append((sid, seed, fi, acc, len(tr), len(te), Ftr.shape[1]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["wav_full", "wav_sel20", "wav_tpl"])
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
        rows = run_wav(X.astype(np.float64), y, sid, remain, args.arm, sr)
        m = np.mean([r[3] for r in rows])
        print(f"[{args.dataset}/{args.arm}] sub-{sid:02d} done: {m*100:.2f}%  ({time.time()-t0:.0f}s)",
              flush=True)
        df = pd.DataFrame(rows, columns=["subject", "seed", "fold", "acc",
                                         "n_train", "n_test", "n_feat"])
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    print(f"\n===== {args.dataset}/{args.arm} 完成 =====")


if __name__ == "__main__":
    main()
