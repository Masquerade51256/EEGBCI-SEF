"""E2a：空间表示修复的 within-subject 检验（XWStroke 全 30 导）。

设计文档：E2_实验设计_20261026.md §3 E2a / §4 验证形态约定。

条件（arms）：
  csp_full    A0 基线：CSP+LDA，全 30 导（功率族参照标尺）
  csp_fixed4  A2 固定运动区 4 导（FC3, Cz, C3, CP3；对齐坐标系下 = 病灶对侧运动区）
  csp_sel4    A1 嵌套个体化通道选择（Li19 式）：训练折内按 μ/β ERD 可分性选 top-4 导，
              CSP 仅用该 4 导；选择过程严格不碰测试折（审计铁律 2）
  eegnet_full 功率族深度参照：EEGNet 全 30 导（可选，GPU）

注：患侧对齐（hemisphere_alignment）对 within-subject 准确率是恒等变换
（固定通道排列不改变 CSP/EEGNet 可达函数类），其价值在跨被试坐标系，
故 E2a within 阶段三个 CSP 条件共用对齐配置，差异只在通道子集。

协议（继承 E1a 干净协议）：
  - GroupKFold 5-fold，按 trial 分组（group_ids），同 trial 重叠窗不跨折
  - 3 seeds（默认 42,123,2026），种子仅影响折划分
  - CSP+LDA：6 分量，数据已 4-40Hz 滤波 + zscore（pipeline 内），CSP 分支无增强
  - EEGNet：fixed 100 epochs，报最终 epoch；增强仅训练折

输出：
  experiments/e2_hypothesis_families/e2a_spatial/{arm}_fold_results.csv
  （列：subject, seed, fold, acc, n_train, n_test, channels）

用法：
  python scripts/e2/e2a_spatial.py --arm csp_full
  python scripts/e2/e2a_spatial.py --arm csp_sel4 --seeds 42,123,2026
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, ".")
from data.datasets import XWStrokeDataset

OUT = "experiments/e2_hypothesis_families/e2a_spatial"
CFG = "configs/dataset/XWStroke_affected_aligned.yaml"
SUBJECTS = list(range(1, 51))

# 对齐坐标系下的通道名（见 data/datasets.py XWSTROKE_FLIP_ORDER 注释）：
#   0 Fp1, 1 Fp2, 2 Fz, 3 F3, 4 F4, 5 F7, 6 F8, 7 FCz, 8 FC3, 9 FC4,
#  10 FT7, 11 FT8, 12 Cz, 13 C3, 14 C4, 15 T3, 16 T4, 17 CPz, 18 CP3, 19 CP4,
#  20 TP7, 21 TP8, 22 Pz, 23 P3, 24 P4, 25 T5, 26 T6, 27 Oz, 28 O1, 29 O2
CH_NAMES = ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4",
            "FT7", "FT8", "Cz", "C3", "C4", "T3", "T4", "CPz", "CP3", "CP4",
            "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2"]
FIXED4 = [8, 12, 13, 18]  # FC3, Cz, C3, CP3：对齐坐标系下即病灶对侧运动区 + 中线
SEL_K = 4
BANDS = [(8, 13), (13, 30)]  # μ, β


def load_subject(sid):
    with open(CFG) as f:
        info = yaml.safe_load(f)
    return XWStrokeDataset(subject_id=sid, dataset_info=info, mode="train")


def group_kfold_indices(group_ids, n_folds, seed):
    from sklearn.model_selection import KFold
    uniq = np.unique(group_ids)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_g, te_g in kf.split(uniq):
        tr = np.where(np.isin(group_ids, uniq[tr_g]))[0]
        te = np.where(np.isin(group_ids, uniq[te_g]))[0]
        yield tr, te


def erd_channel_scores(X_tr, y_tr, fs=250):
    """训练折内各通道 ERD 可分性：μ/β 对数带功率与标签的点二列相关绝对值，两带取最大。

    X_tr: (n, 30, T) 已 4-40Hz 滤波。返回 (30,) 分数。只用训练折数据。
    """
    from scipy.signal import welch
    n_ch = X_tr.shape[1]
    scores = np.zeros(n_ch)
    for lo, hi in BANDS:
        # welch 逐窗逐导求带功率（向量化）
        f, psd = welch(X_tr, fs=fs, nperseg=min(256, X_tr.shape[-1]), axis=-1)
        m = (f >= lo) & (f <= hi)
        bp = np.log(psd[..., m].mean(axis=-1) + 1e-12)  # (n, ch)
        y = y_tr.astype(float)
        y = (y - y.mean()) / (y.std() + 1e-12)
        bp_z = (bp - bp.mean(0)) / (bp.std(0) + 1e-12)
        r = np.abs((bp_z * y[:, None]).mean(0))  # 点二列相关
        scores = np.maximum(scores, r)
    return scores


def run_csp(X, y, g, sid, seeds, chan_mode):
    """chan_mode: 'full' | 'fixed4' | 'sel4'"""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline
    from mne.decoding import CSP
    X = X.astype(np.float64)
    rows = []
    for seed in seeds:
        for fi, (tr, te) in enumerate(group_kfold_indices(g, 5, seed + sid)):
            if chan_mode == "full":
                ch = list(range(X.shape[1]))
            elif chan_mode == "fixed4":
                ch = FIXED4
            else:  # sel4：选择仅用训练折
                ch = np.argsort(-erd_channel_scores(X[tr], y[tr]))[:SEL_K].tolist()
            # 分量数：全导 6；4 导时取 4（CSP 要求偶数且 ≤ 通道数）
            n_csp = 6 if len(ch) >= 6 else len(ch) - (len(ch) % 2)
            clf = Pipeline([("csp", CSP(n_components=n_csp, log=True, norm_trace=False)),
                            ("lda", LinearDiscriminantAnalysis())])
            clf.fit(X[tr][:, ch, :], y[tr])
            acc = float(clf.score(X[te][:, ch, :], y[te]))
            rows.append((sid, seed, fi, acc, len(tr), len(te),
                         ",".join(CH_NAMES[c] for c in ch)))
    return rows


def run_eegnet(ds, seeds, epochs=100):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    from models import register_all_models
    from core.registry import MODELS, build_from_config
    register_all_models()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_np = ds.data  # (n, 1, 30, T)
    y_np = ds.labels
    g = ds.group_ids
    n_ch = X_np.shape[2]
    T = X_np.shape[3]
    rows = []
    for seed in seeds:
        for fi, (tr, te) in enumerate(group_kfold_indices(g, 5, seed + ds.subject_id)):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build_from_config({"type": "EEGNet", "args": {
                "num_channels": n_ch, "num_classes": 2, "num_bands": 1,
                "input_length": T}}, MODELS).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            lossf = nn.CrossEntropyLoss()
            tr_loader = DataLoader(Subset(ds, tr.tolist()), batch_size=32,
                                   shuffle=True, num_workers=0)
            Xte = torch.from_numpy(X_np[te]).float().to(device)
            yte = torch.from_numpy(y_np[te]).long().to(device)
            for _ in range(epochs):
                model.train()
                for xb, yb in tr_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = lossf(model(xb), yb)
                    loss.backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                acc = float((model(Xte).argmax(1) == yte).float().mean().item())
            rows.append((ds.subject_id, seed, fi, acc, len(tr), len(te), "all30"))
            print(f"    sub-{ds.subject_id:02d} seed={seed} fold{fi}: {acc*100:.1f}%",
                  flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["csp_full", "csp_fixed4", "csp_sel4", "eegnet_full"])
    ap.add_argument("--seeds", default="42,123,2026")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    os.makedirs(OUT, exist_ok=True)
    out_csv = os.path.join(OUT, f"{args.arm}_fold_results.csv")
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(zip(prev.subject, prev.seed))
        print(f"[resume] {len(done)} (subject, seed) 已完成")

    for sid in SUBJECTS:
        remain = [s for s in seeds if (sid, s) not in done]
        if not remain:
            continue
        t0 = time.time()
        try:
            ds = load_subject(sid)
        except Exception as e:
            print(f"sub-{sid:02d}: LOAD ERROR {e}", flush=True)
            continue
        X = ds.data[:, 0, :, :]  # (n_win, 30, T)
        if args.arm == "eegnet_full":
            rows = run_eegnet(ds, remain)
        else:
            rows = run_csp(X, ds.labels, ds.group_ids, sid, remain,
                           args.arm.replace("csp_", ""))
        m = np.mean([r[3] for r in rows])
        print(f"[{args.arm}] sub-{sid:02d} done: {m*100:.2f}%  ({time.time()-t0:.0f}s)",
              flush=True)
        df = pd.DataFrame(rows, columns=["subject", "seed", "fold", "acc",
                                         "n_train", "n_test", "channels"])
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    print(f"\n===== {args.arm} 完成 =====")


if __name__ == "__main__":
    main()
