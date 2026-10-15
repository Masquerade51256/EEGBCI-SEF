"""E1a：两数据集 within-subject 对比（统一 7 通道干净协议）。

用法：
  python scripts/e1/e1a_within.py --arm csp    --dataset xw|hs_all|hs_ses01
  python scripts/e1/e1a_within.py --arm eegnet --dataset xw|hs_all|hs_ses01 --seeds 42,123,2026

协议（干净协议铁律）：
  - GroupKFold 5-fold，按 trial 分组（group_ids），同一 trial 的 5 个重叠窗不跨折
  - EEGNet：fixed 100 epochs，报**最终 epoch**测试成绩（禁止 best-epoch 选择）
  - 增强仅训练折（dataset mode='train' 的 __getitem__ 增强；验证折直接取 numpy 绕过）
  - CSP+LDA：6 分量，4-40Hz（数据已被 pipeline 滤波+zscore），无增强

输出：
  experiments/e1_acute_vs_chronic/e1a_within/{dataset}/{arm}_fold_results.csv
  （列：subject, seed, fold, acc, n_train, n_test）
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, ".")
from data.datasets import XWStrokeDataset, HSStrokeDataset

OUT = "experiments/e1_acute_vs_chronic/e1a_within"

DATASETS = {
    "xw": ("configs/dataset/XWStroke_7ch_hsalign.yaml", XWStrokeDataset, list(range(1, 51)), None),
    "hs_all": ("configs/dataset/HSStroke_7ch_250hz.yaml", HSStrokeDataset, list(range(1, 15)), None),
    "hs_ses01": ("configs/dataset/HSStroke_7ch_250hz.yaml", HSStrokeDataset, list(range(1, 15)), [1]),
}


def load_subject(cls, cfg_path, sid, sessions, mode):
    with open(cfg_path) as f:
        info = yaml.safe_load(f)
    if sessions is not None:
        info["dataset"]["sessions"] = sessions
    return cls(subject_id=sid, dataset_info=info, mode=mode)


def group_kfold_indices(group_ids, n_folds, seed):
    from sklearn.model_selection import KFold
    uniq = np.unique(group_ids)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_g, te_g in kf.split(uniq):
        tr = np.where(np.isin(group_ids, uniq[tr_g]))[0]
        te = np.where(np.isin(group_ids, uniq[te_g]))[0]
        yield tr, te


def run_csp(ds, seeds=(42,)):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline
    from mne.decoding import CSP
    X = ds.data[:, 0, :, :].astype(np.float64)  # (n_win, 7, 500)，mne CSP 要求 float64
    y = ds.labels
    g = ds.group_ids
    rows = []
    for seed in seeds:
        for fi, (tr, te) in enumerate(group_kfold_indices(g, 5, seed + ds.subject_id)):
            clf = Pipeline([("csp", CSP(n_components=6, log=True, norm_trace=False)),
                            ("lda", LinearDiscriminantAnalysis())])
            clf.fit(X[tr], y[tr])
            acc = float(clf.score(X[te], y[te]))
            rows.append((ds.subject_id, seed, fi, acc, len(tr), len(te)))
    return rows


def run_eegnet(ds_train_mode, data_np, labels, group_ids, seeds, epochs=100):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset, TensorDataset
    from core.registry import MODELS, build_from_config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_np = data_np  # (n, 1, 7, 500)
    y_np = labels
    g = group_ids
    rows = []
    for seed in seeds:
        for fi, (tr, te) in enumerate(group_kfold_indices(g, 5, seed + ds_train_mode.subject_id)):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build_from_config({"type": "EEGNet", "args": {
                "num_channels": 7, "num_classes": 2, "num_bands": 1, "input_length": 500}}, MODELS).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            lossf = nn.CrossEntropyLoss()
            # 训练折走 dataset.__getitem__（mode='train' 带增强）；验证折直接用 numpy（无增强）
            tr_loader = DataLoader(Subset(ds_train_mode, tr.tolist()), batch_size=32, shuffle=True, num_workers=0)
            Xte = torch.from_numpy(X_np[te]).float().to(device)
            yte = torch.from_numpy(y_np[te]).long().to(device)
            for ep in range(epochs):
                model.train()
                for xb, yb in tr_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = lossf(model(xb), yb)
                    loss.backward()
                    opt.step()
            # fixed-epoch：报最终 epoch
            model.eval()
            with torch.no_grad():
                acc = float((model(Xte).argmax(1) == yte).float().mean().item())
            rows.append((ds_train_mode.subject_id, seed, fi, acc, len(tr), len(te)))
            print(f"    sub-{ds_train_mode.subject_id:02d} seed={seed} fold{fi}: {acc*100:.1f}%", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["csp", "eegnet"])
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--seeds", default="42", help="逗号分隔，如 42,123,2026")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    cfg, cls, sids, sessions = DATASETS[args.dataset]
    out_dir = os.path.join(OUT, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{args.arm}_fold_results.csv")

    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(zip(prev.subject, prev.seed))
        print(f"[resume] {len(done)} (subject, seed) 已完成")

    all_rows = []
    for sid in sids:
        remain = [s for s in seeds if (sid, s) not in done]
        if not remain:
            continue
        t0 = time.time()
        try:
            ds = load_subject(cls, cfg, sid, sessions, mode="train")
        except Exception as e:
            print(f"sub-{sid:02d}: LOAD ERROR {e}", flush=True)
            continue
        if args.arm == "csp":
            rows = run_csp(ds, remain)
        else:
            rows = run_eegnet(ds, ds.data, ds.labels, ds.group_ids, remain)
        all_rows.extend(rows)
        m = np.mean([r[3] for r in rows])
        print(f"[{args.dataset}/{args.arm}] sub-{sid:02d} done: {m*100:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
        # 增量落盘
        df_new = pd.DataFrame(rows, columns=["subject", "seed", "fold", "acc", "n_train", "n_test"])
        df_new.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    print(f"\n===== {args.dataset}/{args.arm} 完成 =====")


if __name__ == "__main__":
    main()
