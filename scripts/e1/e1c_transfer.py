"""E1c：急性（XWStroke）↔ 慢性（HS Stroke）双向跨阶段迁移。

用法：
  python scripts/e1/e1c_transfer.py --direction xw2hs --arm csp
  python scripts/e1/e1c_transfer.py --direction hs2xw --arm eegnet --seeds 42,123,2026

设计：
  - 源域：全部源被试数据混合训练一个模型（EEGNet fixed 100ep / CSP+LDA）
  - 目标域：逐被试测试，报 per-subject 准确率
  - 协议与 E1a 完全一致：7 通道、250Hz、2s 窗、增强仅源域训练侧、EEGNet 3 seeds
  - HS 源域每被试 ~8000 窗，为控制训练规模每被试每 seed 随机抽 2000 窗（保持被试均衡）

输出：
  experiments/e1_acute_vs_chronic/e1c_transfer/{direction}_{arm}.csv
  （列：target_subject, seed, acc, n_test）
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

OUT = "experiments/e1_acute_vs_chronic/e1c_transfer"
MAX_WIN_PER_SUBJECT = 2000  # 源域每被试窗数上限（HS 全量 ~8000/人，控制训练规模）

DOMAINS = {
    "xw": ("configs/dataset/XWStroke_7ch_hsalign.yaml", XWStrokeDataset, list(range(1, 51))),
    "hs": ("configs/dataset/HSStroke_7ch_250hz.yaml", HSStrokeDataset, list(range(1, 15))),
}
DIRECTIONS = {"xw2hs": ("xw", "hs"), "hs2xw": ("hs", "xw")}


def load_subject(cls, cfg_path, sid, mode):
    with open(cfg_path) as f:
        info = yaml.safe_load(f)
    return cls(subject_id=sid, dataset_info=info, mode=mode)


def load_domain(name, mode):
    """加载整域：返回 per-subject [(sid, ds)] 列表"""
    cfg, cls, sids = DOMAINS[name]
    out = []
    for sid in sids:
        try:
            ds = load_subject(cls, cfg, sid, mode)
            out.append((sid, ds))
            print(f"  [{name}] sub-{sid:02d} loaded: {ds.data.shape}", flush=True)
        except Exception as e:
            print(f"  [{name}] sub-{sid:02d}: LOAD ERROR {e}", flush=True)
    return out


def run_csp(source, target, seed):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline
    from mne.decoding import CSP
    rng = np.random.RandomState(seed)
    Xs, ys = [], []
    for sid, ds in source:
        X = ds.data[:, 0, :, :].astype(np.float64)
        idx = rng.choice(len(X), min(len(X), MAX_WIN_PER_SUBJECT), replace=False)
        Xs.append(X[idx])
        ys.append(ds.labels[idx])
    Xs = np.concatenate(Xs)
    ys = np.concatenate(ys)
    clf = Pipeline([("csp", CSP(n_components=6, log=True, norm_trace=False)),
                    ("lda", LinearDiscriminantAnalysis())])
    clf.fit(Xs, ys)
    rows = []
    for sid, ds in target:
        Xt = ds.data[:, 0, :, :].astype(np.float64)
        acc = float(clf.score(Xt, ds.labels))
        rows.append((sid, seed, acc, len(Xt)))
        print(f"  target sub-{sid:02d}: {acc*100:.1f}%", flush=True)
    return rows


def run_eegnet(source, target, seeds, epochs=100):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from models import register_all_models
    from core.registry import MODELS, build_from_config
    register_all_models()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    # 目标域 tensor 预先备好（无增强）
    target_tensors = [(sid, torch.from_numpy(ds.data).float(),
                       torch.from_numpy(ds.labels).long()) for sid, ds in target]
    for seed in seeds:
        rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        # 源域：每被试抽样后合并；训练时手动应用与 E1a 相同的增强
        Xs, ys = [], []
        for sid, ds in source:
            idx = rng.choice(len(ds.data), min(len(ds.data), MAX_WIN_PER_SUBJECT), replace=False)
            Xs.append(ds.data[idx])
            ys.append(ds.labels[idx])
        Xs = np.concatenate(Xs)
        ys = np.concatenate(ys)
        # 用源域 dataset 的增强器逐样本增强（与 E1a 训练侧一致）
        aug = source[0][1].augmentor
        if aug is not None:
            Xs = np.stack([aug.process(x) for x in Xs])
        loader = DataLoader(TensorDataset(torch.from_numpy(Xs).float(),
                                          torch.from_numpy(ys).long()),
                            batch_size=64, shuffle=True, num_workers=0)
        model = build_from_config({"type": "EEGNet", "args": {
            "num_channels": 7, "num_classes": 2, "num_bands": 1, "input_length": 500}}, MODELS).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        t0 = time.time()
        for ep in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = lossf(model(xb), yb)
                loss.backward()
                opt.step()
            if (ep + 1) % 25 == 0:
                print(f"  seed={seed} epoch {ep+1}/{epochs} ({time.time()-t0:.0f}s)", flush=True)
        model.eval()
        for sid, Xt, yt in target_tensors:
            with torch.no_grad():
                preds = []
                for i in range(0, len(Xt), 512):
                    preds.append(model(Xt[i:i+512].to(device)).argmax(1).cpu())
                acc = float((torch.cat(preds) == yt).float().mean().item())
            rows.append((sid, seed, acc, len(Xt)))
        print(f"seed={seed} done ({time.time()-t0:.0f}s)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True, choices=list(DIRECTIONS.keys()))
    ap.add_argument("--arm", required=True, choices=["csp", "eegnet"])
    ap.add_argument("--seeds", default="42")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    src_name, tgt_name = DIRECTIONS[args.direction]
    os.makedirs(OUT, exist_ok=True)
    out_csv = os.path.join(OUT, f"{args.direction}_{args.arm}.csv")

    done_seeds = set()
    if os.path.exists(out_csv):
        done_seeds = set(pd.read_csv(out_csv)["seed"].unique())
    remain = [s for s in seeds if s not in done_seeds]
    if not remain:
        print(f"[resume] {args.direction}/{args.arm} 全部 seed 已完成")
        return

    print(f"=== E1c {args.direction} {args.arm} seeds={remain} ===", flush=True)
    source = load_domain(src_name, mode="train")
    target = load_domain(tgt_name, mode="test")

    if args.arm == "csp":
        rows = []
        for seed in remain:
            rows.extend(run_csp(source, target, seed))
    else:
        rows = run_eegnet(source, target, remain)

    df = pd.DataFrame(rows, columns=["target_subject", "seed", "acc", "n_test"])
    df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
    m = df.groupby("target_subject")["acc"].mean()
    print(f"\n===== {args.direction}/{args.arm} 完成: {m.mean()*100:.2f}% ± {m.std()*100:.2f}% =====")


if __name__ == "__main__":
    main()
