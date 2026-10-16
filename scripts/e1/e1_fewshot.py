"""E1g few-shot：跨阶段迁移的 N-shot 校准曲线（预训练微调 vs 从零训练）。

科学问题：预训练（源阶段全域数据）到底值多少？
  - arm=finetune：源域预训练 EEGNet（同 E1c 协议）→ 目标被试 N 条有标签 trial 全参数微调
  - arm=scratch ：同架构随机初始化 → 同样 N 条 trial 从零训练
  两条曲线之差 = 预训练的净增益。

协议：
  - N ∈ {0, 2, 4, 8, 16, 32}，按类别分层抽样（每类 N/2），trial 级抽样（其 5 窗全入训练）
  - 测试集 = 该被试剩余全部 trial 的窗；N=0 即 zero-shot（应复现 E1c ~50%）
  - 微调 lr=1e-4、50 epochs fixed、无增强；从零 lr=1e-3、50 epochs fixed、无增强
  - 3 seeds（抽样与初始化）；两方向（xw2hs / hs2xw）
  - XW 目标仅 40 trials：N=32 时测试仅剩 8 trials，报数时注明

输出：
  experiments/e1_acute_vs_chronic/e1_fewshot/{direction}_{arm}.csv
  列：target_subject, seed, n_shot, acc, n_test_windows
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "scripts/e1")
from e1c_transfer import DOMAINS, load_domain  # 复用域加载（无 EA 主口径）

OUT = "experiments/e1_acute_vs_chronic/e1_fewshot"
N_LIST = [0, 2, 4, 8, 16, 32]
FT_EPOCHS = 50
PRETRAIN_EPOCHS = 100


def stratified_trial_sample(labels_trials, trial_ids, n, rng):
    """从 trial 级按类分层抽 n 条，返回选中的 trial id 集合"""
    if n == 0:
        return set()
    per_class = n // 2
    chosen = []
    for c in [0, 1]:
        pool = trial_ids[labels_trials == c]
        chosen.extend(rng.choice(pool, min(per_class, len(pool)), replace=False).tolist())
    return set(chosen)


def subject_trial_table(ds):
    """返回 (trial_ids, trial_labels)：每 trial 一行"""
    groups, labels = ds.group_ids, ds.labels
    uniq = np.unique(groups)
    lab = np.array([labels[groups == g][0] for g in uniq])
    return uniq, lab


def windows_of(ds, trial_set):
    mask = np.isin(ds.group_ids, list(trial_set)) if trial_set else np.zeros(len(ds.labels), bool)
    return mask


def train_model(model, X, y, epochs, lr, device, batch_size=32):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(torch.from_numpy(X).float(),
                                      torch.from_numpy(y).long()),
                        batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def evaluate(model, X, y, device):
    import torch
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(X), 512):
            preds.append(model(torch.from_numpy(X[i:i+512]).float().to(device)).argmax(1).cpu())
        return float((torch.cat(preds) == torch.from_numpy(y).long()).float().mean().item())


def build_eegnet(device, seed):
    import torch
    from models import register_all_models
    from core.registry import MODELS, build_from_config
    register_all_models()
    torch.manual_seed(seed)
    np.random.seed(seed)
    return build_from_config({"type": "EEGNet", "args": {
        "num_channels": 7, "num_classes": 2, "num_bands": 1, "input_length": 500}}, MODELS).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True, choices=["xw2hs", "hs2xw"])
    ap.add_argument("--arm", required=True, choices=["finetune", "scratch"])
    ap.add_argument("--seeds", default="42,123,2026")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    import torch  # noqa
    device = "cuda" if torch.cuda.is_available() else "cpu"
    src_name, tgt_name = {"xw2hs": ("xw", "hs"), "hs2xw": ("hs", "xw")}[args.direction]
    os.makedirs(OUT, exist_ok=True)
    out_csv = os.path.join(OUT, f"{args.direction}_{args.arm}.csv")

    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(zip(prev.target_subject, prev.seed, prev.n_shot))

    print(f"=== E1g {args.direction}/{args.arm} seeds={seeds} ===", flush=True)
    print("加载源域...", flush=True)
    source = load_domain(src_name, mode="train")
    print("加载目标域...", flush=True)
    target = load_domain(tgt_name, mode="test")

    # 源域预训练数据（与 E1c 相同的均衡抽样）
    MAX_WIN = 2000
    for seed in seeds:
        rng = np.random.RandomState(seed)
        Xs, ys = [], []
        for sid, ds in source:
            idx = rng.choice(len(ds.data), min(len(ds.data), MAX_WIN), replace=False)
            Xs.append(ds.data[idx])
            ys.append(ds.labels[idx])
        Xs, ys = np.concatenate(Xs), np.concatenate(ys)

        if args.arm == "finetune":
            aug = source[0][1].augmentor
            Xs_in = np.stack([aug.process(x) for x in Xs]) if aug is not None else Xs
            print(f"seed={seed}: 预训练源域模型 ({len(Xs_in)} 窗)...", flush=True)
            pretrained = build_eegnet(device, seed)
            t0 = time.time()
            train_model(pretrained, Xs_in, ys, PRETRAIN_EPOCHS, 1e-3, device, batch_size=64)
            pretrained_state = {k: v.cpu().clone() for k, v in pretrained.state_dict().items()}
            print(f"  预训练完成 ({time.time()-t0:.0f}s)", flush=True)

        # 逐目标被试 × N-shot
        for sid, ds in target:
            trial_ids, trial_labels = subject_trial_table(ds)
            for n in N_LIST:
                if n == 0 and args.arm == "scratch":
                    continue  # 从零训练的 N=0 无意义（随机权重），zero-shot 点仅 finetune 臂有
                if (sid, seed, n) in done:
                    continue
                rng_s = np.random.RandomState(seed * 1000 + sid * 100 + n)
                ft_trials = stratified_trial_sample(trial_labels, trial_ids, n, rng_s)
                ft_mask = windows_of(ds, ft_trials)
                te_mask = ~ft_mask
                if te_mask.sum() == 0:
                    continue
                Xte, yte = ds.data[te_mask], ds.labels[te_mask]

                model = build_eegnet(device, seed)
                if n > 0:
                    if args.arm == "finetune":
                        model.load_state_dict({k: v.to(device) for k, v in pretrained_state.items()})
                        lr = 1e-4
                    else:
                        lr = 1e-3
                    Xtr, ytr = ds.data[ft_mask], ds.labels[ft_mask]
                    train_model(model, Xtr, ytr, FT_EPOCHS, lr, device)
                # n=0 的 finetune 臂 = zero-shot（加载预训练权重直接测）
                elif args.arm == "finetune":
                    model.load_state_dict({k: v.to(device) for k, v in pretrained_state.items()})

                acc = evaluate(model, Xte, yte, device)
                row = pd.DataFrame([(sid, seed, n, acc, int(te_mask.sum()))],
                                   columns=["target_subject", "seed", "n_shot", "acc", "n_test_windows"])
                row.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
                print(f"  seed={seed} sub-{sid:02d} N={n}: {acc*100:.1f}%", flush=True)

    print(f"\n===== {args.direction}/{args.arm} 完成 =====")


if __name__ == "__main__":
    main()
