"""E1b 升级：同阶段患者源 LOSO few-shot 校准曲线（zero-shot LOSO 即 N=0 点）。

科学问题：Q1 框架的最后一条腿——预训练源该用健康人（E1c+ 已证有效）
还是同阶段患者池？本实验给出患者源的对照曲线。

设计（复用 e1_fewshot/e1c_transfer 机制）：
  - 数据集：xw（50 人急性）、hs（14 人慢性，全 session 混合主口径）
  - 留一法：目标被试留出，其余被试混合（每被试 ≤2000 窗）预训练 EEGNet 100ep
  - 目标被试 N-shot 校准：finetune（lr=1e-3, 50ep）vs scratch（lr=1e-3, 50ep）
  - N ∈ {0, 2, 4, 8, 16, 32}，trial 级分层抽样，余下 trial 测试
  - 3 seeds；增量 CSV + resume；N=0 仅 finetune 条件（= LOSO zero-shot）

输出：
  experiments/e1_acute_vs_chronic/e1b_loso/{dataset}_{arm}.csv
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
from e1c_transfer import DOMAINS, load_domain, MAX_WIN_PER_SUBJECT
from e1_fewshot import (N_LIST, FT_EPOCHS, FT_LR, SCRATCH_LR, PRETRAIN_EPOCHS,
                        stratified_trial_sample, subject_trial_table, windows_of,
                        train_model, evaluate, build_eegnet)

OUT = "experiments/e1_acute_vs_chronic/e1b_loso"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["xw", "hs"])
    ap.add_argument("--arm", required=True, choices=["finetune", "scratch"])
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--max-targets", type=int, default=0,
                    help="冒烟测试用：只跑前 k 个目标被试")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    import torch  # noqa
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    out_csv = os.path.join(OUT, f"{args.dataset}_{args.arm}.csv")

    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(zip(prev.target_subject, prev.seed, prev.n_shot))

    print(f"=== E1b {args.dataset}/{args.arm} seeds={seeds} ===", flush=True)
    domain = load_domain(args.dataset, mode="train")
    if args.max_targets:
        domain = domain[:args.max_targets]
    aug = domain[0][1].augmentor

    for seed in seeds:
        # 逐目标被试：LOSO 源域预训练（scratch 条件不需要，但 N=0 行需与 finetune 对齐跳过）
        for sid_t, ds_t in domain:
            pending = [n for n in N_LIST
                       if (sid_t, seed, n) not in done
                       and not (n == 0 and args.arm == "scratch")]
            if not pending:
                continue  # resume：该 (被试, seed) 已全部完成，跳过昂贵的 LOSO 预训练
            trial_ids, trial_labels = subject_trial_table(ds_t)

            if args.arm == "finetune":
                rng = np.random.RandomState(seed)
                Xs, ys = [], []
                for sid_s, ds_s in domain:
                    if sid_s == sid_t:
                        continue
                    idx = rng.choice(len(ds_s.data),
                                     min(len(ds_s.data), MAX_WIN_PER_SUBJECT),
                                     replace=False)
                    Xs.append(ds_s.data[idx])
                    ys.append(ds_s.labels[idx])
                Xs, ys = np.concatenate(Xs), np.concatenate(ys)
                Xs_in = np.stack([aug.process(x) for x in Xs]) if aug is not None else Xs
                pretrained = build_eegnet(device, seed)
                t0 = time.time()
                train_model(pretrained, Xs_in, ys, PRETRAIN_EPOCHS, 1e-3,
                            device, batch_size=64)
                pretrained_state = {k: v.cpu().clone()
                                    for k, v in pretrained.state_dict().items()}
                print(f"seed={seed} target sub-{sid_t:02d}: "
                      f"LOSO 预训练完成 ({time.time()-t0:.0f}s, {len(Xs_in)} 窗)",
                      flush=True)

            for n in N_LIST:
                if n == 0 and args.arm == "scratch":
                    continue  # N=0 只有 finetune 条件有意义（LOSO zero-shot）
                if (sid_t, seed, n) in done:
                    continue
                rng_s = np.random.RandomState(seed * 1000 + sid_t * 100 + n)
                ft_trials = stratified_trial_sample(trial_labels, trial_ids, n, rng_s)
                ft_mask = windows_of(ds_t, ft_trials)
                te_mask = ~ft_mask
                if te_mask.sum() == 0:
                    continue
                Xte, yte = ds_t.data[te_mask], ds_t.labels[te_mask]

                model = build_eegnet(device, seed)
                if args.arm == "finetune":
                    model.load_state_dict({k: v.to(device)
                                           for k, v in pretrained_state.items()})
                if n > 0:
                    lr = FT_LR if args.arm == "finetune" else SCRATCH_LR
                    train_model(model, ds_t.data[ft_mask], ds_t.labels[ft_mask],
                                FT_EPOCHS, lr, device)

                acc = evaluate(model, Xte, yte, device)
                row = pd.DataFrame([(sid_t, seed, n, acc, int(te_mask.sum()))],
                                   columns=["target_subject", "seed", "n_shot",
                                            "acc", "n_test_windows"])
                row.to_csv(out_csv, mode="a",
                           header=not os.path.exists(out_csv), index=False)
                print(f"  seed={seed} sub-{sid_t:02d} N={n}: {acc*100:.1f}%",
                      flush=True)

    print(f"\n===== {args.dataset}/{args.arm} 完成 =====")


if __name__ == "__main__":
    main()
