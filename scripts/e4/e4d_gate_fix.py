"""E4 第 1.5 步：门控路由修复实验（gate_fix 条件）。

设计文档：E4_实验设计_草案_20261105.md §11（2026-11-08 用户确认）。
背景：第 1 步（e4c）门控未实现逐被试分化——50 被试 w_osc 全部收敛于
0.25–0.42 窄带，与 E2 族敏感性画像零相关。失效定位于门控机制与训练策略。

与 e4c gate_l01 的两处差异（其余完全同协议）：

1. 门控蒸馏监督：联合阶段用两支辅助头对训练 batch 的逐 trial 正确性在线构造
   目标权重——仅一支正确 → 该支 0.9；同对/同错 → 0.5/0.5；
   L_gate = MSE(w, target)，系数 BETA=1.0。
2. 温度退火：门控 softmax 温度 τ 在联合 100 epoch 内从 2.0 线性退火至 0.5，
   解除权重表达范围限制。

评估新增：保存测试折逐 trial 的三头预测与门控权重
（gate_fix_test_preds.csv），分支级 oracle 作为副产品回收。

判读线（事先锁定，见草案 §11）：
  ① vs gate_l01 配对 Wilcoxon 显著为正；② 与 broad（56.48%）差距收窄；
  ③ w_osc 跨被试 SD 增大、w_osc×(CSP−SCP) 相关转正；④ 门控利用率缺口。

用法：
  python scripts/e4/e4d_gate_fix.py --subjects 1      # 冒烟
  python scripts/e4/e4d_gate_fix.py                    # 全量 50 被试
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.e4.e4c_dualbranch_gate import (
    DATASETS, prepare_inputs, build_model, augment,
    WARMUP_EPOCHS, JOINT_EPOCHS, ALPHA, BATCH)
from scripts.e2.e2a_spatial import group_kfold_indices

OUT = "experiments/e4_step15_gatefix"

TAU_START, TAU_END = 2.0, 0.5     # 温度退火
BETA = 1.0                        # 蒸馏监督系数
LAMBDA = 0.01                     # 批均值均衡项（与 gate_l01 一致）
W_CORRECT = 0.9                   # 仅一支正确时的目标权重


def run_gate_fix(Xw, Xs, y, sid, seeds):
    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    n = len(y)
    g = np.arange(n)
    rows, pred_rows = [], []
    for seed in seeds:
        for fi, (tr, te) in enumerate(group_kfold_indices(g, 5, seed + sid)):
            torch.manual_seed(seed)
            np.random.seed(seed)
            gen = torch.Generator().manual_seed(seed * 1000 + fi)
            model = build_model(Xw.shape[2], Xw.shape[1],
                                Xw.shape[-1], Xs.shape[-1]).to(device)
            lossf = nn.CrossEntropyLoss()
            ytr = torch.from_numpy(y[tr]).long()
            tr_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.from_numpy(Xw[tr]), torch.from_numpy(Xs[tr]), ytr),
                batch_size=BATCH, shuffle=True, num_workers=0)
            Xw_te = torch.from_numpy(Xw[te]).to(device)
            Xs_te = torch.from_numpy(Xs[te]).to(device)
            yte = torch.from_numpy(y[te]).long().to(device)

            # ---- 阶段 1：分支预热（与 e4c 一致）----
            for branch, head in (("osc", "head_osc"), ("slow", "head_slow")):
                params = list(getattr(model, branch).parameters()) + \
                         list(getattr(model, head).parameters())
                opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
                for _ in range(WARMUP_EPOCHS):
                    model.train()
                    for xb_w, xb_s, yb in tr_loader:
                        xb_w, xb_s, yb = xb_w.to(device), xb_s.to(device), yb.to(device)
                        xb_w, xb_s = augment(xb_w, xb_s, gen)
                        opt.zero_grad()
                        out = (model.head_osc(model.osc(xb_w)) if branch == "osc"
                               else model.head_slow(model.slow(xb_s)))
                        loss = lossf(out, yb)
                        loss.backward()
                        opt.step()

            # ---- 阶段 2：联合训练（蒸馏监督 + 温度退火）----
            branch_params = list(model.osc.parameters()) + list(model.slow.parameters()) + \
                list(model.head_fuse.parameters()) + list(model.head_osc.parameters()) + \
                list(model.head_slow.parameters())
            params = [{"params": branch_params, "lr": 3e-4},
                      {"params": model.gate.parameters(), "lr": 1e-3}]
            opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-2)
            for ep in range(JOINT_EPOCHS):
                model.gate.tau = TAU_START + (TAU_END - TAU_START) * ep / (JOINT_EPOCHS - 1)
                model.train()
                for xb_w, xb_s, yb in tr_loader:
                    xb_w, xb_s, yb = xb_w.to(device), xb_s.to(device), yb.to(device)
                    xb_w, xb_s = augment(xb_w, xb_s, gen)
                    opt.zero_grad()
                    lf, lo, ls, w = model(xb_w, xb_s, mode="gate")
                    loss = lossf(lf, yb) + ALPHA * (lossf(lo, yb) + lossf(ls, yb))
                    # 批均值均衡项（λ=0.01，与 gate_l01 一致）
                    wm = w.mean(0)
                    loss = loss + LAMBDA * ((wm - 0.5) ** 2).sum()
                    # 门控蒸馏监督：目标权重由两支逐 trial 正确性在线构造
                    with torch.no_grad():
                        co = lo.argmax(1) == yb
                        cs = ls.argmax(1) == yb
                        t = torch.full((len(yb), 2), 0.5, device=device)
                        only_o = co & ~cs
                        only_s = cs & ~co
                        t[only_o] = torch.tensor([W_CORRECT, 1 - W_CORRECT], device=device)
                        t[only_s] = torch.tensor([1 - W_CORRECT, W_CORRECT], device=device)
                    loss = loss + BETA * ((w - t) ** 2).mean()
                    loss.backward()
                    opt.step()

            # ---- 评估（trial 级，保存逐 trial 预测与权重）----
            model.eval()
            with torch.no_grad():
                lf, lo, ls, w = model(Xw_te, Xs_te, mode="gate")
                pf, po, ps = lf.argmax(1), lo.argmax(1), ls.argmax(1)
                acc = float((pf == yte).float().mean().item())
                acc_o = float((po == yte).float().mean().item())
                acc_s = float((ps == yte).float().mean().item())
                wo, ws = float(w[:, 0].mean().item()), float(w[:, 1].mean().item())
                for j in range(len(te)):
                    pred_rows.append((sid, seed, fi, int(te[j]), int(yte[j].item()),
                                      int(pf[j].item()), int(po[j].item()),
                                      int(ps[j].item()),
                                      float(w[j, 0].item()), float(w[j, 1].item())))
            rows.append((sid, seed, fi, acc, acc_o, acc_s, wo, ws,
                         len(tr), len(te), "gate_fix"))
            print(f"    sub-{sid:02d} seed={seed} fold{fi}: fuse={acc*100:.1f}% "
                  f"osc={acc_o*100:.1f}% slow={acc_s*100:.1f}% w_osc={wo:.2f}",
                  flush=True)
    return rows, pred_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xw", choices=list(DATASETS.keys()))
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--subjects", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    dspec = DATASETS[args.dataset]
    subjects = dspec["subjects"] if args.subjects is None else \
        [int(s) for s in args.subjects.split(",")]

    out_dir = os.path.join(OUT, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "gate_fix_fold_results.csv")
    pred_csv = os.path.join(out_dir, "gate_fix_test_preds.csv")
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(zip(prev.subject, prev.seed))
        print(f"[resume] {len(done)} (subject, seed) 已完成")

    for sid in subjects:
        remain = [s for s in seeds if (sid, s) not in done]
        if not remain:
            continue
        t0 = time.time()
        try:
            X, y = dspec["loader"](sid)
        except Exception as e:
            print(f"sub-{sid:02d}: LOAD ERROR {e}", flush=True)
            continue
        Xw, Xs = prepare_inputs(X, dspec["sr"])
        rows, pred_rows = run_gate_fix(Xw, Xs, y, sid, remain)
        m = np.mean([r[3] for r in rows])
        print(f"[{args.dataset}/gate_fix] sub-{sid:02d} done: {m*100:.2f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows, columns=["subject", "seed", "fold", "acc",
                                         "acc_osc", "acc_slow",
                                         "gate_osc_mean", "gate_slow_mean",
                                         "n_train", "n_test", "channels"])
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
        dp = pd.DataFrame(pred_rows, columns=["subject", "seed", "fold", "trial",
                                              "y", "pred_fuse", "pred_osc", "pred_slow",
                                              "w_osc", "w_slow"])
        dp.to_csv(pred_csv, mode="a", header=not os.path.exists(pred_csv), index=False)

    print(f"\n===== {args.dataset}/gate_fix 完成 =====")


if __name__ == "__main__":
    main()
