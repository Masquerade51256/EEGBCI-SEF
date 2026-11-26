"""E4 第 3 步 S3-D v2a：教师降维版蒸馏分支（wPLI 220 维 → 折内 PCA K 维）。

背景（E4 记录 §8.5，2026-11-26 用户确认"救一下"）：
保真度诊断坐实 v1 失效模式——280 维回归目标在 32 训练 trial 下被纯记忆化
（训练 R²=0.996，测试 0/50 被试 R²>0）。v2a 唯一改动：wPLI 教师经**折内
PCA**（只在训练折内拟合，守折纪律）压缩至 K 维；非周期 60 维保持原样
（保真度诊断中唯一留有痕迹的成分：r_s1_ap × 单体成绩 ρ=0.406, p=0.003）。
回归目标 280 → 80 维；架构、两阶段训练、融合/单体、协议全部与 e4n 相同，
单因子可归因。

判读线（2026-11-26 事先锁定）：
  ① 阶段 1 测试折 wPLI R² > 0（e4o 保真度脚本检验，--arm S3DPCA --teacher pca）；
     ①不过则直接归档，不讨论融合；
  ② ①通过后：融合在独有选区 8 人（sub-09/10/11/24/37/39/43/47）显著为正。

输出：experiments/e4_step3/{dataset}/S3DPCA_fold_results.csv（单体 S3DPCAonly_*）
用法：
  python scripts/e4/e4p_distill2_pca.py --dataset bcic --subjects 1 --seeds 42
  bash scripts/e4/run_e4p.sh        # BCIC 效度
  bash scripts/e4/run_e4p_xw.sh     # XW 主实验
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.e4.e4a_broadband_veto import LOADERS
from scripts.e4.e4e_step2_factorial import make_twoband
from scripts.e4.e4n_step3_distill import (REGIONS, teacher_features,
                                          build_s3d_model)
from scripts.e2.e2a_spatial import group_kfold_indices

OUT = "experiments/e4_step3"
ARM = "S3DPCA"
PCA_K = 20  # wPLI 教师主成分数（判读线①锁定的唯一改动量）


def apply_teacher_pca(Tz, tr, ap_dim, k=PCA_K):
    """wPLI 块折内 PCA：只在训练折内拟合，返回 (Tz_compressed, 解释方差比)。"""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=k)
    pca.fit(Tz[tr, ap_dim:])
    ev = float(pca.explained_variance_ratio_.sum())
    Tz2 = np.concatenate([Tz[:, :ap_dim],
                          pca.transform(Tz[:, ap_dim:])], axis=1)
    return Tz2.astype(np.float32), ev


def run_subject(Xs, y, Tch, sid, seeds, out_dir, distill_only=False):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    n_ch, T = Xs.shape[2], Xs.shape[-1]
    ap_dim = 2 * n_ch
    n_teacher = ap_dim + PCA_K
    tidx = np.arange(len(y))
    rows, pred_rows = [], []
    for seed in seeds:
        for fi, (tr, te) in enumerate(group_kfold_indices(tidx, 5, seed + sid)):
            torch.manual_seed(seed)
            np.random.seed(seed)
            pre_cls, NetCls = build_s3d_model(n_ch, T, n_teacher,
                                              distill_only=distill_only)
            pre = pre_cls.to(device)
            # 教师 z-score：只用训练折统计量；随后 wPLI 块折内 PCA
            mu = Tch[tr].mean(0, keepdims=True)
            sd = Tch[tr].std(0, keepdims=True) + 1e-8
            Tz = (Tch - mu) / sd
            Tz, ev = apply_teacher_pca(Tz, tr, ap_dim)
            # —— 阶段 1：无标签模仿预训练 ——
            opt = torch.optim.Adam(pre.parameters(), lr=1e-3)
            lossf = nn.MSELoss()
            loader = DataLoader(TensorDataset(torch.from_numpy(Xs[tr]),
                                              torch.from_numpy(Tz[tr]).float()),
                                batch_size=16, shuffle=True, num_workers=0)
            for _ in range(100):
                pre.train()
                for xb, tb in loader:
                    xb, tb = xb.to(device), tb.to(device)
                    opt.zero_grad()
                    loss = lossf(pre(xb), tb)
                    loss.backward()
                    opt.step()
            # —— 阶段 2：任务微调（融合或单体）——
            model = NetCls(pre.trunk).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            lossf = nn.CrossEntropyLoss()
            loader = DataLoader(TensorDataset(torch.from_numpy(Xs[tr]),
                                              torch.from_numpy(y[tr]).long()),
                                batch_size=16, shuffle=True, num_workers=0)
            Xte = torch.from_numpy(Xs[te]).to(device)
            yte = torch.from_numpy(y[te]).long().to(device)
            for _ in range(100):
                model.train()
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = lossf(model(xb), yb)
                    loss.backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                logits = model(Xte)
                pred = logits.argmax(1)
                acc = float((pred == yte).float().mean().item())
            pred_np, y_np = pred.cpu().numpy(), yte.cpu().numpy()
            for j in range(len(te)):
                pred_rows.append((sid, seed, fi, int(te[j]), int(y_np[j]),
                                  int(pred_np[j])))
            rows.append((sid, seed, fi, acc, len(tr), len(te), ARM, ev))
            print(f"    sub-{sid:02d} seed={seed} fold{fi}: {acc*100:.1f}% "
                  f"(pca_ev={ev:.2f})", flush=True)
            mdir = os.path.join(out_dir, "models")
            os.makedirs(mdir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(mdir,
                                    f"{ARM}_sub{sid:02d}_seed{seed}_fold{fi}.pt"))
    return rows, pred_rows


def main():
    global ARM
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xw", choices=list(LOADERS.keys()))
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--subjects", default=None)
    ap.add_argument("--distill-only", action="store_true",
                    help="蒸馏分支单体（ARM=S3DPCAonly），无 B 主干")
    args = ap.parse_args()
    if args.distill_only:
        ARM = "S3DPCAonly"
    seeds = [int(s) for s in args.seeds.split(",")]
    dspec = LOADERS[args.dataset]
    subjects = dspec["subjects"] if args.subjects is None else \
        [int(s) for s in args.subjects.split(",")]

    out_dir = os.path.join(OUT, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{ARM}_fold_results.csv")
    pred_csv = os.path.join(out_dir, f"{ARM}_test_preds.csv")
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
        Xs = make_twoband(X, dspec["sr"])
        Tch = teacher_features(X, args.dataset, dspec["sr"]).astype(np.float32)
        rows, pred_rows = run_subject(Xs, y.astype(int), Tch, sid, remain,
                                      out_dir, distill_only=args.distill_only)
        m = np.mean([r[3] for r in rows])
        print(f"[{args.dataset}/{ARM}] sub-{sid:02d} done: {m*100:.2f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows, columns=["subject", "seed", "fold", "acc",
                                         "n_train", "n_test", "channels",
                                         "pca_ev"])
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv),
                  index=False)
        dp = pd.DataFrame(pred_rows, columns=["subject", "seed", "fold",
                                              "trial", "y", "pred"])
        dp.to_csv(pred_csv, mode="a", header=not os.path.exists(pred_csv),
                  index=False)

    print(f"\n===== {args.dataset}/{ARM} 完成 =====")


if __name__ == "__main__":
    main()
