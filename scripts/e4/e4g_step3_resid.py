"""E4 第 3 步条件 S3-R：B 主干 + 残差分支（无尺度先验通用 CNN）。

正式设计：E4_实验设计_草案_20261105.md §13（2026-11-10 用户确认）。
定位：翻译表"自由派"分支——不为任何手工族定制结构，给网络一个通用
分支去吸收主族（振荡/慢波/连接）之外"翻译不了但真实存在"的信息形式，
服务对象是独有选区 8 人（sub-09/10/11/24/37/39/43/47，名单锁定）。

架构（融合 = 拼接自由混合，无门控）：
  输入 (n, 2, ch, T) 双带（同第 2 步 B）
   ├─ B 主干：与 e4e 条件 B 逐层一致
   └─ 残差分支：双带全量输入（B, 2, ch, T）
        → 时域卷积 6 滤波器 × k=63（0.25 s，通用小核，不带慢/快尺度先验）
        → BN → ELU → 空间 depthwise → BN → ELU → AvgPool(1,4) → Dropout
        → 二级时域可分离卷积 k=15 → BN → ELU → AvgPool(1,8) → flatten
      （容量取 §7.5 最低档，~2.5k 参数）
   concat(feat_B, feat_res) → Dropout → Linear(→2)

与振荡支的区分（事先声明，防"重复造 EEGNet"质疑）：
  - 输入为 0.1–40 Hz 全信息（双带全量），不限于振荡带；
  - 无 log-方差池化（不对应功率族语义）、无滑窗/尺度切分；
  - 允许网络自行发现双谱型跨频耦合、非周期谱形、复杂度等
    无手工族对应的信息形态。

训练口径与第 0/1/1.5/2 步完全一致：100 epoch、Adam lr=1e-3、batch 16、
CE、无增强；GroupKFold 5 折按 trial × 3 seeds（seed+sid 同前划分）。

输出：experiments/e4_step3/{dataset}/
  S3R_fold_results.csv / S3R_test_preds.csv
  models/S3R_sub*_seed*_fold*.pt   state_dict（遮蔽分析用）

用法：
  python scripts/e4/e4g_step3_resid.py --dataset bcic --subjects 1 --seeds 42  # 冒烟
  bash scripts/e4/run_e4g.sh                                                 # BCIC 效度
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
from scripts.e2.e2a_spatial import group_kfold_indices

OUT = "experiments/e4_step3"
ARM = "S3R"


def build_s3r_model(n_ch, T):
    """B 主干 + 残差分支（通用 CNN），拼接融合。"""
    import torch
    import torch.nn as nn
    from models.layers import Conv2dWithConstraint

    F1, D, T2, P1, P2, DROP = 8, 2, 30, 4, 8, 0.25
    FR = 6  # 残差支时域滤波器数（每输入带）

    class ResidBranch(nn.Module):
        """无尺度先验通用 CNN：(B, 2, ch, T) → (B, d)。"""
        def __init__(self):
            super().__init__()
            fr1 = 2 * FR                      # 双带合并后 12 通道
            self.tconv = nn.Conv2d(2, fr1, (1, 63), padding=(0, 31), bias=False)
            self.bn1 = nn.BatchNorm2d(fr1)
            self.elu = nn.ELU()
            self.spatial = nn.Sequential(
                Conv2dWithConstraint(fr1, fr1 * 2, (n_ch, 1), padding=0,
                                     groups=fr1, bias=False, max_norm=1),
                nn.BatchNorm2d(fr1 * 2), nn.ELU(),
                nn.AvgPool2d((1, 4), stride=4), nn.Dropout(DROP))
            fr2 = fr1 * 2
            self.sep = nn.Sequential(
                nn.Conv2d(fr2, fr2, (1, 15), padding=(0, 7), groups=fr2),
                nn.Conv2d(fr2, fr2, 1, bias=False),
                nn.BatchNorm2d(fr2), nn.ELU(),
                nn.AvgPool2d((1, 8), stride=8))

        def forward(self, x):                     # x: (B, 2, ch, T)
            T_in = x.shape[-1]
            h = self.elu(self.bn1(self.tconv(x)[..., :T_in]))
            h = self.spatial(h)
            h = self.sep(h)
            return h.flatten(1)

    class S3RNet(nn.Module):
        def __init__(self):
            super().__init__()
            f1 = 2 * F1
            self.slow_conv = nn.Conv2d(1, F1, (1, 125), padding=(0, 62), bias=False)
            self.fast_conv = nn.Conv2d(1, F1, (1, 125), padding=(0, 62), bias=False)
            self.bn_t = nn.BatchNorm2d(f1)
            self.spatial = nn.Sequential(
                Conv2dWithConstraint(f1, f1 * D, (n_ch, 1), padding=0, groups=f1,
                                     bias=False, max_norm=1),
                nn.BatchNorm2d(f1 * D), nn.ELU(),
                nn.AvgPool2d((1, P1), stride=P1), nn.Dropout(DROP))
            f2 = f1 * D
            self.sep = nn.Sequential(
                nn.Conv2d(f2, f2, (1, T2), padding=(0, T2 // 2), groups=f2),
                nn.Conv2d(f2, f2, 1, bias=False),
                nn.BatchNorm2d(f2), nn.ELU())
            self.pool = nn.AvgPool2d((1, P2), stride=P2)
            self.resid = ResidBranch()
            self.drop = nn.Dropout(DROP)
            dummy = torch.zeros(1, 2, n_ch, T)
            with torch.no_grad():
                feat = self.features(dummy)
            self.head = nn.Linear(feat.shape[1], 2)

        def features(self, x):                    # x: (B, 2, ch, T)
            T_in = x.shape[-1]
            xb = torch.cat([self.slow_conv(x[:, 0:1])[..., :T_in],
                            self.fast_conv(x[:, 1:2])[..., :T_in]], dim=1)
            xb = self.bn_t(xb)
            xb = self.spatial(xb)
            xb = self.sep(xb)
            feat_b = self.pool(xb).flatten(1)
            feat_r = self.resid(x)                # 残差支吃双带全量
            return self.drop(torch.cat([feat_b, feat_r], dim=1))

        def forward(self, x):
            return self.head(self.features(x))

    return S3RNet()


def run_subject(Xs, y, sid, seeds, out_dir):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    tidx = np.arange(len(y))
    rows, pred_rows = [], []
    for seed in seeds:
        for fi, (tr, te) in enumerate(group_kfold_indices(tidx, 5, seed + sid)):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build_s3r_model(Xs.shape[2], Xs.shape[-1]).to(device)
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
                pred_rows.append((sid, seed, fi, int(te[j]), int(y_np[j]), int(pred_np[j])))
            rows.append((sid, seed, fi, acc, len(tr), len(te), ARM))
            print(f"    sub-{sid:02d} seed={seed} fold{fi}: {acc*100:.1f}%", flush=True)
            mdir = os.path.join(out_dir, "models")
            os.makedirs(mdir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(mdir, f"{ARM}_sub{sid:02d}_seed{seed}_fold{fi}.pt"))
    return rows, pred_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xw", choices=list(LOADERS.keys()))
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--subjects", default=None)
    args = ap.parse_args()
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
        rows, pred_rows = run_subject(Xs, y.astype(int), sid, remain, out_dir)
        m = np.mean([r[3] for r in rows])
        print(f"[{args.dataset}/{ARM}] sub-{sid:02d} done: {m*100:.2f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows, columns=["subject", "seed", "fold", "acc",
                                         "n_train", "n_test", "channels"])
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
        dp = pd.DataFrame(pred_rows, columns=["subject", "seed", "fold", "trial",
                                              "y", "pred"])
        dp.to_csv(pred_csv, mode="a", header=not os.path.exists(pred_csv), index=False)

    print(f"\n===== {args.dataset}/{ARM} 完成 =====")


if __name__ == "__main__":
    main()
