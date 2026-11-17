"""E4 第 3 步条件 S3-C v3：B 主干 + 协方差池化分支（"算连接"替代"学连接"）。

背景与判词：E4 记录 §6（2026-11-11）——注意力图翻译形态在能力层面不成立
（单体 41.97% 低于机会、与 E2 连接族零相关）。连接信息轴本身不判负，
本条件改用协方差/双线性池化路线（2026-11-17 用户确认启动）。

设计逻辑：注意力权重是 token 混合权重，从不测量导联间共涨落；wPLI/iCoh
是二阶共变统计量。本分支逐 trial 解析地"算出"连接矩阵，学习只发生在
低维读出端——样本复杂度友好（40 trial 下无 30×30 结构可学）。

架构（融合 = 拼接自由混合，无门控）：
  输入 (n, 2, ch, T) 双带（同第 2 步 B）
   ├─ B 主干：与 e4e 条件 B 逐层一致
   └─ 协方差分支：fast 带
        → 固定子带滤波器组（θ 4–8 / α 8–13 / β 13–30 / γ 30–40 Hz，不可学习）
        → 每带可学习空间投影 W_b: ch→K=4（类 FBCSP 源空间，4×ch×4 ≈ 480 参）
        → 源空间逐 trial 协方差 C_b = S_b S_bᵀ/T（einsum，可微）
        → 相关化归一（C/√diag diagᵀ，抗容积传导的幅值尺度）
        → 上三角 K(K+1)/2=10 × 4 带 = 40 维 → Linear(40→16) + LayerNorm
   concat(feat_B, feat_cov) → Dropout → Linear(→2)

风险声明（事先写入）：零滞后协方差受容积传导污染（wPLI 的设计动机即剔除
零滞后成分）；缓解=源空间降维 + 相关化归一，若仍阴性则连接轴网络翻译
整体搁置。单体条件（--cov-only, ARM=S3COVonly）按 2026-11-11 规则必配。

训练口径同前：100 epoch、Adam lr=1e-3、batch 16、CE、无增强；
GroupKFold 5 折按 trial × 3 seeds（seed+sid 同前划分）。

输出：experiments/e4_step3/{dataset}/
  S3COV_fold_results.csv / S3COV_test_preds.csv（单体为 S3COVonly_*）
  models/S3COV_sub*_seed*_fold*.pt   state_dict（含学习到的空间投影 W）

用法：
  python scripts/e4/e4j_step3_cov.py --dataset bcic --subjects 1 --seeds 42   # 冒烟
  bash scripts/e4/run_e4j.sh                                                 # BCIC 效度
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
ARM = "S3COV"
SUBBANDS = [(4., 8.), (8., 13.), (13., 30.), (30., 40.)]
K_SRC = 4                       # 每带源空间维度


def build_s3cov_model(n_ch, T, cov_only=False):
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    from scipy.signal import firwin
    from models.layers import Conv2dWithConstraint

    F1, D, T2, P1, P2, DROP = 8, 2, 30, 4, 8, 0.25
    NB = len(SUBBANDS)
    IU = torch.triu_indices(K_SRC, K_SRC)          # 上三角（含对角）10 维/带

    class CovBranch(nn.Module):
        """协方差池化分支：fast 带 → 子带 → 源空间协方差 → (B, 16)。"""
        def __init__(self, d=16):
            super().__init__()
            kern = np.stack([firwin(125, [lo, hi], pass_zero=False, fs=250)
                             for lo, hi in SUBBANDS])
            self.register_buffer("filtbank",
                                 torch.tensor(kern).float()[:, None, None, :])
            self.W = nn.Parameter(torch.randn(NB, n_ch, K_SRC) * 0.05)
            n_feat = NB * K_SRC * (K_SRC + 1) // 2
            self.readout = nn.Linear(n_feat, d)
            self.norm = nn.LayerNorm(d)

        def forward(self, xf):                    # xf: (B, 1, ch, T)
            T_in = xf.shape[-1]
            xb = Fn.conv2d(xf, self.filtbank, padding=(0, 62))[..., :T_in]
            S = torch.einsum("bnct,nkc->bnkt", xb, self.W)   # (B, NB, K, T)
            C = torch.einsum("bnkt,bnlt->bnkl", S, S) / S.shape[-1]
            dg = C.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).sqrt()
            Cn = C / torch.einsum("bnk,bnl->bnkl", dg, dg)   # 相关化
            iu0, iu1 = IU[0].to(C.device), IU[1].to(C.device)
            v = Cn[:, :, iu0, iu1]                           # (B, NB, 10)
            return self.norm(self.readout(v.reshape(v.shape[0], -1)))

    if cov_only:
        class CovOnlyNet(nn.Module):
            """协方差分支单体：检验提取能力（区分能力缺失 vs 主干冗余）。"""
            def __init__(self):
                super().__init__()
                self.cov = CovBranch()
                self.drop = nn.Dropout(DROP)
                self.head = nn.Linear(16, 2)

            def forward(self, x):
                return self.head(self.drop(self.cov(x[:, 1:2])))

        return CovOnlyNet()

    class S3COVNet(nn.Module):
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
            self.cov = CovBranch()
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
            feat_c = self.cov(x[:, 1:2])          # 协方差分支只吃 fast 带
            return self.drop(torch.cat([feat_b, feat_c], dim=1))

        def forward(self, x):
            return self.head(self.features(x))

    return S3COVNet()


def run_subject(Xs, y, sid, seeds, out_dir, cov_only=False):
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
            model = build_s3cov_model(Xs.shape[2], Xs.shape[-1],
                                      cov_only=cov_only).to(device)
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
    global ARM
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xw", choices=list(LOADERS.keys()))
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--subjects", default=None)
    ap.add_argument("--cov-only", action="store_true",
                    help="协方差分支单体（ARM=S3COVonly），无 B 主干")
    args = ap.parse_args()
    if args.cov_only:
        ARM = "S3COVonly"
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
        rows, pred_rows = run_subject(Xs, y.astype(int), sid, remain, out_dir,
                                      cov_only=args.cov_only)
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
