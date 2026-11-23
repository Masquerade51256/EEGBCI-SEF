"""E4 第 3 步条件 S3-C v4：B 主干 + 虚部相干（iCoh）分支。

背景与判词：E4 记录 §6/§7、草案 §13.7③⑤——v3 协方差池化单体能力成立
（51.65%）但信息对 B 冗余（遮蔽≈0、决策层融合 −1.25pp n.s.）；v3 风险
事先声明指出其零滞后协方差受容积传导污染。E2 手工连接族主力特征
wPLI/iCoh 的设计动机即剔除零滞后成分。v4 检验遗留假设：
**虚部相干信息是否也对 B 冗余**（2026-11-23 用户确认启动）。

设计逻辑：与 v3 同为"算出来而非学出来"，但统计量从零滞后协方差换成
交叉谱虚部相干——剔除容积传导的零滞后分量，只保留有相位延迟的共涨落。

架构（融合 = 拼接自由混合，无门控，同 v3）：
  输入 (n, 2, ch, T) 双带（同第 2 步 B）
   ├─ B 主干：与 e4e 条件 B 逐层一致
   └─ iCoh 分支：fast 带
        → 固定子带滤波器组（θ 4–8 / α 8–13 / β 13–30 / γ 30–40 Hz，不可学习）
        → 每带可学习空间投影 W_b: ch→K=4（同 v3）
        → rfft 交叉谱 → 虚部相干 iCoh_kl = Σ_f|Im(C_kl)|/(Σ_f√(P_k P_l)+eps)
          （功率加权带内聚合，低功率频点自动降权；对角恒零，取严格上三角）
        → 6 对 × 4 带 = 24 维 → Linear(24→16) + LayerNorm
   concat(feat_B, feat_icoh) → Dropout → Linear(→2)

判读逻辑（事先写入）：单体成立且融合在独有选区为正 → 连接族网络翻译复活；
单体成立但融合冗余 → 连接轴正式判负、无遗留假设；单体不成立 → 可微
iCoh 形态不可学，连接族转蒸馏处置。单体条件（--icoh-only,
ARM=S3ICOHonly）按 2026-11-11 规则必配。

训练口径同前：100 epoch、Adam lr=1e-3、batch 16、CE、无增强；
GroupKFold 5 折按 trial × 3 seeds（seed+sid 同前划分）。

输出：experiments/e4_step3/{dataset}/
  S3ICOH_fold_results.csv / S3ICOH_test_preds.csv（单体为 S3ICOHonly_*）
  models/S3ICOH_sub*_seed*_fold*.pt

用法：
  python scripts/e4/e4l_step3_icoh.py --dataset bcic --subjects 1 --seeds 42  # 冒烟
  bash scripts/e4/run_e4l.sh                                                  # BCIC 效度
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
ARM = "S3ICOH"
SUBBANDS = [(4., 8.), (8., 13.), (13., 30.), (30., 40.)]
K_SRC = 4                       # 每带源空间维度


def build_s3icoh_model(n_ch, T, icoh_only=False):
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    from scipy.signal import firwin
    from models.layers import Conv2dWithConstraint

    F1, D, T2, P1, P2, DROP = 8, 2, 30, 4, 8, 0.25
    NB = len(SUBBANDS)
    IU = torch.triu_indices(K_SRC, K_SRC, offset=1)   # 严格上三角（6 对/带）

    class ICohBranch(nn.Module):
        """虚部相干分支：fast 带 → 子带 → 源空间 → 交叉谱虚部相干 → (B, 16)。"""
        def __init__(self, d=16):
            super().__init__()
            kern = np.stack([firwin(125, [lo, hi], pass_zero=False, fs=250)
                             for lo, hi in SUBBANDS])
            self.register_buffer("filtbank",
                                 torch.tensor(kern).float()[:, None, None, :])
            self.W = nn.Parameter(torch.randn(NB, n_ch, K_SRC) * 0.05)
            n_feat = NB * K_SRC * (K_SRC - 1) // 2
            self.readout = nn.Linear(n_feat, d)
            self.norm = nn.LayerNorm(d)

        def forward(self, xf):                    # xf: (B, 1, ch, T)
            T_in = xf.shape[-1]
            xb = Fn.conv2d(xf, self.filtbank, padding=(0, 62))[..., :T_in]
            S = torch.einsum("bnct,nck->bnkt", xb, self.W)   # (B, NB, K, T)
            Fc = torch.fft.rfft(S.float(), dim=-1)           # (B, NB, K, F)
            cross = (Fc.unsqueeze(3) * Fc.conj().unsqueeze(2))  # (B,NB,K,K,F)
            im = cross.imag.abs()                             # |Im(C_kl)|
            pw = Fc.real.pow(2) + Fc.imag.pow(2)              # 功率谱
            w = (pw.unsqueeze(3) * pw.unsqueeze(2)).sqrt()    # √(P_k P_l)
            icoh = im.sum(-1) / (w.sum(-1) + 1e-8)            # (B, NB, K, K)
            iu0, iu1 = IU[0].to(Fc.device), IU[1].to(Fc.device)
            v = icoh[:, :, iu0, iu1]                          # (B, NB, 6)
            return self.norm(self.readout(v.reshape(v.shape[0], -1)))

    if icoh_only:
        class ICohOnlyNet(nn.Module):
            """iCoh 分支单体：检验提取能力（区分能力缺失 vs 主干冗余）。"""
            def __init__(self):
                super().__init__()
                self.icoh = ICohBranch()
                self.drop = nn.Dropout(DROP)
                self.head = nn.Linear(16, 2)

            def forward(self, x):
                return self.head(self.drop(self.icoh(x[:, 1:2])))

        return ICohOnlyNet()

    class S3ICOHNet(nn.Module):
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
            self.icoh = ICohBranch()
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
            feat_c = self.icoh(x[:, 1:2])         # iCoh 分支只吃 fast 带
            return self.drop(torch.cat([feat_b, feat_c], dim=1))

        def forward(self, x):
            return self.head(self.features(x))

    return S3ICOHNet()


def run_subject(Xs, y, sid, seeds, out_dir, icoh_only=False):
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
            model = build_s3icoh_model(Xs.shape[2], Xs.shape[-1],
                                       icoh_only=icoh_only).to(device)
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
    ap.add_argument("--icoh-only", action="store_true",
                    help="iCoh 分支单体（ARM=S3ICOHonly），无 B 主干")
    args = ap.parse_args()
    if args.icoh_only:
        ARM = "S3ICOHonly"
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
                                      icoh_only=args.icoh_only)
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
