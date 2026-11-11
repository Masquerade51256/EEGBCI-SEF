"""E4 第 3 步条件 S3-C v2：B 主干 + 连接分支 v2（token 富化 + 窗级注意力）。

正式设计：E4_实验设计_草案_20261105.md §13 + 2026-11-11 用户确认的两点改进。
针对 v1（S3C）已确诊的两个缺陷：
  ① token 内容贫瘠（每导联仅 4 个卷积特征的全段平均）
     → 注入手工先验：固定子带滤波器组（θ 4–8 / α 8–13 / β 13–30 / γ 30–40 Hz）
       的逐窗 log 带功率（4 维，不可学习）+ 可学习卷积特征（4 维）= 8 维 token；
  ② 样本复杂度（全 trial 一张图，32 训练 trial 学 30×30 边）
     → 窗级注意力：2 s 滑窗 / 0.5 s 步长（XW 5 窗 / BCIC 3 窗），每窗一张
       连接图，trial 级聚合——等效比较样本 ×5，守折纪律（窗不跨折）。

架构（融合 = 拼接自由混合，无门控）：
  输入 (n, 2, ch, T) 双带（同第 2 步 B）
   ├─ B 主干：与 e4e 条件 B 逐层一致
   └─ 连接分支 v2：fast 带 → 滑窗 → 每窗：
        固定滤波器组 log 带功率 (4) ⊕ 共享时域卷积 Fc=4 均值 (4) → token(8)
        → Linear(8→16) → LayerNorm → MHSA(4 头) + 残差 + LayerNorm
        → 导联均值池化 → 窗向量(16) → 窗均值 → (n, 16)
   concat(feat_B, feat_conn) → Dropout → Linear(→2)

判读线（沿用 §13.4，修正判负措辞，2026-11-11 用户确认）：
  若注意力矩阵仍均匀（熵 ≈ ln(870)），判词为"注意力图这一翻译形态未通过
  验证"；连接信息轴本身不判负（E2 已证其对部分被试有效），协方差池化等
  替代设计保留为开放方向。

训练口径同前：100 epoch、Adam lr=1e-3、batch 16、CE、无增强；
GroupKFold 5 折按 trial × 3 seeds（seed+sid 同前划分）。

输出：experiments/e4_step3/{dataset}/
  S3C2_fold_results.csv / S3C2_test_preds.csv
  attn/S3C2_sub{sid}.npz（逐 trial 注意力矩阵，窗均值）+ models/ state_dict

用法：
  python scripts/e4/e4i_step3_conn2.py --dataset bcic --subjects 1 --seeds 42  # 冒烟
  bash scripts/e4/run_e4i.sh                                                  # BCIC 效度
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
ARM = "S3C2"
WIN, STRIDE = 500, 125          # 2 s / 0.5 s @ 250 Hz
SUBBANDS = [(4., 8.), (8., 13.), (13., 30.), (30., 40.)]


def build_s3c2_model(n_ch, T, conn_only=False):
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    from scipy.signal import firwin
    from models.layers import Conv2dWithConstraint

    F1, D, T2, P1, P2, DROP = 8, 2, 30, 4, 8, 0.25

    class ConnBranchV2(nn.Module):
        """窗级导联自注意力：固定带功率先验 + 可学习特征 → token → MHSA。"""
        def __init__(self, fc=4, d=16, heads=4, fs=250):
            super().__init__()
            kern = np.stack([firwin(125, [lo, hi], pass_zero=False, fs=250)
                             for lo, hi in SUBBANDS])          # (nb, 125)
            self.register_buffer("filtbank",
                                 torch.tensor(kern).float()[:, None, None, :])
            self.tconv = nn.Conv2d(1, fc, (1, 125), padding=(0, 62), bias=False)
            self.bn = nn.BatchNorm2d(fc)
            self.elu = nn.ELU()
            self.proj = nn.Linear(len(SUBBANDS) + fc, d)
            self.norm1 = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.norm2 = nn.LayerNorm(d)

        def forward(self, xf):                    # xf: (B, 1, ch, T)
            B = xf.shape[0]
            xw = xf.unfold(-1, WIN, STRIDE)       # (B, 1, ch, n_win, win)
            nw = xw.shape[3]
            xw = xw.permute(0, 3, 1, 2, 4).reshape(B * nw, 1, -1, WIN)
            # 固定子带 log 功率（不可学习先验）
            bp = Fn.conv2d(xw, self.filtbank, padding=(0, 62))[..., :WIN]
            bp = torch.log(bp.pow(2).mean(-1) + 1e-12)        # (B*nw, nb, ch)
            bp = bp.permute(0, 2, 1)                          # (B*nw, ch, nb)
            # 可学习时域特征
            h = self.elu(self.bn(self.tconv(xw)[..., :WIN]))  # (B*nw, Fc, ch, win)
            h = h.mean(-1).permute(0, 2, 1)                   # (B*nw, ch, Fc)
            tok = self.proj(torch.cat([bp, h], dim=-1))       # (B*nw, ch, d)
            tok = self.norm1(tok)
            a, aw = self.attn(tok, tok, tok, need_weights=True,
                              average_attn_weights=False)
            tok = self.norm2(tok + a)
            feat = tok.mean(dim=1)                            # (B*nw, d)
            feat = feat.reshape(B, nw, -1).mean(dim=1)        # (B, d) 窗聚合
            aw = aw.reshape(B, nw, *aw.shape[1:]).mean(dim=1) # (B, heads, ch, ch)
            return feat, aw

    if conn_only:
        class ConnOnlyNet(nn.Module):
            """连接分支 v2 单体：检验分支独立提取信息的能力上限
            （区分"分支学不到"与"被 B 主干冗余覆盖"）。"""
            def __init__(self):
                super().__init__()
                self.conn = ConnBranchV2()
                self.drop = nn.Dropout(DROP)
                self.head = nn.Linear(16, 2)

            def features(self, x):
                feat_c, aw = self.conn(x[:, 1:2])
                return self.drop(feat_c), aw

            def forward(self, x, return_attn=False):
                feat, aw = self.features(x)
                out = self.head(feat)
                return (out, aw) if return_attn else out

        return ConnOnlyNet()

    class S3C2Net(nn.Module):
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
            self.conn = ConnBranchV2()
            self.drop = nn.Dropout(DROP)
            dummy = torch.zeros(1, 2, n_ch, T)
            with torch.no_grad():
                feat, _ = self.features(dummy)
            self.head = nn.Linear(feat.shape[1], 2)

        def features(self, x):                    # x: (B, 2, ch, T)
            T_in = x.shape[-1]
            xb = torch.cat([self.slow_conv(x[:, 0:1])[..., :T_in],
                            self.fast_conv(x[:, 1:2])[..., :T_in]], dim=1)
            xb = self.bn_t(xb)
            xb = self.spatial(xb)
            xb = self.sep(xb)
            feat_b = self.pool(xb).flatten(1)
            feat_c, aw = self.conn(x[:, 1:2])     # 连接分支只吃 fast 带
            feat = self.drop(torch.cat([feat_b, feat_c], dim=1))
            return feat, aw

        def forward(self, x, return_attn=False):
            feat, aw = self.features(x)
            out = self.head(feat)
            return (out, aw) if return_attn else out

    return S3C2Net()


def run_subject(Xs, y, sid, seeds, out_dir, conn_only=False):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    tidx = np.arange(len(y))
    rows, pred_rows = [], []
    attn_store = {"w": [], "trial": [], "y": [], "seed": [], "fold": []}
    for seed in seeds:
        for fi, (tr, te) in enumerate(group_kfold_indices(tidx, 5, seed + sid)):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build_s3c2_model(Xs.shape[2], Xs.shape[-1],
                                     conn_only=conn_only).to(device)
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
                logits, aw = model(Xte, return_attn=True)
                attn_store["w"].append(aw.cpu().numpy())
                attn_store["trial"].append(te)
                attn_store["y"].append(yte.cpu().numpy())
                attn_store["seed"] += [seed] * len(te)
                attn_store["fold"] += [fi] * len(te)
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
    adir = os.path.join(out_dir, "attn")
    os.makedirs(adir, exist_ok=True)
    np.savez_compressed(os.path.join(adir, f"{ARM}_sub{sid:02d}.npz"),
                        w=np.concatenate(attn_store["w"], 0),
                        trial=np.concatenate(attn_store["trial"], 0),
                        y=np.concatenate(attn_store["y"], 0),
                        seed=np.array(attn_store["seed"]),
                        fold=np.array(attn_store["fold"]))
    return rows, pred_rows


def main():
    global ARM
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xw", choices=list(LOADERS.keys()))
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--subjects", default=None)
    ap.add_argument("--conn-only", action="store_true",
                    help="连接分支单体（ARM=S3C2conn），无 B 主干")
    args = ap.parse_args()
    if args.conn_only:
        ARM = "S3C2conn"
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
                                      conn_only=args.conn_only)
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
