"""E4 第 3 步条件 S3-C：B 主干 + 连接分支（导联自注意力）。

正式设计：E4_实验设计_草案_20261105.md §13（2026-11-10 用户确认）。
翻译表"功能连接"行：wPLI/iCoh 导联对边特征 → 导联自注意力层
（注意力矩阵 = 可学习连接图）。

架构（融合 = 拼接自由混合，无门控）：
  输入 (n, 2, ch, T) 双带（0.1–4 / 4–40 Hz，各带独立 zscore，同第 2 步 B）
   ├─ B 主干：逐带时域卷积 → BN → 空间 depthwise → 可分离卷积 → 平均池化 → flatten
   │    （与 e4e_step2_factorial 条件 B 逐层一致）
   └─ 连接分支：fast 带 (n, ch, T)
        → 共享时域卷积 Fc=4 × k=125（每导联同套滤波器）→ BN → ELU
        → 时间维平均池化 → 每导联 token (n, ch, 4)
        → Linear(4→16) → 单层 MHSA（4 头，d=16）+ 残差 + LayerNorm
        → 导联均值池化 → (n, 16)
   concat(feat_B, feat_conn) → Dropout → Linear(→2)
  分支参数量 ~1.7k（预算 ≤3k）；整网 ~9k（预算 ~10k）。

设计决策（事先声明）：
  - 连接分支只取 fast 带——E2 连接族（wPLI/iCoh）定义在振荡频段，
    慢带的"连接"会被漂移主导，无对应手工族语义；
  - 读出用导联均值池化（注意力池化已在第 2 步 C/D 判负，不再引入）。

训练口径与第 0/1/1.5/2 步完全一致：100 epoch、Adam lr=1e-3、batch 16、
CE、无增强；GroupKFold 5 折按 trial × 3 seeds（seed+sid 同前划分）。

输出：experiments/e4_step3/{dataset}/
  S3C_fold_results.csv        折级：subject,seed,fold,acc,n_train,n_test,channels
  S3C_test_preds.csv          逐 trial：subject,seed,fold,trial,y,pred
  attn/S3C_sub{sid}.npz       连接注意力矩阵（测试 trial × heads × ch × ch + 标签）
  models/S3C_sub*_seed*_fold*.pt   state_dict（遮蔽分析用）

用法：
  python scripts/e4/e4f_step3_conn.py --dataset bcic --subjects 1 --seeds 42   # 冒烟
  bash scripts/e4/run_e4f.sh                                                  # BCIC 效度全量
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
ARM = "S3C"


def build_s3c_model(n_ch, T):
    """B 主干（复刻 e4e 条件 B）+ 连接分支，拼接融合。"""
    import torch
    import torch.nn as nn
    from models.layers import Conv2dWithConstraint

    F1, D, T2, P1, P2, DROP = 8, 2, 30, 4, 8, 0.25

    class ConnBranch(nn.Module):
        """导联自注意力分支：fast 带 (B,1,ch,T) → 导联 token → MHSA → (B,d)。"""
        def __init__(self, n_ch, fc=4, d=16, heads=4):
            super().__init__()
            self.tconv = nn.Conv2d(1, fc, (1, 125), padding=(0, 62), bias=False)
            self.bn = nn.BatchNorm2d(fc)
            self.elu = nn.ELU()
            self.proj = nn.Linear(fc, d)
            self.norm1 = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.norm2 = nn.LayerNorm(d)

        def forward(self, xf):                    # xf: (B, 1, ch, T)
            T_in = xf.shape[-1]
            h = self.tconv(xf)[..., :T_in]        # (B, Fc, ch, T)
            h = self.elu(self.bn(h))
            tok = h.mean(dim=-1)                  # (B, Fc, ch) 时间维平均
            tok = tok.permute(0, 2, 1)            # (B, ch, Fc)
            tok = self.proj(tok)                  # (B, ch, d)
            tok = self.norm1(tok)
            a, aw = self.attn(tok, tok, tok, need_weights=True,
                              average_attn_weights=False)   # aw: (B, heads, ch, ch)
            tok = self.norm2(tok + a)
            return tok.mean(dim=1), aw            # (B, d), (B, heads, ch, ch)

    class S3CNet(nn.Module):
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
            self.conn = ConnBranch(n_ch)
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
            feat_b = self.pool(xb).flatten(1)     # (B, f2*T'')
            feat_c, aw = self.conn(x[:, 1:2])     # 连接分支只吃 fast 带
            feat = self.drop(torch.cat([feat_b, feat_c], dim=1))
            return feat, aw

        def forward(self, x, return_attn=False):
            feat, aw = self.features(x)
            out = self.head(feat)
            return (out, aw) if return_attn else out

    return S3CNet()


def run_subject(Xs, y, sid, seeds, out_dir):
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
            model = build_s3c_model(Xs.shape[2], Xs.shape[-1]).to(device)
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
