"""E4 第 2 步：broad 主干因子检验——输入分带 × 读出方式 × 慢带感受野。

正式设计：E4_实验设计_草案_20261105.md §12（2026-11-08 用户确认）。

条件（主干 = 第 0 步 broad 的 EEGNet，训练口径完全一致：100 epoch、Adam lr=1e-3、
batch 16、CE、无增强；GroupKFold 5 折按 trial × 3 seeds，与第 0/1/1.5 步同划分）：

  A   单带 broad + 平均池化 = 第 0 步已有结果，直接引用（本脚本不跑）
  B   双带输入（0.1–4 / 4–40 Hz，各带独立 trial 级 zscore + 独立时域卷积）+ 平均池化
  C   单带 + 注意力池化（打分器 MLP(F2→16→1, tanh)，softmax over 时间位置）
  D   双带 + 注意力池化
  E   双带 + 慢带 2 s 长核（500 点 @250 Hz，振荡带核不变）+ 平均池化

输出：experiments/e4_step2_factorial/{dataset}/
  {arm}_fold_results.csv   折级：subject,seed,fold,acc,n_train,n_test,channels
  {arm}_test_preds.csv     逐 trial：subject,seed,fold,trial,y,pred
  attn/{arm}_sub{sid}.npz  注意力权重（臂 C/D：测试 trial × 时间位置 + 标签）
  models/{arm}_sub{sid}_seed{seed}_fold{fi}.pt   双带臂（B/D/E）state_dict（带遮蔽分析用）

用法：
  python scripts/e4/e4e_step2_factorial.py --dataset bcic --arm B --subjects 1   # 冒烟
  bash scripts/e4/run_e4e.sh                                                    # 全量
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.e4.e4a_broadband_veto import bandpass_np, to_250hz, make_broad, LOADERS
from scripts.e2.e2a_spatial import group_kfold_indices

OUT = "experiments/e4_step2_factorial"
ARMS = ["B", "C", "D", "E"]
TWO_BAND = {"B", "D", "E"}
ATTN = {"C", "D"}
LONG_SLOW = {"E"}


def make_twoband(X, sr):
    """双带输入：0.1–4 Hz 与 4–40 Hz 分带 -> 250 Hz -> 各带独立基线校正 +
    trial 级 zscore（消除慢波/振荡数量级差异）。返回 (n, 2, ch, T) float32。"""
    from scipy.signal import butter, filtfilt
    Xf = to_250hz(X, sr)
    fs = 250
    b_lo, a_lo = butter(4, 4.0 / (fs / 2), btype="low")
    b_hi, a_hi = butter(4, [4.0 / (fs / 2), 40.0 / (fs / 2)], btype="band")
    Xslow = filtfilt(b_lo, a_lo, Xf, axis=-1)
    Xfast = filtfilt(b_hi, a_hi, Xf, axis=-1)
    out = []
    for Xb in (Xslow, Xfast):
        q = Xb.shape[-1] // 4
        Xb = Xb - Xb[..., :q].mean(axis=-1, keepdims=True)     # 基线校正
        mu = Xb.mean(axis=-1, keepdims=True)
        sd = Xb.std(axis=-1, keepdims=True) + 1e-12
        out.append((Xb - mu) / sd)                             # 各带独立 zscore
    X2 = np.stack(out, axis=1)                                 # (n, 2, ch, T)
    return np.ascontiguousarray(X2).astype(np.float32)


def build_step2_model(arm, n_ch, T):
    """主干复刻 models/EEGNet.py 的 broad 配置；按臂改输入分带与读出。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    from models.layers import Conv2dWithConstraint

    F1, D, T2, P1, P2, DROP = 8, 2, 30, 4, 8, 0.25
    SLOW_K = 500 if arm in LONG_SLOW else 125

    class AttnPool(nn.Module):
        """打分器 MLP(F2→16→1, tanh) + softmax over 时间位置 -> 加权和。"""
        def __init__(self, f2):
            super().__init__()
            self.score = nn.Sequential(nn.Linear(f2, 16), nn.Tanh(), nn.Linear(16, 1))

        def forward(self, x):                     # x: (B, F2, 1, T')
            h = x.squeeze(2).permute(0, 2, 1)     # (B, T', F2)
            a = torch.softmax(self.score(h), dim=1)          # (B, T', 1)
            return (a * h).sum(1), a.squeeze(-1)             # (B, F2), (B, T')

    class Step2Net(nn.Module):
        def __init__(self):
            super().__init__()
            two = arm in TWO_BAND
            f1 = 2 * F1 if two else F1
            if two:
                self.slow_conv = nn.Conv2d(1, F1, (1, SLOW_K), padding=(0, SLOW_K // 2), bias=False)
                self.fast_conv = nn.Conv2d(1, F1, (1, 125), padding=(0, 125 // 2), bias=False)
            else:
                self.slow_conv = nn.Conv2d(1, F1, (1, 125), padding=(0, 125 // 2), bias=False)
                self.fast_conv = None
            self.bn_t = nn.BatchNorm2d(f1)
            self.spatial = nn.Sequential(
                Conv2dWithConstraint(f1, f1 * D, (n_ch, 1), padding=0, groups=f1,
                                     bias=False, max_norm=1),
                nn.BatchNorm2d(f1 * D), nn.ELU(),
                nn.AvgPool2d((1, P1), stride=P1), nn.Dropout(DROP))
            f2 = f1 * D
            # 与 models/EEGNet.py 的 temporal 块严格同序：conv→pointwise→BN→ELU→pool→dropout
            self.sep = nn.Sequential(
                nn.Conv2d(f2, f2, (1, T2), padding=(0, T2 // 2), groups=f2),
                nn.Conv2d(f2, f2, 1, bias=False),
                nn.BatchNorm2d(f2), nn.ELU())
            self.pool = AttnPool(f2) if arm in ATTN else nn.AvgPool2d((1, P2), stride=P2)
            self.drop = nn.Dropout(DROP)
            dummy = torch.zeros(1, 2 if two else 1, n_ch, T)
            with torch.no_grad():
                feat, _ = self.features(dummy)
            self.head = nn.Linear(feat.shape[1], 2)

        def features(self, x):                    # x: (B, nband, ch, T)
            T_in = x.shape[-1]
            if self.fast_conv is not None:
                x = torch.cat([self.slow_conv(x[:, 0:1])[..., :T_in],
                               self.fast_conv(x[:, 1:2])[..., :T_in]], dim=1)
            else:
                x = self.slow_conv(x)[..., :T_in]
            x = self.bn_t(x)
            x = self.spatial(x)
            x = self.sep(x)
            if isinstance(self.pool, AttnPool):
                feat, aw = self.pool(x)           # (B, F2), (B, T')
            else:
                x = self.pool(x)
                feat, aw = x.flatten(1), None
            feat = self.drop(feat)
            return feat, aw

        def forward(self, x, return_attn=False):
            feat, aw = self.features(x)
            out = self.head(feat)
            return (out, aw) if return_attn else out

    return Step2Net()


def run_arm(Xs, y, sid, seeds, arm, out_dir):
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
            model = build_step2_model(arm, Xs.shape[2], Xs.shape[-1]).to(device)
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
                if arm in ATTN:
                    logits, aw = model(Xte, return_attn=True)
                    attn_store["w"].append(aw.cpu().numpy())
                    attn_store["trial"].append(te)
                    attn_store["y"].append(yte.cpu().numpy())
                    attn_store["seed"] += [seed] * len(te)
                    attn_store["fold"] += [fi] * len(te)
                else:
                    logits = model(Xte)
                pred = logits.argmax(1)
                acc = float((pred == yte).float().mean().item())
            pred_np, y_np = pred.cpu().numpy(), yte.cpu().numpy()
            for j in range(len(te)):
                pred_rows.append((sid, seed, fi, int(te[j]), int(y_np[j]), int(pred_np[j])))
            rows.append((sid, seed, fi, acc, len(tr), len(te), arm))
            print(f"    sub-{sid:02d} seed={seed} fold{fi}: {acc*100:.1f}%", flush=True)
            if arm in TWO_BAND:                    # 存模型供带遮蔽分析
                mdir = os.path.join(out_dir, "models")
                os.makedirs(mdir, exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(mdir, f"{arm}_sub{sid:02d}_seed{seed}_fold{fi}.pt"))
    if arm in ATTN and attn_store["w"]:
        adir = os.path.join(out_dir, "attn")
        os.makedirs(adir, exist_ok=True)
        np.savez_compressed(os.path.join(adir, f"{arm}_sub{sid:02d}.npz"),
                            w=np.concatenate(attn_store["w"], 0),
                            trial=np.concatenate(attn_store["trial"], 0),
                            y=np.concatenate(attn_store["y"], 0),
                            seed=np.array(attn_store["seed"]),
                            fold=np.array(attn_store["fold"]))
    return rows, pred_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
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
    out_csv = os.path.join(out_dir, f"{args.arm}_fold_results.csv")
    pred_csv = os.path.join(out_dir, f"{args.arm}_test_preds.csv")
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
        if args.arm in TWO_BAND:
            Xs = make_twoband(X, dspec["sr"])
        else:
            Xs, _ = make_broad(X, dspec["sr"])
            Xs = Xs[:, None, :, :]                # (n, 1, ch, T)
        rows, pred_rows = run_arm(Xs, y.astype(int), sid, remain, args.arm, out_dir)
        m = np.mean([r[3] for r in rows])
        print(f"[{args.dataset}/{args.arm}] sub-{sid:02d} done: {m*100:.2f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows, columns=["subject", "seed", "fold", "acc",
                                         "n_train", "n_test", "channels"])
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
        dp = pd.DataFrame(pred_rows, columns=["subject", "seed", "fold", "trial",
                                              "y", "pred"])
        dp.to_csv(pred_csv, mode="a", header=not os.path.exists(pred_csv), index=False)

    print(f"\n===== {args.dataset}/{args.arm} 完成 =====")


if __name__ == "__main__":
    main()
