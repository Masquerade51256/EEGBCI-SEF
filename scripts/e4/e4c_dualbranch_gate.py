"""E4 第 1 步：双分支软门控主实验（慢波支 + 振荡支 + 端到端学习门控）。

设计文档：E4_实验设计_草案_20261105.md §3/§7；结果记录：E4_实验结果记录.md。
第 0/0.5 步已锁定输入表示约束（E4_实验结果记录.md §2.3）：
  - 慢支：0.1–40 Hz 必须保留、整段 4 s 长程上下文、分支内降采样 50 Hz；
  - 振荡支：4–40 Hz + 2 s 滑窗（0.5 s 步长）+ 逐窗 zscore 即可；
  - 归一化方式对性能无影响（按训练稳定性取窗级 zscore / trial 级原值）。

结构（容量低档起跑，全模型 ~3k 参数）：
  振荡支：多尺度时域卷积（kernel 10/20/31 @250Hz ≈ 40–125 ms，各 F=8）
          → depthwise 空间卷积 → log 方差池化 → 窗嵌入均值汇聚到 trial 级（24 维）
  慢支：  降采样 50 Hz（T=200/150）→ 长核时域卷积（kernel 50 = 1 s，F=8）
          → depthwise 空间卷积 → 3 等分子窗均值读出（保留尾段，24 维）
  门控：  concat(e_osc, e_slow) → hidden 16 → softmax(·/τ=2.0)，末层零初始化
          （初始精确均匀），训练期 logit 抖动 σ=0.2
  融合：  trial 级 e = w_osc·e_osc + w_slow·e_slow → 线性头

训练（草案 §7）：
  1. 分支预热 25 epoch：各分支带临时线性头独立训练（振荡支窗级，慢支 trial 级）；
  2. 联合 100 epoch：L = L_融合 + 0.3·(L_慢 + L_振荡) + λ·L_均衡；
     分支 lr 3e-4，门控 lr 1e-3，AdamW wd 1e-2，batch 16 trial；
  3. 增强（仅训练折）：通道 dropout p=0.15（两支共享掩码）+ 幅值抖动 ×0.9–1.1
     + 慢支随机时间滚动 ±0.25 s。

臂（同一套折划分与 seeds，全部 trial 级评估）：
  gate_l0 / gate_l01 / gate_l10   门控融合，λ = 0 / 0.01 / 0.1（λ=0 臂 = 防塌必要性证据）
  uniform                         均匀平均融合（门控替换为固定 0.5/0.5，其余同训）
  slow_only / osc_only            单支 + 自身线性头，全程 125 epoch

审计：GroupKFold 5 折按 trial 分组（振荡支重叠窗随 trial 不跨折）、
seeds 42/123/2026 仅影响折划分、增强仅训练折、禁止测试集反馈调参。

输出：experiments/e4_step1_dualbranch/{dataset}/{arm}_fold_results.csv
  列：subject, seed, fold, acc, acc_osc, acc_slow, gate_osc_mean, gate_slow_mean,
      n_train, n_test, channels（=臂名）

用法：
  python scripts/e4/e4c_dualbranch_gate.py --dataset bcic --arm gate_l01 --subjects 1
  python scripts/e4/e4c_dualbranch_gate.py --dataset xw --arm gate_l01
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.e2.e2e_slowpot import load_xw_trials, load_bcic_trials
from scripts.e2.e2a_spatial import group_kfold_indices
from scripts.e4.e4a_broadband_veto import bandpass_np, to_250hz

OUT = "experiments/e4_step1_dualbranch"

DATASETS = {
    "xw":   dict(loader=load_xw_trials,   sr=500, subjects=list(range(1, 51))),
    "bcic": dict(loader=load_bcic_trials, sr=250, subjects=list(range(1, 10))),
}

ARMS = ["gate_l0", "gate_l01", "gate_l10", "uniform", "slow_only", "osc_only"]
LAMBDA = {"gate_l0": 0.0, "gate_l01": 0.01, "gate_l10": 0.1}

WARMUP_EPOCHS = 25
JOINT_EPOCHS = 100
ALPHA = 0.3
TAU = 2.0
BATCH = 16


# ------------------------- 输入预处理 -------------------------

def prepare_inputs(X, sr):
    """X: (n, ch, T) 未滤波 -> 振荡支窗级输入 Xw (n, W, ch, 500)，慢支 trial 输入 Xs (n, ch, T//5)。
    所有操作均为 trial 内操作，无跨 trial 泄漏。"""
    # 振荡支：4–40 Hz -> 250 Hz -> 2 s 窗 / 0.5 s 步长 -> 逐窗逐通道 zscore
    Xo = to_250hz(bandpass_np(X, 4.0, 40.0, sr), sr)
    win, stride = 500, 125
    starts = list(range(0, Xo.shape[-1] - win + 1, stride))
    n, ch = Xo.shape[:2]
    Xw = np.stack([Xo[:, :, s:s + win] for s in starts], axis=1)   # (n, W, ch, 500)
    mu = Xw.mean(-1, keepdims=True)
    sd = Xw.std(-1, keepdims=True) + 1e-12
    Xw = ((Xw - mu) / sd).astype(np.float32)
    # 慢支：0.1–40 Hz -> 250 Hz -> 降采样 50 Hz -> 基线校正（减前 1/4 段均值）
    Xs = to_250hz(bandpass_np(X, 0.1, 40.0, sr), sr)[:, :, ::5]    # (n, ch, T//5)
    q = Xs.shape[-1] // 4
    Xs = Xs - Xs[..., :q].mean(axis=-1, keepdims=True)
    Xs = np.ascontiguousarray(Xs).astype(np.float32)
    return Xw, Xs


# ------------------------- 模型 -------------------------

def build_model(n_ch, n_win, T_win, T_slow):
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn

    EMB = 24            # 两支等维，便于加权融合

    class OscBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList([
                nn.Conv2d(1, 8, (1, k), padding=(0, k // 2), bias=False)
                for k in (10, 20, 31)])
            self.bn_t = nn.BatchNorm2d(24)
            self.spatial = nn.Conv2d(24, 24, (n_ch, 1), groups=24, bias=False)
            self.bn_s = nn.BatchNorm2d(24)

        def forward(self, xw):                    # (B, W, ch, T)
            B, W, C, T = xw.shape
            x = xw.reshape(B * W, 1, C, T)
            x = torch.cat([c(x) for c in self.convs], dim=1)
            x = Fn.elu(self.bn_t(x))
            x = Fn.elu(self.bn_s(self.spatial(x)))         # (B*W, 24, 1, T)
            x = torch.log(torch.var(x, dim=-1) + 1e-6)     # log-var 池化 (B*W, 24, 1)
            return x.reshape(B, W, EMB).mean(1)            # 窗->trial 均值汇聚

    class SlowBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal = nn.Conv2d(1, 8, (1, 50), padding=(0, 25), bias=False)
            self.bn_t = nn.BatchNorm2d(8)
            self.spatial = nn.Conv2d(8, 8, (n_ch, 1), groups=8, bias=False)
            self.bn_s = nn.BatchNorm2d(8)

        def forward(self, xs):                    # (B, ch, T_slow)
            x = xs.unsqueeze(1)
            x = Fn.elu(self.bn_t(self.temporal(x)))
            x = Fn.elu(self.bn_s(self.spatial(x)))         # (B, 8, 1, T)
            parts = torch.chunk(x.squeeze(2), 3, dim=-1)   # 3 等分子窗，保留尾段
            e = torch.cat([p.mean(-1) for p in parts], dim=-1)   # (B, 24)
            return e

    class Gate(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2 * EMB, 16)
            self.fc2 = nn.Linear(16, 2)
            nn.init.zeros_(self.fc2.weight)                # 初始精确均匀
            nn.init.zeros_(self.fc2.bias)

        def forward(self, e):
            logit = self.fc2(Fn.elu(self.fc1(e)))
            if self.training:
                logit = logit + torch.randn_like(logit) * 0.2    # logit 抖动
            return torch.softmax(logit / TAU, dim=-1)

    class DualBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.osc = OscBranch()
            self.slow = SlowBranch()
            self.gate = Gate()
            self.head_fuse = nn.Linear(EMB, 2)
            self.head_osc = nn.Linear(EMB, 2)
            self.head_slow = nn.Linear(EMB, 2)

        def forward(self, xw, xs, mode="gate"):
            eo = self.osc(xw)
            es = self.slow(xs)
            lo = self.head_osc(eo)
            ls = self.head_slow(es)
            if mode == "osc_only":
                return lo, lo, ls, None
            if mode == "slow_only":
                return ls, lo, ls, None
            if mode == "uniform":
                ef = 0.5 * eo + 0.5 * es
                w = torch.full((xw.shape[0], 2), 0.5, device=xw.device)
            else:
                w = self.gate(torch.cat([eo, es], dim=-1))
                ef = w[:, 0:1] * eo + w[:, 1:2] * es
            return self.head_fuse(ef), lo, ls, w

    return DualBranch()


# ------------------------- 训练 -------------------------

def augment(xw, xs, gen):
    """训练折增强：通道 dropout（两支共享掩码）+ 幅值抖动 + 慢支时间滚动。"""
    import torch
    B, W, C, T = xw.shape
    Ts = xs.shape[-1]
    mask = (torch.rand(B, C, generator=gen, device=xw.device) > 0.15).float()
    xw = xw * mask[:, None, :, None]
    xs = xs * mask[:, :, None]
    amp = 0.9 + 0.2 * torch.rand(B, 1, 1, 1, generator=gen, device=xw.device)
    xw = xw * amp
    xs = xs * amp.squeeze(-1)
    shift = int(torch.randint(-12, 13, (1,), generator=gen).item())   # ±0.25 s @50Hz
    if shift != 0:
        xs = torch.roll(xs, shifts=shift, dims=-1)
    return xw, xs


def run_arm(Xw, Xs, y, sid, seeds, arm):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(y)
    g = np.arange(n)                              # 每 trial 一组（振荡支窗随 trial）
    mode = {"gate_l0": "gate", "gate_l01": "gate", "gate_l10": "gate",
            "uniform": "uniform", "slow_only": "slow_only", "osc_only": "osc_only"}[arm]
    lam = LAMBDA.get(arm, 0.0)
    rows = []
    for seed in seeds:
        for fi, (tr, te) in enumerate(group_kfold_indices(g, 5, seed + sid)):
            torch.manual_seed(seed)
            np.random.seed(seed)
            gen = torch.Generator().manual_seed(seed * 1000 + fi)
            model = build_model(Xw.shape[2], Xw.shape[1],
                                Xw.shape[-1], Xs.shape[-1]).to(device)
            lossf = nn.CrossEntropyLoss()
            ytr = torch.from_numpy(y[tr]).long()
            Xw_tr = torch.from_numpy(Xw[tr])
            Xs_tr = torch.from_numpy(Xs[tr])
            tr_loader = DataLoader(TensorDataset(Xw_tr, Xs_tr, ytr),
                                   batch_size=BATCH, shuffle=True, num_workers=0)
            Xw_te = torch.from_numpy(Xw[te]).to(device)
            Xs_te = torch.from_numpy(Xs[te]).to(device)
            yte = torch.from_numpy(y[te]).long().to(device)

            # ---- 阶段 1：分支预热（各支带临时头独立训练）----
            if mode in ("gate", "uniform"):
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
                            if branch == "osc":
                                out = model.head_osc(model.osc(xb_w))
                            else:
                                out = model.head_slow(model.slow(xb_s))
                            loss = lossf(out, yb)
                            loss.backward()
                            opt.step()

            # ---- 阶段 2：联合 / 单支训练 ----
            branch_params = list(model.osc.parameters()) + list(model.slow.parameters()) + \
                list(model.head_fuse.parameters()) + list(model.head_osc.parameters()) + \
                list(model.head_slow.parameters())
            gate_params = list(model.gate.parameters())
            params = [{"params": branch_params, "lr": 3e-4}]
            if mode == "gate":
                params.append({"params": gate_params, "lr": 1e-3})
            opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-2)
            epochs = JOINT_EPOCHS if mode in ("gate", "uniform") else \
                WARMUP_EPOCHS + JOINT_EPOCHS
            for _ in range(epochs):
                model.train()
                for xb_w, xb_s, yb in tr_loader:
                    xb_w, xb_s, yb = xb_w.to(device), xb_s.to(device), yb.to(device)
                    xb_w, xb_s = augment(xb_w, xb_s, gen)
                    opt.zero_grad()
                    lf, lo, ls, w = model(xb_w, xb_s, mode=mode)
                    if mode == "osc_only":
                        loss = lossf(lo, yb)
                    elif mode == "slow_only":
                        loss = lossf(ls, yb)
                    else:
                        loss = lf_ce = lossf(lf, yb) + ALPHA * (lossf(lo, yb) + lossf(ls, yb))
                        if lam > 0:
                            wm = w.mean(0)
                            loss = lf_ce + lam * ((wm - 0.5) ** 2).sum()
                    loss.backward()
                    opt.step()

            # ---- 评估（trial 级）----
            model.eval()
            with torch.no_grad():
                lf, lo, ls, w = model(Xw_te, Xs_te, mode=mode)
                acc = float((lf.argmax(1) == yte).float().mean().item())
                acc_o = float((lo.argmax(1) == yte).float().mean().item())
                acc_s = float((ls.argmax(1) == yte).float().mean().item())
                if w is None:
                    wo = ws = float("nan")
                else:
                    wo = float(w[:, 0].mean().item())
                    ws = float(w[:, 1].mean().item())
            rows.append((sid, seed, fi, acc, acc_o, acc_s, wo, ws,
                         len(tr), len(te), arm))
            print(f"    sub-{sid:02d} seed={seed} fold{fi}: fuse={acc*100:.1f}% "
                  f"osc={acc_o*100:.1f}% slow={acc_s*100:.1f}% w_osc={wo:.2f}",
                  flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--dataset", default="xw", choices=list(DATASETS.keys()))
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--subjects", default=None, help="如 '1,2' 限定被试（冒烟测试用）")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    dspec = DATASETS[args.dataset]
    subjects = dspec["subjects"] if args.subjects is None else \
        [int(s) for s in args.subjects.split(",")]

    out_dir = os.path.join(OUT, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{args.arm}_fold_results.csv")
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
        rows = run_arm(Xw, Xs, y, sid, remain, args.arm)
        m = np.mean([r[3] for r in rows])
        print(f"[{args.dataset}/{args.arm}] sub-{sid:02d} done: {m*100:.2f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows, columns=["subject", "seed", "fold", "acc",
                                         "acc_osc", "acc_slow",
                                         "gate_osc_mean", "gate_slow_mean",
                                         "n_train", "n_test", "channels"])
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    print(f"\n===== {args.dataset}/{args.arm} 完成 =====")


if __name__ == "__main__":
    main()
