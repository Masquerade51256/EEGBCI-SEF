"""E4 第 3 步条件 S3-D：B 主干 + 蒸馏分支（非周期 + 固定区域网格 wPLI）。

背景与设计（草案 §13.7⑥，2026-11-23 用户确认）：
连接轴解析翻译判负闭合（E4 记录 §7.4），瓶颈定位于"先学投影再算统计"
的两段式误差串联；蒸馏路线把空间覆盖从学习问题变为设计选择。教师集：
  ① 非周期：每导 log-log 谱（4–40Hz）线性拟合斜率+截距（E2 e2cd 同口径，
     60/44 维）——E2 中"可蒸馏性：非周期 ≫ 双谱 > 排列熵"的优先族；
  ② 固定解剖区域网格 wPLI：区域均值信号 → 4 子带 Hilbert → wPLI
     （E2 e2b 同公式），L/R 分区显式保留偏侧化（代偿连接的候选载体），
     XW 11 区 55 对 ×4 带 = 220 维；BCIC 9 区 36 对 ×4 带 = 144 维。

两阶段训练（守折纪律：教师 z-score 均值/方差只在训练折内估计；
教师计算本身为逐 trial 确定性信号处理，无折内拟合参数）：
  阶段 1（无标签模仿预训练）：分支 trunk + 回归头 MSE 回归教师，100 epoch，
     不接触标签；
  阶段 2（任务微调）：弃回归头，融合=B+分支拼接 / 单体=分支独立分类，
     CE 100 epoch。单体条件按 2026-11-11 规则必配（--distill-only）。

判读逻辑（事先写入）：单体成立且融合在独有选区为正 → 蒸馏路线兑现、
连接族以蒸馏形态复活；单体成立但融合冗余 → 蒸馏形态冗余，族信息正式
移交"直接特征拼接"对照评估；单体不成立 → 教师信息在 XW 不可蒸馏。

训练口径同前：Adam lr=1e-3、batch 16、CE；GroupKFold 5 折 × 3 seeds。

输出：experiments/e4_step3/{dataset}/S3D_fold_results.csv（单体 S3Donly_*）
用法：
  python scripts/e4/e4n_step3_distill.py --dataset bcic --subjects 1 --seeds 42
  bash scripts/e4/run_e4n.sh        # BCIC 效度
  bash scripts/e4/run_e4n_xw.sh     # XW 主实验
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
from scripts.e2.e2b_connectivity import band_hilbert, BANDS_4
from scripts.e2.e2cd_nonlinear_aperiodic import feat_aperiodic

OUT = "experiments/e4_step3"
ARM = "S3D"

# 固定解剖区域网格（对齐坐标系导联索引；L/R 显式分离，覆盖全脑）
REGIONS = {
    "xw": [("FL", [0, 3, 5]), ("FR", [1, 4, 6]),
           ("FCL", [8, 10]), ("FCR", [9, 11]),
           ("CL", [13, 15]), ("CR", [14, 16]),
           ("CPL", [18, 20]), ("CPR", [19, 21]),
           ("PL", [23, 25]), ("PR", [24, 26]),
           ("MID", [2, 7, 12, 17, 22, 27, 28, 29])],
    "bcic": [("FCL", [1, 2]), ("FCR", [4, 5]),
             ("CL", [6, 7, 8]), ("CR", [10, 11, 12]),
             ("CPL", [13, 14]), ("CPR", [16, 17]),
             ("PL", [18]), ("PR", [20]),
             ("MID", [0, 3, 9, 15, 19, 21])],
}


def teacher_features(X, dataset, fs=250):
    """X: (n, ch, T) → 教师矩阵 (n, D)：非周期 + 区域网格 wPLI。"""
    ap = feat_aperiodic(X, fs)                       # (n, ch*2)
    regs = REGIONS[dataset]
    R = np.stack([X[:, idx].mean(axis=1) for _, idx in regs], axis=1)  # (n,R,T)
    iu = np.triu_indices(len(regs), k=1)
    wplis = []
    for lo, hi in BANDS_4:
        Z = band_hilbert(R, lo, hi, fs)              # (n, R, T) 解析信号
        im = (Z[:, iu[0]] * np.conj(Z[:, iu[1]])).imag
        wpli = np.abs(im.mean(-1)) / (np.abs(im).mean(-1) + 1e-12)
        wplis.append(wpli)
    return np.concatenate([ap] + wplis, axis=1)


def build_s3d_model(n_ch, T, n_teacher, distill_only=False):
    import torch
    import torch.nn as nn
    from models.layers import Conv2dWithConstraint

    F1, D, T2, P1, P2, DROP = 8, 2, 30, 4, 8, 0.25
    FR = 6

    class DistillTrunk(nn.Module):
        """通用 CNN trunk（同 S3-R 残差支架构）：(B, 2, ch, T) → (B, d)。"""
        def __init__(self):
            super().__init__()
            fr1 = 2 * FR
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

        def forward(self, x):
            T_in = x.shape[-1]
            h = self.elu(self.bn1(self.tconv(x)[..., :T_in]))
            h = self.spatial(h)
            h = self.sep(h)
            return h.flatten(1)

    class PretrainNet(nn.Module):
        """阶段 1：trunk + 回归头（模仿教师）。"""
        def __init__(self):
            super().__init__()
            self.trunk = DistillTrunk()
            dummy = torch.zeros(1, 2, n_ch, T)
            with torch.no_grad():
                f = self.trunk(dummy)
            self.reg = nn.Linear(f.shape[1], n_teacher)

        def forward(self, x):
            return self.reg(self.trunk(x))

    if distill_only:
        class DistillOnlyNet(nn.Module):
            def __init__(self, trunk):
                super().__init__()
                self.trunk = trunk
                self.drop = nn.Dropout(DROP)
                dummy = torch.zeros(1, 2, n_ch, T)
                with torch.no_grad():
                    f = trunk(dummy)
                self.head = nn.Linear(f.shape[1], 2)

            def forward(self, x):
                return self.head(self.drop(self.trunk(x)))

        return PretrainNet(), DistillOnlyNet

    class S3DNet(nn.Module):
        def __init__(self, trunk):
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
            self.trunk = trunk
            self.drop = nn.Dropout(DROP)
            dummy = torch.zeros(1, 2, n_ch, T)
            with torch.no_grad():
                feat = self.features(dummy)
            self.head = nn.Linear(feat.shape[1], 2)

        def features(self, x):
            T_in = x.shape[-1]
            xb = torch.cat([self.slow_conv(x[:, 0:1])[..., :T_in],
                            self.fast_conv(x[:, 1:2])[..., :T_in]], dim=1)
            xb = self.bn_t(xb)
            xb = self.spatial(xb)
            xb = self.sep(xb)
            feat_b = self.pool(xb).flatten(1)
            feat_d = self.trunk(x)
            return self.drop(torch.cat([feat_b, feat_d], dim=1))

        def forward(self, x):
            return self.head(self.features(x))

    return PretrainNet(), S3DNet


def run_subject(Xs, y, Tch, sid, seeds, out_dir, distill_only=False):
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
            pre_cls, NetCls = build_s3d_model(
                Xs.shape[2], Xs.shape[-1], Tch.shape[1],
                distill_only=distill_only)
            pre = pre_cls.to(device)
            # 教师 z-score：只用训练折统计量
            mu = Tch[tr].mean(0, keepdims=True)
            sd = Tch[tr].std(0, keepdims=True) + 1e-8
            Tz = (Tch - mu) / sd
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
            # 蒸馏保真度（诊断量，测试折 trunk 输出与教师的相关不涉标签）
            model.eval()
            with torch.no_grad():
                logits = model(Xte)
                pred = logits.argmax(1)
                acc = float((pred == yte).float().mean().item())
            pred_np, y_np = pred.cpu().numpy(), yte.cpu().numpy()
            for j in range(len(te)):
                pred_rows.append((sid, seed, fi, int(te[j]), int(y_np[j]),
                                  int(pred_np[j])))
            rows.append((sid, seed, fi, acc, len(tr), len(te), ARM))
            print(f"    sub-{sid:02d} seed={seed} fold{fi}: {acc*100:.1f}%",
                  flush=True)
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
                    help="蒸馏分支单体（ARM=S3Donly），无 B 主干")
    args = ap.parse_args()
    if args.distill_only:
        ARM = "S3Donly"
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
                                         "n_train", "n_test", "channels"])
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv),
                  index=False)
        dp = pd.DataFrame(pred_rows, columns=["subject", "seed", "fold",
                                              "trial", "y", "pred"])
        dp.to_csv(pred_csv, mode="a", header=not os.path.exists(pred_csv),
                  index=False)

    print(f"\n===== {args.dataset}/{ARM} 完成 =====")


if __name__ == "__main__":
    main()
