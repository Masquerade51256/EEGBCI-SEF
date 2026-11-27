"""E4 第 3 步 S3-D v3a：阶段 1 正则化三件套版蒸馏分支。

背景（E4 记录 §8.5/§8.6，2026-11-27 用户确认"把蒸馏做好"）：
v1/v2a 判负边界复核发现阶段 1 为裸跑（无增强、Adam 无 weight decay、
容量未降档）——"纯记忆化"诊断的标准治疗手段均未动用。v3a 单因子改动
（vs v1，教师保持 raw 280 维、阶段 2 与 v1 完全一致）：
  ① 训练折内无标签增强：每训练 trial ×3 副本（时间循环滚动 ±0.25 s +
     幅值抖动 ×0.9–1.1 + 通道 dropout p=0.15），**教师对增强后输入同步
     重算**（教师是输入的确定性函数，保证输入-目标不错位），阶段 1 样本
     池 32 → 128；z-score 统计量仍只用原始训练 trial 估计；
  ② 阶段 1 优化器 Adam → AdamW（weight decay 1e-2）；
  ③ trunk 容量减半（FR 6→3，§7.5 容量纪律方向）。
保真度指标（测试折 R²/Pearson r 分组）在阶段 1 结束时内嵌计算
（复用 e4o.r2_by_group），无需事后复演。

判读线（2026-11-26 锁定，沿用）：① 阶段 1 测试折 wPLI R² > 0（先 10 被试
× seed 42 探针）；①不过则归档不讨论融合；② ①过后融合在独有选区 8 人
（sub-09/10/11/24/37/39/43/47）显著为正。

输出：experiments/e4_step3/{dataset}/S3DREG_fold_results.csv（单体 S3DREGonly_*）
用法：
  python scripts/e4/e4q_distill3_reg.py --dataset xw --subjects 1 --seeds 42
  bash scripts/e4/run_e4q.sh        # BCIC 效度
  bash scripts/e4/run_e4p_xw.sh     # （v2a 用）XW；v3a 用 run_e4q_xw.sh
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
from scripts.e4.e4n_step3_distill import REGIONS, teacher_features
from scripts.e4.e4o_distill_fidelity import r2_by_group
from scripts.e2.e2a_spatial import group_kfold_indices

OUT = "experiments/e4_step3"
ARM = "S3DREG"
N_AUG = 3  # 每训练 trial 的增强副本数


def augment_trials(X, sr=250):
    """训练折内无标签增强：X (n, ch, T) → (N_AUG*n, ch, T)。
    时间循环滚动 ±0.25 s + 幅值抖动 ×0.9–1.1 + 通道 dropout p=0.15。"""
    n, ch, T = X.shape
    max_roll = int(0.25 * sr)
    outs = []
    for _ in range(N_AUG):
        Xa = np.empty_like(X)
        for i in range(n):
            r = np.random.randint(-max_roll, max_roll + 1)
            t = np.roll(X[i], r, axis=-1) * np.random.uniform(0.9, 1.1)
            mask = np.random.rand(ch) >= 0.15
            t[~mask] = 0.0
            Xa[i] = t
        outs.append(Xa)
    return np.concatenate(outs, axis=0)


def build_s3dreg_model(n_ch, T, n_teacher, distill_only=False):
    """同 e4n.build_s3d_model，唯一差别：trunk FR 6→3（容量减半）。"""
    import torch
    import torch.nn as nn
    from models.layers import Conv2dWithConstraint

    F1, D, T2, P1, P2, DROP = 8, 2, 30, 4, 8, 0.25
    FR = 3  # ← v3a 容量降档（v1 为 6）

    class DistillTrunk(nn.Module):
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
                dev = next(trunk.parameters()).device
                self.to(dev)
                dummy = torch.zeros(1, 2, n_ch, T, device=dev)
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
            dev = next(trunk.parameters()).device
            self.to(dev)
            dummy = torch.zeros(1, 2, n_ch, T, device=dev)
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


def run_subject(X, Xs, y, Tch, sid, seeds, out_dir, distill_only, sr):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    n_ch, T = Xs.shape[2], Xs.shape[-1]
    ap_dim = 2 * n_ch
    tidx = np.arange(len(y))
    rows, pred_rows = [], []
    for seed in seeds:
        for fi, (tr, te) in enumerate(group_kfold_indices(tidx, 5, seed + sid)):
            torch.manual_seed(seed)
            np.random.seed(seed)
            pre_cls, NetCls = build_s3dreg_model(n_ch, T, Tch.shape[1],
                                                 distill_only=distill_only)
            pre = pre_cls.to(device)
            # 教师 z-score：只用原始训练 trial 统计量
            mu = Tch[tr].mean(0, keepdims=True)
            sd = Tch[tr].std(0, keepdims=True) + 1e-8
            Tz = ((Tch - mu) / sd).astype(np.float32)
            # —— 增强副本：教师对增强后输入同步重算 ——
            Xa = augment_trials(X[tr], sr)                      # (N_AUG*k, ch, T)
            Ta = teacher_features(Xa, dset_key, sr).astype(np.float32)
            Tza = ((Ta - mu) / sd).astype(np.float32)
            Xsa = make_twoband(Xa, sr).astype(np.float32)
            Xs_pool = np.concatenate([Xs[tr], Xsa], axis=0)
            Tz_pool = np.concatenate([Tz[tr], Tza], axis=0)
            # —— 阶段 1：无标签模仿预训练（AdamW + 增强池）——
            opt = torch.optim.AdamW(pre.parameters(), lr=1e-3,
                                    weight_decay=1e-2)
            lossf = nn.MSELoss()
            loader = DataLoader(TensorDataset(
                torch.from_numpy(Xs_pool),
                torch.from_numpy(Tz_pool).float()),
                batch_size=16, shuffle=True, num_workers=0)
            for _ in range(100):
                pre.train()
                for xb, tb in loader:
                    xb, tb = xb.to(device), tb.to(device)
                    opt.zero_grad()
                    loss = lossf(pre(xb), tb)
                    loss.backward()
                    opt.step()
            # —— 阶段 1 保真度（测试折，诊断量不涉标签）——
            pre.eval()
            with torch.no_grad():
                p_te = pre(torch.from_numpy(Xs[te]).to(device)).cpu().numpy()
            r2_ap, r2_wp, r_ap, r_wp, ndeg = r2_by_group(p_te, Tz[te], ap_dim)
            # —— 阶段 2：任务微调（与 v1 完全一致：Adam、无增强）——
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
            rows.append((sid, seed, fi, acc, len(tr), len(te), ARM,
                         r2_ap, r2_wp, r_ap, r_wp, ndeg))
            print(f"    sub-{sid:02d} seed={seed} fold{fi}: {acc*100:.1f}% | "
                  f"s1_test R2 ap={r2_ap:.3f} wpli={r2_wp:.3f} "
                  f"r ap={r_ap:.3f} wpli={r_wp:.3f}", flush=True)
            mdir = os.path.join(out_dir, "models")
            os.makedirs(mdir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(mdir,
                                    f"{ARM}_sub{sid:02d}_seed{seed}_fold{fi}.pt"))
    return rows, pred_rows


def main():
    global ARM, dset_key
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xw", choices=list(LOADERS.keys()))
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--subjects", default=None)
    ap.add_argument("--distill-only", action="store_true",
                    help="蒸馏分支单体（ARM=S3DREGonly），无 B 主干")
    args = ap.parse_args()
    dset_key = args.dataset
    if args.distill_only:
        ARM = "S3DREGonly"
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
        rows, pred_rows = run_subject(X, Xs, y.astype(int), Tch, sid, remain,
                                      out_dir, args.distill_only, dspec["sr"])
        m = np.mean([r[3] for r in rows])
        print(f"[{args.dataset}/{ARM}] sub-{sid:02d} done: {m*100:.2f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows, columns=["subject", "seed", "fold", "acc",
                                         "n_train", "n_test", "channels",
                                         "r2_s1_ap_test", "r2_s1_wpli_test",
                                         "r_s1_ap_test", "r_s1_wpli_test",
                                         "n_degen_dims"])
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv),
                  index=False)
        dp = pd.DataFrame(pred_rows, columns=["subject", "seed", "fold",
                                              "trial", "y", "pred"])
        dp.to_csv(pred_csv, mode="a", header=not os.path.exists(pred_csv),
                  index=False)

    print(f"\n===== {args.dataset}/{ARM} 完成 =====")


dset_key = "xw"

if __name__ == "__main__":
    main()
