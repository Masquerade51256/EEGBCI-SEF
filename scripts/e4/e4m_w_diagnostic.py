"""E4 第 3 步 W 诊断：v3/v4 空间投影的学习质量分析（2026-11-23）。

动机（E4 记录 §7.4 判词 2）：v4 失败定位于"先学投影再算统计"的两段式误差
串联；用户提出 K=4 覆盖假说（投影集中于运动区、未覆盖代偿区域）。本脚本
直接检查训练后的 W 回答三个问题：
  Q1 W 学到了稳定结构吗？（跨种子×折的子空间一致性）
  Q2 W 的空间能量集中在哪里？（运动区 vs 全脑覆盖）
  Q3 v3（能力成立）与 v4（能力缺失）的 W 有系统差异吗？

指标：
  - 子空间一致性：每带 W_b (ch×K) 的列子空间跨 15 次运行（3 种子×5 折）
    两两比较，mean canonical correlation²（对源排列/旋转不变）；
    基线 = 同尺度随机高斯投影的期望一致性（蒙特卡洛 200 次）；
  - 空间能量：|W|² 对带与源聚合 → 30 导联能量份额；报告运动区
    （FC/C/CP 排：FCz,FC3,FC4,Cz,C3,C4,CPz,CP3,CP4）能量占比。

输出：experiments/e4_step3/xw/wdiag_summary.csv（拉回本地画图）
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.e2.e2a_spatial import XW_CH_NAMES

MODELS = "experiments/e4_step3/xw/models"
OUT_CSV = "experiments/e4_step3/xw/wdiag_summary.csv"
MOTOR = [i for i, n in enumerate(XW_CH_NAMES)
         if n in ("FCz", "FC3", "FC4", "Cz", "C3", "C4", "CPz", "CP3", "CP4")]
FRONTAL = [i for i, n in enumerate(XW_CH_NAMES)
           if n.startswith(("Fp", "F")) and i not in MOTOR]
PARIETO_OCC = [i for i, n in enumerate(XW_CH_NAMES)
               if n.startswith(("P", "O", "T5", "T6"))]


def subspace_consistency(Ws):
    """W: list of (NB, ch, K)。逐带跨运行两两 mean CC²，再对带平均。"""
    from scipy.linalg import orth
    nb = Ws[0].shape[0]
    per_band = []
    for b in range(nb):
        Qs = [orth(W[b]) for W in Ws]          # (ch, K) 正交基
        cc = []
        for i in range(len(Qs)):
            for j in range(i + 1, len(Qs)):
                sv = np.linalg.svd(Qs[i].T @ Qs[j], compute_uv=False)
                cc.append(float((sv ** 2).mean()))
        per_band.append(float(np.mean(cc)))
    return float(np.mean(per_band))


def random_baseline(n_ch=30, k=4, nb=4, n_rep=200, scale=0.05):
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(n_rep):
        Ws = [rng.standard_normal((nb, n_ch, k)) * scale for _ in range(15)]
        vals.append(subspace_consistency(Ws))
    return float(np.mean(vals))


def energy_share(W):
    """W: (NB, ch, K) → 30 导联能量份额 + 分区占比。"""
    e = (W ** 2).sum(axis=(0, 2))              # (ch,)
    e = e / e.sum()
    return e, float(e[MOTOR].sum()), float(e[FRONTAL].sum()), \
        float(e[PARIETO_OCC].sum())


def main():
    rows = []
    rand_ref = random_baseline()
    print(f"[baseline] 随机投影子空间一致性: {rand_ref:.3f}", flush=True)
    for arm, key in [("S3COV", "cov.W"), ("S3ICOH", "icoh.W")]:
        for sid in range(1, 51):
            files = sorted(glob.glob(
                os.path.join(MODELS, f"{arm}_sub{sid:02d}_seed*_fold*.pt")))
            if not files:
                continue
            import torch
            Ws = []
            for f in files:
                sd = torch.load(f, map_location="cpu", weights_only=True)
                if key in sd:
                    Ws.append(sd[key].numpy())
            if len(Ws) < 2:
                continue
            cons = subspace_consistency(Ws)
            Wm = np.mean(Ws, axis=0)
            e, m_motor, m_front, m_po = energy_share(Wm)
            rows.append((arm, sid, len(Ws), cons, m_motor, m_front, m_po))
            print(f"{arm} sub-{sid:02d}: n={len(Ws)} consistency={cons:.3f} "
                  f"motor={m_motor:.2f} frontal={m_front:.2f} "
                  f"parieto-occ={m_po:.2f}", flush=True)
    df = pd.DataFrame(rows, columns=["arm", "subject", "n_runs", "consistency",
                                     "motor_share", "frontal_share",
                                     "parieto_occ_share"])
    df["random_baseline"] = rand_ref
    df.to_csv(OUT_CSV, index=False)
    print(f"\n===== 汇总 =====")
    print(df.groupby("arm")[["consistency", "motor_share", "frontal_share",
                             "parieto_occ_share"]].mean().round(3))
    print(f"随机基线: {rand_ref:.3f}")


if __name__ == "__main__":
    main()
