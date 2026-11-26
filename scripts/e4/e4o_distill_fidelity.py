"""E4 第 3 步 S3-D 蒸馏保真度诊断（e4o，只读诊断，不产生新分类结果）。

动机（2026-11-24 讨论，用户确认执行）：
S3-D XW 判负的核心证据是"单体 × E2 连接族 ρ=−0.04"，但存在诊断缺口——
从未检查阶段 1 蒸馏的拟合质量。本脚本区分两种失效：
  ① 教师没教进去（阶段 1 回归 R² 低）→ 学生架构/训练配比问题，v2 可修；
  ② 教进去了但任务无用（R² 高、分类无增益）→ 教师信息与标签弱相关，v2 无意义。

双重视角：
  A. 阶段 1 复演：按 e4n 原协议（同 seed、同折划分、同超参）重跑阶段 1
     无标签回归（100 epoch，仅 32 训练 trial，秒级），在测试折上计算
     教师回归 R²（训练折 R² 一并记录，用于识别纯记忆化过拟合）；
     阶段 1 是原运行中 torch.manual_seed 后的首个 RNG 消费者，
     复演与原阶段 1 等价（GPU 数值非确定性量级可忽略）。
  B. 微调后存活探针：加载已保存的 S3Donly 模型（750 个），提取微调后
     trunk 表示，训练折 ridge 回归 → 教师，测试折 R²——回答"教师信息
     是否在任务微调后仍线性存在于分支表示中"。

R² 分组报告：非周期（前 2*n_ch 维）/ 区域网格 wPLI（其余维）。
R² 定义：逐维 1 − SS_res/SS_var（SS_var 以被评估集合自身均值计），组内取均值。

输出：experiments/e4_step3/xw/S3D_fidelity.csv
用法：
  python scripts/e4/e4o_distill_fidelity.py --dataset xw --subjects 1 --seeds 42
  bash scripts/e4/run_e4o_xw.sh
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
from scripts.e4.e4n_step3_distill import (REGIONS, teacher_features,
                                          build_s3d_model)
from scripts.e2.e2a_spatial import group_kfold_indices

OUT = "experiments/e4_step3"


def r2_by_group(pred, true, ap_dim):
    """pred/true: (n, D) → (r2_ap, r2_wpli)，逐维 R² 组内均值。"""
    def _r2(p, t):
        ss_res = ((t - p) ** 2).sum(0)
        ss_var = ((t - t.mean(0, keepdims=True)) ** 2).sum(0) + 1e-12
        return 1.0 - ss_res / ss_var
    r2 = _r2(pred, true)
    return float(r2[:ap_dim].mean()), float(r2[ap_dim:].mean())


def run_subject(Xs, y, Tch, sid, seeds, out_dir):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.linear_model import Ridge

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_ch, T = Xs.shape[2], Xs.shape[-1]
    ap_dim = 2 * n_ch
    tidx = np.arange(len(y))
    rows = []
    for seed in seeds:
        for fi, (tr, te) in enumerate(group_kfold_indices(tidx, 5, seed + sid)):
            # —— A. 阶段 1 复演（与 e4n 同 RNG 顺序）——
            torch.manual_seed(seed)
            np.random.seed(seed)
            pre_cls, OnlyCls = build_s3d_model(n_ch, T, Tch.shape[1],
                                               distill_only=True)
            pre = pre_cls.to(device)
            mu = Tch[tr].mean(0, keepdims=True)
            sd = Tch[tr].std(0, keepdims=True) + 1e-8
            Tz = ((Tch - mu) / sd).astype(np.float32)
            opt = torch.optim.Adam(pre.parameters(), lr=1e-3)
            lossf = nn.MSELoss()
            loader = DataLoader(TensorDataset(torch.from_numpy(Xs[tr]),
                                              torch.from_numpy(Tz[tr])),
                                batch_size=16, shuffle=True, num_workers=0)
            for _ in range(100):
                pre.train()
                for xb, tb in loader:
                    xb, tb = xb.to(device), tb.to(device)
                    opt.zero_grad()
                    loss = lossf(pre(xb), tb)
                    loss.backward()
                    opt.step()
            pre.eval()
            with torch.no_grad():
                p_te = pre(torch.from_numpy(Xs[te]).to(device)).cpu().numpy()
                p_tr = pre(torch.from_numpy(Xs[tr]).to(device)).cpu().numpy()
            r2_ap, r2_wp = r2_by_group(p_te, Tz[te], ap_dim)
            r2_ap_tr, r2_wp_tr = r2_by_group(p_tr, Tz[tr], ap_dim)

            # —— B. 微调后 trunk 存活探针（加载已存 S3Donly 模型）——
            r2f_ap = r2f_wp = np.nan
            mpath = os.path.join(out_dir, "models",
                                 f"S3Donly_sub{sid:02d}_seed{seed}_fold{fi}.pt")
            if os.path.exists(mpath):
                pre2 = build_s3d_model(n_ch, T, Tch.shape[1],
                                       distill_only=True)[0]
                mdl = OnlyCls(pre2.trunk)
                mdl.load_state_dict(torch.load(mpath, map_location=device))
                mdl.eval()
                with torch.no_grad():
                    e_tr = mdl.trunk(torch.from_numpy(Xs[tr]).to(device)) \
                        .cpu().numpy()
                    e_te = mdl.trunk(torch.from_numpy(Xs[te]).to(device)) \
                        .cpu().numpy()
                rg = Ridge(alpha=1.0)
                rg.fit(e_tr, Tz[tr])
                r2f_ap, r2f_wp = r2_by_group(rg.predict(e_te), Tz[te], ap_dim)

            rows.append((sid, seed, fi, r2_ap, r2_wp, r2_ap_tr, r2_wp_tr,
                         r2f_ap, r2f_wp))
            print(f"    sub-{sid:02d} seed={seed} fold{fi}: "
                  f"s1_test ap={r2_ap:.3f} wpli={r2_wp:.3f} | "
                  f"s1_train ap={r2_ap_tr:.3f} wpli={r2_wp_tr:.3f} | "
                  f"ft_test ap={r2f_ap:.3f} wpli={r2f_wp:.3f}", flush=True)
    return rows


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
    out_csv = os.path.join(out_dir, "S3D_fidelity.csv")
    cols = ["subject", "seed", "fold", "r2_s1_ap_test", "r2_s1_wpli_test",
            "r2_s1_ap_train", "r2_s1_wpli_train", "r2_ft_ap_test",
            "r2_ft_wpli_test"]
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
        rows = run_subject(Xs, y.astype(int), Tch, sid, remain, out_dir)
        print(f"[{args.dataset}/fidelity] sub-{sid:02d} done "
              f"({time.time()-t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv),
                  index=False)

    print(f"\n===== {args.dataset}/S3D_fidelity 完成 =====")


if __name__ == "__main__":
    main()
