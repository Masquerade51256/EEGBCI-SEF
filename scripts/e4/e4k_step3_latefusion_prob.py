"""E4 第 3 步补充：决策层（概率级）后期融合——B 主干 × 协方差分支单体。

动机（2026-11-18 用户确认）：v3 证据链显示 分支有信息（单体 51.65%）
→ 拼接融合未让其进入决策（遮蔽 ≈ 0）→ 融合增益绕道训练动力学。
后期融合（概率平均）在决策层汇合两模型输出，分支信息必然进入判定，
绕开拼接的维度失衡（992:16）问题。

做法：加载存盘的 B 条件模型（e4_step2_factorial/xw/models/B_*.pt）与
S3COVonly 单体模型（e4_step3/xw/models/S3COVonly_*.pt），在同一测试折上
导出 softmax 概率。融合权重固定 w=0.5（无参，不碰测试折调权）。

输出：experiments/e4_step3/{dataset}/latefusion_probs.csv
  列：subject, seed, fold, trial, y, p_B, p_COV（类别 1 的概率）

用法：
  python scripts/e4/e4k_step3_latefusion_prob.py --dataset xw --subjects 1  # 冒烟
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.e4.e4a_broadband_veto import LOADERS
from scripts.e4.e4e_step2_factorial import make_twoband, build_step2_model
from scripts.e4.e4j_step3_cov import build_s3cov_model
from scripts.e2.e2a_spatial import group_kfold_indices

STEP2 = "experiments/e4_step2_factorial"
OUT = "experiments/e4_step3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xw", choices=list(LOADERS.keys()))
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--subjects", default=None)
    args = ap.parse_args()
    import torch

    seeds = [int(s) for s in args.seeds.split(",")]
    dspec = LOADERS[args.dataset]
    subjects = dspec["subjects"] if args.subjects is None else \
        [int(s) for s in args.subjects.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)

    out_dir = os.path.join(OUT, args.dataset)
    out_csv = os.path.join(out_dir, "latefusion_probs.csv")
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(prev.subject)
        print(f"[resume] {len(done)} 被试已完成")

    for sid in subjects:
        if sid in done:
            continue
        t0 = time.time()
        try:
            X, y = dspec["loader"](sid)
        except Exception as e:
            print(f"sub-{sid:02d}: LOAD ERROR {e}", flush=True)
            continue
        Xs = make_twoband(X, dspec["sr"])
        y = y.astype(int)
        tidx = np.arange(len(y))
        rows = []
        for seed in seeds:
            for fi, (tr, te) in enumerate(group_kfold_indices(tidx, 5, seed + sid)):
                mb = os.path.join(STEP2, args.dataset, "models",
                                  f"B_sub{sid:02d}_seed{seed}_fold{fi}.pt")
                mc = os.path.join(out_dir, "models",
                                  f"S3COVonly_sub{sid:02d}_seed{seed}_fold{fi}.pt")
                if not (os.path.exists(mb) and os.path.exists(mc)):
                    print(f"    sub-{sid:02d} seed={seed} fold{fi}: MODEL MISSING",
                          flush=True)
                    continue
                mB = build_step2_model("B", Xs.shape[2], Xs.shape[-1]).to(device)
                mB.load_state_dict(torch.load(mb, map_location=device))
                mB.eval()
                mC = build_s3cov_model(Xs.shape[2], Xs.shape[-1],
                                       cov_only=True).to(device)
                mC.load_state_dict(torch.load(mc, map_location=device))
                mC.eval()
                Xte = torch.from_numpy(Xs[te]).to(device)
                with torch.no_grad():
                    pB = torch.softmax(mB(Xte), dim=1)[:, 1].cpu().numpy()
                    pC = torch.softmax(mC(Xte), dim=1)[:, 1].cpu().numpy()
                for j in range(len(te)):
                    rows.append((sid, seed, fi, int(te[j]), int(y[te[j]]),
                                 float(pB[j]), float(pC[j])))
        df = pd.DataFrame(rows, columns=["subject", "seed", "fold", "trial",
                                         "y", "p_B", "p_COV"])
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
        print(f"[{args.dataset}/latefusion] sub-{sid:02d} done "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n===== {args.dataset}/latefusion 完成 =====")


if __name__ == "__main__":
    main()
