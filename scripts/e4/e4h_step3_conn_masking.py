"""E4 第 3 步 S3-C 机制闭环（判读线④）：连接分支遮蔽敏感度分析。

设计：E4_实验设计_草案_20261105.md §13.4（2026-11-10 用户确认执行）。

做法：加载 S3-C 存盘模型（models/S3C_sub*_seed*_fold*.pt），在同一测试折上
评估两次——完整模型 vs 连接分支输出置零（遮蔽），得逐折敏感度
sensitivity = acc_full − acc_masked。下游分析：逐被试敏感度 × E2 连接族
（wPLI/iCoh）逐被试表现的 Spearman 相关（本地离线完成）。

实现说明：S3CNet.features 返回 concat(feat_B, feat_conn)，feat_conn 为最后
CONN_DIM=16 维；eval 模式下 dropout 为恒等，故遮蔽 = 将特征最后 16 维置零。

输出：experiments/e4_step3/{dataset}/S3C_masking.csv
  列：subject, seed, fold, acc_full, acc_masked

用法：
  python scripts/e4/e4h_step3_conn_masking.py --dataset xw --subjects 1  # 冒烟
  bash scripts/e4/run_e4h.sh                                            # 全量
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
from scripts.e4.e4f_step3_conn import build_s3c_model
from scripts.e2.e2a_spatial import group_kfold_indices

OUT = "experiments/e4_step3"
CONN_DIM = 16     # S3-C 连接分支输出维度（d=16，见 e4f_step3_conn.ConnBranch）


def run_subject(Xs, y, sid, seeds, out_dir):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    tidx = np.arange(len(y))
    rows = []
    for seed in seeds:
        for fi, (tr, te) in enumerate(group_kfold_indices(tidx, 5, seed + sid)):
            mpath = os.path.join(out_dir, "models",
                                 f"S3C_sub{sid:02d}_seed{seed}_fold{fi}.pt")
            if not os.path.exists(mpath):
                print(f"    sub-{sid:02d} seed={seed} fold{fi}: MODEL MISSING",
                      flush=True)
                continue
            model = build_s3c_model(Xs.shape[2], Xs.shape[-1]).to(device)
            model.load_state_dict(torch.load(mpath, map_location=device))
            model.eval()
            Xte = torch.from_numpy(Xs[te]).to(device)
            yte = torch.from_numpy(y[te]).long().to(device)
            with torch.no_grad():
                feat, _ = model.features(Xte)
                acc_full = float((model.head(feat).argmax(1) == yte)
                                 .float().mean().item())
                feat_m = feat.clone()
                feat_m[:, -CONN_DIM:] = 0.0           # 遮蔽连接分支
                acc_masked = float((model.head(feat_m).argmax(1) == yte)
                                   .float().mean().item())
            rows.append((sid, seed, fi, acc_full, acc_masked))
            print(f"    sub-{sid:02d} seed={seed} fold{fi}: full {acc_full*100:.1f}%"
                  f" masked {acc_masked*100:.1f}%", flush=True)
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
    out_csv = os.path.join(out_dir, "S3C_masking.csv")
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
        rows = run_subject(Xs, y.astype(int), sid, remain, out_dir)
        print(f"[{args.dataset}/S3C-masking] sub-{sid:02d} done "
              f"({time.time()-t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows, columns=["subject", "seed", "fold",
                                         "acc_full", "acc_masked"])
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    print(f"\n===== {args.dataset}/S3C-masking 完成 =====")


if __name__ == "__main__":
    main()
