"""E2l：SCP × Connectivity 特征级拼接实验（XW 急性期，trial 级）。

动机（E2 记录 §3.4 遗留项）：
  Connectivity × SlowPotential 是 28 对配对中"增益相关最低的高价值组合"
  （r=0.07，最互斥）。配对 oracle 59.1% 提示两族覆盖不同人群，但增益相关低
  有两种互斥解释：
    (a) 互斥人群：两族各自只在一部分被试上有效，同一人身上无协同；
    (b) 同人协同：两族信息在同一被试的同一 trial 内互补，拼接可超单族。
  特征级拼接是 (a)/(b) 的直接判别检验：若 concat 显著超过两单族 max → (b) 成立；
  若 concat ≈ max（或被高维稀释而更低）→ (a) 成立。

粒度对齐：
  SCP 特征本就是 trial 级（40 trial × 120 维，E2e scp_full 口径）；
  Connectivity 原为窗级（2s 窗 × 5 窗/trial，E2b 口径），此处对每 trial 的
  5 个窗特征取均值得到 trial 级表示（40 trial × 1740 维）。
  两条加载链的标签一致性（affected 映射 + 右瘫翻转）在运行时被试级断言：
  窗标签按 group 取众数必须与 load_xw_trials 标签完全一致，否则中止。

条件（arms）：
  scp_trial        SCP 120 维单用（复核臂，应 ≈ E2e scp_full 54.70%）
  conn_trial_full  conn 1740 维单用 + 收缩 LDA
  conn_trial_sel50 conn 折内 top-50 边筛选 + 收缩 LDA（对应族代表臂 conn_sel50）
  concat_full      [SCP | conn] 1860 维拼接 + 收缩 LDA
  concat_sel50     [SCP | conn_top50] 170 维拼接 + 收缩 LDA

协议（审计铁律）：
  - GroupKFold 5 折，每 trial 一组（group = trial 索引），seeds 42/123/2026，
    折划分种子 seed+sid，与 E2a–E2k 完全同源；
  - zscore 仅训练折拟合（SCP/conn 两块分别归一化后再拼接，避免量纲悬殊）；
  - top-50 边筛选仅用训练折（edge_scores，|点二列相关|）；
  - 分类器统一收缩 LDA（shrinkage="auto", solver="lsqr"；n_train=32 下
    p=50 的 plain LDA 不可逆，故 sel50 臂与 E2b 窗级版的分类器略有差异，特此声明）。

输出：
  experiments/e2_hypothesis_families/e2l_scpconn/xw/{arm}_fold_results.csv
  （列：subject,seed,fold,acc,n_train,n_test）

用法：
  python scripts/e2/e2l_scpconn_concat.py --arm concat_sel50 --subjects 1   # 冒烟
  python scripts/e2/e2l_scpconn_concat.py --arm all                        # 全量
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.e2.e2a_spatial import load_subject, group_kfold_indices
from scripts.e2.e2b_connectivity import connectivity_features, BANDS_2, edge_scores
from scripts.e2.e2e_slowpot import load_xw_trials, scp_features

OUT = "experiments/e2_hypothesis_families/e2l_scpconn"

ARMS = ["scp_trial", "conn_trial_full", "conn_trial_sel50",
        "concat_full", "concat_sel50"]


def subject_features(sid):
    """返回 Fscp (n_trial,120), Fconn (n_trial,1740), y (n_trial,) 并断言对齐。"""
    ds = load_subject("xw", sid)                    # 窗级：250Hz, 4-40Hz, zscore
    X4, y4 = load_xw_trials(sid)                    # trial 级：500Hz 原始 4s
    Fscp = scp_features(X4, 500)                    # (n_trial, 120)

    Fw = connectivity_features(ds.data, BANDS_2, fs=250)   # (n_win, 1740)
    g = np.asarray(ds.group_ids)
    uniq = np.unique(g)
    # --- 对齐断言：窗组 = trial，标签两链一致 ---
    assert len(uniq) == len(y4), (
        f"sub-{sid}: 窗组数 {len(uniq)} != trial 数 {len(y4)}")
    y_win = np.array([np.bincount(ds.labels[g == u].astype(int)).argmax()
                      for u in uniq])
    n_mismatch = int((y_win != y4).sum())
    assert n_mismatch == 0, (
        f"sub-{sid}: 窗级与 trial 级标签不一致 {n_mismatch}/{len(y4)}，中止")
    Fconn = np.stack([Fw[g == u].mean(0) for u in uniq])   # (n_trial, 1740)
    return Fscp, Fconn, y4


def run_arm(Fscp, Fconn, y, sid, seeds, arm):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    groups = np.arange(len(y))
    rows = []
    for seed in seeds:
        for fi, (tr, te) in enumerate(group_kfold_indices(groups, 5, seed + sid)):
            # 折内 zscore：两块分别拟合（拼接臂共享同一归一化，保证可比）
            mu_s, sd_s = Fscp[tr].mean(0), Fscp[tr].std(0) + 1e-12
            mu_c, sd_c = Fconn[tr].mean(0), Fconn[tr].std(0) + 1e-12
            Zs_tr, Zs_te = (Fscp[tr] - mu_s) / sd_s, (Fscp[te] - mu_s) / sd_s
            Zc_tr, Zc_te = (Fconn[tr] - mu_c) / sd_c, (Fconn[te] - mu_c) / sd_c
            if arm == "scp_trial":
                Ftr, Fte = Zs_tr, Zs_te
            elif arm == "conn_trial_full":
                Ftr, Fte = Zc_tr, Zc_te
            elif arm == "conn_trial_sel50":
                edges = np.argsort(-edge_scores(Zc_tr, y[tr]))[:50]
                Ftr, Fte = Zc_tr[:, edges], Zc_te[:, edges]
            elif arm == "concat_full":
                Ftr = np.concatenate([Zs_tr, Zc_tr], 1)
                Fte = np.concatenate([Zs_te, Zc_te], 1)
            elif arm == "concat_sel50":
                edges = np.argsort(-edge_scores(Zc_tr, y[tr]))[:50]
                Ftr = np.concatenate([Zs_tr, Zc_tr[:, edges]], 1)
                Fte = np.concatenate([Zs_te, Zc_te[:, edges]], 1)
            else:
                raise ValueError(arm)
            clf = LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")
            clf.fit(Ftr, y[tr])
            rows.append((sid, seed, fi, float(clf.score(Fte, y[te])),
                         len(tr), len(te)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=ARMS + ["all"])
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--subjects", default="1-50")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if "-" in args.subjects:
        a, b = args.subjects.split("-")
        subjects = list(range(int(a), int(b) + 1))
    else:
        subjects = [int(s) for s in args.subjects.split(",")]
    arms = ARMS if args.arm == "all" else [args.arm]

    for arm in arms:
        out_dir = os.path.join(OUT, "xw")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"{arm}_fold_results.csv")
        done = set()
        if os.path.exists(out_csv):          # resume
            done = set(pd.read_csv(out_csv)["subject"].unique())
        t0 = time.time()
        for sid in subjects:
            if sid in done:
                continue
            Fscp, Fconn, y = subject_features(sid)
            rows = run_arm(Fscp, Fconn, y, sid, seeds, arm)
            df = pd.DataFrame(rows, columns=["subject", "seed", "fold", "acc",
                                             "n_train", "n_test"])
            df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv)
                      or os.path.getsize(out_csv) == 0, index=False)
            print(f"[{arm}] sub-{sid:02d}  acc={df.acc.mean()*100:.2f}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        print(f"[{arm}] DONE -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
