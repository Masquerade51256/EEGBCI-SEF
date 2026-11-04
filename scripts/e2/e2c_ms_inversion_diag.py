# -*- coding: utf-8 -*-
"""E2c 微状态"系统性低于机会概率"的机制诊断（XW 急性期）。

背景：ms_knn 群体均值 47.66%（显著低于 50%），18/50 被试跨 3 seeds 一致反转。
两个候选机制：
  M1 类别条件分布不对称（致密类被 KNN 系统性偏爱）——不随时间变化；
  M2 特征-标签映射随 session 时间漂移且符号反转——时间距离越大反转越强。

诊断设计（全部被试，KNN 臂，严格折内模板/归一化）：
  A 前半 session 训练 -> 后半测试（最大时间间隔）
  B 后半训练 -> 前半测试（反向，检验方向不对称）
  C 随机半分对照（2 seeds，时间交错，时间间隔最小）
  D 散布度描述：全部数据拟合模板后，两类特征在 z 空间中的平均类内距离差

判读：
  - A/B 显著低于 C -> 支持 M2（非平稳反转）；
  - A/B 与 C 相当且都低于 50%，且反转幅度与散布不对称度相关 -> 支持 M1。

输出：experiments/e2_hypothesis_families/e2c_microstate/xw/ms_inversion_diag.csv
用法：python scripts/e2/e2c_ms_inversion_diag.py
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.e2.e2a_spatial import DATASETS, load_subject
from scripts.e2.e2c_microstate import fit_templates, segment, ms_features, K_STATES

OUT = "experiments/e2_hypothesis_families/e2c_microstate/xw"
os.makedirs(OUT, exist_ok=True)


def knn_eval(Xtr, ytr, Xte, yte, seed):
    """折内模板+归一化+KNN，返回 (acc, 预测为患侧的比例)"""
    from sklearn.neighbors import KNeighborsClassifier
    T = fit_templates(Xtr, seed=seed)
    Ftr = ms_features(segment(Xtr, T)); Fte = ms_features(segment(Xte, T))
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-12
    Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
    clf = KNeighborsClassifier(n_neighbors=5).fit(Ftr, ytr)
    pred = clf.predict(Fte)
    return float((pred == yte).mean()), float((pred == 0).mean())


def dispersion_asym(X, y, seed=0):
    """全部数据拟合模板（描述性诊断，不用于分类），返回 患侧类内均距 − 健侧类内均距"""
    T = fit_templates(X, seed=seed)
    F = ms_features(segment(X, T))
    mu, sd = F.mean(0), F.std(0) + 1e-12
    F = (F - mu) / sd
    out = {}
    for cls, nm in [(0, "aff"), (1, "una")]:
        Fc = F[y == cls]
        cent = Fc.mean(0)
        out[nm] = float(np.linalg.norm(Fc - cent, axis=1).mean())
    return out["aff"] - out["una"], out["aff"], out["una"]


def main():
    rows = []
    for sid in DATASETS["xw"]["subjects"]:
        t0 = time.time()
        try:
            ds = load_subject("xw", sid)
        except Exception as e:
            print(f"sub-{sid:02d}: LOAD ERROR {e}", flush=True)
            continue
        X = ds.data[:, 0, :, :]; y = ds.labels; g = ds.group_ids
        trials = np.unique(g)                      # 假定编号顺序 = 时间顺序
        half = len(trials) // 2
        tr_early = np.isin(g, trials[:half]); tr_late = np.isin(g, trials[half:])

        acc_a, bias_a = knn_eval(X[tr_early], y[tr_early], X[tr_late], y[tr_late], seed=0)
        acc_b, bias_b = knn_eval(X[tr_late], y[tr_late], X[tr_early], y[tr_early], seed=1)
        cs, cbs = [], []
        for seed in (42, 123):
            rng = np.random.default_rng(seed)
            perm = rng.permutation(trials)
            tr_mask = np.isin(g, perm[:half])
            acc_c, bias_c = knn_eval(X[tr_mask], y[tr_mask], X[~tr_mask], y[~tr_mask], seed=seed)
            cs.append(acc_c); cbs.append(bias_c)
        asym, d_aff, d_una = dispersion_asym(X, y)
        rows.append(dict(subject=sid, acc_A_early2late=acc_a, bias_A=bias_a,
                         acc_B_late2early=acc_b, bias_B=bias_b,
                         acc_C_random=np.mean(cs), bias_C=np.mean(cbs),
                         disp_asym=asym, disp_aff=d_aff, disp_una=d_una))
        print(f"sub-{sid:02d} done: A={acc_a*100:.1f} B={acc_b*100:.1f} "
              f"C={np.mean(cs)*100:.1f} asym={asym:+.2f} ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "ms_inversion_diag.csv"), index=False)

    df = pd.DataFrame(rows)
    from scipy import stats
    ab = df[["acc_A_early2late", "acc_B_late2early"]].mean(axis=1)
    print("\n=== 汇总 ===")
    print("A/B(时间远) 均值 %.3f vs C(时间交错) 均值 %.3f, Wilcoxon p=%.4f"
          % (ab.mean(), df.acc_C_random.mean(), stats.wilcoxon(ab, df.acc_C_random).pvalue))
    print("预测偏差(预测为患侧比例): A %.3f B %.3f C %.3f"
          % (df.bias_A.mean(), df.bias_B.mean(), df.bias_C.mean()))
    cv_path = "experiments/e2_hypothesis_families/e2c_microstate/xw/ms_knn_fold_results.csv"
    cv = pd.read_csv(cv_path) if os.path.exists(cv_path) else None
    if cv is not None:
        gm = cv.groupby("subject")["acc"].mean()
        d2 = df.set_index("subject").loc[gm.index]
        from scipy.stats import spearmanr
        rho, p = spearmanr(d2.disp_asym, gm.values)
        print("散布不对称度 × 标准CV准确率: rho=%.3f p=%.4f" % (rho, p))
    print("saved", os.path.join(OUT, "ms_inversion_diag.csv"))


if __name__ == "__main__":
    main()
