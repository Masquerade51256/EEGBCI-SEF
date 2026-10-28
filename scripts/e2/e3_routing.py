"""E3：个体化特征路由可行性实验（闭环检验第一阶段，XWStroke 急性）。

问题：仅靠患者自己的校准数据，能否自动为他选对信息形式（特征族）？
对比线：最佳单族（可实现下限） < 硬路由 / stacking（可实现路由） < oracle（上限）。

专家（4 个，均为 E2 验证过的管线，EEGNet 因嵌套成本排除）：
  csp    功率族：CSP+LDA，窗口级 → trial 概率取窗平均
  conn   连接族：conn_sel50（折内选 top-50 边）+ LDA，窗口级 → trial 平均
  scp    慢电位族：scp_full（0-4Hz 全导 120 维），trial 级
  aper   非周期族：1/f 斜率+截距，窗口级 → trial 平均

协议 A（全校准嵌套，主读数）：
  外层 GroupKFold 5 折（按 trial，seeds 42/123/2026，与 E2 一致）；
  内层在校准 trials 上 4 折 OOF 估计各族准确率 → 硬路由 argmax；
  stacking：内层 OOF 概率 → LogisticRegression 元学习器（5→4 参数，小数据可行）。

协议 B（N-shot 校准曲线）：
  N ∈ {8,16,32}，20 次随机校准子集；内层 2 折选族（硬路由）+ 均匀集成对照；
  在剩余 trials 上测试。回答"多少校准数据够选对族"。

输出：
  experiments/e3_routing/xw/full_calib_results.csv
  experiments/e3_routing/xw/nshot_results.csv

用法：
  python scripts/e2/e3_routing.py [--seeds 42,123,2026]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.e2.e2a_spatial import DATASETS, load_subject, group_kfold_indices
from scripts.e2.e2b_connectivity import connectivity_features, edge_scores, BANDS_2
from scripts.e2.e2e_slowpot import load_xw_trials, scp_features, LOADERS
import scripts.e2.e2e_slowpot as e2e
from scripts.e2.e2cd_nonlinear_aperiodic import feat_aperiodic

OUT = "experiments/e3_routing"
EXPERTS = ["csp", "conn", "scp", "aper"]


# ---------- 专家拟合/预测（返回 trial 级概率） ----------

def _agg_trial(prob_win, g, trial_ids):
    return np.array([prob_win[g == t].mean() for t in trial_ids])


def fit_predict_csp(Xtr_w, ytr_w, gtr, Xte_w, gte, tr_ids, te_ids):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline
    from mne.decoding import CSP
    clf = Pipeline([("csp", CSP(n_components=6, log=True, norm_trace=False)),
                    ("lda", LinearDiscriminantAnalysis())])
    clf.fit(Xtr_w, ytr_w)
    return (_agg_trial(clf.predict_proba(Xtr_w)[:, 1], gtr, tr_ids),
            _agg_trial(clf.predict_proba(Xte_w)[:, 1], gte, te_ids))


def fit_predict_conn(Ftr_w, ytr_w, gtr, Fte_w, gte, tr_ids, te_ids):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    edges = np.argsort(-edge_scores(Ftr_w, ytr_w))[:50]     # 选择仅用本段训练数据
    mu, sd = Ftr_w[:, edges].mean(0), Ftr_w[:, edges].std(0) + 1e-12
    clf = LinearDiscriminantAnalysis()
    clf.fit((Ftr_w[:, edges] - mu) / sd, ytr_w)
    ptr = clf.decision_function((Ftr_w[:, edges] - mu) / sd)
    pte = clf.decision_function((Fte_w[:, edges] - mu) / sd)
    sig = lambda z: 1 / (1 + np.exp(-z))
    return (_agg_trial(sig(ptr), gtr, tr_ids), _agg_trial(sig(pte), gte, te_ids))


def fit_predict_trial_level(Ftr, ytr, Fte):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-12
    clf = LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")
    clf.fit((Ftr - mu) / sd, ytr)
    sig = lambda z: 1 / (1 + np.exp(-z))
    return sig(clf.decision_function((Ftr - mu) / sd)), sig(clf.decision_function((Fte - mu) / sd))


def fit_predict_aper(Ftr_w, ytr_w, gtr, Fte_w, gte, tr_ids, te_ids):
    ptr, pte = fit_predict_trial_level(Ftr_w, ytr_w, Fte_w)
    return _agg_trial(ptr, gtr, tr_ids), _agg_trial(pte, gte, te_ids)


class SubjectData:
    """每个被试预计算全部特征，避免嵌套内重复计算。"""

    def __init__(self, sid):
        ds = load_subject("xw", sid)
        self.Xw = ds.data[:, 0, :, :].astype(np.float64)  # (n_win, 30, 500)
        self.yw = ds.labels
        self.g = ds.group_ids
        self.trial_ids = np.unique(self.g)
        # trial 级标签（已验证与慢电位 loader 一致）
        self.yt = np.array([self.yw[self.g == t][0] for t in self.trial_ids])
        # 预计算特征
        self.Fconn = connectivity_features(self.Xw, BANDS_2)
        self.Faper = feat_aperiodic(self.Xw)
        e2e.LOADERS_CUR = LOADERS["xw"]
        Xt, _ = load_xw_trials(sid)
        self.Fscp = scp_features(Xt, LOADERS["xw"][1])     # trial 级

    def _win_mask(self, trial_subset):
        return np.isin(self.g, trial_subset)

    def expert_trial_probs(self, tr_ids, te_ids):
        """返回 dict expert -> (p_train_trials, p_test_trials)"""
        out = {}
        mtr, mte = self._win_mask(tr_ids), self._win_mask(te_ids)
        # csp
        out["csp"] = fit_predict_csp(self.Xw[mtr], self.yw[mtr], self.g[mtr],
                                     self.Xw[mte], self.g[mte],
                                     tr_ids, te_ids)
        out["conn"] = fit_predict_conn(self.Fconn[mtr], self.yw[mtr], self.g[mtr],
                                       self.Fconn[mte], self.g[mte],
                                       tr_ids, te_ids)
        out["aper"] = fit_predict_aper(self.Faper[mtr], self.yw[mtr], self.g[mtr],
                                       self.Faper[mte], self.g[mte],
                                       tr_ids, te_ids)
        itr = np.isin(self.trial_ids, tr_ids)
        ite = np.isin(self.trial_ids, te_ids)
        out["scp"] = fit_predict_trial_level(self.Fscp[itr], self.yt[itr],
                                             self.Fscp[ite])
        return out


def inner_oof_probs(sd: SubjectData, calib_ids, y_calib, n_inner=4, seed=0):
    """校准集内层 OOF：每专家的 trial 概率矩阵 + 准确率（分层折，防小样本单类）。"""
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=min(n_inner, len(calib_ids) // 2),
                         shuffle=True, random_state=seed)
    P = np.zeros((len(calib_ids), len(EXPERTS)))
    for itr, ite in kf.split(calib_ids, y_calib):
        probs = sd.expert_trial_probs(calib_ids[itr], calib_ids[ite])
        for j, ex in enumerate(EXPERTS):
            P[ite, j] = probs[ex][1]
    accs = {ex: float(((P[:, j] > 0.5).astype(int) == y_calib).mean())
            for j, ex in enumerate(EXPERTS)}
    return P, accs


def protocol_a(sd, seeds):
    """全校准嵌套：硬路由 / stacking / 单族 / oracle"""
    from sklearn.linear_model import LogisticRegression
    rows = []
    for seed in seeds:
        for fi, (tr_i, te_i) in enumerate(group_kfold_indices(
                np.arange(len(sd.trial_ids)), 5, seed + sd.sid)):
            tr_ids, te_ids = sd.trial_ids[tr_i], sd.trial_ids[te_i]
            y_tr = np.array([sd.yt[sd.trial_ids == t][0] for t in tr_ids])
            y_te = np.array([sd.yt[sd.trial_ids == t][0] for t in te_ids])
            P_oof, accs = inner_oof_probs(sd, tr_ids, y_tr, seed=seed * 10 + fi)
            probs = sd.expert_trial_probs(tr_ids, te_ids)
            Pte = np.column_stack([probs[ex][1] for ex in EXPERTS])
            # 单族
            for j, ex in enumerate(EXPERTS):
                rows.append((sd.sid, seed, fi, f"single_{ex}",
                             float(((Pte[:, j] > 0.5).astype(int) == y_te).mean())))
            # 硬路由
            best = int(np.argmax([accs[ex] for ex in EXPERTS]))
            rows.append((sd.sid, seed, fi, "hard_routing",
                         float(((Pte[:, best] > 0.5).astype(int) == y_te).mean())))
            rows.append((sd.sid, seed, fi, "hard_routing_choice", str(best)))
            # stacking
            meta = LogisticRegression(C=1.0).fit(P_oof, y_tr)
            rows.append((sd.sid, seed, fi, "stacking",
                         float(meta.score(Pte, y_te))))
            # 均匀集成
            rows.append((sd.sid, seed, fi, "uniform_ens",
                         float(((Pte.mean(1) > 0.5).astype(int) == y_te).mean())))
            # oracle
            test_accs = [((Pte[:, j] > 0.5).astype(int) == y_te).mean()
                         for j in range(len(EXPERTS))]
            rows.append((sd.sid, seed, fi, "oracle", float(max(test_accs))))
    return rows


def protocol_b(sd, seeds, n_list=(8, 16, 32), n_rep=5):
    """N-shot：硬路由 + 均匀集成 + 单族"""
    rng_master = np.random.default_rng(0)
    rows = []
    n_all = len(sd.trial_ids)
    for seed in seeds:
        for N in n_list:
            for rep in range(n_rep):
                rng = np.random.default_rng(seed * 1000 + rep)
                # 分层抽样：每类各取 N/2，保证校准集两类都在
                pos0 = np.where(sd.yt == 0)[0]
                pos1 = np.where(sd.yt == 1)[0]
                calib_pos = np.concatenate([
                    rng.choice(pos0, N // 2, replace=False),
                    rng.choice(pos1, N - N // 2, replace=False)])
                test_pos = np.setdiff1d(np.arange(n_all), calib_pos)
                tr_ids, te_ids = sd.trial_ids[calib_pos], sd.trial_ids[test_pos]
                y_tr = sd.yt[calib_pos]
                y_te = sd.yt[test_pos]
                # 内层 2 折选族
                _, accs = inner_oof_probs(sd, tr_ids, y_tr, n_inner=2,
                                          seed=int(rng.integers(1e9)))
                probs = sd.expert_trial_probs(tr_ids, te_ids)
                Pte = np.column_stack([probs[ex][1] for ex in EXPERTS])
                best = int(np.argmax([accs[ex] for ex in EXPERTS]))
                rows.append((sd.sid, seed, N, rep, "hard_routing",
                             float(((Pte[:, best] > 0.5).astype(int) == y_te).mean())))
                rows.append((sd.sid, seed, N, rep, "uniform_ens",
                             float(((Pte.mean(1) > 0.5).astype(int) == y_te).mean())))
                for j, ex in enumerate(EXPERTS):
                    rows.append((sd.sid, seed, N, rep, f"single_{ex}",
                                 float(((Pte[:, j] > 0.5).astype(int) == y_te).mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--part", default="ab", choices=["a", "b", "ab"])
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    out_dir = os.path.join(OUT, "xw")
    os.makedirs(out_dir, exist_ok=True)
    csv_a = os.path.join(out_dir, "full_calib_results.csv")
    csv_b = os.path.join(out_dir, "nshot_results.csv")

    done_a, done_b = set(), set()
    if os.path.exists(csv_a):
        prev = pd.read_csv(csv_a)
        done_a = set(zip(prev.subject, prev.seed))
    if os.path.exists(csv_b):
        prev = pd.read_csv(csv_b)
        done_b = set(prev.subject)

    for sid in DATASETS["xw"]["subjects"]:
        t0 = time.time()
        try:
            sd = SubjectData(sid)
            sd.sid = sid
        except Exception as e:
            print(f"sub-{sid:02d}: LOAD ERROR {e}", flush=True)
            continue
        if "a" in args.part:
            remain = [s for s in seeds if (sid, s) not in done_a]
            if remain:
                rows = protocol_a(sd, remain)
                df = pd.DataFrame(rows, columns=["subject", "seed", "fold",
                                                 "condition", "value"])
                df.to_csv(csv_a, mode="a", header=not os.path.exists(csv_a),
                          index=False)
        if "b" in args.part and sid not in done_b:
            rows = protocol_b(sd, seeds)
            df = pd.DataFrame(rows, columns=["subject", "seed", "n_shot", "rep",
                                             "condition", "acc"])
            df.to_csv(csv_b, mode="a", header=not os.path.exists(csv_b),
                      index=False)
        print(f"sub-{sid:02d} done ({time.time()-t0:.0f}s)", flush=True)

    print("\n===== E3 完成 =====")


if __name__ == "__main__":
    main()
