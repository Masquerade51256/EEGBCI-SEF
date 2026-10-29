"""E2j：源空间特征的 within-subject 检验（XWStroke 主战场 + BCIC 阳性对照 + HS 参考）。

设计文档：E2_实验设计_20261026.md §6 E2j。
假设族：病灶会畸变头皮电位分布，判别信息在皮层源空间可能比传感器空间更规整。
实现（无个体 MRI）：球形头模型（按各数据集电极位置自动拟合）+ 体积源空间
（球内 10mm 网格）+ eLORETA 逆解（ad-hoc 噪声协方差，数据无关，无泄漏）。

配对消融约定：与 E2a csp_full 用**完全相同的窗口数据与折划分**——只换
"传感器空间 CSP 功率"→"源空间频段功率"，分类器同为 LDA 系。

特征（每窗口一个样本）：
  窗口数据（复用 E2a 的 load_subject：250Hz、4-40Hz、2s 窗、逐窗 zscore）
  → eLORETA 源时间序列（逆算子每数据集只算一次，线性变换）
  → Welch 谱在 mu(8-13)/beta(13-30) 两带的功率（三朝向功率求和）→ log
  → 特征 = n_src × 2

条件（arms）：
  src_full   全源特征 + 收缩 LDA
  src_sel50  折内嵌套特征选择（训练折 |点二列相关| top-50）+ LDA

协议：GroupKFold 5 折按 trial（group_ids），seeds 42/123/2026；折内标准化。

注：HS 仅 11 导，逆解极度病态，其结果只作参考、不作结论依据。

输出：
  experiments/e2_hypothesis_families/e2j_source/{dataset}/{arm}_fold_results.csv

用法：
  python scripts/e2/e2j_source.py --dataset xw --arm src_full
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.e2.e2a_spatial import DATASETS, group_kfold_indices, load_subject

OUT = "experiments/e2_hypothesis_families/e2j_source"
SR = 250
BANDS = [(8.0, 13.0), (13.0, 30.0)]   # mu, beta

MONTAGE = {"xw": "standard_alphabetic",   # T3/T4/T5/T6 旧名
           "bcic": "standard_1020",
           "hs": "standard_1020",
           "hs_all": "standard_1020"}


def build_inverse(ch_names):
    """球形头模型 + 体积源空间 + eLORETA 逆算子。每数据集只调一次。"""
    import mne
    mne.set_log_level("ERROR")
    info = mne.create_info(ch_names=list(ch_names), sfreq=SR, ch_types="eeg")
    montage = mne.channels.make_standard_montage(
        MONTAGE.get(CUR_DATASET, "standard_1020"))
    info.set_montage(montage, on_missing="warn")
    # 球形头模型（按电极位置拟合，3 层 EEG 标准电导）
    bem = mne.make_sphere_model(r0="auto", head_radius="auto", info=info,
                                relative_radii=(0.90, 0.92, 0.97, 1.0),
                                sigmas=(0.33, 1.0, 0.004, 0.33))
    src = mne.setup_volume_source_space(
        subject=None, pos=10.0, bem=bem, sphere=None, verbose="ERROR")
    try:
        fwd = mne.make_forward_solution(info, trans=None, src=src, bem=bem,
                                        eeg=True, meg=False, verbose="ERROR")
    except Exception:
        fwd = mne.make_forward_solution(info, trans=np.eye(4), src=src, bem=bem,
                                        eeg=True, meg=False, verbose="ERROR")
    noise_cov = mne.make_ad_hoc_cov(info)
    inv = mne.minimum_norm.make_inverse_operator(
        info, fwd, noise_cov, loose=1.0, depth=0.8, fixed=False, verbose="ERROR")
    return inv, info


def source_bandpower(X, inv, ch_names):
    """X: (n_win, n_ch, T) -> (n_win, n_src*2) log 频段功率（三朝向合并）。"""
    import mne
    from scipy.signal import welch
    mne.set_log_level("ERROR")
    n, n_ch, T = X.shape
    # 逆算子是线性算子：用 apply_inverse_epochs 批量施加
    info = mne.create_info(ch_names=list(ch_names), sfreq=SR, ch_types="eeg")
    montage = mne.channels.make_standard_montage(
        MONTAGE.get(CUR_DATASET, "standard_1020"))
    info.set_montage(montage, on_missing="warn")
    epochs = mne.EpochsArray(X, info, tmin=0.0, verbose="ERROR")
    stcs = mne.minimum_norm.apply_inverse_epochs(
        epochs, inv, lambda2=1.0 / 9.0, method="eLORETA",
        pick_ori=None, verbose="ERROR")
    # 所有 epoch 的源序列堆成 (n, 3*n_src, T)
    S = np.stack([s.data for s in stcs])
    freqs, P = welch(S, fs=SR, axis=-1, nperseg=min(256, T))
    feats = []
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs <= hi)
        bp = P[..., m].mean(axis=-1)          # (n, 3*n_src)
        bp = bp.reshape(n, 3, -1).sum(axis=1)  # 三朝向功率求和 -> (n, n_src)
        feats.append(np.log(bp + 1e-20))
    return np.concatenate(feats, axis=1)


def run_src(X, y, g, sid, seeds, arm, inv, ch_names):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    F = source_bandpower(X, inv, ch_names)    # 逐窗独立，无泄漏
    rows = []
    for seed in seeds:
        for fi, (tr, te) in enumerate(group_kfold_indices(g, 5, seed + sid)):
            Ftr, Fte = F[tr], F[te]
            mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-12
            Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
            if arm == "src_sel50":
                yz = (y[tr] - y[tr].mean()) / (y[tr].std() + 1e-12)
                sc = np.abs((Ftr * yz[:, None]).mean(0))
                idx = np.argsort(-sc)[:50]
                Ftr, Fte = Ftr[:, idx], Fte[:, idx]
                clf = LinearDiscriminantAnalysis()
            else:
                clf = LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")
            clf.fit(Ftr, y[tr])
            acc = float(clf.score(Fte, y[te]))
            rows.append((sid, seed, fi, acc, len(tr), len(te), F.shape[1]))
    return rows


CUR_DATASET = "xw"


def main():
    global CUR_DATASET
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["src_full", "src_sel50"])
    ap.add_argument("--dataset", default="xw", choices=list(DATASETS.keys()))
    ap.add_argument("--seeds", default="42,123,2026")
    args = ap.parse_args()
    CUR_DATASET = args.dataset
    seeds = [int(s) for s in args.seeds.split(",")]

    ch_names = DATASETS[args.dataset]["ch_names"]
    t0 = time.time()
    inv, _ = build_inverse(ch_names)
    print(f"[{args.dataset}] inverse operator built ({time.time()-t0:.0f}s)", flush=True)

    out_dir = os.path.join(OUT, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{args.arm}_fold_results.csv")
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(zip(prev.subject, prev.seed))
        print(f"[resume] {len(done)} (subject, seed) 已完成")

    for sid in DATASETS[args.dataset]["subjects"]:
        remain = [s for s in seeds if (sid, s) not in done]
        if not remain:
            continue
        t0 = time.time()
        try:
            ds = load_subject(args.dataset, sid)
        except Exception as e:
            print(f"sub-{sid:02d}: LOAD ERROR {e}", flush=True)
            continue
        X = ds.data[:, 0, :, :].astype(np.float64)
        y, g = ds.labels, ds.group_ids
        rows = run_src(X, y, g, sid, remain, args.arm, inv, ch_names)
        m = np.mean([r[3] for r in rows])
        print(f"[{args.dataset}/{args.arm}] sub-{sid:02d} done: {m*100:.2f}%  ({time.time()-t0:.0f}s)",
              flush=True)
        df = pd.DataFrame(rows, columns=["subject", "seed", "fold", "acc",
                                         "n_train", "n_test", "n_feat"])
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    print(f"\n===== {args.dataset}/{args.arm} 完成 =====")


if __name__ == "__main__":
    main()
