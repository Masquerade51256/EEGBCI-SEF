"""E2e：慢电位（MRCP 类）特征的 within-subject 检验（XWStroke 主战场 + BCIC 阳性对照）。

设计文档：E2_实验设计_20261026.md §3 E2e。
假设族：判别信息存在于 0–4 Hz 慢速电位（准备电位/MRCP 类）的幅值与斜率中，
该信息被标准管线的 4–40 Hz 滤波 + 逐窗 zscore 物理删除。

**为什么必须独立加载、绕过标准管线**：
  - 标准管线的 temporal_standardization（逐窗 zscore）会抹平慢电位的幅值信息；
  - 需要保留 0–4 Hz 频段（标准管线滤掉）。
  XW：直接读 sourcedata 的 4s .mat（500 Hz 未滤波原始分段）；
  BCIC：复刻 BCICIV2aDataset._load_gdf_session（raw.filter(0., 40.) 本就保留慢电位）。

特征（每 trial，整段 4s/3s epoch 为一个样本，不做滑窗）：
  低通 4 Hz（butter4 filtfilt）→ 基线校正（减前 1/4 段均值）→
  每通道：后 3 个等长子窗均值 + 后半段线性斜率 = 4 特征/通道。

条件（arms）：
  scp_full   全导特征（30×4=120 维）+ 收缩 LDA
  scp_motor  运动区 7 导（FC3/FC4/Cz/C3/C4/CP3/CP4 或其 BCIC 对应）+ LDA
  scp_sel20  折内嵌套特征选择（训练折 |点二列相关| top-20）+ LDA

协议：GroupKFold 5 折（每 trial 即一组），seeds 42/123/2026；
标签口径与 E2a 一致（XW 用 affected 映射 + 右瘫翻转对齐；BCIC 取左/右手两类）。

输出：
  experiments/e2_hypothesis_families/e2e_slowpot/{dataset}/{arm}_fold_results.csv

用法：
  python scripts/e2/e2e_slowpot.py --dataset xw --arm scp_full
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, ".")
from data.datasets import _load_paralysis_side, _flip_xwstroke_channels
from scripts.e2.e2a_spatial import DATASETS, group_kfold_indices

OUT = "experiments/e2_hypothesis_families/e2e_slowpot"

XW_DIR = "src/datasets/21679035/sourcedata"
XW_SR = 500
XW_EPOCH_S = 4.0
# 运动区 7 导（对齐坐标系，见 e2a_spatial 通道表）
XW_MOTOR = [8, 9, 12, 13, 14, 18, 19]   # FC3, FC4, Cz, C3, C4, CP3, CP4
BCIC_MOTOR = [1, 5, 7, 9, 11, 13, 17]  # FC3,FC4,C3,Cz,C4,CP3,CP4


def load_xw_trials(sid):
    """读 4s .mat -> (n_trials, 30, 2000) float, labels(0=affected,1=unaffected)"""
    import yaml
    with open(DATASETS["xw"]["cfg"]) as f:
        info = yaml.safe_load(f)
    fpath = os.path.join(
        XW_DIR, f"sub-{sid:02d}", f"sub-{sid:02d}_task-motor-imagery_eeg_4s.mat")
    mat = sio.loadmat(fpath)
    data = mat["eeg_data_4s"]
    labels = mat["labels"].flatten().astype(int)
    if data.shape[1] > data.shape[2]:      # (trials, time, ch) -> (trials, ch, time)
        data = np.transpose(data, (0, 2, 1))
    if set(np.unique(labels)) == {1, 2}:
        labels = labels - 1                # -> 0=left, 1=right
    # affected 映射 + 右瘫翻转（与 XWStrokeDataset 一致）
    side = _load_paralysis_side(sid, info)
    if side == "right":
        data = _flip_xwstroke_channels(data)
        labels = 1 - labels
    return data[:, :30, :].astype(np.float64), labels


def load_bcic_trials(sid):
    """复刻 BCICIV2aDataset._load_gdf_session（0–40Hz，3s epoch），取左/右手。"""
    import mne
    data_dir = "src/datasets/BCICIV2a"
    all_data, all_labels = [], []
    for is_train in ("T", "E"):
        gdf = os.path.join(data_dir, f"A{sid:02d}{is_train}.gdf")
        matf = os.path.join(data_dir, f"A{sid:02d}{is_train}.mat")
        raw = mne.io.read_raw_gdf(gdf, preload=True, verbose=False)
        events, _ = mne.events_from_annotations(raw)
        raw.load_data()
        raw.filter(0., 40., fir_design="firwin", verbose=False)
        raw.info["bads"] += ["EOG-left", "EOG-central", "EOG-right"]
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,
                               stim=False, exclude="bads")
        if is_train == "T":
            event_id = ({"769": 7, "770": 8, "771": 9, "772": 10} if sid != 4
                        else {"769": 5, "770": 6, "771": 7, "772": 8})
        else:
            event_id = {"783": 7}
        epochs = mne.Epochs(raw, events, event_id, 0, 3, proj=True, picks=picks,
                            baseline=None, preload=True, verbose=False)
        all_data.append(epochs.get_data() * 1e6)
        all_labels.append(sio.loadmat(matf)["classlabel"].reshape(-1) - 1)
    data = np.concatenate(all_data, 0)
    labels = np.concatenate(all_labels)
    keep = np.isin(labels, [0, 1])          # 左/右手
    return data[keep].astype(np.float64), labels[keep]


def load_hs_trials(sid, sessions=(1,)):
    """HS Stroke 慢性期：直接读每会话 .mat（未滤波 4s epoch，256 Hz）。
    -> (n_trials, 11, 1024), labels(0=left,1=right)"""
    import glob
    data_dir = "src/datasets/HSStroke"
    sub = f"sub-{sid:02d}"
    pattern = os.path.join(data_dir, sub, "ses-*", "eeg", "*_eeg.mat")
    mat_files = sorted(glob.glob(pattern))
    if sessions != "all":
        wanted = {f"ses-{int(s):02d}" for s in sessions}
        mat_files = [f for f in mat_files
                     if os.path.basename(os.path.dirname(os.path.dirname(f))) in wanted]
    if not mat_files:
        raise FileNotFoundError(f"No HSStroke data for {sub} sessions={sessions}")
    data_list, label_list = [], []
    for f in mat_files:
        m = sio.loadmat(f, squeeze_me=True)
        data_list.append(np.asarray(m["EEGdata"], dtype=np.float64))
        label_list.append(np.asarray(m["EEGdatalabel"]).reshape(-1).astype(int) - 1)
    return np.concatenate(data_list, 0), np.concatenate(label_list)


def load_hs_trials_ses01(sid):
    return load_hs_trials(sid, sessions=(1,))


def load_hs_trials_all(sid):
    return load_hs_trials(sid, sessions="all")


LOADERS = {"xw": (load_xw_trials, XW_SR, XW_EPOCH_S, XW_MOTOR),
           "bcic": (load_bcic_trials, 250, 3.0, BCIC_MOTOR),
           "hs": (load_hs_trials_ses01, 256, 4.0, list(range(11))),
           "hs_all": (load_hs_trials_all, 256, 4.0, list(range(11)))}


def scp_features(X, sr):
    """X: (n, ch, T) -> (n, ch*4)。低通 4Hz + 基线校正 + 幅值/斜率。"""
    from scipy.signal import butter, filtfilt
    b, a = butter(4, 4.0 / (sr / 2), btype="low")
    Xf = filtfilt(b, a, X, axis=-1)
    T = Xf.shape[-1]
    q = T // 4
    Xf = Xf - Xf[..., :q].mean(axis=-1, keepdims=True)   # 基线：前 1/4 段
    feats = [Xf[..., q:2*q].mean(-1), Xf[..., 2*q:3*q].mean(-1),
             Xf[..., 3*q:].mean(-1)]
    t = np.arange(2*q, T) / sr
    t_c = t - t.mean()
    slope = (Xf[..., 2*q:] * t_c).sum(-1) / (t_c**2).sum()  # 后半段线性斜率
    feats.append(slope)
    return np.concatenate(feats, axis=1)                    # (n, ch*4)


def run_scp(X, y, sid, seeds, arm, motor_idx, n_ch):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    F = scp_features(X, sr=LOADERS_CUR[1])                  # 逐 trial 独立，无泄漏
    g = np.arange(len(y))                                   # 每 trial 一组
    rows = []
    for seed in seeds:
        for fi, (tr, te) in enumerate(group_kfold_indices(g, 5, seed + sid)):
            Ftr, Fte = F[tr], F[te]
            mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-12
            Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
            if arm == "scp_motor":
                idx = np.concatenate([np.arange(c*4, c*4+4) for c in motor_idx])
                Ftr, Fte = Ftr[:, idx], Fte[:, idx]
                clf = LinearDiscriminantAnalysis()
            elif arm == "scp_sel20":
                yz = (y[tr] - y[tr].mean()) / (y[tr].std() + 1e-12)
                sc = np.abs((Ftr * yz[:, None]).mean(0))
                idx = np.argsort(-sc)[:20]
                Ftr, Fte = Ftr[:, idx], Fte[:, idx]
                clf = LinearDiscriminantAnalysis()
            else:
                clf = LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")
            clf.fit(Ftr, y[tr])
            acc = float(clf.score(Fte, y[te]))
            rows.append((sid, seed, fi, acc, len(tr), len(te), F.shape[1]))
    return rows


LOADERS_CUR = None


def main():
    global LOADERS_CUR
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["scp_full", "scp_motor", "scp_sel20"])
    ap.add_argument("--dataset", default="xw", choices=list(LOADERS.keys()))
    ap.add_argument("--seeds", default="42,123,2026")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    loader, sr, epoch_s, motor_idx = LOADERS[args.dataset]
    LOADERS_CUR = (loader, sr, epoch_s, motor_idx)

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
            X, y = loader(sid)
        except Exception as e:
            print(f"sub-{sid:02d}: LOAD ERROR {e}", flush=True)
            continue
        rows = run_scp(X, y, sid, remain, args.arm, motor_idx, X.shape[1])
        m = np.mean([r[3] for r in rows])
        print(f"[{args.dataset}/{args.arm}] sub-{sid:02d} done: {m*100:.2f}%  ({time.time()-t0:.0f}s)",
              flush=True)
        df = pd.DataFrame(rows, columns=["subject", "seed", "fold", "acc",
                                         "n_train", "n_test", "n_feat"])
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    print(f"\n===== {args.dataset}/{args.arm} 完成 =====")


if __name__ == "__main__":
    main()
