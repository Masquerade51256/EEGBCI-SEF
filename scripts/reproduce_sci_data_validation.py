"""
Python reproduction of the technical validation pipeline from:
Liu et al. (2024), Scientific Data 11:131
"An EEG motor imagery dataset for brain computer interface in acute stroke patients"

This script reproduces the MATLAB-based analysis (`code/TWFB_DGFMDM.m`)
in pure Python. Four methods from the paper are evaluated:

1. CSP + LDA          (paper: 55.57%)
2. FBCSP + SVM        (paper: 57.57%)
3. TSLDA + DGFMDM     (paper: 61.20%)
4. TWFB + DGFMDM      (paper: 72.21%)  <- best performing

Notes on equivalence:
- The data loading, filtering, and covariance estimation closely follow the
  original MATLAB code.
- For the Riemannian classifiers (TSLDA and TWFB+DGFMDM), we use `pyriemann`'s
  `TangentSpace+LDA` and `FgMDM` as the closest open-source Python equivalents
  to the MATLAB Covariance Toolbox's `tslda` and `fgmdm`.
- On this dataset, `pyriemann`'s `FgMDM` can show numerical sensitivity
  ("Convergence not reached" warnings) and may yield slightly different
  accuracies from the MATLAB `fgmdm` implementation. If exact MATLAB parity
  is required, consider calling the original functions via MATLAB Engine.

Usage:
    conda activate BCI310
    python scripts/reproduce_sci_data_validation.py

Requirements:
    scipy, numpy, scikit-learn, mne, pyriemann
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, iirnotch, lfilter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from mne.decoding import CSP
from pyriemann.classification import FgMDM
from pyriemann.tangentspace import TangentSpace


def notch_filter(data, fs, fa=50, q=6, axis=0):
    """Notch filter (MATLAB NotchFilter equivalent)."""
    wo = fa / (fs / 2)
    bw = wo / q
    b, a = iirnotch(wo, bw)
    return lfilter(b, a, data, axis=axis)


def bandpass_filter(data, fs, fl, fh, order=2, axis=0):
    """Butterworth bandpass filter (MATLAB BandpassFilter equivalent)."""
    fn = fs / 2
    b, a = butter(order, [fl / fn, fh / fn], btype="band")
    return lfilter(b, a, data, axis=axis)


def load_subject_raw(subject_id, data_dir):
    """
    Load raw .mat file for a subject and return the continuous 2D array
    (time x channels) plus labels, exactly as MATLAB PlotAccuracy.m does.
    """
    sid = f"{subject_id:02d}"
    mat_path = Path(data_dir) / f"sub-{sid}" / f"sub-{sid}_task-motor-imagery_eeg.mat"
    mat = loadmat(mat_path)
    rawdata = mat["eeg"]["rawdata"][0, 0]          # (40, 33, 4000)
    labels = mat["eeg"]["label"][0, 0].ravel()      # (40,)
    # MATLAB: eeg0 = permute(eeg0,[1,3,2]); data = []; for t=1:size(eeg0,1); data=[data;squeeze(eeg0(t,:,:))]; end
    data2d = rawdata.transpose(0, 2, 1).reshape(-1, 33)
    return data2d, labels


def extract_trials(data2d, fs, channels, pre_trig=800, post_trig=2000, win_start=800, win_len=2000):
    """
    Extract epochs around marker==2 triggers.
    Default parameters match MATLAB TWFB_DGFMDM.m:
      - take -1.6s to +4s around trigger (2800 samples)
      - after filtering, keep 0s to +4s (2000 samples)
    Returns: (n_trials, n_samples, n_channels)
    """
    marker = data2d[:, -1]
    triggers = np.where(marker == 2)[0]
    n_ch = len(channels)
    eeg = np.zeros((len(triggers), win_len, n_ch))
    total_len = pre_trig + post_trig
    for i, trig in enumerate(triggers):
        seg = data2d[trig - pre_trig : trig + post_trig, :][:, channels]
        # MATLAB pipeline: notch -> bandpass -> window
        seg = notch_filter(seg, fs, axis=0)
        # bandpass will be applied later per frequency band
        eeg[i] = seg[win_start : win_start + win_len, :]
    return eeg


def compute_covariances(eeg):
    """
    Compute un-normalized covariance matrices as in MATLAB:
        COV(:,:,i) = SS' * SS   where SS is (time, channels)
    Input eeg: (n_trials, n_times, n_channels)
    Output:    (n_trials, n_channels, n_channels)
    """
    return np.array([trial.T @ trial for trial in eeg])


def twfb_dgfmdm(eeg, labels, fs, freq_bands, n_repeats=10, train_size=24):
    """
    TWFB + DGFMDM reproduction.
    Mirrors MATLAB TWFB_DGFMDM.m:
      - loop over frequency bands
      - 10 random train/test splits (24/16)
      - pick best band per repeat
    """
    n_trials = len(labels)
    test_size = n_trials - train_size
    accs = []
    left_nums = []
    right_nums = []

    for h in range(n_repeats):
        np.random.seed(h)
        perm = np.random.permutation(n_trials)
        train_idx = perm[:train_size]
        test_idx = perm[train_size:]

        best_acc = 0.0
        best_left = 0
        best_right = 0
        best_total_left = 0
        best_total_right = 0

        for fl, fh in freq_bands:
            eeg_filt = np.array([bandpass_filter(trial, fs, fl, fh, axis=0) for trial in eeg])
            covs = compute_covariances(eeg_filt)

            clf = FgMDM(metric="riemann")
            clf.fit(covs[train_idx], labels[train_idx])
            pred = clf.predict(covs[test_idx])

            acc = np.mean(pred == labels[test_idx]) * 100
            if acc > best_acc:
                best_acc = acc
                best_total_left = np.sum(labels[test_idx] == 1)
                best_total_right = np.sum(labels[test_idx] == 2)
                best_left = np.sum((pred == labels[test_idx]) & (pred == 1))
                best_right = np.sum((pred == labels[test_idx]) & (pred == 2))

        accs.append(best_acc)
        left_nums.append([best_total_left, best_left])
        right_nums.append([best_total_right, best_right])

    return (
        np.mean(accs),
        np.array(left_nums),
        np.array(right_nums),
    )


def tslda_dgfmdm(eeg, labels, fs, freq_bands, n_repeats=10, train_size=24):
    """
    TSLDA + DGFMDM reproduction (simplified to a single 8-30 Hz band,
    matching the MATLAB TSLDA_DGFMDM.m logic where LowFreq/UpFreq are used).
    Here we follow the paper validation: one band [8,30] Hz.
    """
    n_trials = len(labels)
    eeg_filt = np.array([bandpass_filter(trial, fs, 8, 30, axis=0) for trial in eeg])
    covs = compute_covariances(eeg_filt)

    accs = []
    left_nums = []
    right_nums = []

    for h in range(n_repeats):
        np.random.seed(h)
        perm = np.random.permutation(n_trials)
        train_idx = perm[:train_size]
        test_idx = perm[train_size:]

        # TSLDA (Tangent Space + LDA)
        ts = TangentSpace(metric="riemann")
        X_train = ts.fit_transform(covs[train_idx])
        X_test = ts.transform(covs[test_idx])
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, labels[train_idx])
        pred_ts = lda.predict(X_test)
        acc_ts = np.mean(pred_ts == labels[test_idx]) * 100

        # DGFMDM (FgMDM)
        clf = FgMDM(metric="riemann")
        clf.fit(covs[train_idx], labels[train_idx])
        pred_fg = clf.predict(covs[test_idx])
        acc_fg = np.mean(pred_fg == labels[test_idx]) * 100

        # Pick the better of the two, as MATLAB TSLDA_DGFMDM.m does
        if acc_ts > acc_fg:
            pred = pred_ts
            acc = acc_ts
        else:
            pred = pred_fg
            acc = acc_fg

        accs.append(acc)
        left_nums.append([
            np.sum(labels[test_idx] == 1),
            np.sum((pred == labels[test_idx]) & (pred == 1)),
        ])
        right_nums.append([
            np.sum(labels[test_idx] == 2),
            np.sum((pred == labels[test_idx]) & (pred == 2)),
        ])

    return np.mean(accs), np.array(left_nums), np.array(right_nums)


def csp_lda(eeg, labels, fs, n_repeats=10, train_size=24):
    """
    CSP + LDA reproduction.
    MATLAB CSP_LDA.m uses a single [8,30] Hz band and 4 CSP components.
    """
    n_trials = len(labels)
    eeg_filt = np.array([bandpass_filter(trial, fs, 8, 30, axis=0) for trial in eeg])
    # MNE CSP expects (n_epochs, n_channels, n_times)
    X = eeg_filt.transpose(0, 2, 1)
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    X_csp = csp.fit_transform(X, labels)

    accs = []
    left_nums = []
    right_nums = []

    for h in range(n_repeats):
        np.random.seed(h)
        perm = np.random.permutation(n_trials)
        train_idx = perm[:train_size]
        test_idx = perm[train_size:]

        lda = LinearDiscriminantAnalysis()
        lda.fit(X_csp[train_idx], labels[train_idx])
        pred = lda.predict(X_csp[test_idx])
        acc = np.mean(pred == labels[test_idx]) * 100

        accs.append(acc)
        left_nums.append([
            np.sum(labels[test_idx] == 1),
            np.sum((pred == labels[test_idx]) & (pred == 1)),
        ])
        right_nums.append([
            np.sum(labels[test_idx] == 2),
            np.sum((pred == labels[test_idx]) & (pred == 2)),
        ])

    return np.mean(accs), np.array(left_nums), np.array(right_nums)


def fbcsp_svm(eeg, labels, fs, n_repeats=10, train_size=24):
    """
    FBCSP + SVM reproduction.
    MATLAB FBCSP_SVM.m uses filter banks 8-12, 9-13, ..., 26-30 Hz (19 bands)
    with 4 CSP components per band and an SVM classifier.
    MNE does not ship FBCSP, so we build it manually with per-band CSPs.
    """
    n_trials = len(labels)
    # Build 19 overlapping bands: [8,12] to [26,30]
    bands = [(f, f + 4) for f in range(8, 27)]
    # Pre-filter data for each band and stack CSP features
    X = eeg.transpose(0, 2, 1)  # (trials, channels, times)
    X_fb_list = []
    for fl, fh in bands:
        eeg_filt = np.array([bandpass_filter(trial.T, fs, fl, fh, axis=0).T for trial in X])
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        X_fb_list.append(csp.fit_transform(eeg_filt, labels))
    X_fb = np.hstack(X_fb_list)

    accs = []
    left_nums = []
    right_nums = []

    for h in range(n_repeats):
        np.random.seed(h)
        perm = np.random.permutation(n_trials)
        train_idx = perm[:train_size]
        test_idx = perm[train_size:]

        svm = SVC(kernel="linear", C=1.0)
        svm.fit(X_fb[train_idx], labels[train_idx])
        pred = svm.predict(X_fb[test_idx])
        acc = np.mean(pred == labels[test_idx]) * 100

        accs.append(acc)
        left_nums.append([
            np.sum(labels[test_idx] == 1),
            np.sum((pred == labels[test_idx]) & (pred == 1)),
        ])
        right_nums.append([
            np.sum(labels[test_idx] == 2),
            np.sum((pred == labels[test_idx]) & (pred == 2)),
        ])

    return np.mean(accs), np.array(left_nums), np.array(right_nums)


def build_confusion(left_num, right_num):
    """Build aggregated confusion matrix from left/right counts."""
    total_left = int(np.sum(left_num[:, 0]))
    correct_left = int(np.sum(left_num[:, 1]))
    total_right = int(np.sum(right_num[:, 0]))
    correct_right = int(np.sum(right_num[:, 1]))
    return np.array([
        [correct_left, total_left - correct_left],
        [total_right - correct_right, correct_right],
    ])


def cal_evaluate_index(cm):
    """Calculate Kappa, Sensitivity, Precision from confusion matrix."""
    tp = cm[1, 1]
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    total = cm.sum()

    observed = (tp + tn) / total
    expected = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (total ** 2)
    kappa = (observed - expected) / (1 - expected) if expected != 1 else 0.0

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return kappa, sensitivity, precision


def main():
    parser = argparse.ArgumentParser(description="Reproduce SciData 2024 validation")
    parser.add_argument("--data-dir", default="src/datasets/21679035/sourcedata", help="Root of sub-XX folders")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 51)), help="Subject IDs to process")
    parser.add_argument("--repeats", type=int, default=10, help="Number of random repeats per subject")
    parser.add_argument("--train-size", type=int, default=24, help="Training trials per split")
    args = parser.parse_args()

    fs = 500
    # MATLAB TWFB_DGFMDM.m hardcodes these 8 frequency bands
    freq_bands = [(8, 12), (8, 20), (8, 30), (12, 20), (15, 20), (15, 30), (20, 30), (8, 15)]
    # MATLAB uses channels [1:17 19:30] -> 0-based indices [0:16, 18:29]
    channels = np.concatenate([np.arange(0, 17), np.arange(18, 30)]).astype(int)

    results = {
        "CSP+LDA": {"accs": [], "left": [], "right": []},
        "FBCSP+SVM": {"accs": [], "left": [], "right": []},
        "TSLDA+DGFMDM": {"accs": [], "left": [], "right": []},
        "TWFB+DGFMDM": {"accs": [], "left": [], "right": []},
    }

    for sid in args.subjects:
        print(f"\nProcessing Subject {sid} ...")
        data2d, labels = load_subject_raw(sid, args.data_dir)
        eeg = extract_trials(data2d, fs, channels)

        acc, left, right = csp_lda(eeg, labels, fs, n_repeats=args.repeats, train_size=args.train_size)
        results["CSP+LDA"]["accs"].append(acc)
        results["CSP+LDA"]["left"].append(left)
        results["CSP+LDA"]["right"].append(right)
        print(f"  CSP+LDA:        {acc:.2f}%")

        acc, left, right = fbcsp_svm(eeg, labels, fs, n_repeats=args.repeats, train_size=args.train_size)
        results["FBCSP+SVM"]["accs"].append(acc)
        results["FBCSP+SVM"]["left"].append(left)
        results["FBCSP+SVM"]["right"].append(right)
        print(f"  FBCSP+SVM:      {acc:.2f}%")

        acc, left, right = tslda_dgfmdm(eeg, labels, fs, freq_bands, n_repeats=args.repeats, train_size=args.train_size)
        results["TSLDA+DGFMDM"]["accs"].append(acc)
        results["TSLDA+DGFMDM"]["left"].append(left)
        results["TSLDA+DGFMDM"]["right"].append(right)
        print(f"  TSLDA+DGFMDM:   {acc:.2f}%")

        acc, left, right = twfb_dgfmdm(eeg, labels, fs, freq_bands, n_repeats=args.repeats, train_size=args.train_size)
        results["TWFB+DGFMDM"]["accs"].append(acc)
        results["TWFB+DGFMDM"]["left"].append(left)
        results["TWFB+DGFMDM"]["right"].append(right)
        print(f"  TWFB+DGFMDM:    {acc:.2f}%")

    print("\n" + "=" * 60)
    print("OVERALL RESULTS (reproduced in Python)")
    print("=" * 60)
    for method_name, res in results.items():
        mean_acc = np.mean(res["accs"])
        cm = build_confusion(np.vstack(res["left"]), np.vstack(res["right"]))
        kappa, sens, prec = cal_evaluate_index(cm)
        print(f"\n{method_name}")
        print(f"  Average Accuracy: {mean_acc:.2f}%")
        print(f"  Kappa:            {kappa:.4f}")
        print(f"  Precision:        {prec:.4f}")
        print(f"  Sensitivity:      {sens:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

    print("\nPaper-reported averages for comparison:")
    print("  CSP+LDA:        55.57%")
    print("  FBCSP+SVM:      57.57%")
    print("  TSLDA+DGFMDM:   61.20%")
    print("  TWFB+DGFMDM:    72.21%")


if __name__ == "__main__":
    with warnings.catch_warnings():
        # pyriemann often warns about mean-convergence on small trial counts
        warnings.simplefilter("ignore")
        main()
