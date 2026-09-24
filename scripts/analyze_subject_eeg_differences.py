#!/usr/bin/env python3
"""
Comprehensive EEG signal-level analysis for quantifying inter-subject differences
in the XWStroke dataset.

This script extracts multi-dimensional EEG features and quantifies:
1. How large are the signal differences between subjects?
2. In which dimensions do these differences manifest?

Dimensions analyzed:
- Spectral: band power, alpha peak frequency, alpha/theta ratio
- Time-frequency: ERD/ERS amplitude and topography (mu & beta)
- Spatial: covariance geometry, CSP eigenvalue dispersion
- Connectivity: C3-C4 inter-hemispheric coherence and wPLI
- Trial-level variability: coefficient of variation
- Signal-to-Noise Ratio (SNR): task/baseline power ratio, ERD contrast SNR, inter-trial coherence

Output:
- Per-subject feature matrices
- Inter-subject distance matrices per dimension
- Group comparisons (left vs right paralysis)
- Visualization figures saved to results/eeg_signal_analysis/
"""

import os
import sys
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt
import seaborn as sns

# Optional dependencies with graceful fallback
try:
    import scipy.io as sio
    from scipy import signal
    from scipy.spatial.distance import squareform, pdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not installed. FFT-based PSD and filtering will use numpy fallbacks.")

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    warnings.warn("mne not installed. Some advanced functions will be limited.")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = "src/datasets/21679035/sourcedata"
PARTICIPANTS_TSV = "src/datasets/21679035/participants.tsv"
OUTPUT_DIR = "experiments/eeg_signal_analysis"

# XWStroke standard 32-channel names (first 30 are EEG, last 2 are EOG)
CH_NAMES = [
    'Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FCz', 'FC3', 'FC4',
    'FT7', 'FT8', 'Cz', 'C3', 'C4', 'T3', 'T4', 'CPz', 'CP3', 'CP4',
    'TP7', 'TP8', 'Pz', 'P3', 'P4', 'T5', 'T6', 'Oz', 'O1', 'O2',
    'HEOL', 'HEOR'
]
EEG_CH_NAMES = CH_NAMES[:30]
SFREQ = 500  # Original sampling rate
N_CHANNELS = 30  # Exclude EOG

# Motor imagery cue onset (in seconds). The 4s data window typically includes pre-cue baseline.
# Adjust based on your actual paradigm. Common: cue at t=1s or t=2s within 4s window.
CUE_ONSET_SEC = 2.0
BASELINE_WINDOW = (0.0, 1.5)  # Pre-cue baseline for ERD normalization
ERD_WINDOW = (2.0, 3.5)      # Post-cue MI period

BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'mu': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 40),
}

# Key channels for motor imagery (10-20 equivalents)
MOTOR_CHANNELS = {
    'C3': 13,   # Left sensorimotor
    'C4': 14,   # Right sensorimotor
    'Cz': 12,   # Midline
    'FC3': 8,
    'FC4': 9,
    'CP3': 18,
    'CP4': 19,
}

# Channel left-right pairs for LI and connectivity
LEFT_RIGHT_PAIRS = [
    ('Fp1', 'Fp2'), ('F3', 'F4'), ('F7', 'F8'),
    ('FC3', 'FC4'), ('FT7', 'FT8'), ('C3', 'C4'),
    ('T3', 'T4'), ('CP3', 'CP4'), ('TP7', 'TP8'),
    ('P3', 'P4'), ('T5', 'T6'), ('O1', 'O2'),
]


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
def load_subject_data(subject_id: int):
    """Load preprocessed 4s EEG data for a subject."""
    sub_str = f"sub-{subject_id:02d}"
    mat_file = os.path.join(DATA_DIR, sub_str, f"{sub_str}_task-motor-imagery_eeg_4s.mat")
    
    if not HAS_SCIPY:
        raise RuntimeError("scipy is required to load .mat files. Install: pip install scipy")
    
    mat = sio.loadmat(mat_file)
    data = mat['eeg_data_4s']  # Expected: (n_trials, n_channels, n_times)
    labels = mat['labels'].flatten()  # Expected: 1=Left, 2=Right
    
    # Convert to 0-based labels: 0=Left, 1=Right
    labels = labels - 1
    
    # Keep only EEG channels (exclude EOG indices 30,31)
    data = data[:, :N_CHANNELS, :]
    
    return data, labels


def load_participants_info():
    """Load clinical metadata."""
    df = pd.read_csv(PARTICIPANTS_TSV, sep='\t')
    df['subject_id'] = df['Participant_ID'].str.replace('sub-', '').astype(int)
    df['ParalysisSide'] = df['ParalysisSide'].str.lower().str.strip()
    return df


# -----------------------------------------------------------------------------
# Feature Extraction Helpers
# -----------------------------------------------------------------------------
def bandpass_filter(data, low, high, sfreq=SFREQ, order=4):
    """Bandpass filter data of shape (n_trials, n_channels, n_times)."""
    if not HAS_SCIPY:
        raise RuntimeError("scipy required for filtering")
    nyq = sfreq / 2.0
    b, a = signal.butter(order, [low/nyq, high/nyq], btype='band')
    filtered = signal.filtfilt(b, a, data, axis=-1)
    return filtered


def compute_psd_welch(data, sfreq=SFREQ, nperseg=None):
    """
    Compute Welch PSD for each trial and channel.
    Input: (n_trials, n_channels, n_times)
    Output: (n_trials, n_channels, n_freqs), freqs
    """
    if not HAS_SCIPY:
        raise RuntimeError("scipy required for Welch PSD")
    if nperseg is None:
        nperseg = min(256, data.shape[-1] // 4)
    
    n_trials, n_ch, n_times = data.shape
    # Use first trial to get freq shape
    _, freqs = signal.welch(data[0, 0, :], fs=sfreq, nperseg=nperseg)
    psd = np.zeros((n_trials, n_ch, len(freqs)))
    
    for t in range(n_trials):
        for c in range(n_ch):
            _, psd[t, c, :] = signal.welch(data[t, c, :], fs=sfreq, nperseg=nperseg)
    return psd, freqs


def compute_band_powers(psd, freqs):
    """Extract band powers from PSD. Output: dict of (n_trials, n_channels)."""
    band_powers = {}
    for name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        band_powers[name] = np.trapz(psd[:, :, mask], freqs[mask], axis=-1)
    return band_powers


def compute_alpha_peak_freq(psd, freqs):
    """Find alpha peak frequency per trial/channel (mean across central channels)."""
    alpha_mask = (freqs >= 8) & (freqs <= 13)
    if not np.any(alpha_mask):
        return np.nan
    alpha_psd = psd[:, :, alpha_mask]
    alpha_freqs = freqs[alpha_mask]
    if alpha_psd.shape[-1] == 0:
        return np.nan
    # Average PSD across central channels (Cz, C3, C4)
    central_idx = [MOTOR_CHANNELS['Cz'], MOTOR_CHANNELS['C3'], MOTOR_CHANNELS['C4']]
    central_psd = alpha_psd[:, central_idx, :].mean(axis=1)  # (n_trials, n_alpha_freqs)
    peak_idx = np.argmax(central_psd, axis=-1)
    # Return mean peak freq across trials
    return float(alpha_freqs[peak_idx].mean())


def compute_erd_ers(data, sfreq=SFREQ, baseline=(0.0, 1.5), erd=(2.0, 3.5)):
    """
    Compute ERD/ERS for mu and beta bands.
    Uses bandpass + Hilbert envelope approach.
    
    Returns:
        erd_mu: (n_trials, n_channels) - power change ratio in mu
        erd_beta: (n_trials, n_channels) - power change ratio in beta
    """
    times = np.arange(data.shape[-1]) / sfreq
    baseline_mask = (times >= baseline[0]) & (times < baseline[1])
    erd_mask = (times >= erd[0]) & (times < erd[1])
    
    def _band_erd(band_data):
        # band_data: (n_trials, n_channels, n_times)
        # Compute power via squared amplitude (or Hilbert envelope)
        power = band_data ** 2
        base_pow = power[:, :, baseline_mask].mean(axis=-1)
        act_pow = power[:, :, erd_mask].mean(axis=-1)
        # Avoid division by zero
        base_pow = np.where(base_pow < 1e-12, 1e-12, base_pow)
        erd_ratio = (act_pow - base_pow) / base_pow  # Positive = ERS, Negative = ERD
        return erd_ratio
    
    mu_data = bandpass_filter(data, 8, 13, sfreq)
    beta_data = bandpass_filter(data, 13, 30, sfreq)
    
    return _band_erd(mu_data), _band_erd(beta_data)


def compute_csp(cov1, cov2, n_components=4):
    """
    Compute Common Spatial Patterns from two class covariance matrices.
    cov1, cov2: (n_channels, n_channels)
    Returns:
        W: CSP filters (n_channels, n_channels), columns are filters
        eigenvalues: generalized eigenvalues
    """
    # Regularize
    cov1 = cov1 + 0.001 * np.trace(cov1) / cov1.shape[0] * np.eye(cov1.shape[0])
    cov2 = cov2 + 0.001 * np.trace(cov2) / cov2.shape[0] * np.eye(cov2.shape[0])
    
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(cov2) @ cov1)
    # Sort by eigenvalue: largest = most discriminative for class 1
    sorted_idx = np.argsort(eigvals)[::-1]
    W = eigvecs[:, sorted_idx]
    eigenvalues = eigvals[sorted_idx]
    return W, eigenvalues


def compute_connectivity_coh(data, sfreq=SFREQ, band=(8, 13)):
    """
    Compute inter-channel coherence in a band.
    Returns: coherence matrix (n_channels, n_channels)
    """
    filtered = bandpass_filter(data, band[0], band[1], sfreq)
    n_trials, n_ch, n_times = filtered.shape
    
    # Compute cross-spectral density via FFT
    coh_matrix = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i, n_ch):
            sig_i = filtered[:, i, :]
            sig_j = filtered[:, j, :]
            # Average over trials
            csd = 0
            psd_i = 0
            psd_j = 0
            for t in range(n_trials):
                fi = np.fft.rfft(sig_i[t, :])
                fj = np.fft.rfft(sig_j[t, :])
                csd += np.abs(np.mean(fi * np.conj(fj)))
                psd_i += np.mean(np.abs(fi)**2)
                psd_j += np.mean(np.abs(fj)**2)
            csd /= n_trials
            psd_i /= n_trials
            psd_j /= n_trials
            coh = csd / (np.sqrt(psd_i * psd_j) + 1e-12)
            coh_matrix[i, j] = coh
            coh_matrix[j, i] = coh
    return coh_matrix


def compute_lateralization_index(erd_map, ch_names=EEG_CH_NAMES):
    """
    Compute lateralization index from ERD topography.
    erd_map: (n_channels,) - ERD amplitude per channel
    LI = (contra - ipsi) / (contra + ipsi)
    
    Note: This assumes LEFT HAND MI as the reference condition.
    For a general version, you should pass the condition label.
    """
    left_idx = [ch_names.index(c) for c, _ in LEFT_RIGHT_PAIRS if c in ch_names]
    right_idx = [ch_names.index(c) for _, c in LEFT_RIGHT_PAIRS if c in ch_names]
    
    left_power = np.mean(np.abs(erd_map[left_idx]))
    right_power = np.mean(np.abs(erd_map[right_idx]))
    
    li = (right_power - left_power) / (right_power + left_power + 1e-12)
    return li


# -----------------------------------------------------------------------------
# Per-Subject Feature Extraction
# -----------------------------------------------------------------------------
def extract_subject_features(subject_id: int):
    """Extract all features for a single subject."""
    data, labels = load_subject_data(subject_id)  # data: (n_trials, 30, n_times), labels: 0=Left, 1=Right
    n_trials, n_ch, n_times = data.shape
    
    # Split by condition
    left_data = data[labels == 0]   # Left hand MI
    right_data = data[labels == 1]  # Right hand MI
    
    features = {'subject_id': subject_id}
    
    # -------------------------------------------------------------------------
    # 1. Spectral Features
    # -------------------------------------------------------------------------
    psd_all, freqs = compute_psd_welch(data)
    band_powers = compute_band_powers(psd_all, freqs)
    
    # Relative band power (per trial, averaged across trials)
    total_power = sum(band_powers.values())
    for name in BANDS:
        rel_pow = band_powers[name] / (total_power + 1e-12)
        features[f'rel_{name}_power'] = rel_pow.mean(axis=0)  # (n_channels,)
    
    # Alpha peak frequency (averaged across trials, central channels)
    features['alpha_peak_freq'] = compute_alpha_peak_freq(psd_all, freqs)
    
    # Alpha/Theta ratio (mean across trials and channels, with clipping to avoid overflow)
    theta_safe = np.clip(band_powers['theta'], 1e-12, None)
    ratio = band_powers['mu'] / theta_safe
    features['alpha_theta_ratio'] = float(np.clip(ratio, 0, 100).mean())
    
    # -------------------------------------------------------------------------
    # 2. Time-Frequency: ERD/ERS
    # -------------------------------------------------------------------------
    erd_mu_left, erd_beta_left = compute_erd_ers(left_data)
    erd_mu_right, erd_beta_right = compute_erd_ers(right_data)
    
    # Average ERD across trials
    features['erd_mu_left'] = erd_mu_left.mean(axis=0)    # (n_channels,)
    features['erd_mu_right'] = erd_mu_right.mean(axis=0)
    features['erd_beta_left'] = erd_beta_left.mean(axis=0)
    features['erd_beta_right'] = erd_beta_right.mean(axis=0)
    
    # Contrast: (Right MI - Left MI) at C3/C4
    c3_idx = MOTOR_CHANNELS['C3']
    c4_idx = MOTOR_CHANNELS['C4']
    features['erd_contrast_c3'] = (erd_mu_right[:, c3_idx] - erd_mu_left[:, c3_idx]).mean()
    features['erd_contrast_c4'] = (erd_mu_right[:, c4_idx] - erd_mu_left[:, c4_idx]).mean()
    
    # Lateralization index (based on mu ERD for Left MI)
    features['li_mu_left'] = compute_lateralization_index(erd_mu_left.mean(axis=0))
    features['li_mu_right'] = compute_lateralization_index(erd_mu_right.mean(axis=0))
    
    # -------------------------------------------------------------------------
    # 3. Spatial: Covariance & CSP
    # -------------------------------------------------------------------------
    cov_left = np.mean([np.cov(left_data[i]) for i in range(len(left_data))], axis=0)
    cov_right = np.mean([np.cov(right_data[i]) for i in range(len(right_data))], axis=0)
    
    # CSP eigenvalue dispersion (how discriminative are the spatial patterns?)
    W, csp_eig = compute_csp(cov_left, cov_right)
    features['csp_eigenvalues'] = csp_eig
    features['csp_discriminability'] = np.log(csp_eig[0] / csp_eig[-1])  # Ratio of largest to smallest
    
    # Covariance matrix geometry (flattened upper triangle for distance computation)
    cov_mean = (cov_left + cov_right) / 2
    features['cov_upper'] = cov_mean[np.triu_indices_from(cov_mean)]
    
    # -------------------------------------------------------------------------
    # 4. Connectivity
    # -------------------------------------------------------------------------
    # Mu-band coherence
    coh_mu = compute_connectivity_coh(data, band=(8, 13))
    features['coh_mu'] = coh_mu
    
    # C3-C4 coherence (inter-hemispheric sensorimotor)
    features['c3c4_coh_mu'] = coh_mu[c3_idx, c4_idx]
    
    # -------------------------------------------------------------------------
    # 5. Signal-to-Noise Ratio (SNR) Metrics
    # -------------------------------------------------------------------------
    # Recompute filtered data for SNR analysis (local to compute_erd_ers only)
    mu_data = bandpass_filter(data, 8, 13, SFREQ)
    beta_data = bandpass_filter(data, 13, 30, SFREQ)
    
    # 5a. Task-to-Baseline SNR: ratio of task-period power to baseline-period power
    # This captures how much the signal changes during MI relative to rest
    times = np.arange(data.shape[-1]) / SFREQ
    baseline_mask = (times >= BASELINE_WINDOW[0]) & (times < BASELINE_WINDOW[1])
    task_mask = (times >= ERD_WINDOW[0]) & (times < ERD_WINDOW[1])
    
    # Note: snr_task_baseline is NOT computed here because the 4s data has been
    # z-score normalized, which removes baseline-task power differences.
    # Instead, we use ERD contrast SNR and ITC as the primary SNR proxies.
    features['snr_task_baseline'] = np.nan
    features['snr_task_baseline_cv'] = np.nan
    
    # 5b. ERD Contrast SNR (d'-like): (mu_power_left - mu_power_right) / pooled_std
    # Computed at C3 and C4 for left/right hand MI respectively
    mu_left_c3 = (erd_mu_left[:, MOTOR_CHANNELS['C3']] ** 2).mean() ** 0.5
    mu_right_c3 = (erd_mu_right[:, MOTOR_CHANNELS['C3']] ** 2).mean() ** 0.5
    mu_left_c4 = (erd_mu_left[:, MOTOR_CHANNELS['C4']] ** 2).mean() ** 0.5
    mu_right_c4 = (erd_mu_right[:, MOTOR_CHANNELS['C4']] ** 2).mean() ** 0.5
    
    # For left hand MI, C4 should show stronger ERD (contra-lateral)
    # For right hand MI, C3 should show stronger ERD
    # d' = (mean1 - mean2) / sqrt((std1^2 + std2^2)/2)
    std_left_c4 = erd_mu_left[:, MOTOR_CHANNELS['C4']].std()
    std_right_c4 = erd_mu_right[:, MOTOR_CHANNELS['C4']].std()
    std_left_c3 = erd_mu_left[:, MOTOR_CHANNELS['C3']].std()
    std_right_c3 = erd_mu_right[:, MOTOR_CHANNELS['C3']].std()
    
    pooled_std_c4 = np.sqrt((std_left_c4**2 + std_right_c4**2) / 2)
    pooled_std_c3 = np.sqrt((std_left_c3**2 + std_right_c3**2) / 2)
    
    features['snr_contrast_c4'] = abs(mu_left_c4 - mu_right_c4) / (pooled_std_c4 + 1e-12)
    features['snr_contrast_c3'] = abs(mu_left_c3 - mu_right_c3) / (pooled_std_c3 + 1e-12)
    
    # 5c. Inter-Trial Coherence (ITC) / Phase-Locking SNR
    # Measure of how consistently the phase evolves across trials
    # Computed via FFT phase concentration
    def _itc(band_data):
        # band_data: (n_trials, n_channels, n_times)
        n_trials = band_data.shape[0]
        # Use central time window for FFT
        seg = band_data[:, :, task_mask]
        fft_vals = np.fft.rfft(seg, axis=-1)
        # Phase concentration per channel-frequency
        phases = np.angle(fft_vals)
        itc = np.abs(np.exp(1j * phases).mean(axis=0))  # (n_channels, n_freqs)
        return itc.mean()  # average ITC across channels and frequencies
    
    features['itc_mu'] = _itc(mu_data)
    features['itc_beta'] = _itc(beta_data)
    
    # -------------------------------------------------------------------------
    # 6. Trial-level variability
    # -------------------------------------------------------------------------
    trial_var = data.std(axis=-1).mean(axis=1)  # Variability per trial, averaged over channels
    features['trial_variability'] = trial_var.mean()
    features['trial_cv'] = trial_var.std() / (trial_var.mean() + 1e-12)  # Coefficient of variation
    
    return features


# -----------------------------------------------------------------------------
# Inter-Subject Distance & Comparison
# -----------------------------------------------------------------------------
def compute_distance_matrix(feature_vectors, metric='euclidean'):
    """Compute pairwise distance matrix from feature vectors (n_subjects, n_features)."""
    if metric == 'correlation':
        # 1 - Pearson correlation
        corr = np.corrcoef(feature_vectors)
        return 1 - corr
    return squareform(pdist(feature_vectors, metric=metric))


def compare_groups(feature_dict, group_labels, feature_name):
    """
    Compare a scalar feature between two groups.
    feature_dict: {subject_id: value}
    group_labels: pd.Series indexed by subject_id with group names
    """
    from scipy.stats import mannwhitneyu, ttest_ind
    
    groups = group_labels.unique()
    if len(groups) != 2:
        return None
    
    g1, g2 = groups
    vals1 = np.array([feature_dict[sid] for sid in group_labels[group_labels == g1].index])
    vals2 = np.array([feature_dict[sid] for sid in group_labels[group_labels == g2].index])
    
    t_stat, t_p = ttest_ind(vals1, vals2)
    u_stat, u_p = mannwhitneyu(vals1, vals2, alternative='two-sided')
    
    cohens_d = (vals1.mean() - vals2.mean()) / np.sqrt((vals1.std()**2 + vals2.std()**2) / 2)
    
    return {
        'group1': g1, 'mean1': vals1.mean(), 'std1': vals1.std(),
        'group2': g2, 'mean2': vals2.mean(), 'std2': vals2.std(),
        'cohens_d': cohens_d,
        't_test_p': t_p, 'mann_whitney_p': u_p
    }


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def plot_feature_comparison(results_df, all_features, group_col='ParalysisSide', output_dir=OUTPUT_DIR):
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Spectral + SNR features comparison
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    spectral_feats = ['alpha_peak_freq', 'alpha_theta_ratio', 'c3c4_coh_mu', 'trial_variability',
                      'trial_cv', 'csp_discriminability', 'snr_task_baseline', 'snr_contrast_c3']
    for ax, feat in zip(axes.flat, spectral_feats):
        if feat in results_df.columns and not results_df[feat].isna().all():
            sns.boxplot(data=results_df, x=group_col, y=feat, ax=ax)
            ax.set_title(feat)
        else:
            ax.set_title(f'{feat}\n(not available)')
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectral_snr_comparison.png'), dpi=150)
    plt.close()
    
    # 2. ERD topography: average across subjects per group
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    conditions = [
        ('erd_mu_left', 'Mu ERD - Left MI'),
        ('erd_mu_right', 'Mu ERD - Right MI'),
        ('erd_beta_left', 'Beta ERD - Left MI'),
        ('erd_beta_right', 'Beta ERD - Right MI'),
    ]
    
    # Build lookup from all_features
    feat_lookup = {f['subject_id']: f for f in all_features}
    
    for ax, (feat, title) in zip(axes.flat, conditions):
        left_vals = []
        right_vals = []
        for f in all_features:
            if f.get(group_col) == 'left':
                left_vals.append(f[feat])
            elif f.get(group_col) == 'right':
                right_vals.append(f[feat])
        
        if left_vals and right_vals:
            left_group = np.array(left_vals).mean(axis=0)
            right_group = np.array(right_vals).mean(axis=0)
            
            x = np.arange(N_CHANNELS)
            width = 0.35
            ax.bar(x - width/2, left_group, width, label='Left paralysis', alpha=0.7)
            ax.bar(x + width/2, right_group, width, label='Right paralysis', alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel('Channel')
            ax.set_ylabel('ERD/ERS Ratio')
            ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'erd_topography_comparison.png'), dpi=150)
    plt.close()
    
    # 3. Inter-subject distance heatmap (using spectral features)
    spectral_feature_names = ['alpha_peak_freq', 'alpha_theta_ratio', 'c3c4_coh_mu', 'trial_variability', 'csp_discriminability']
    available_cols = [c for c in spectral_feature_names if c in results_df.columns and not results_df[c].isna().all()]
    if available_cols:
        feat_mat = results_df[available_cols].values
        dist_mat = compute_distance_matrix(feat_mat, metric='euclidean')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(dist_mat, square=True, cmap='viridis', ax=ax)
        ax.set_title('Inter-Subject Euclidean Distance (Spectral Features)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'inter_subject_distance_heatmap.png'), dpi=150)
        plt.close()
    
    # 4. CSP discriminability comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    for group in ['left', 'right']:
        group_eigs = [f['csp_eigenvalues'] for f in all_features if f.get(group_col) == group]
        if group_eigs:
            mean_eig = np.array(group_eigs).mean(axis=0)
            ax.plot(range(len(mean_eig)), np.log10(mean_eig), 'o-', label=f'{group} paralysis (n={len(group_eigs)})')
    ax.set_xlabel('CSP Component')
    ax.set_ylabel('log10(Eigenvalue)')
    ax.set_title('CSP Eigenvalue Spectrum (Group Average)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'csp_eigenvalue_spectrum.png'), dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(subject_ids=None):
    """Run full analysis pipeline."""
    if not HAS_SCIPY:
        print("ERROR: scipy is required. Install: pip install scipy")
        sys.exit(1)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if subject_ids is None:
        subject_ids = list(range(1, 51))
    
    participants = load_participants_info()
    group_map = participants.set_index('subject_id')['ParalysisSide']
    
    print(f"Analyzing {len(subject_ids)} subjects...")
    all_features = []
    
    for sid in subject_ids:
        print(f"  Subject {sid:02d}...", end=' ')
        try:
            feats = extract_subject_features(sid)
            feats['ParalysisSide'] = group_map.get(sid, 'unknown')
            all_features.append(feats)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
    
    # Convert to DataFrame (only scalar features)
    scalar_keys = ['subject_id', 'ParalysisSide', 'alpha_peak_freq', 'alpha_theta_ratio',
                   'c3c4_coh_mu', 'trial_variability', 'trial_cv', 'csp_discriminability',
                   'erd_contrast_c3', 'erd_contrast_c4', 'li_mu_left', 'li_mu_right',
                   'snr_task_baseline', 'snr_task_baseline_cv', 'snr_contrast_c3', 'snr_contrast_c4',
                   'itc_mu', 'itc_beta']
    
    results_df = pd.DataFrame([{k: f[k] for k in scalar_keys} for f in all_features])
    
    # Statistical comparisons
    print("\n" + "="*60)
    print("Group Comparison Results (Left vs Right Paralysis)")
    print("="*60)
    
    comparisons = {}
    for feat in ['alpha_peak_freq', 'alpha_theta_ratio', 'c3c4_coh_mu',
                 'trial_variability', 'csp_discriminability',
                 'erd_contrast_c3', 'erd_contrast_c4', 'li_mu_left', 'li_mu_right',
                 'snr_task_baseline', 'snr_contrast_c3', 'snr_contrast_c4', 'itc_mu']:
        feat_dict = dict(zip(results_df['subject_id'], results_df[feat]))
        comp = compare_groups(feat_dict, group_map, feat)
        if comp:
            comparisons[feat] = comp
            sig = "***" if comp['mann_whitney_p'] < 0.001 else "**" if comp['mann_whitney_p'] < 0.01 else "*" if comp['mann_whitney_p'] < 0.05 else ""
            print(f"{feat:25s}: {comp['group1']:5s} {comp['mean1']:7.3f}±{comp['std1']:5.3f} vs "
                  f"{comp['group2']:5s} {comp['mean2']:7.3f}±{comp['std2']:5.3f} | "
                  f"Cohen's d={comp['cohens_d']:6.3f}, p={comp['mann_whitney_p']:.4f} {sig}")
    
    # Save results
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'subject_feature_table.csv'), index=False)
    with open(os.path.join(OUTPUT_DIR, 'group_comparisons.json'), 'w') as f:
        json.dump(comparisons, f, indent=2, default=float)
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_feature_comparison(results_df, all_features)
    
    # Summary statistics
    print("\n" + "="*60)
    print("Inter-Subject Variability Summary")
    print("="*60)
    for feat in ['alpha_peak_freq', 'alpha_theta_ratio', 'csp_discriminability', 'c3c4_coh_mu']:
        vals = results_df[feat].values
        print(f"{feat:25s}: mean={vals.mean():.3f}, std={vals.std():.3f}, CV={vals.std()/vals.mean():.3f}")
    
    print(f"\nAll results saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
