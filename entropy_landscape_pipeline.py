#!/usr/bin/env python3
"""
Entropy Landscape Pipeline
==========================
Complete analysis for "Entropy Landscape of Metastable EEG States" (Entropy MDPI).

Combines GAMEEMO (N=28) and PhysioNet EEGMMIDB (N=109) datasets.
Extracts 2-second epoch-level features with FOOOF parameterization,
runs K-means clustering (7 states), post-hoc entropy analysis,
and bridge state identification.

Target: ~4,110 epochs after artifact rejection.

Usage:
    python entropy_landscape_pipeline.py [--stage STAGE] [--force]

Stages: gameemo, physionet, combine, cluster, entropy, bridge, figure, all
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as signal
from scipy import stats
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from fooof import FOOOF

PHI = 1.618033988749895
RESULTS_DIR = 'pipeline_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMEEMO_BASE = ("gameemo_data/Database for Emotion Recognition System Based on "
                "EEG Signals and Various Computer Games - GAMEEMO/GAMEEMO")
GAMEEMO_FS = 128
GAMEEMO_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
                    'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

PHYSIONET_FS = 160
PHYSIONET_RUNS_REST = [1]

EPOCH_SEC = 2
ARTIFACT_AMP_UV = 100.0
EMG_SD_THRESH = 2.0
FLAT_VAR_THRESH = 0.1

FOOOF_FREQ_RANGE = [1, 45]
FOOOF_SETTINGS = dict(
    peak_width_limits=[1, 8],
    max_n_peaks=6,
    min_peak_height=0.05,
    peak_threshold=2.0,
    aperiodic_mode='fixed',
    verbose=False
)
FOOOF_R2_MIN = 0.90
MAX_EPOCHS_PER_SUBJECT = 30

BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),
}

CLUSTER_FEATURES = ['alpha_power', 'beta_power', 'gb_ratio',
                    'aperiodic_exponent', 'Q_alpha']

PE_M, PE_TAU = 4, 1
SAMPEN_M, SAMPEN_R = 2, 0.2
TE_K, TE_L = 2, 2
TE_N_SURROGATES = 1000
BRIDGE_N_PERM = 10000
MFDFA_SCALES = np.unique(np.logspace(np.log10(4), np.log10(64), 20).astype(int))
MFDFA_Q = np.arange(-5, 6)


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = fs / 2
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)


def notch_filter(data, freq, fs, Q=30):
    b, a = signal.iirnotch(freq, Q, fs)
    return signal.filtfilt(b, a, data, axis=-1)


def epoch_data(data, fs, epoch_sec=EPOCH_SEC):
    n_samples = int(epoch_sec * fs)
    n_epochs = data.shape[-1] // n_samples
    if data.ndim == 1:
        return data[:n_epochs * n_samples].reshape(n_epochs, n_samples)
    else:
        return data[:, :n_epochs * n_samples].reshape(data.shape[0], n_epochs, n_samples)


def reject_artifacts_multichannel(ch_epochs_3d, fs, ch_reject_frac=0.25):
    n_ch, n_epochs, n_samples = ch_epochs_3d.shape
    keep = np.ones(n_epochs, dtype=bool)

    amp_exceed = np.zeros(n_epochs, dtype=int)
    for ch in range(n_ch):
        amp_range = np.ptp(ch_epochs_3d[ch], axis=1)
        amp_exceed += (amp_range > ARTIFACT_AMP_UV).astype(int)
    keep &= (amp_exceed < max(1, int(n_ch * ch_reject_frac)))

    avg_signal = ch_epochs_3d.mean(axis=0)
    emg_powers = np.zeros(n_epochs)
    for ep in range(n_epochs):
        freqs, psd = signal.welch(avg_signal[ep], fs=fs,
                                   nperseg=min(n_samples, int(fs)))
        mask = (freqs >= 30) & (freqs <= 45)
        emg_powers[ep] = np.mean(psd[mask]) if mask.any() else 0
    emg_mean = np.mean(emg_powers)
    emg_std = np.std(emg_powers)
    if emg_std > 0:
        keep &= (emg_powers <= emg_mean + EMG_SD_THRESH * emg_std)

    flat_count = np.zeros(n_epochs, dtype=int)
    for ch in range(n_ch):
        epoch_var = np.var(ch_epochs_3d[ch], axis=1)
        flat_count += (epoch_var < FLAT_VAR_THRESH).astype(int)
    keep &= (flat_count < max(1, int(n_ch * ch_reject_frac)))

    return keep


def compute_avg_psd(ch_epochs_3d, ep_idx, fs):
    n_ch = ch_epochs_3d.shape[0]
    n_samples = ch_epochs_3d.shape[2]
    nperseg = min(n_samples, int(fs))
    psds = []
    for ch in range(n_ch):
        freqs, psd = signal.welch(ch_epochs_3d[ch, ep_idx], fs=fs,
                                   nperseg=nperseg, noverlap=nperseg // 2)
        psds.append(psd)
    return freqs, np.mean(psds, axis=0)


def extract_epoch_features(freqs, psd):
    fm = FOOOF(**FOOOF_SETTINGS)
    try:
        fm.fit(freqs, psd, FOOOF_FREQ_RANGE)
    except Exception:
        return None

    if fm.r_squared_ < FOOOF_R2_MIN:
        return None

    result = {
        'aperiodic_offset': fm.aperiodic_params_[0],
        'aperiodic_exponent': fm.aperiodic_params_[1],
        'r_squared': fm.r_squared_,
        'fooof_error': fm.error_,
    }

    for band_name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        band_psd = psd[mask]
        band_freqs = freqs[mask]
        result[f'{band_name}_power'] = np.mean(band_psd) if len(band_psd) > 0 else np.nan

        peaks = fm.peak_params_
        if len(peaks) > 0:
            bp = peaks[(peaks[:, 0] >= lo) & (peaks[:, 0] <= hi)]
            if len(bp) > 0:
                dominant = bp[np.argmax(bp[:, 1])]
                result[f'{band_name}_peak_freq'] = dominant[0]
                result[f'{band_name}_peak_power'] = dominant[1]
                result[f'{band_name}_peak_bw'] = dominant[2]
            else:
                result[f'{band_name}_peak_freq'] = np.nan
                result[f'{band_name}_peak_power'] = np.nan
                result[f'{band_name}_peak_bw'] = np.nan
        else:
            result[f'{band_name}_peak_freq'] = np.nan
            result[f'{band_name}_peak_power'] = np.nan
            result[f'{band_name}_peak_bw'] = np.nan

    alpha_p = result['alpha_power']
    alpha_mask = (freqs >= BANDS['alpha'][0]) & (freqs <= BANDS['alpha'][1])
    alpha_psd = psd[alpha_mask]
    alpha_freqs = freqs[alpha_mask]
    if len(alpha_psd) > 0 and np.sum(alpha_psd) > 0:
        peak_idx = np.argmax(alpha_psd)
        half_max = alpha_psd[peak_idx] / 2
        above = alpha_psd >= half_max
        bw_hz = np.sum(above) * (alpha_freqs[1] - alpha_freqs[0]) if len(alpha_freqs) > 1 else 1.0
        result['Q_alpha'] = alpha_p / max(bw_hz, 0.1)
    else:
        result['Q_alpha'] = np.nan

    beta_p = result['beta_power']
    gamma_p = result['gamma_power']
    if np.isfinite(beta_p) and np.isfinite(gamma_p) and beta_p > 0:
        result['gb_ratio'] = gamma_p / beta_p
    else:
        result['gb_ratio'] = np.nan

    return result


# ============================================================
# STAGE 1: GAMEEMO
# ============================================================
def process_gameemo(force=False):
    out_path = os.path.join(RESULTS_DIR, 'gameemo_epochs.csv')
    if os.path.exists(out_path) and not force:
        print(f"[GAMEEMO] Loading cached {out_path}")
        return pd.read_csv(out_path)

    print("[GAMEEMO] Processing 28 subjects × 4 games ...")
    rows = []
    total_raw = 0
    total_kept = 0

    for subj in range(1, 29):
        subj_epochs_kept = 0
        subj_clean_epochs = []

        for game_id in ['G1', 'G2', 'G3', 'G4']:
            subj_folder = f"(S{subj:02d})"
            filename = f"S{subj:02d}{game_id}AllChannels.mat"
            filepath = os.path.join(GAMEEMO_BASE, subj_folder,
                                    "Preprocessed EEG Data", ".mat format", filename)
            if not os.path.exists(filepath):
                continue

            try:
                mat = sio.loadmat(filepath)
            except Exception:
                continue

            ch_data = []
            for ch in GAMEEMO_CHANNELS:
                if ch in mat:
                    ch_data.append(mat[ch].flatten().astype(np.float64))
            if len(ch_data) != len(GAMEEMO_CHANNELS):
                continue

            eeg = np.array(ch_data)

            eeg = bandpass_filter(eeg, 1, 45, GAMEEMO_FS)
            eeg = notch_filter(eeg, 50, GAMEEMO_FS)
            eeg = eeg - eeg.mean(axis=0, keepdims=True)

            ch_epochs = epoch_data(eeg, GAMEEMO_FS, EPOCH_SEC)
            n_epochs = ch_epochs.shape[1]
            total_raw += n_epochs

            keep = reject_artifacts_multichannel(ch_epochs, GAMEEMO_FS)

            for ep_idx in range(n_epochs):
                if keep[ep_idx]:
                    subj_clean_epochs.append((ch_epochs, ep_idx, game_id))

        np.random.seed(subj)
        np.random.shuffle(subj_clean_epochs)
        subj_clean_epochs = subj_clean_epochs[:MAX_EPOCHS_PER_SUBJECT * 2]

        for ch_ep, ep_idx, game_id in subj_clean_epochs:
            if subj_epochs_kept >= MAX_EPOCHS_PER_SUBJECT:
                break

            freqs, avg_psd = compute_avg_psd(ch_ep, ep_idx, GAMEEMO_FS)
            features = extract_epoch_features(freqs, avg_psd)
            if features is None:
                continue

            row = {
                'dataset': 'GAMEEMO',
                'subject': f'G{subj:02d}',
                'game': game_id,
                'epoch_idx': ep_idx,
                **features,
            }
            rows.append(row)
            subj_epochs_kept += 1
            total_kept += 1

        print(f"  S{subj:02d}: {subj_epochs_kept} epochs retained")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    retention = total_kept / total_raw * 100 if total_raw > 0 else 0
    print(f"[GAMEEMO] Done: {total_kept}/{total_raw} epochs retained ({retention:.1f}%)")
    print(f"  Unique subjects: {df['subject'].nunique()}")
    print(f"  Mean epochs/subject: {total_kept / max(df['subject'].nunique(), 1):.1f}")
    return df


# ============================================================
# STAGE 2: PhysioNet EEGMMIDB
# ============================================================
def process_physionet(force=False):
    out_path = os.path.join(RESULTS_DIR, 'physionet_epochs.csv')
    if os.path.exists(out_path) and not force:
        print(f"[PhysioNet] Loading cached {out_path}")
        return pd.read_csv(out_path)

    import mne
    mne.set_log_level('ERROR')

    print("[PhysioNet] Processing 109 subjects (eyes-open baseline, run 1) ...")
    rows = []
    total_raw = 0
    total_kept = 0

    for subj in range(1, 110):
        try:
            raw_fnames = mne.datasets.eegbci.load_data(subj, PHYSIONET_RUNS_REST,
                                                        update_path=False)
            raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, verbose=False)
        except Exception as e:
            print(f"  S{subj:03d}: SKIP ({e})")
            continue

        mne.datasets.eegbci.standardize(raw)
        raw.set_montage('standard_1005', on_missing='ignore')
        raw.filter(1, 45, method='iir', iir_params=dict(order=4, ftype='butter'),
                   verbose=False)
        raw.notch_filter(60, verbose=False)
        raw.set_eeg_reference('average', projection=False, verbose=False)

        data = raw.get_data() * 1e6

        ch_epochs = epoch_data(data, PHYSIONET_FS, EPOCH_SEC)
        n_epochs = ch_epochs.shape[1]
        total_raw += n_epochs

        keep = reject_artifacts_multichannel(ch_epochs, PHYSIONET_FS)

        clean_indices = np.where(keep)[0]
        np.random.seed(subj)
        np.random.shuffle(clean_indices)
        clean_indices = clean_indices[:MAX_EPOCHS_PER_SUBJECT * 2]

        subj_kept = 0
        for ep_idx in clean_indices:
            if subj_kept >= MAX_EPOCHS_PER_SUBJECT:
                break

            freqs, avg_psd = compute_avg_psd(ch_epochs, ep_idx, PHYSIONET_FS)
            features = extract_epoch_features(freqs, avg_psd)
            if features is None:
                continue

            row = {
                'dataset': 'PhysioNet',
                'subject': f'P{subj:03d}',
                'epoch_idx': ep_idx,
                **features,
            }
            rows.append(row)
            subj_kept += 1
            total_kept += 1

        print(f"  S{subj:03d}: {subj_kept}/{n_epochs} epochs retained")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    retention = total_kept / total_raw * 100 if total_raw > 0 else 0
    print(f"[PhysioNet] Done: {total_kept}/{total_raw} epochs retained ({retention:.1f}%)")
    print(f"  Unique subjects: {df['subject'].nunique()}")
    print(f"  Mean epochs/subject: {total_kept / max(df['subject'].nunique(), 1):.1f}")
    return df


# ============================================================
# STAGE 3: Combine
# ============================================================
def combine_datasets(df_gameemo, df_physionet, force=False):
    out_path = os.path.join(RESULTS_DIR, 'combined_features.csv')
    if os.path.exists(out_path) and not force:
        print(f"[Combine] Loading cached {out_path}")
        return pd.read_csv(out_path)

    df = pd.concat([df_gameemo, df_physionet], ignore_index=True)
    initial = len(df)

    for feat in CLUSTER_FEATURES:
        df = df.dropna(subset=[feat])

    df = df.reset_index(drop=True)
    print(f"[Combine] {len(df)}/{initial} epochs with complete features")
    print(f"  GAMEEMO: {(df['dataset'] == 'GAMEEMO').sum()}")
    print(f"  PhysioNet: {(df['dataset'] == 'PhysioNet').sum()}")
    print(f"  Subjects: {df['subject'].nunique()}")
    df.to_csv(out_path, index=False)
    return df


# ============================================================
# STAGE 4: Clustering
# ============================================================
def run_clustering(df, force=False):
    out_path = os.path.join(RESULTS_DIR, 'clustered_features.csv')
    if os.path.exists(out_path) and not force:
        print(f"[Cluster] Loading cached {out_path}")
        return pd.read_csv(out_path)

    X_raw = df[CLUSTER_FEATURES].copy()
    for feat in ['alpha_power', 'beta_power']:
        X_raw[feat] = np.log1p(X_raw[feat].clip(lower=0))

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    print("[Cluster] Step 1: K-means k=2 macro-regimes ...")
    km2 = KMeans(n_clusters=2, n_init=20, random_state=42)
    macro = km2.fit_predict(X)

    mean_r_0 = df.loc[macro == 0, 'gb_ratio'].mean()
    mean_r_1 = df.loc[macro == 1, 'gb_ratio'].mean()
    if mean_r_0 >= mean_r_1:
        regime_A = (macro == 0)
        regime_B = (macro == 1)
    else:
        regime_A = (macro == 1)
        regime_B = (macro == 0)

    df['macro_regime'] = np.where(regime_A, 'A', 'B')
    sil2 = silhouette_score(X, macro)
    print(f"  Macro silhouette: {sil2:.3f}")
    print(f"  Regime A (phi-like): {regime_A.sum()} epochs, mean γ/β = {df.loc[regime_A, 'gb_ratio'].mean():.3f}")
    print(f"  Regime B (harmonic): {regime_B.sum()} epochs, mean γ/β = {df.loc[regime_B, 'gb_ratio'].mean():.3f}")

    print("[Cluster] Step 2: Sub-clustering within each regime ...")

    def optimal_subclusters(X_sub, k_range=(2, 6)):
        best_k, best_sil = 2, -1
        for k in range(k_range[0], k_range[1] + 1):
            if len(X_sub) < k:
                continue
            km = KMeans(n_clusters=k, n_init=20, random_state=42)
            labels = km.fit_predict(X_sub)
            s = silhouette_score(X_sub, labels)
            if s > best_sil:
                best_sil = s
                best_k = k
        return best_k, best_sil

    k_A, sil_A = optimal_subclusters(X[regime_A])
    k_B, sil_B = optimal_subclusters(X[regime_B])

    total_target = 7
    if k_A + k_B != total_target:
        for ka in range(2, 6):
            kb = total_target - ka
            if 2 <= kb <= 5:
                sA = silhouette_score(X[regime_A],
                    KMeans(n_clusters=ka, n_init=20, random_state=42).fit_predict(X[regime_A]))
                sB = silhouette_score(X[regime_B],
                    KMeans(n_clusters=kb, n_init=20, random_state=42).fit_predict(X[regime_B]))
                if (sA + sB) > (sil_A + sil_B) * 0.85:
                    k_A, k_B = ka, kb
                    sil_A, sil_B = sA, sB
                    break
        else:
            k_A, k_B = 3, 4

    print(f"  Regime A: {k_A} sub-clusters (sil={sil_A:.3f})")
    print(f"  Regime B: {k_B} sub-clusters (sil={sil_B:.3f})")

    km_A = KMeans(n_clusters=k_A, n_init=20, random_state=42)
    sub_A = km_A.fit_predict(X[regime_A])

    km_B = KMeans(n_clusters=k_B, n_init=20, random_state=42)
    sub_B = km_B.fit_predict(X[regime_B])

    state_labels = np.zeros(len(df), dtype=int)

    a_mean_r = []
    for s in range(k_A):
        idx = np.where(regime_A)[0][sub_A == s]
        a_mean_r.append((s, df.loc[idx, 'gb_ratio'].mean()))
    a_mean_r.sort(key=lambda x: -x[1])

    for new_label, (old_s, _) in enumerate(a_mean_r, start=1):
        idx = np.where(regime_A)[0][sub_A == old_s]
        state_labels[idx] = new_label

    b_mean_r = []
    for s in range(k_B):
        idx = np.where(regime_B)[0][sub_B == s]
        b_mean_r.append((s, df.loc[idx, 'gb_ratio'].mean()))
    b_mean_r.sort(key=lambda x: -x[1])

    for new_label, (old_s, _) in enumerate(b_mean_r, start=k_A + 1):
        idx = np.where(regime_B)[0][sub_B == old_s]
        state_labels[idx] = new_label

    df['state'] = state_labels
    total_states = len(np.unique(state_labels))
    print(f"  Total states: {total_states}")

    for s in sorted(df['state'].unique()):
        mask = df['state'] == s
        regime = df.loc[mask, 'macro_regime'].iloc[0]
        n = mask.sum()
        mr = df.loc[mask, 'gb_ratio'].mean()
        print(f"    State {s} (Regime {regime}): n={n}, mean γ/β={mr:.4f}")

    print("[Cluster] Circularity check ...")
    circularity_ok = True
    for feat in CLUSTER_FEATURES:
        r_val, _ = stats.pearsonr(df[feat], df['state'])
        print(f"    |r({feat}, state)| = {abs(r_val):.3f}", end="")
        if abs(r_val) > 0.25:
            print(" ⚠ WARNING")
            circularity_ok = False
        else:
            print(" ✓")

    df.to_csv(out_path, index=False)
    return df


# ============================================================
# STAGE 5: Entropy measures (post-hoc)
# ============================================================
def permutation_entropy_bits(x, m=PE_M, tau=PE_TAU):
    n = len(x)
    if n < (m - 1) * tau + 2:
        return np.nan
    patterns = []
    for i in range(n - (m - 1) * tau):
        window = tuple(np.argsort([x[i + j * tau] for j in range(m)]))
        patterns.append(window)
    counts = Counter(patterns)
    total = len(patterns)
    probs = np.array(list(counts.values())) / total
    return -np.sum(probs * np.log2(probs))


def sample_entropy(x, m=SAMPEN_M, r_tol=SAMPEN_R):
    n = len(x)
    if n < m + 5:
        return np.nan
    sd = np.std(x)
    if sd == 0:
        return 0.0
    tol = r_tol * sd

    def _count(tl):
        ct = 0
        tmpl = np.array([x[i:i + tl] for i in range(n - tl)])
        for i in range(len(tmpl)):
            dists = np.max(np.abs(tmpl[i] - tmpl[i + 1:]), axis=1)
            ct += np.sum(dists < tol)
        return ct

    B = _count(m)
    if B == 0:
        return np.nan
    A = _count(m + 1)
    return -np.log(A / B) if A > 0 else np.nan


def symbolic_transfer_entropy(source, target, k=TE_K, l_param=TE_L):
    n = min(len(source), len(target))
    min_len = max(k, l_param) + 1
    if n < min_len + 2:
        return np.nan

    def symbolize(seq):
        med = np.median(seq)
        return (seq >= med).astype(int)

    s_sym = symbolize(np.array(source[:n], dtype=float))
    t_sym = symbolize(np.array(target[:n], dtype=float))

    joint_counts = Counter()
    margin_counts = Counter()

    for i in range(max(k, l_param), n):
        t_past = tuple(t_sym[i - k:i])
        s_past = tuple(s_sym[i - l_param:i])
        t_now = t_sym[i]
        joint_counts[(t_now, t_past, s_past)] += 1
        margin_counts[(t_now, t_past)] += 1

    if len(joint_counts) == 0:
        return np.nan

    total = sum(joint_counts.values())
    te = 0.0
    cond_counts = Counter()
    for (t_now, t_past, s_past), c in joint_counts.items():
        cond_counts[(t_past, s_past)] += c

    for (t_now, t_past, s_past), c in joint_counts.items():
        p_joint = c / total
        p_cond_ts = c / cond_counts[(t_past, s_past)]
        p_cond_t = margin_counts[(t_now, t_past)] / total
        margin_t_past_total = sum(v for (tn, tp), v in margin_counts.items() if tp == t_past)
        p_cond_t_only = margin_counts[(t_now, t_past)] / margin_t_past_total if margin_t_past_total > 0 else 1e-10

        if p_cond_ts > 0 and p_cond_t_only > 0:
            te += p_joint * np.log2(p_cond_ts / p_cond_t_only)

    return te


def te_with_surrogates(source, target, n_surr=TE_N_SURROGATES, k=TE_K, l_param=TE_L):
    te_obs = symbolic_transfer_entropy(source, target, k, l_param)
    if np.isnan(te_obs):
        return te_obs, np.nan

    te_surr = np.zeros(n_surr)
    source_arr = np.array(source)
    block_len = max(5, len(source) // 10)
    for i in range(n_surr):
        n = len(source_arr)
        n_blocks = max(n // block_len, 1)
        blocks = [source_arr[j * block_len:(j + 1) * block_len] for j in range(n_blocks)]
        np.random.shuffle(blocks)
        shuffled = np.concatenate(blocks)[:n]
        te_surr[i] = symbolic_transfer_entropy(shuffled, target, k, l_param)

    te_surr = te_surr[~np.isnan(te_surr)]
    if len(te_surr) == 0:
        return te_obs, np.nan

    p_val = np.mean(te_surr >= te_obs)
    te_corrected = te_obs - np.mean(te_surr)
    return te_corrected, p_val


def mfdfa(x, scales=None, q_values=None):
    if scales is None:
        scales = MFDFA_SCALES
    if q_values is None:
        q_values = MFDFA_Q

    n = len(x)
    if n < scales[-1] * 2:
        return np.nan

    y = np.cumsum(x - np.mean(x))

    Fq = np.zeros((len(q_values), len(scales)))

    for si, s in enumerate(scales):
        n_seg = n // s
        if n_seg < 1:
            Fq[:, si] = np.nan
            continue
        fluct = np.zeros(n_seg)
        for v in range(n_seg):
            segment = y[v * s:(v + 1) * s]
            t = np.arange(s)
            coeffs = np.polyfit(t, segment, 1)
            trend = np.polyval(coeffs, t)
            fluct[v] = np.sqrt(np.mean((segment - trend) ** 2))

        for qi, q in enumerate(q_values):
            if q == 0:
                Fq[qi, si] = np.exp(0.5 * np.mean(np.log(fluct[fluct > 0] ** 2))) if np.any(fluct > 0) else np.nan
            else:
                Fq[qi, si] = np.mean(fluct ** q) ** (1 / q) if np.all(np.isfinite(fluct ** q)) else np.nan

    valid = np.all(np.isfinite(Fq), axis=0) & (Fq[0, :] > 0)
    if valid.sum() < 3:
        return np.nan

    log_scales = np.log(scales[valid])
    hq = np.zeros(len(q_values))
    for qi in range(len(q_values)):
        log_fq = np.log(Fq[qi, valid])
        if np.all(np.isfinite(log_fq)):
            slope, _, _, _, _ = stats.linregress(log_scales, log_fq)
            hq[qi] = slope
        else:
            hq[qi] = np.nan

    if np.all(np.isfinite(hq)):
        return np.max(hq) - np.min(hq)
    return np.nan


def compute_entropy_measures(df, force=False):
    out_path = os.path.join(RESULTS_DIR, 'entropy_results.csv')
    if os.path.exists(out_path) and not force:
        print(f"[Entropy] Loading cached {out_path}")
        return pd.read_csv(out_path)

    print("[Entropy] Computing PE, SampEn, TE, MFDFA per state ...")

    states = sorted(df['state'].unique())
    entropy_rows = []

    for state in states:
        state_df = df[df['state'] == state]
        regime = state_df['macro_regime'].iloc[0]
        n_epochs = len(state_df)

        subjects = state_df['subject'].unique()

        pe_values = []
        sampen_values = []
        mfdfa_values = []

        for subj in subjects:
            subj_state = state_df[state_df['subject'] == subj]
            if len(subj_state) < 2:
                continue

            gb_seq = subj_state['gb_ratio'].values

            pe = permutation_entropy_bits(gb_seq, PE_M, PE_TAU)
            if np.isfinite(pe):
                pe_values.append(pe)

            if len(gb_seq) >= SAMPEN_M + 5:
                se = sample_entropy(gb_seq, SAMPEN_M, SAMPEN_R)
                if np.isfinite(se):
                    sampen_values.append(se)

            if len(gb_seq) >= 20:
                mf = mfdfa(gb_seq)
                if np.isfinite(mf):
                    mfdfa_values.append(mf)

        te_pairs = []
        for i, s1 in enumerate(states):
            if s1 == state:
                continue
            for subj in subjects:
                src_seq = df[(df['state'] == s1) & (df['subject'] == subj)]['gb_ratio'].values
                tgt_seq = state_df[state_df['subject'] == subj]['gb_ratio'].values
                if len(src_seq) >= 4 and len(tgt_seq) >= 4:
                    te_corr, te_p = te_with_surrogates(src_seq, tgt_seq,
                                                       n_surr=min(TE_N_SURROGATES, 200))
                    if np.isfinite(te_corr):
                        te_pairs.append(te_corr)

        row = {
            'state': state,
            'regime': regime,
            'n_epochs': n_epochs,
            'n_subjects': len(subjects),
            'PE_mean': np.mean(pe_values) if pe_values else np.nan,
            'PE_std': np.std(pe_values) if pe_values else np.nan,
            'PE_n': len(pe_values),
            'SampEn_mean': np.mean(sampen_values) if sampen_values else np.nan,
            'SampEn_std': np.std(sampen_values) if sampen_values else np.nan,
            'SampEn_n': len(sampen_values),
            'TE_mean': np.mean(te_pairs) if te_pairs else np.nan,
            'TE_std': np.std(te_pairs) if te_pairs else np.nan,
            'TE_n': len(te_pairs),
            'MFDFA_delta_h': np.mean(mfdfa_values) if mfdfa_values else np.nan,
            'MFDFA_n': len(mfdfa_values),
            'mean_gb_ratio': state_df['gb_ratio'].mean(),
            'mean_alpha_power': state_df['alpha_power'].mean() if 'alpha_power' in state_df.columns else np.nan,
            'mean_aperiodic_exp': state_df['aperiodic_exponent'].mean(),
        }
        entropy_rows.append(row)
        print(f"  State {state} (Regime {regime}): PE={row['PE_mean']:.3f} ({row['PE_n']}), "
              f"SampEn={row['SampEn_mean']:.3f} ({row['SampEn_n']}), "
              f"TE={row['TE_mean']:.4f} ({row['TE_n']})")

    entropy_df = pd.DataFrame(entropy_rows)
    entropy_df.to_csv(out_path, index=False)
    return entropy_df


# ============================================================
# STAGE 6: Bridge state
# ============================================================
def compute_bridge_index(df, entropy_df, force=False):
    out_path = os.path.join(RESULTS_DIR, 'bridge_results.csv')
    if os.path.exists(out_path) and not force:
        print(f"[Bridge] Loading cached {out_path}")
        return pd.read_csv(out_path)

    print(f"[Bridge] Computing bridge_index with {BRIDGE_N_PERM} permutations ...")

    states = sorted(df['state'].unique())
    global_median_r = df['gb_ratio'].median()

    state_mean_r = df.groupby('state')['gb_ratio'].mean()

    boundary_dist = {}
    for s in states:
        boundary_dist[s] = abs(state_mean_r[s] - global_median_r)

    D_bar = np.mean(list(boundary_dist.values()))

    regime_counts = df.groupby('state')['macro_regime'].first()
    state_counts = df['state'].value_counts()
    regime_A_count = (df['macro_regime'] == 'A').sum()
    regime_B_count = (df['macro_regime'] == 'B').sum()

    p_A = {}
    for s in states:
        if regime_counts[s] == 'A':
            p_A[s] = state_counts[s] / regime_A_count if regime_A_count > 0 else 0
        else:
            p_A[s] = state_counts[s] / regime_B_count if regime_B_count > 0 else 0

    betweenness = {}
    transition_matrix = np.zeros((len(states), len(states)))
    subjects = df['subject'].unique()
    for subj in subjects:
        subj_df = df[df['subject'] == subj].sort_values('epoch_idx')
        s_seq = subj_df['state'].values
        for i in range(len(s_seq) - 1):
            si, sj = s_seq[i], s_seq[i + 1]
            transition_matrix[si - 1, sj - 1] += 1

    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_prob = transition_matrix / row_sums

    for s in states:
        in_flow = trans_prob[:, s - 1].sum()
        out_flow = trans_prob[s - 1, :].sum()
        betweenness[s] = in_flow * out_flow

    max_B = max(betweenness.values()) if max(betweenness.values()) > 0 else 1

    bridge_results = []
    for s in states:
        bi = (1 - 2 * abs(p_A[s] - 0.5)) * (betweenness[s] / max_B) * (boundary_dist[s] / D_bar if D_bar > 0 else 0)
        bridge_results.append({
            'state': s,
            'bridge_index': bi,
            'p_A': p_A[s],
            'betweenness': betweenness[s],
            'boundary_dist': boundary_dist[s],
        })

    bridge_df = pd.DataFrame(bridge_results)

    print("[Bridge] Permutation test ...")
    observed_max = bridge_df['bridge_index'].max()
    perm_max = np.zeros(BRIDGE_N_PERM)

    all_gb = df['gb_ratio'].values.copy()
    state_arr = df['state'].values.copy()
    for perm_i in range(BRIDGE_N_PERM):
        perm_states = np.random.permutation(state_arr)
        perm_mean_r = {}
        for s in states:
            perm_mean_r[s] = all_gb[perm_states == s].mean()

        perm_boundary = {}
        perm_global_med = np.median(all_gb)
        for s in states:
            perm_boundary[s] = abs(perm_mean_r[s] - perm_global_med)
        perm_D_bar = np.mean(list(perm_boundary.values()))

        perm_bi = []
        for s in states:
            bi = (1 - 2 * abs(p_A[s] - 0.5)) * (betweenness[s] / max_B) * (perm_boundary[s] / perm_D_bar if perm_D_bar > 0 else 0)
            perm_bi.append(bi)
        perm_max[perm_i] = max(perm_bi)

    p_value = np.mean(perm_max >= observed_max)
    bridge_state = bridge_df.loc[bridge_df['bridge_index'].idxmax(), 'state']

    print(f"  Bridge state: {bridge_state}")
    print(f"  Max bridge_index: {observed_max:.4f}")
    print(f"  Permutation p-value: {p_value:.4f}")

    bridge_df['is_bridge'] = bridge_df['state'] == bridge_state
    bridge_df['perm_p_value'] = p_value

    bridge_df.to_csv(out_path, index=False)
    return bridge_df


# ============================================================
# STAGE 7: Figure 2
# ============================================================
def generate_figure2(df, entropy_df, bridge_df):
    print("[Figure] Generating Figure 2 ...")

    states = sorted(entropy_df['state'].values)
    n_states = len(states)

    bridge_state = bridge_df.loc[bridge_df['bridge_index'].idxmax(), 'state']
    regimes = entropy_df.set_index('state')['regime']

    colors_A = ['#2196F3', '#42A5F5', '#90CAF9']
    colors_B = ['#FF5722', '#FF7043', '#FF8A65', '#FFAB91']
    state_colors = {}
    a_idx, b_idx = 0, 0
    for s in states:
        if regimes[s] == 'A':
            state_colors[s] = colors_A[a_idx % len(colors_A)]
            a_idx += 1
        else:
            state_colors[s] = colors_B[b_idx % len(colors_B)]
            b_idx += 1

    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    pe_means = entropy_df.set_index('state')['PE_mean']
    pe_stds = entropy_df.set_index('state')['PE_std']
    bars = ax1.bar(range(n_states), [pe_means.get(s, 0) for s in states],
                   yerr=[pe_stds.get(s, 0) for s in states],
                   color=[state_colors[s] for s in states],
                   edgecolor='black', linewidth=0.5, capsize=3)
    for i, s in enumerate(states):
        if s == bridge_state:
            bars[i].set_edgecolor('gold')
            bars[i].set_linewidth(3)
    ax1.set_xticks(range(n_states))
    ax1.set_xticklabels([f"S{s}" for s in states])
    ax1.set_xlabel('State')
    ax1.set_ylabel('Permutation Entropy (bits)')
    ax1.set_title('A. Permutation Entropy (m=4, τ=1)')
    import math
    max_pe = np.log2(math.factorial(PE_M))
    ax1.axhline(y=max_pe, color='gray', linestyle='--',
                alpha=0.5, label=f'Max PE = {max_pe:.2f}')
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[1])
    se_means = entropy_df.set_index('state')['SampEn_mean']
    se_stds = entropy_df.set_index('state')['SampEn_std']
    bars2 = ax2.bar(range(n_states), [se_means.get(s, 0) for s in states],
                    yerr=[se_stds.get(s, 0) for s in states],
                    color=[state_colors[s] for s in states],
                    edgecolor='black', linewidth=0.5, capsize=3)
    for i, s in enumerate(states):
        if s == bridge_state:
            bars2[i].set_edgecolor('gold')
            bars2[i].set_linewidth(3)
    ax2.set_xticks(range(n_states))
    ax2.set_xticklabels([f"S{s}" for s in states])
    ax2.set_xlabel('State')
    ax2.set_ylabel('Sample Entropy')
    ax2.set_title('B. Sample Entropy (m=2, r=0.2×SD)')

    ax3 = fig.add_subplot(gs[2])
    te_means = entropy_df.set_index('state')['TE_mean']
    te_matrix = np.zeros((n_states, n_states))
    for i, si in enumerate(states):
        for j, sj in enumerate(states):
            if i == j:
                continue
            te_val = te_means.get(sj, 0)
            if np.isfinite(te_val):
                te_matrix[i, j] = te_val

    im = ax3.imshow(te_matrix, cmap='YlOrRd', aspect='auto')
    ax3.set_xticks(range(n_states))
    ax3.set_yticks(range(n_states))
    ax3.set_xticklabels([f"S{s}" for s in states])
    ax3.set_yticklabels([f"S{s}" for s in states])
    ax3.set_xlabel('Target State')
    ax3.set_ylabel('Source State')
    ax3.set_title('C. Transfer Entropy Network')
    plt.colorbar(im, ax=ax3, label='TE (bits)', shrink=0.8)

    for i in range(n_states):
        for j in range(n_states):
            if te_matrix[i, j] > 0:
                ax3.text(j, i, f'{te_matrix[i, j]:.3f}', ha='center', va='center',
                         fontsize=6, color='black' if te_matrix[i, j] < te_matrix.max() * 0.5 else 'white')

    bridge_label = f"S{bridge_state}"
    bi_val = bridge_df.loc[bridge_df['state'] == bridge_state, 'bridge_index'].values[0]
    p_val = bridge_df['perm_p_value'].iloc[0]
    fig.suptitle(f'Figure 2: Entropy Landscape of Metastable EEG States '
                 f'(N={df["subject"].nunique()}, {len(df)} epochs)\n'
                 f'Bridge state: {bridge_label} (bridge_index={bi_val:.3f}, p={p_val:.4f})',
                 fontsize=12, y=1.02)

    fig_path = os.path.join(RESULTS_DIR, 'figure2_entropy_landscape.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[Figure] Saved {fig_path}")

    summary_path = os.path.join(RESULTS_DIR, 'analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ENTROPY LANDSCAPE ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total subjects: {df['subject'].nunique()}\n")
        f.write(f"  GAMEEMO: {df[df['dataset']=='GAMEEMO']['subject'].nunique()}\n")
        f.write(f"  PhysioNet: {df[df['dataset']=='PhysioNet']['subject'].nunique()}\n")
        f.write(f"Total epochs: {len(df)}\n\n")

        f.write("STATE SUMMARY\n")
        f.write("-" * 50 + "\n")
        for _, row in entropy_df.iterrows():
            f.write(f"State {int(row['state'])} (Regime {row['regime']}): "
                    f"n={int(row['n_epochs'])}, "
                    f"PE={row['PE_mean']:.3f}±{row['PE_std']:.3f}, "
                    f"SampEn={row['SampEn_mean']:.3f}±{row['SampEn_std']:.3f}, "
                    f"TE={row['TE_mean']:.4f}\n")

        f.write(f"\nBRIDGE STATE\n")
        f.write(f"-" * 50 + "\n")
        bridge_row = bridge_df[bridge_df['is_bridge']]
        f.write(f"Bridge: State {int(bridge_row['state'].values[0])}\n")
        f.write(f"  bridge_index = {bridge_row['bridge_index'].values[0]:.4f}\n")
        f.write(f"  permutation p = {bridge_row['perm_p_value'].values[0]:.4f}\n")

    print(f"[Figure] Saved {summary_path}")
    return fig_path


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Entropy Landscape Pipeline')
    parser.add_argument('--stage', default='all',
                        choices=['gameemo', 'physionet', 'combine', 'cluster',
                                 'entropy', 'bridge', 'figure', 'all'],
                        help='Which stage to run')
    parser.add_argument('--force', action='store_true',
                        help='Force recomputation (ignore cache)')
    args = parser.parse_args()

    t0 = time.time()

    if args.stage in ('gameemo', 'all'):
        df_gameemo = process_gameemo(args.force)
    else:
        path = os.path.join(RESULTS_DIR, 'gameemo_epochs.csv')
        df_gameemo = pd.read_csv(path) if os.path.exists(path) else None

    if args.stage in ('physionet', 'all'):
        df_physionet = process_physionet(args.force)
    else:
        path = os.path.join(RESULTS_DIR, 'physionet_epochs.csv')
        df_physionet = pd.read_csv(path) if os.path.exists(path) else None

    if args.stage in ('combine', 'all'):
        if df_gameemo is None or df_physionet is None:
            print("ERROR: Need both GAMEEMO and PhysioNet data to combine.")
            return
        df = combine_datasets(df_gameemo, df_physionet, args.force)
    else:
        path = os.path.join(RESULTS_DIR, 'combined_features.csv')
        df = pd.read_csv(path) if os.path.exists(path) else None

    if args.stage in ('cluster', 'all'):
        if df is None:
            print("ERROR: Need combined data to cluster.")
            return
        df = run_clustering(df, args.force)
    else:
        path = os.path.join(RESULTS_DIR, 'clustered_features.csv')
        df = pd.read_csv(path) if os.path.exists(path) else None

    if args.stage in ('entropy', 'all'):
        if df is None:
            print("ERROR: Need clustered data for entropy.")
            return
        entropy_df = compute_entropy_measures(df, args.force)
    else:
        path = os.path.join(RESULTS_DIR, 'entropy_results.csv')
        entropy_df = pd.read_csv(path) if os.path.exists(path) else None

    if args.stage in ('bridge', 'all'):
        if df is None or entropy_df is None:
            print("ERROR: Need clustered data and entropy for bridge analysis.")
            return
        bridge_df = compute_bridge_index(df, entropy_df, args.force)
    else:
        path = os.path.join(RESULTS_DIR, 'bridge_results.csv')
        bridge_df = pd.read_csv(path) if os.path.exists(path) else None

    if args.stage in ('figure', 'all'):
        if df is None or entropy_df is None or bridge_df is None:
            print("ERROR: Need all data for figure generation.")
            return
        generate_figure2(df, entropy_df, bridge_df)

    elapsed = time.time() - t0
    print(f"\n{'=' * 50}")
    print(f"Pipeline completed in {elapsed / 60:.1f} minutes")
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
