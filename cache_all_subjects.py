#!/usr/bin/env python3
"""Cache all subject features to JSON - FAST version skipping FOOOF."""
import numpy as np
import json
import os
import glob
import warnings
warnings.filterwarnings('ignore')

from scipy.io import loadmat
from scipy.signal import butter, filtfilt, welch

PHI = 1.6180339887
E_MINUS_1 = float(np.e - 1)
BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
BAND_ORDER = ['delta', 'theta', 'alpha', 'beta', 'gamma']
ADJACENT_PAIRS = [('theta', 'delta'), ('alpha', 'theta'), ('beta', 'alpha'), ('gamma', 'beta')]

def preprocess_eeg(data, fs, bp_low=1.0, bp_high=45.0, notch_freq=None):
    from scipy.signal import iirnotch
    data = data - np.mean(data, axis=-1, keepdims=True)
    nyq = fs / 2.0
    if bp_high >= nyq:
        bp_high = nyq - 1
    b, a = butter(4, [bp_low / nyq, bp_high / nyq], btype='band')
    for ch in range(data.shape[0]):
        data[ch] = filtfilt(b, a, data[ch])
    if notch_freq and notch_freq < nyq:
        b_n, a_n = iirnotch(notch_freq, Q=30, fs=fs)
        for ch in range(data.shape[0]):
            data[ch] = filtfilt(b_n, a_n, data[ch])
    return data

def compute_avg_psd(data, fs, nperseg=None):
    if nperseg is None:
        nperseg = min(int(4 * fs), data.shape[1])
    avg_psd = None
    for ch in range(data.shape[0]):
        freqs, psd = welch(data[ch], fs=fs, nperseg=nperseg)
        if avg_psd is None:
            avg_psd = psd.copy()
        else:
            avg_psd += psd
    avg_psd /= data.shape[0]
    return freqs, avg_psd

def spectral_centroid(freqs, psd, lo, hi):
    idx = (freqs >= lo) & (freqs <= hi)
    f_band = freqs[idx]
    p_band = psd[idx]
    total = np.sum(p_band)
    if total == 0:
        return (lo + hi) / 2.0
    return float(np.sum(f_band * p_band) / total)

def bandpower(freqs, psd, lo, hi):
    idx = (freqs >= lo) & (freqs <= hi)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    return float(np.sum(psd[idx]) * df)

def extract_fast(freqs, psd, subject_id, dataset, condition, fs):
    centroids = {}
    powers = {}
    for band, (lo, hi) in BANDS.items():
        centroids[band] = spectral_centroid(freqs, psd, lo, hi)
        powers[band] = bandpower(freqs, psd, lo, hi)
    ratios = {}
    for num_band, den_band in ADJACENT_PAIRS:
        key = f"{num_band}/{den_band}"
        ratios[key] = centroids[num_band] / centroids[den_band] if centroids[den_band] > 0 else None
    skip1_ratios = {}
    for num_band, den_band in [('alpha', 'delta'), ('beta', 'theta'), ('gamma', 'alpha')]:
        key = f"{num_band}/{den_band}"
        skip1_ratios[key] = centroids[num_band] / centroids[den_band] if centroids[den_band] > 0 else None
    skip2_ratios = {}
    for num_band, den_band in [('beta', 'delta'), ('gamma', 'theta')]:
        key = f"{num_band}/{den_band}"
        skip2_ratios[key] = centroids[num_band] / centroids[den_band] if centroids[den_band] > 0 else None
    full_span = centroids['gamma'] / centroids['delta'] if centroids['delta'] > 0 else None
    alpha_theta = ratios.get('alpha/theta')
    power_ratios = {}
    for num_band, den_band in ADJACENT_PAIRS:
        key = f"P_{num_band}/P_{den_band}"
        power_ratios[key] = powers[num_band] / powers[den_band] if powers[den_band] > 0 else None

    return {
        'subject_id': subject_id, 'dataset': dataset, 'condition': condition, 'fs': fs,
        'centroids': centroids, 'powers': powers,
        'adjacent_ratios': ratios, 'skip1_ratios': skip1_ratios,
        'skip2_ratios': skip2_ratios, 'full_span_ratio': full_span,
        'alpha_theta_ratio': alpha_theta, 'power_ratios': power_ratios,
        'fooof': {'aperiodic_offset': None, 'aperiodic_slope': None, 'n_peaks': 0, 'peaks': []},
    }

def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)

    print("Caching Alpha_Waves...")
    aw = []
    seen = set()
    for pattern in ['alpha_s[0-9][0-9].mat', 'alpha_subj_[0-9][0-9].mat']:
        for fpath in sorted(glob.glob(pattern)):
            subj_name = os.path.basename(fpath).replace('.mat', '')
            num = ''.join(c for c in subj_name if c.isdigit())
            key = f"{pattern.split('[')[0]}_{num}"
            if key in seen:
                continue
            seen.add(key)
            try:
                mat = loadmat(fpath)
                data = mat['SIGNAL'].astype(np.float64).T
                data = preprocess_eeg(data, 512, notch_freq=50)
                freqs, psd = compute_avg_psd(data, 512)
                feat = extract_fast(freqs, psd, f"AW_{subj_name}", 'Alpha_Waves', 'rest', 512)
                aw.append(sanitize(feat))
            except Exception as e:
                print(f"  Skip {fpath}: {e}")
    print(f"  Cached {len(aw)} subjects")
    with open('outputs/aw_cached_subjects.json', 'w') as f:
        json.dump(aw, f)

    print("Caching OpenNeuro ds003969...")
    import mne
    mne.set_log_level('ERROR')
    ds = []
    bdf_files = sorted(glob.glob('ds003969/sub-*/eeg/*task-think1*_eeg.bdf'))
    for fpath in bdf_files:
        subj = os.path.basename(fpath).split('_')[0]
        try:
            raw = mne.io.read_raw_bdf(fpath, preload=True, verbose=False)
            eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
            if len(eeg_picks) == 0:
                continue
            raw.pick(eeg_picks)
            raw.filter(1, 45, verbose=False)
            fs = raw.info['sfreq']
            data = raw.get_data()
            if data.shape[1] < int(4 * fs):
                continue
            freqs, psd = compute_avg_psd(data, fs)
            feat = extract_fast(freqs, psd, f"DS3969_{subj}", 'OpenNeuro_ds003969', 'rest_think', fs)
            ds.append(sanitize(feat))
            if len(ds) % 20 == 0:
                print(f"  Processed {len(ds)} subjects...")
        except:
            pass
    print(f"  Cached {len(ds)} subjects")
    with open('outputs/ds003969_cached_subjects.json', 'w') as f:
        json.dump(ds, f)

    print(f"\nDone. Alpha_Waves: {len(aw)}, ds003969: {len(ds)}")
