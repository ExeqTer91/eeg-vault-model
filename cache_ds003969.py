#!/usr/bin/env python3
"""Cache ds003969 subjects incrementally - saves after each subject."""
import numpy as np
import json
import os
import glob
import warnings
warnings.filterwarnings('ignore')

from scipy.signal import butter, filtfilt, welch

BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
ADJACENT_PAIRS = [('theta', 'delta'), ('alpha', 'theta'), ('beta', 'alpha'), ('gamma', 'beta')]

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
    f_band, p_band = freqs[idx], psd[idx]
    total = np.sum(p_band)
    if total == 0:
        return (lo + hi) / 2.0
    return float(np.sum(f_band * p_band) / total)

def bandpower(freqs, psd, lo, hi):
    idx = (freqs >= lo) & (freqs <= hi)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    return float(np.sum(psd[idx]) * df)

def extract_fast(freqs, psd, subject_id, fs):
    centroids = {band: spectral_centroid(freqs, psd, lo, hi) for band, (lo, hi) in BANDS.items()}
    powers = {band: bandpower(freqs, psd, lo, hi) for band, (lo, hi) in BANDS.items()}
    ratios = {}
    for num_band, den_band in ADJACENT_PAIRS:
        key = f"{num_band}/{den_band}"
        ratios[key] = centroids[num_band] / centroids[den_band] if centroids[den_band] > 0 else None
    skip1 = {}
    for n, d in [('alpha','delta'),('beta','theta'),('gamma','alpha')]:
        skip1[f"{n}/{d}"] = centroids[n]/centroids[d] if centroids[d] > 0 else None
    skip2 = {}
    for n, d in [('beta','delta'),('gamma','theta')]:
        skip2[f"{n}/{d}"] = centroids[n]/centroids[d] if centroids[d] > 0 else None
    power_ratios = {}
    for n, d in ADJACENT_PAIRS:
        power_ratios[f"P_{n}/P_{d}"] = powers[n]/powers[d] if powers[d] > 0 else None
    return {
        'subject_id': subject_id, 'dataset': 'OpenNeuro_ds003969',
        'condition': 'rest_think', 'fs': fs,
        'centroids': centroids, 'powers': powers,
        'adjacent_ratios': ratios, 'skip1_ratios': skip1, 'skip2_ratios': skip2,
        'full_span_ratio': centroids['gamma']/centroids['delta'] if centroids['delta'] > 0 else None,
        'alpha_theta_ratio': ratios.get('alpha/theta'), 'power_ratios': power_ratios,
        'fooof': {'aperiodic_offset': None, 'aperiodic_slope': None},
    }

def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj

if __name__ == '__main__':
    cache_path = 'outputs/ds003969_cached_subjects.json'
    existing = []
    done_ids = set()
    if os.path.exists(cache_path):
        existing = json.load(open(cache_path))
        done_ids = {s['subject_id'] for s in existing}
        print(f"Resuming from {len(existing)} already cached subjects")

    import mne
    mne.set_log_level('ERROR')
    bdf_files = sorted(glob.glob('ds003969/sub-*/eeg/*task-think1*_eeg.bdf'))
    print(f"Found {len(bdf_files)} BDF files")

    count = 0
    for fpath in bdf_files:
        subj = os.path.basename(fpath).split('_')[0]
        sid = f"DS3969_{subj}"
        if sid in done_ids:
            continue
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
            feat = extract_fast(freqs, psd, sid, fs)
            existing.append(sanitize(feat))
            count += 1
            if count % 5 == 0:
                with open(cache_path, 'w') as f:
                    json.dump(existing, f)
                print(f"  Saved {len(existing)} subjects (batch {count})")
        except Exception as e:
            pass

    with open(cache_path, 'w') as f:
        json.dump(existing, f)
    print(f"Final: {len(existing)} subjects cached")
