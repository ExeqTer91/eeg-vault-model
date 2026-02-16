#!/usr/bin/env python3
"""
Phase 2: Process PhysioNet EEGBCI (N=109) into epoch-level features
matching the N=15 format for identical test replication.
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
import warnings, os, sys
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2

def power_law(f, a, b):
    return a * f**(-b)

def compute_epoch_features(data, sfreq, epoch_id, subject):
    rows = []
    nperseg = min(256, len(data))
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg)

    mask = freqs > 0
    freqs_pos = freqs[mask]
    psd_pos = psd[mask]

    bands = {
        'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
        'beta': (13, 30), 'gamma': (30, 45)
    }
    powers = {}
    for name, (lo, hi) in bands.items():
        idx = (freqs_pos >= lo) & (freqs_pos <= hi)
        powers[name] = np.trapezoid(psd_pos[idx], freqs_pos[idx]) if idx.any() else 0

    beta_idx = (freqs_pos >= 13) & (freqs_pos <= 30)
    gamma_idx = (freqs_pos >= 30) & (freqs_pos <= 45)

    if beta_idx.any() and psd_pos[beta_idx].sum() > 0:
        beta_cf = np.average(freqs_pos[beta_idx], weights=psd_pos[beta_idx])
    else:
        beta_cf = 21.5

    if gamma_idx.any() and psd_pos[gamma_idx].sum() > 0:
        gamma_cf = np.average(freqs_pos[gamma_idx], weights=psd_pos[gamma_idx])
    else:
        gamma_cf = 37.0

    r = gamma_cf / beta_cf

    try:
        fit_idx = (freqs_pos >= 2) & (freqs_pos <= 40)
        popt, _ = curve_fit(power_law, freqs_pos[fit_idx], psd_pos[fit_idx],
                            p0=[1, 1], maxfev=500)
        aperiodic_exp = popt[1]
    except:
        aperiodic_exp = 1.0

    delta_score = np.log(abs(r - 2.0) + 1e-6) - np.log(abs(r - PHI) + 1e-6)
    hit_phi = 1 if abs(r - PHI) < abs(r - 2.0) else 0

    return {
        'subject': subject, 'epoch_id': epoch_id,
        'delta_power': powers['delta'], 'theta_power': powers['theta'],
        'alpha_power': powers['alpha'], 'beta_power': powers['beta'],
        'gamma_power': powers['gamma'],
        'beta_cf': beta_cf, 'gamma_cf': gamma_cf,
        'aperiodic_exponent': aperiodic_exp, 'r': r,
        'delta_score': delta_score, 'hit_phi': hit_phi
    }


def process_subject(subject_id):
    try:
        import mne
        mne.set_log_level('ERROR')
        runs = [1]
        raw_fnames = mne.datasets.eegbci.load_data(subject_id, runs, update_path=False)
        raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, verbose=False)
        raw.filter(1, 45, verbose=False)

        sfreq = raw.info['sfreq']
        data = raw.get_data()

        epoch_len = int(2.0 * sfreq)
        n_epochs = data.shape[1] // epoch_len
        rows = []

        for ch_idx in range(0, min(data.shape[0], 8), 2):
            for ep in range(n_epochs):
                start = ep * epoch_len
                segment = data[ch_idx, start:start+epoch_len]
                feat = compute_epoch_features(segment, sfreq, ep, subject_id)
                feat['channel'] = ch_idx
                rows.append(feat)

        return rows
    except Exception as e:
        print(f"  Subject {subject_id} failed: {e}")
        return []


def assign_states(df, n_states=6):
    from sklearn.cluster import MiniBatchKMeans
    features = ['aperiodic_exponent', 'r', 'beta_cf', 'gamma_cf']
    X = df[features].values
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    km = MiniBatchKMeans(n_clusters=n_states, random_state=42, batch_size=1000)
    df['state'] = km.fit_predict(X)

    state_delta = df.groupby('state')['delta_score'].mean()
    phi_states = [s for s in range(n_states) if state_delta.get(s, 0) < -0.05]
    harmonic_states = [s for s in range(n_states) if state_delta.get(s, 0) > 0.02]
    bridge_states = [s for s in range(n_states) if -0.05 <= state_delta.get(s, 0) <= 0.02]
    df['regime'] = df['state'].apply(lambda s:
        'phi-like' if s in phi_states else
        ('harmonic' if s in harmonic_states else 'bridge'))
    return df


def main():
    out_path = 'epoch_features_n109.csv'
    if os.path.exists(out_path):
        print(f"Already exists: {out_path}")
        df = pd.read_csv(out_path)
        print(f"  {len(df)} rows, {df['subject'].nunique()} subjects")
        return

    partial_path = 'epoch_features_n109_partial.csv'
    if os.path.exists(partial_path):
        existing = pd.read_csv(partial_path)
        all_rows = existing.to_dict('records')
        done_subs = set(existing['subject'].unique())
        print(f"Resuming from checkpoint: {len(done_subs)} subjects done")
    else:
        all_rows = []
        done_subs = set()

    for sid in range(1, 110):
        if sid in done_subs:
            continue
        print(f"Processing subject {sid}/109...", end=' ', flush=True)
        rows = process_subject(sid)
        all_rows.extend(rows)
        print(f"{len(rows)} epochs")

        if sid % 20 == 0:
            pd.DataFrame(all_rows).to_csv(f'epoch_features_n109_partial.csv', index=False)
            print(f"  Checkpoint saved: {len(all_rows)} total rows")

    df = pd.DataFrame(all_rows)
    print(f"\nTotal: {len(df)} rows, {df['subject'].nunique()} subjects")

    df = assign_states(df)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
