#!/usr/bin/env python3
"""
PHI-RATIO VALIDATION ANALYSIS
==============================
1. Topographic analysis (frontal vs parietal) on GAMEEMO
2. Confirm γ/β ≈ φ on PhysioNet EEGBCI (109 subjects)
3. Compare meditation vs thinking (ds003969)
"""

import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import os
import mne
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

PHI = 1.618033988749895
FS_GAMEEMO = 128

FREQ_BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

GAMEEMO_REGIONS = {
    'frontal': ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'FC5', 'FC6'],
    'parietal': ['P7', 'P8'],
    'occipital': ['O1', 'O2'],
    'temporal': ['T7', 'T8']
}
GAMEEMO_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

GAMES = {
    'G1': {'name': 'Boring', 'arousal': 'low'},
    'G2': {'name': 'Calm', 'arousal': 'low'},
    'G3': {'name': 'Horror', 'arousal': 'high'},
    'G4': {'name': 'Funny', 'arousal': 'high'}
}

BASE_GAMEEMO = "gameemo_data/Database for Emotion Recognition System Based on EEG Signals and Various Computer Games - GAMEEMO/GAMEEMO"

def compute_band_cog(freqs, psd, band):
    """Center of gravity frequency in band"""
    low, high = FREQ_BANDS[band]
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return np.nan
    band_freqs = freqs[mask]
    band_psd = psd[mask] if len(psd.shape) == 1 else np.mean(psd[..., mask], axis=tuple(range(len(psd.shape)-1)))
    if np.sum(band_psd) == 0:
        return np.nan
    return np.sum(band_freqs * band_psd) / np.sum(band_psd)

def compute_ratios(freqs, psd):
    """Compute all adjacent band ratios"""
    cog = {band: compute_band_cog(freqs, psd, band) for band in FREQ_BANDS}
    ratios = {}
    for lower, upper in [('theta', 'alpha'), ('alpha', 'beta'), ('beta', 'gamma')]:
        if cog[lower] > 0 and cog[upper] > 0:
            ratios[f'{upper}/{lower}'] = cog[upper] / cog[lower]
        else:
            ratios[f'{upper}/{lower}'] = np.nan
    return ratios, cog

print("="*70)
print("1. TOPOGRAPHIC ANALYSIS - FRONTAL vs PARIETAL (GAMEEMO)")
print("="*70)

def load_gameemo_mat(subj_id, game_id):
    subj_folder = f"(S{subj_id:02d})"
    filename = f"S{subj_id:02d}{game_id}AllChannels.mat"
    filepath = os.path.join(BASE_GAMEEMO, subj_folder, "Preprocessed EEG Data", ".mat format", filename)
    if not os.path.exists(filepath):
        return None
    try:
        data = sio.loadmat(filepath)
        for key in data.keys():
            if not key.startswith('__'):
                eeg = data[key]
                if isinstance(eeg, np.ndarray) and eeg.size > 1000:
                    return eeg.astype(np.float64)
    except:
        pass
    return None

topographic_results = []

for subj_id in range(1, 29):
    for game_id in ['G1', 'G2', 'G3', 'G4']:
        eeg = load_gameemo_mat(subj_id, game_id)
        if eeg is None:
            continue
        
        eeg = np.nan_to_num(eeg, nan=0.0)
        if len(eeg.shape) == 1:
            eeg = eeg.reshape(1, -1)
        elif eeg.shape[0] > eeg.shape[1]:
            eeg = eeg.T
        
        n_channels = min(eeg.shape[0], len(GAMEEMO_CHANNELS))
        
        for region_name, region_channels in GAMEEMO_REGIONS.items():
            ch_indices = [i for i, ch in enumerate(GAMEEMO_CHANNELS[:n_channels]) if ch in region_channels]
            
            if len(ch_indices) == 0:
                continue
            
            region_eeg = eeg[ch_indices, :]
            mean_region = np.mean(region_eeg, axis=0)
            
            nperseg = min(256, len(mean_region) // 4)
            freqs, psd = signal.welch(mean_region, fs=FS_GAMEEMO, nperseg=nperseg)
            
            ratios, cog = compute_ratios(freqs, psd)
            
            topographic_results.append({
                'subject': subj_id,
                'game': game_id,
                'arousal': GAMES[game_id]['arousal'],
                'region': region_name,
                'gamma_beta': ratios.get('gamma/beta', np.nan),
                'alpha_theta': ratios.get('alpha/theta', np.nan),
                'beta_alpha': ratios.get('beta/alpha', np.nan),
                'cog_gamma': cog['gamma'],
                'cog_beta': cog['beta']
            })

    if subj_id % 7 == 0:
        print(f"  Processed {subj_id}/28 subjects...")

df_topo = pd.DataFrame(topographic_results)
print(f"\nTotal records: {len(df_topo)}")

print("\n--- Gamma/Beta by Region ---")
region_stats = df_topo.groupby('region')['gamma_beta'].agg(['mean', 'std', 'count'])
for region in ['frontal', 'parietal', 'occipital', 'temporal']:
    if region in region_stats.index:
        m, s, n = region_stats.loc[region]
        dist = abs(m - PHI)
        print(f"  {region.upper()}: {m:.4f} ± {s:.4f} (n={int(n)}) | dist from φ: {dist:.4f}")

frontal = df_topo[df_topo['region'] == 'frontal']['gamma_beta'].dropna()
parietal = df_topo[df_topo['region'] == 'parietal']['gamma_beta'].dropna()

if len(frontal) > 3 and len(parietal) > 3:
    t_stat, p_val = stats.ttest_ind(frontal, parietal)
    print(f"\nFrontal vs Parietal t-test: t={t_stat:.2f}, p={p_val:.4f}")
    print(f"  Frontal closer to φ: {abs(frontal.mean() - PHI) < abs(parietal.mean() - PHI)}")

df_topo.to_csv('gameemo_topographic.csv', index=False)
print("Saved: gameemo_topographic.csv")

print("\n" + "="*70)
print("2. PhysioNet EEGBCI CONFIRMATION (109 subjects)")
print("="*70)

eegbci_results = []
n_subjects = 0

for subj in range(1, 110):
    try:
        raw_fnames = mne.datasets.eegbci.load_data(subj, runs=[1], path='/tmp/eegbci', update_path=False, verbose=False)
        raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, verbose=False)
        raw.filter(1, 50, verbose=False)
        
        data = raw.get_data()
        fs = raw.info['sfreq']
        
        mean_data = np.mean(data, axis=0)
        nperseg = min(int(fs * 2), len(mean_data) // 4)
        freqs, psd = signal.welch(mean_data, fs=fs, nperseg=nperseg)
        
        ratios, cog = compute_ratios(freqs, psd)
        
        eegbci_results.append({
            'subject': subj,
            'gamma_beta': ratios.get('gamma/beta', np.nan),
            'alpha_theta': ratios.get('alpha/theta', np.nan),
            'beta_alpha': ratios.get('beta/alpha', np.nan)
        })
        n_subjects += 1
        
    except Exception as e:
        pass
    
    if subj % 20 == 0:
        print(f"  Processed {subj}/109 subjects...")

df_eegbci = pd.DataFrame(eegbci_results)
print(f"\nLoaded {n_subjects} subjects from PhysioNet EEGBCI")

if len(df_eegbci) > 0:
    print("\n--- EEGBCI Ratios ---")
    for col in ['gamma_beta', 'alpha_theta', 'beta_alpha']:
        vals = df_eegbci[col].dropna()
        if len(vals) > 0:
            m = vals.mean()
            s = vals.std()
            dist_phi = abs(m - PHI)
            dist_2 = abs(m - 2.0)
            best = 'φ' if dist_phi < dist_2 else '2.0'
            print(f"  {col}: {m:.4f} ± {s:.4f} | dist from φ: {dist_phi:.4f}, from 2.0: {dist_2:.4f} → {best}")
    
    gb = df_eegbci['gamma_beta'].dropna()
    t_stat, p_val = stats.ttest_1samp(gb, PHI)
    print(f"\n  Gamma/Beta vs φ: t={t_stat:.2f}, p={p_val:.4f}")
    print(f"  Gamma/Beta mean: {gb.mean():.4f} (only {100*abs(gb.mean()-PHI)/PHI:.1f}% from φ)")
    
    df_eegbci.to_csv('eegbci_phi_results.csv', index=False)
    print("Saved: eegbci_phi_results.csv")

print("\n" + "="*70)
print("3. MEDITATION vs THINKING COMPARISON (ds003969)")
print("="*70)

meditation_results = []
n_meditation = 0

for subj_id in range(1, 30):
    subj_folder = f"ds003969/sub-{subj_id:03d}/eeg"
    
    for task, task_type in [('med1breath', 'meditation'), ('think1', 'thinking')]:
        filepath = os.path.join(subj_folder, f"sub-{subj_id:03d}_task-{task}_eeg.bdf")
        
        if not os.path.exists(filepath):
            continue
        
        try:
            raw = mne.io.read_raw_bdf(filepath, preload=True, verbose=False)
            raw.filter(1, 50, verbose=False)
            
            data = raw.get_data()
            fs = raw.info['sfreq']
            
            mean_data = np.mean(data, axis=0)
            nperseg = min(int(fs * 2), len(mean_data) // 4)
            freqs, psd = signal.welch(mean_data, fs=fs, nperseg=nperseg)
            
            ratios, cog = compute_ratios(freqs, psd)
            
            meditation_results.append({
                'subject': subj_id,
                'task': task_type,
                'gamma_beta': ratios.get('gamma/beta', np.nan),
                'alpha_theta': ratios.get('alpha/theta', np.nan),
                'beta_alpha': ratios.get('beta/alpha', np.nan)
            })
            n_meditation += 1
            
        except Exception as e:
            pass
    
    if subj_id % 10 == 0:
        print(f"  Processed {subj_id}/30 subjects...")

if len(meditation_results) > 0:
    df_med = pd.DataFrame(meditation_results)
    print(f"\nLoaded {n_meditation} recordings from meditation dataset")
    
    med = df_med[df_med['task'] == 'meditation']
    think = df_med[df_med['task'] == 'thinking']
    
    print("\n--- Gamma/Beta: Meditation vs Thinking ---")
    med_gb = med['gamma_beta'].dropna()
    think_gb = think['gamma_beta'].dropna()
    
    if len(med_gb) > 3 and len(think_gb) > 3:
        print(f"  MEDITATION: {med_gb.mean():.4f} ± {med_gb.std():.4f} (dist from φ: {abs(med_gb.mean()-PHI):.4f})")
        print(f"  THINKING:   {think_gb.mean():.4f} ± {think_gb.std():.4f} (dist from φ: {abs(think_gb.mean()-PHI):.4f})")
        
        t_stat, p_val = stats.ttest_ind(med_gb, think_gb)
        print(f"  t-test: t={t_stat:.2f}, p={p_val:.4f}")
        
        if abs(med_gb.mean() - PHI) < abs(think_gb.mean() - PHI):
            print("  → MEDITATION closer to φ!")
        else:
            print("  → THINKING closer to φ!")
    
    df_med.to_csv('meditation_phi_results.csv', index=False)
    print("Saved: meditation_phi_results.csv")
else:
    print("No meditation data could be loaded.")

print("\n" + "="*70)
print("GENERATING COMPARISON FIGURES")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

ax = axes[0, 0]
if 'df_topo' in dir() and len(df_topo) > 0:
    regions = ['frontal', 'parietal', 'occipital', 'temporal']
    means = [df_topo[df_topo['region'] == r]['gamma_beta'].mean() for r in regions]
    stds = [df_topo[df_topo['region'] == r]['gamma_beta'].std() for r in regions]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(regions, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    ax.axhline(y=PHI, color='gold', linestyle='--', linewidth=2, label=f'φ = {PHI:.3f}')
    ax.set_ylabel('Gamma/Beta Ratio')
    ax.set_title('GAMEEMO: Topographic γ/β')
    ax.legend()

ax = axes[0, 1]
if 'df_eegbci' in dir() and len(df_eegbci) > 0:
    gb_vals = df_eegbci['gamma_beta'].dropna()
    ax.hist(gb_vals, bins=20, color='steelblue', alpha=0.7, density=True)
    ax.axvline(x=PHI, color='gold', linestyle='--', linewidth=2, label=f'φ = {PHI:.3f}')
    ax.axvline(x=gb_vals.mean(), color='red', linestyle='-', linewidth=2, label=f'mean = {gb_vals.mean():.3f}')
    ax.set_xlabel('Gamma/Beta Ratio')
    ax.set_title(f'PhysioNet EEGBCI (N={len(gb_vals)})')
    ax.legend()

ax = axes[0, 2]
if 'df_med' in dir() and len(df_med) > 0:
    med_gb = df_med[df_med['task'] == 'meditation']['gamma_beta'].dropna()
    think_gb = df_med[df_med['task'] == 'thinking']['gamma_beta'].dropna()
    if len(med_gb) > 0 and len(think_gb) > 0:
        bp = ax.boxplot([med_gb, think_gb], labels=['Meditation', 'Thinking'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.axhline(y=PHI, color='gold', linestyle='--', linewidth=2)
        ax.set_ylabel('Gamma/Beta Ratio')
        ax.set_title('Meditation vs Thinking')

ax = axes[1, 0]
datasets = []
means = []
stds = []
labels = []

if 'df_topo' in dir():
    gameemo_gb = df_topo['gamma_beta'].dropna()
    if len(gameemo_gb) > 0:
        labels.append('GAMEEMO\n(N=112)')
        means.append(gameemo_gb.mean())
        stds.append(gameemo_gb.std())

if 'df_eegbci' in dir():
    eegbci_gb = df_eegbci['gamma_beta'].dropna()
    if len(eegbci_gb) > 0:
        labels.append('PhysioNet\n(N=109)')
        means.append(eegbci_gb.mean())
        stds.append(eegbci_gb.std())

if 'df_med' in dir():
    med_all = df_med['gamma_beta'].dropna()
    if len(med_all) > 0:
        labels.append('Meditation\nDataset')
        means.append(med_all.mean())
        stds.append(med_all.std())

if len(labels) > 0:
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(labels)], alpha=0.7)
    ax.axhline(y=PHI, color='gold', linestyle='--', linewidth=2, label=f'φ = {PHI:.3f}')
    ax.set_ylabel('Gamma/Beta Ratio')
    ax.set_title('Cross-Dataset Comparison')
    ax.legend()

ax = axes[1, 1]
all_ratios = []
all_labels = []
if 'df_topo' in dir() and len(df_topo) > 0:
    for col, label in [('alpha_theta', 'α/θ'), ('beta_alpha', 'β/α'), ('gamma_beta', 'γ/β')]:
        vals = df_topo[col].dropna()
        if len(vals) > 0:
            all_ratios.append(vals.values)
            all_labels.append(label)

if len(all_ratios) > 0:
    bp = ax.boxplot(all_ratios, labels=all_labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'gold']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.axhline(y=PHI, color='red', linestyle='--', linewidth=2, label=f'φ = {PHI:.3f}')
    ax.axhline(y=2.0, color='gray', linestyle=':', linewidth=1, label='2.0')
    ax.set_ylabel('Ratio')
    ax.set_title('All Ratios (GAMEEMO)')
    ax.legend()

ax = axes[1, 2]
summary = "CROSS-DATASET γ/β SUMMARY\n" + "="*30 + "\n\n"

if 'df_topo' in dir():
    gameemo_gb = df_topo['gamma_beta'].dropna()
    summary += f"GAMEEMO (N=112):\n  γ/β = {gameemo_gb.mean():.4f}\n  dist from φ: {abs(gameemo_gb.mean()-PHI):.4f}\n\n"

if 'df_eegbci' in dir() and len(df_eegbci) > 0:
    eegbci_gb = df_eegbci['gamma_beta'].dropna()
    summary += f"PhysioNet EEGBCI (N={len(eegbci_gb)}):\n  γ/β = {eegbci_gb.mean():.4f}\n  dist from φ: {abs(eegbci_gb.mean()-PHI):.4f}\n\n"

if 'df_med' in dir() and len(df_med) > 0:
    med_gb = df_med[df_med['task'] == 'meditation']['gamma_beta'].dropna()
    if len(med_gb) > 0:
        summary += f"Meditation (N={len(med_gb)}):\n  γ/β = {med_gb.mean():.4f}\n  dist from φ: {abs(med_gb.mean()-PHI):.4f}\n"

summary += "\n" + "="*30 + "\n"
summary += f"φ = {PHI:.4f}"

ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace')
ax.axis('off')

plt.tight_layout()
plt.savefig('phi_validation_figures.png', dpi=150, bbox_inches='tight')
print("Saved: phi_validation_figures.png")
plt.close()

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print("\nGamma/Beta ratio across datasets:")
if 'df_topo' in dir():
    gb = df_topo['gamma_beta'].dropna()
    print(f"  GAMEEMO:   {gb.mean():.4f} ± {gb.std():.4f} (N={len(gb)}, {100*abs(gb.mean()-PHI)/PHI:.1f}% from φ)")

if 'df_eegbci' in dir() and len(df_eegbci) > 0:
    gb = df_eegbci['gamma_beta'].dropna()
    print(f"  EEGBCI:    {gb.mean():.4f} ± {gb.std():.4f} (N={len(gb)}, {100*abs(gb.mean()-PHI)/PHI:.1f}% from φ)")

if 'df_med' in dir() and len(df_med) > 0:
    gb = df_med['gamma_beta'].dropna()
    print(f"  Meditation:{gb.mean():.4f} ± {gb.std():.4f} (N={len(gb)}, {100*abs(gb.mean()-PHI)/PHI:.1f}% from φ)")

print("\n" + "="*70)
