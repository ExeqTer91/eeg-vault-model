import numpy as np
import pandas as pd
import json
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from eeg_features import (
    compute_bandpower, compute_psd, compute_gfp,
    compute_plv, compute_coherence_matrix,
    compute_kuramoto, compute_mean_plv, BANDS, bandpass_filter
)
from scipy.signal import butter, filtfilt, iirnotch

FS = 512
BP_LOW = 1.0
BP_HIGH = 45.0
NOTCH = 50
ALPHA_BAND = (8, 12)

def preprocess(data, fs):
    data = data - np.mean(data, axis=0, keepdims=True)
    nyq = fs / 2.0
    b, a = butter(4, [BP_LOW / nyq, BP_HIGH / nyq], btype='band')
    for ch in range(data.shape[0]):
        data[ch] = filtfilt(b, a, data[ch])
    b_n, a_n = iirnotch(NOTCH, Q=30, fs=fs)
    for ch in range(data.shape[0]):
        data[ch] = filtfilt(b_n, a_n, data[ch])
    return data

mat_files = sorted(glob.glob('alpha_s[0-9][0-9].mat'))
print(f"Found {len(mat_files)} unique alpha subject files\n")

all_results = []

for i, fpath in enumerate(mat_files):
    subj = os.path.basename(fpath).replace('.mat', '')
    print(f"[{i+1}/{len(mat_files)}] Processing {subj}...", end=' ', flush=True)

    mat = loadmat(fpath)
    data = mat['SIGNAL'].astype(np.float64).T
    n_ch, n_samp = data.shape
    duration = n_samp / FS

    data = preprocess(data, FS)

    bp = compute_bandpower(data, FS)

    plv_matrix = compute_plv(data, FS, band=ALPHA_BAND)
    mean_plv = compute_mean_plv(plv_matrix)

    n_upper = n_ch * (n_ch - 1) // 2
    coh_vals = []
    from itertools import combinations
    nperseg = min(int(4 * FS), n_samp)
    from scipy.signal import coherence as scipy_coherence
    for ci, cj in combinations(range(n_ch), 2):
        freqs_c, cxy = scipy_coherence(data[ci], data[cj], fs=FS, nperseg=nperseg)
        idx = np.logical_and(freqs_c >= ALPHA_BAND[0], freqs_c <= ALPHA_BAND[1])
        coh_vals.append(np.mean(cxy[idx]))
    mean_coh = np.mean(coh_vals)

    r_t, mean_r = compute_kuramoto(data, FS, band=ALPHA_BAND)
    gfp = compute_gfp(data)

    res = {
        'subject': subj,
        'n_channels': n_ch,
        'n_samples': n_samp,
        'duration_s': round(duration, 1),
        'mean_plv_alpha': round(float(mean_plv), 4),
        'mean_coherence_alpha': round(float(mean_coh), 4),
        'mean_kuramoto_r': round(float(mean_r), 4),
        'std_kuramoto_r': round(float(np.std(r_t)), 4),
        'mean_gfp': round(float(np.mean(gfp)), 4),
    }
    for band_name, vals in bp.items():
        res[f'mean_bp_{band_name}'] = round(float(np.mean(vals)), 8)

    all_results.append(res)
    print(f"PLV={mean_plv:.4f}, Coh={mean_coh:.4f}, Kuramoto r={mean_r:.4f}")

print("\n" + "="*70)
print("BATCH ANALYSIS COMPLETE")
print("="*70)

df = pd.DataFrame(all_results)
print(f"\nProcessed {len(df)} subjects\n")

print("--- Synchrony Metrics (Alpha 8-12 Hz) ---")
for col in ['mean_plv_alpha', 'mean_coherence_alpha', 'mean_kuramoto_r']:
    vals = df[col]
    print(f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, min={vals.min():.4f}, max={vals.max():.4f}")

print("\n--- Bandpower (mean across channels) ---")
for band in BANDS:
    col = f'mean_bp_{band}'
    vals = df[col]
    print(f"  {band:>6s}: mean={vals.mean():.6e}, std={vals.std():.6e}")

os.makedirs('outputs', exist_ok=True)
df.to_csv('outputs/batch_results.csv', index=False)
with open('outputs/batch_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

from scipy.stats import pearsonr

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
metrics = ['mean_plv_alpha', 'mean_coherence_alpha', 'mean_kuramoto_r']
labels = ['PLV', 'Coherence', 'Kuramoto r']
bp_data = [df[m].values for m in metrics]
bp_plot = ax.boxplot(bp_data, positions=range(len(metrics)), widths=0.5, patch_artist=True)
colors = ['#FF6B6B', '#4ECDC4', '#9B59B6']
for patch, color in zip(bp_plot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(labels)
ax.set_title('Alpha-Band Synchrony Metrics')
ax.set_ylabel('Value')

ax = axes[0, 1]
band_means = [df[f'mean_bp_{b}'].values for b in BANDS]
band_labels = list(BANDS.keys())
bp_plot2 = ax.boxplot(band_means, positions=range(len(BANDS)), widths=0.5, patch_artist=True)
colors2 = ['#3498DB', '#2ECC71', '#F1C40F', '#E67E22', '#E74C3C']
for patch, color in zip(bp_plot2['boxes'], colors2):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticks(range(len(BANDS)))
ax.set_xticklabels(band_labels)
ax.set_title('Bandpower Distribution')
ax.set_ylabel('Power')
ax.set_yscale('log')

ax = axes[1, 0]
ax.scatter(df['mean_plv_alpha'], df['mean_kuramoto_r'], c='steelblue', alpha=0.7, s=60)
ax.set_xlabel('Mean PLV (Alpha)')
ax.set_ylabel('Mean Kuramoto r')
ax.set_title('PLV vs Kuramoto Synchrony')
r_corr, p_corr = pearsonr(df['mean_plv_alpha'], df['mean_kuramoto_r'])
ax.text(0.05, 0.95, f'r={r_corr:.3f}, p={p_corr:.2e}', transform=ax.transAxes, va='top', fontsize=10)

ax = axes[1, 1]
ax.scatter(df['mean_bp_alpha'], df['mean_kuramoto_r'], c='darkorange', alpha=0.7, s=60)
ax.set_xlabel('Mean Alpha Bandpower')
ax.set_ylabel('Mean Kuramoto r')
ax.set_title('Alpha Power vs Synchrony')
r_corr2, p_corr2 = pearsonr(df['mean_bp_alpha'], df['mean_kuramoto_r'])
ax.text(0.05, 0.95, f'r={r_corr2:.3f}, p={p_corr2:.2e}', transform=ax.transAxes, va='top', fontsize=10)

plt.suptitle(f'EEG Batch Analysis: N={len(df)} Subjects (Alpha Waves Dataset, 512 Hz)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/batch_analysis_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure saved: outputs/batch_analysis_summary.png")

print("\n--- Full Results Table ---")
print(df[['subject', 'duration_s', 'mean_plv_alpha', 'mean_coherence_alpha', 'mean_kuramoto_r', 'mean_bp_alpha']].to_string(index=False))

print("\n--- Key Correlations ---")
print(f"  PLV vs Kuramoto r:     r={r_corr:.4f}, p={p_corr:.2e}")
print(f"  Alpha Power vs Kuram.: r={r_corr2:.4f}, p={p_corr2:.2e}")

print("\nAll outputs saved to outputs/")
