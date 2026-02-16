import numpy as np
import pandas as pd
import json
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.stats import pearsonr, ttest_1samp

PHI = 1.6180339887
FS = 512
BP_LOW = 1.0
BP_HIGH = 45.0
NOTCH = 50

BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),
}

ADJACENT_RATIOS = [
    ('theta', 'delta'),
    ('alpha', 'theta'),
    ('beta', 'alpha'),
    ('gamma', 'beta'),
]

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

def spectral_centroid(freqs, psd, lo, hi):
    idx = np.logical_and(freqs >= lo, freqs <= hi)
    f_band = freqs[idx]
    p_band = psd[idx]
    total = np.sum(p_band)
    if total == 0:
        return (lo + hi) / 2.0
    return np.sum(f_band * p_band) / total

def compute_centroids_and_bandpower(data, fs):
    nperseg = min(int(4 * fs), data.shape[-1])
    n_ch = data.shape[0]

    all_centroids = {name: [] for name in BANDS}
    all_bp = {name: [] for name in BANDS}

    for ch in range(n_ch):
        freqs, psd = welch(data[ch], fs=fs, nperseg=nperseg)
        for name, (lo, hi) in BANDS.items():
            c = spectral_centroid(freqs, psd, lo, hi)
            all_centroids[name].append(c)
            idx = np.logical_and(freqs >= lo, freqs <= hi)
            all_bp[name].append(np.trapezoid(psd[idx], freqs[idx]))

    mean_centroids = {name: np.mean(vals) for name, vals in all_centroids.items()}
    mean_bp = {name: np.mean(vals) for name, vals in all_bp.items()}
    return mean_centroids, mean_bp

def phi_error(ratio):
    return abs(ratio - PHI) / PHI

def compute_pci(ratio):
    return np.log(abs(ratio - 2.0) / abs(ratio - PHI)) if abs(ratio - PHI) > 1e-10 else 10.0

mat_files = sorted(glob.glob('alpha_s[0-9][0-9].mat')) + sorted(glob.glob('alpha_subj_[0-9][0-9].mat'))
mat_files = mat_files[:35]
print(f"Processing {len(mat_files)} subjects\n")
print("Using SPECTRAL CENTROIDS (power-weighted mean frequency per band)\n")

all_rows = []

for i, fpath in enumerate(mat_files):
    subj = os.path.basename(fpath).replace('.mat', '')
    mat = loadmat(fpath)
    data = mat['SIGNAL'].astype(np.float64).T
    data = preprocess(data, FS)

    centroids, bp = compute_centroids_and_bandpower(data, FS)

    freq_ratios = {}
    for num, den in ADJACENT_RATIOS:
        freq_ratios[f'{num}/{den}'] = centroids[num] / centroids[den] if centroids[den] > 0 else np.nan

    power_ratios = {}
    for num, den in ADJACENT_RATIOS:
        power_ratios[f'{num}/{den}'] = bp[num] / bp[den] if bp[den] > 0 else np.nan

    at_ratio = freq_ratios['alpha/theta']
    pci = compute_pci(at_ratio)

    freq_errors = {k: phi_error(v) for k, v in freq_ratios.items()}
    poi = np.mean(list(freq_errors.values()))

    row = {
        'subject': subj,
        'centroid_delta': centroids['delta'],
        'centroid_theta': centroids['theta'],
        'centroid_alpha': centroids['alpha'],
        'centroid_beta': centroids['beta'],
        'centroid_gamma': centroids['gamma'],
        'freq_ratio_alpha_theta': at_ratio,
        'PCI': pci,
        'POI': poi,
    }
    for k, v in freq_ratios.items():
        row[f'freq_ratio_{k}'] = v
    for k, v in freq_errors.items():
        row[f'freq_error_{k}'] = v
    for k, v in bp.items():
        row[f'bp_{k}'] = v
    for k, v in power_ratios.items():
        row[f'power_ratio_{k}'] = v

    all_rows.append(row)
    print(f"[{i+1}/{len(mat_files)}] {subj}: theta={centroids['theta']:.2f} Hz, alpha={centroids['alpha']:.2f} Hz, "
          f"alpha/theta={at_ratio:.4f}, PCI={pci:.4f}")

df = pd.DataFrame(all_rows)

print("\n" + "="*70)
print("RESULTS — SPECTRAL CENTROID FREQUENCY RATIOS")
print("="*70)

print("\n--- Group Mean Spectral Centroids ---")
for band in BANDS:
    col = f'centroid_{band}'
    print(f"  {band:>6s}: {df[col].mean():.2f} Hz (SD={df[col].std():.2f})")

print(f"\n--- Group Mean Frequency Ratios (centroid-based) ---")
for num, den in ADJACENT_RATIOS:
    col = f'freq_ratio_{num}/{den}'
    err_col = f'freq_error_{num}/{den}'
    print(f"  {num}/{den}: mean={df[col].mean():.4f} (SD={df[col].std():.4f}), phi error={df[err_col].mean()*100:.2f}%")

at_mean = df['freq_ratio_alpha_theta'].mean()
at_vals = df['freq_ratio_alpha_theta'].values
at_err_pct = abs(at_mean - PHI) / PHI * 100
t_stat, p_val = ttest_1samp(at_vals, PHI)

print(f"\n--- Alpha/Theta Key Stats ---")
print(f"  Mean alpha/theta ratio = {at_mean:.4f}")
print(f"  Phi                    = {PHI:.4f}")
print(f"  Difference             = {abs(at_mean - PHI):.4f}")
print(f"  Error                  = {at_err_pct:.2f}%")
print(f"  One-sample t-test vs phi: t={t_stat:.4f}, p={p_val:.4f}")

print(f"\n--- PCI Stats ---")
pci_vals = df['PCI'].values
pci_positive = np.sum(pci_vals > 0)
print(f"  Mean PCI = {np.mean(pci_vals):.4f} (SD={np.std(pci_vals):.4f})")
print(f"  PCI > 0 (phi-organized): {pci_positive}/{len(pci_vals)} ({pci_positive/len(pci_vals)*100:.1f}%)")

print(f"\n--- POI Stats ---")
print(f"  Mean POI = {df['POI'].mean():.4f} (SD={df['POI'].std():.4f})")

print("\n--- Per-Subject Table ---")
show_cols = ['subject', 'centroid_theta', 'centroid_alpha', 'freq_ratio_alpha_theta', 'PCI', 'POI']
print(df[show_cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

os.makedirs('outputs', exist_ok=True)
df.to_csv('outputs/poi_centroid_results.csv', index=False)

fig = plt.figure(figsize=(20, 16))

ax1 = fig.add_subplot(2, 3, 1)
ratio_cols = [f'freq_ratio_{n}/{d}' for n, d in ADJACENT_RATIOS]
ratio_labels = [f'{n}/{d}' for n, d in ADJACENT_RATIOS]
means = [df[c].mean() for c in ratio_cols]
stds = [df[c].std() for c in ratio_cols]
colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']
bars = ax1.bar(range(len(means)), means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
ax1.axhline(PHI, color='gold', linewidth=2.5, linestyle='--', label=f'phi = {PHI:.4f}')
ax1.set_xticks(range(len(means)))
ax1.set_xticklabels(ratio_labels, fontsize=11)
ax1.set_ylabel('Frequency Ratio', fontsize=12)
ax1.set_title('Spectral Centroid Ratios vs Phi', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
for i_b, (m, s) in enumerate(zip(means, stds)):
    ax1.text(i_b, m + s + 0.02, f'{m:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2 = fig.add_subplot(2, 3, 2)
ax2.hist(at_vals, bins=12, color='#E74C3C', alpha=0.7, edgecolor='black')
ax2.axvline(PHI, color='gold', linewidth=2.5, linestyle='--', label=f'phi = {PHI:.4f}')
ax2.axvline(at_mean, color='blue', linewidth=2, linestyle='-', label=f'mean = {at_mean:.4f}')
ax2.set_xlabel('Alpha/Theta Centroid Ratio', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Alpha/Theta Ratio Distribution', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)

ax3 = fig.add_subplot(2, 3, 3)
error_cols = [f'freq_error_{n}/{d}' for n, d in ADJACENT_RATIOS]
error_means = [df[c].mean() * 100 for c in error_cols]
error_stds = [df[c].std() * 100 for c in error_cols]
bars2 = ax3.bar(range(len(error_means)), error_means, yerr=error_stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
ax3.set_xticks(range(len(error_means)))
ax3.set_xticklabels(ratio_labels, fontsize=11)
ax3.set_ylabel('% Error from Phi', fontsize=12)
ax3.set_title('Phi Error by Centroid Ratio', fontsize=13, fontweight='bold')
for i_b, (m, s) in enumerate(zip(error_means, error_stds)):
    ax3.text(i_b, m + s + 0.5, f'{m:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax4 = fig.add_subplot(2, 3, 4)
ax4.scatter(df['centroid_theta'], df['centroid_alpha'], c='steelblue', s=80, alpha=0.7, edgecolors='black')
r_ta, p_ta = pearsonr(df['centroid_theta'], df['centroid_alpha'])
ax4.set_xlabel('Theta Centroid (Hz)', fontsize=12)
ax4.set_ylabel('Alpha Centroid (Hz)', fontsize=12)
ax4.set_title('Theta vs Alpha Centroids', fontsize=13, fontweight='bold')
ax4.text(0.05, 0.95, f'r={r_ta:.3f}, p={p_ta:.3f}', transform=ax4.transAxes, va='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax5 = fig.add_subplot(2, 3, 5)
pci_colors = ['#2ECC71' if p > 0 else '#E74C3C' for p in pci_vals]
subjects_short = [s.replace('alpha_s', 'S') for s in df['subject']]
ax5.bar(range(len(pci_vals)), pci_vals, color=pci_colors, alpha=0.8, edgecolor='black')
ax5.axhline(0, color='black', linewidth=1)
ax5.set_xticks(range(len(subjects_short)))
ax5.set_xticklabels(subjects_short, rotation=45, fontsize=8)
ax5.set_ylabel('PCI', fontsize=12)
ax5.set_title(f'PCI per Subject ({pci_positive}/{len(pci_vals)} phi-organized)', fontsize=13, fontweight='bold')

ax6 = fig.add_subplot(2, 3, 6)
poi_vals = df['POI'].values
ax6.hist(poi_vals, bins=12, color='#9B59B6', alpha=0.7, edgecolor='black')
ax6.axvline(np.mean(poi_vals), color='blue', linewidth=2, linestyle='-', label=f'mean = {np.mean(poi_vals):.4f}')
ax6.set_xlabel('Phi-Organization Index (POI)', fontsize=12)
ax6.set_ylabel('Count', fontsize=12)
ax6.set_title('POI Distribution', fontsize=13, fontweight='bold')
ax6.legend(fontsize=10)

plt.suptitle(f'Phi-Organization Analysis (Spectral Centroids) — N={len(df)} Subjects\n'
             f'Alpha/Theta = {at_mean:.4f} vs Phi = {PHI:.4f} (error = {at_err_pct:.2f}%, p = {p_val:.4f})',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/poi_centroid_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure saved: outputs/poi_centroid_analysis.png")

print("\n" + "="*70)
print("KEY FINDING")
print("="*70)
print(f"\n  Alpha/Theta centroid ratio = {at_mean:.4f}")
print(f"  Phi                        = {PHI:.4f}")
print(f"  Difference                 = {abs(at_mean - PHI):.4f}")
print(f"  Percent error              = {at_err_pct:.2f}%")
print(f"  t-test vs phi              = t={t_stat:.3f}, p={p_val:.4f}")
print(f"  PCI > 0 (phi-organized)    = {pci_positive}/{len(pci_vals)} ({pci_positive/len(pci_vals)*100:.1f}%)")
print(f"  Group POI                  = {df['POI'].mean():.4f}")
