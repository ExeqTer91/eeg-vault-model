import numpy as np
import pandas as pd
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.stats import pearsonr

PHI = 1.6180339887
FS = 512
BP_LOW = 1.0
BP_HIGH = 45.0
NOTCH = 50

BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
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

def compute_bandpower(data, fs):
    nperseg = min(int(4 * fs), data.shape[-1])
    n_ch = data.shape[0]
    bp = {name: np.zeros(n_ch) for name in BANDS}
    for ch in range(n_ch):
        freqs, psd = welch(data[ch], fs=fs, nperseg=nperseg)
        for name, (lo, hi) in BANDS.items():
            idx = np.logical_and(freqs >= lo, freqs <= hi)
            bp[name][ch] = np.trapezoid(psd[idx], freqs[idx])
    return bp

def phi_error(ratio):
    return abs(ratio - PHI) / PHI

def compute_poi(ratios):
    errors = [phi_error(r) for r in ratios]
    return np.mean(errors)

import glob
mat_files = sorted(glob.glob('alpha_s[0-9][0-9].mat'))
print(f"Processing {len(mat_files)} subjects\n")

all_rows = []

for i, fpath in enumerate(mat_files):
    subj = os.path.basename(fpath).replace('.mat', '')
    mat = loadmat(fpath)
    data = mat['SIGNAL'].astype(np.float64).T
    data = preprocess(data, FS)

    bp = compute_bandpower(data, FS)

    mean_bp = {name: np.mean(vals) for name, vals in bp.items()}

    ratios = {}
    for num, den in ADJACENT_RATIOS:
        r = mean_bp[num] / mean_bp[den] if mean_bp[den] > 0 else np.nan
        ratios[f'{num}/{den}'] = r

    errors = {k: phi_error(v) for k, v in ratios.items()}
    poi = compute_poi(list(ratios.values()))

    row = {'subject': subj}
    row.update({f'bp_{k}': v for k, v in mean_bp.items()})
    row.update({f'ratio_{k}': v for k, v in ratios.items()})
    row.update({f'error_{k}': v for k, v in errors.items()})
    row['POI'] = poi

    all_rows.append(row)
    print(f"[{i+1}/{len(mat_files)}] {subj}: alpha/theta={ratios['alpha/theta']:.4f}, POI={poi:.4f}")

df = pd.DataFrame(all_rows)

print("\n" + "="*70)
print("RESULTS")
print("="*70)

print("\n--- Group Mean Band Ratios ---")
for num, den in ADJACENT_RATIOS:
    col = f'ratio_{num}/{den}'
    vals = df[col]
    err_col = f'error_{num}/{den}'
    errs = df[err_col]
    print(f"  {num}/{den}: mean={vals.mean():.4f} (SD={vals.std():.4f}), phi error={errs.mean()*100:.2f}%")

print(f"\n  phi = {PHI:.4f}")

print(f"\n--- Group Mean POI ---")
print(f"  POI = {df['POI'].mean():.4f} (SD={df['POI'].std():.4f})")

print("\n--- Per-Subject Table ---")
cols_to_show = ['subject'] + [f'ratio_{n}/{d}' for n, d in ADJACENT_RATIOS] + ['POI']
print(df[cols_to_show].to_string(index=False))

os.makedirs('outputs', exist_ok=True)
df.to_csv('outputs/poi_results.csv', index=False)

fig = plt.figure(figsize=(18, 14))

ax1 = fig.add_subplot(2, 3, 1)
ratio_cols = [f'ratio_{n}/{d}' for n, d in ADJACENT_RATIOS]
ratio_labels = [f'{n}/{d}' for n, d in ADJACENT_RATIOS]
means = [df[c].mean() for c in ratio_cols]
stds = [df[c].std() for c in ratio_cols]
colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']
bars = ax1.bar(range(len(means)), means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
ax1.axhline(PHI, color='gold', linewidth=2, linestyle='--', label=f'phi = {PHI:.4f}')
ax1.set_xticks(range(len(means)))
ax1.set_xticklabels(ratio_labels, fontsize=11)
ax1.set_ylabel('Ratio', fontsize=12)
ax1.set_title('Adjacent Band Power Ratios vs Phi', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
for i_b, (m, s) in enumerate(zip(means, stds)):
    ax1.text(i_b, m + s + 0.05, f'{m:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2 = fig.add_subplot(2, 3, 2)
error_cols = [f'error_{n}/{d}' for n, d in ADJACENT_RATIOS]
error_means = [df[c].mean() * 100 for c in error_cols]
error_stds = [df[c].std() * 100 for c in error_cols]
bars2 = ax2.bar(range(len(error_means)), error_means, yerr=error_stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
ax2.set_xticks(range(len(error_means)))
ax2.set_xticklabels(ratio_labels, fontsize=11)
ax2.set_ylabel('% Error from Phi', fontsize=12)
ax2.set_title('Phi Error by Band Ratio', fontsize=13, fontweight='bold')
for i_b, (m, s) in enumerate(zip(error_means, error_stds)):
    ax2.text(i_b, m + s + 1, f'{m:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3 = fig.add_subplot(2, 3, 3)
at_vals = df['ratio_alpha/theta'].values
ax3.hist(at_vals, bins=12, color='#E74C3C', alpha=0.7, edgecolor='black')
ax3.axvline(PHI, color='gold', linewidth=2.5, linestyle='--', label=f'phi = {PHI:.4f}')
ax3.axvline(np.mean(at_vals), color='blue', linewidth=2, linestyle='-', label=f'mean = {np.mean(at_vals):.4f}')
ax3.set_xlabel('Alpha/Theta Ratio', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('Alpha/Theta Ratio Distribution', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)

ax4 = fig.add_subplot(2, 3, 4)
poi_vals = df['POI'].values
ax4.hist(poi_vals, bins=12, color='#9B59B6', alpha=0.7, edgecolor='black')
ax4.axvline(np.mean(poi_vals), color='blue', linewidth=2, linestyle='-', label=f'mean POI = {np.mean(poi_vals):.4f}')
ax4.set_xlabel('Phi-Organization Index (POI)', fontsize=12)
ax4.set_ylabel('Count', fontsize=12)
ax4.set_title('POI Distribution (N=19)', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)

ax5 = fig.add_subplot(2, 3, 5)
subjects = [s.replace('alpha_s', 'S') for s in df['subject']]
x_pos = range(len(subjects))
for idx, (num, den) in enumerate(ADJACENT_RATIOS):
    col = f'ratio_{num}/{den}'
    ax5.plot(x_pos, df[col].values, 'o-', color=colors[idx], label=f'{num}/{den}', alpha=0.8, markersize=5)
ax5.axhline(PHI, color='gold', linewidth=2, linestyle='--', label='phi')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(subjects, rotation=45, fontsize=8)
ax5.set_ylabel('Ratio', fontsize=12)
ax5.set_title('Per-Subject Band Ratios', fontsize=13, fontweight='bold')
ax5.legend(fontsize=8, loc='upper right')

ax6 = fig.add_subplot(2, 3, 6)
at_err = df['error_alpha/theta'].values * 100
bp_alpha = df['bp_alpha'].values
ax6.scatter(bp_alpha, at_err, c='#E74C3C', s=60, alpha=0.7, edgecolors='black')
r_val, p_val = pearsonr(bp_alpha, at_err)
ax6.set_xlabel('Alpha Bandpower', fontsize=12)
ax6.set_ylabel('Alpha/Theta Phi-Error (%)', fontsize=12)
ax6.set_title('Alpha Power vs Phi Proximity', fontsize=13, fontweight='bold')
ax6.text(0.05, 0.95, f'r={r_val:.3f}, p={p_val:.3f}', transform=ax6.transAxes, va='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle(f'Phi-Organization Index (POI) Analysis — N={len(df)} Subjects\n'
             f'Alpha/Theta = {df["ratio_alpha/theta"].mean():.4f} | Phi = {PHI:.4f} | Error = {df["error_alpha/theta"].mean()*100:.2f}%',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/poi_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure saved: outputs/poi_analysis.png")

print("\n" + "="*70)
print("KEY FINDING")
print("="*70)
at_mean = df['ratio_alpha/theta'].mean()
at_err_pct = abs(at_mean - PHI) / PHI * 100
print(f"\n  Alpha/Theta ratio = {at_mean:.4f}")
print(f"  Phi               = {PHI:.4f}")
print(f"  Error             = {at_err_pct:.2f}%")
print(f"  Group POI         = {df['POI'].mean():.4f}")
print(f"\n  Other ratios: theta/delta={df['ratio_theta/delta'].mean():.4f} ({df['error_theta/delta'].mean()*100:.1f}% err)")
print(f"               beta/alpha={df['ratio_beta/alpha'].mean():.4f} ({df['error_beta/alpha'].mean()*100:.1f}% err)")
print(f"               gamma/beta={df['ratio_gamma/beta'].mean():.4f} ({df['error_gamma/beta'].mean()*100:.1f}% err)")
