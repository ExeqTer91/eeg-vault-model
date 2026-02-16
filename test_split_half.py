import numpy as np
import pandas as pd
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, welch

PHI = 1.6180339887
FS = 512
BP_LOW = 1.0
BP_HIGH = 45.0
NOTCH = 50
N_WINDOWS = 10

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

def compute_pci(ratio):
    d_phi = abs(ratio - PHI)
    d_harm = abs(ratio - 2.0)
    if d_phi < 1e-10:
        return 10.0
    return np.log(d_harm / d_phi)

def compute_window_ratios(data, fs, n_windows):
    n_samples = data.shape[1]
    win_size = n_samples // n_windows
    ratios = []
    pcis = []

    for w in range(n_windows):
        start = w * win_size
        end = start + win_size
        segment = data[:, start:end]

        nperseg = min(int(4 * fs), segment.shape[1])
        if nperseg < int(fs):
            continue

        theta_centroids = []
        alpha_centroids = []
        for ch in range(segment.shape[0]):
            freqs, psd = welch(segment[ch], fs=fs, nperseg=nperseg)
            theta_centroids.append(spectral_centroid(freqs, psd, 4, 8))
            alpha_centroids.append(spectral_centroid(freqs, psd, 8, 13))

        theta_c = np.mean(theta_centroids)
        alpha_c = np.mean(alpha_centroids)
        ratio = alpha_c / theta_c
        pci = compute_pci(ratio)
        ratios.append(ratio)
        pcis.append(pci)

    return np.array(ratios), np.array(pcis)

def icc_oneway(data_matrix):
    n_subjects = data_matrix.shape[0]
    k = data_matrix.shape[1]
    grand_mean = np.mean(data_matrix)
    subject_means = np.mean(data_matrix, axis=1)

    ss_between = k * np.sum((subject_means - grand_mean)**2)
    ss_within = np.sum((data_matrix - subject_means[:, None])**2)

    ms_between = ss_between / (n_subjects - 1)
    ms_within = ss_within / (n_subjects * (k - 1))

    icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
    return icc

mat_files = sorted(glob.glob('alpha_s[0-9][0-9].mat')) + sorted(glob.glob('alpha_subj_[0-9][0-9].mat'))
mat_files = mat_files[:35]
n_subjects = len(mat_files)
print(f"Processing {n_subjects} subjects with {N_WINDOWS} windows each\n")

all_ratios = []
all_pcis = []
subj_names = []
min_windows = N_WINDOWS

for i, fpath in enumerate(mat_files):
    subj = os.path.basename(fpath).replace('.mat', '')
    subj_names.append(subj.replace('alpha_s', 'S'))
    mat = loadmat(fpath)
    data = mat['SIGNAL'].astype(np.float64).T
    data = preprocess(data, FS)

    ratios, pcis = compute_window_ratios(data, FS, N_WINDOWS)
    all_ratios.append(ratios)
    all_pcis.append(pcis)
    min_windows = min(min_windows, len(ratios))

    print(f"[{i+1}/{n_subjects}] {subj}: mean_ratio={np.mean(ratios):.4f} (SD={np.std(ratios):.4f}), "
          f"PCI_sign_stable={np.mean((pcis > 0) == (pcis[0] > 0))*100:.0f}%")

k = min_windows
ratio_matrix = np.zeros((n_subjects, k))
pci_matrix = np.zeros((n_subjects, k))
for s in range(n_subjects):
    ratio_matrix[s] = all_ratios[s][:k]
    pci_matrix[s] = all_pcis[s][:k]

icc_ratio = icc_oneway(ratio_matrix)
icc_pci = icc_oneway(pci_matrix)

within_sd = np.mean([np.std(ratio_matrix[s]) for s in range(n_subjects)])
between_sd = np.std(np.mean(ratio_matrix, axis=1))
within_pci_sd = np.mean([np.std(pci_matrix[s]) for s in range(n_subjects)])
between_pci_sd = np.std(np.mean(pci_matrix, axis=1))

pci_sign_stability = np.zeros(n_subjects)
for s in range(n_subjects):
    majority_sign = np.sign(np.median(pci_matrix[s]))
    if majority_sign == 0:
        majority_sign = 1
    pci_sign_stability[s] = np.mean(np.sign(pci_matrix[s]) == majority_sign)

first_half = ratio_matrix[:, :k//2]
second_half = ratio_matrix[:, k//2:k]
first_means = np.mean(first_half, axis=1)
second_means = np.mean(second_half, axis=1)
from scipy.stats import pearsonr, spearmanr
r_split, p_split = pearsonr(first_means, second_means)
rho_split, p_rho = spearmanr(first_means, second_means)

print("\n" + "="*70)
print("WITHIN-SUBJECT SPLIT-HALF RELIABILITY RESULTS")
print(f"N = {n_subjects} subjects, K = {k} windows per subject")
print("="*70)

print(f"\n--- ICC (Intraclass Correlation) ---")
print(f"  ICC (alpha/theta ratio): {icc_ratio:.4f}")
print(f"  ICC (PCI):               {icc_pci:.4f}")

print(f"\n--- Variance Decomposition (alpha/theta ratio) ---")
print(f"  Within-subject SD:  {within_sd:.4f}")
print(f"  Between-subject SD: {between_sd:.4f}")
print(f"  Ratio (between/within): {between_sd/within_sd:.2f}")

print(f"\n--- Variance Decomposition (PCI) ---")
print(f"  Within-subject SD:  {within_pci_sd:.4f}")
print(f"  Between-subject SD: {between_pci_sd:.4f}")

print(f"\n--- PCI Sign Stability ---")
print(f"  Mean sign stability: {np.mean(pci_sign_stability):.4f}")
print(f"  Median sign stability: {np.median(pci_sign_stability):.4f}")
print(f"  Min: {np.min(pci_sign_stability):.4f}")
print(f"  Subjects with >= 80% sign stability: {np.sum(pci_sign_stability >= 0.8)}/{n_subjects}")

print(f"\n--- Split-Half Correlation ---")
print(f"  Pearson r: {r_split:.4f} (p = {p_split:.4e})")
print(f"  Spearman rho: {rho_split:.4f} (p = {p_rho:.4e})")

print(f"\n--- Per-Subject Details ---")
for s in range(n_subjects):
    print(f"  {subj_names[s]:>4s}: mean_ratio={np.mean(ratio_matrix[s]):.4f}, "
          f"SD={np.std(ratio_matrix[s]):.4f}, "
          f"PCI_sign_stab={pci_sign_stability[s]:.2f}")

os.makedirs('outputs', exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

ax = axes[0, 0]
bp_data = [ratio_matrix[s] for s in range(n_subjects)]
bp_plot = ax.boxplot(bp_data, positions=range(n_subjects), widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor='#3498DB', alpha=0.5))
ax.axhline(PHI, color='gold', linewidth=2, linestyle='--', label=f'phi = {PHI:.3f}')
ax.set_xticks(range(n_subjects))
ax.set_xticklabels(subj_names, rotation=45, fontsize=8)
ax.set_ylabel('Alpha/Theta Ratio', fontsize=11)
ax.set_title(f'Within-Subject Ratio Variability\n(ICC = {icc_ratio:.3f})', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

ax = axes[0, 1]
ax.scatter(first_means, second_means, c='steelblue', s=70, alpha=0.7, edgecolors='black')
lim = [min(first_means.min(), second_means.min()) - 0.05, max(first_means.max(), second_means.max()) + 0.05]
ax.plot(lim, lim, 'k--', alpha=0.3)
ax.set_xlabel('First Half Mean Ratio', fontsize=11)
ax.set_ylabel('Second Half Mean Ratio', fontsize=11)
ax.set_title(f'Split-Half Reliability\nr = {r_split:.3f}, p = {p_split:.4f}', fontsize=12, fontweight='bold')
ax.set_xlim(lim)
ax.set_ylim(lim)

ax = axes[0, 2]
colors_sign = ['#2ECC71' if p >= 0.8 else '#E74C3C' for p in pci_sign_stability]
ax.bar(range(n_subjects), pci_sign_stability, color=colors_sign, alpha=0.8, edgecolor='black')
ax.axhline(0.8, color='gray', linestyle=':', linewidth=1, label='80% threshold')
ax.set_xticks(range(n_subjects))
ax.set_xticklabels(subj_names, rotation=45, fontsize=8)
ax.set_ylabel('PCI Sign Stability', fontsize=11)
ax.set_title('PCI Sign Stability per Subject', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

ax = axes[1, 0]
within_sds = [np.std(ratio_matrix[s]) for s in range(n_subjects)]
ax.bar(range(n_subjects), within_sds, color='#E74C3C', alpha=0.7, edgecolor='black')
ax.axhline(between_sd, color='blue', linewidth=2, linestyle='--', label=f'Between-subj SD = {between_sd:.4f}')
ax.set_xticks(range(n_subjects))
ax.set_xticklabels(subj_names, rotation=45, fontsize=8)
ax.set_ylabel('Within-Subject SD', fontsize=11)
ax.set_title('Within vs Between Subject Variability', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

ax = axes[1, 1]
for s in range(n_subjects):
    ax.plot(range(k), ratio_matrix[s], '-o', alpha=0.5, markersize=3)
ax.axhline(PHI, color='gold', linewidth=2, linestyle='--', label='phi')
ax.set_xlabel('Window', fontsize=11)
ax.set_ylabel('Alpha/Theta Ratio', fontsize=11)
ax.set_title('Per-Window Ratio Trajectories', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

ax = axes[1, 2]
summary_data = {
    'ICC (ratio)': icc_ratio,
    'ICC (PCI)': icc_pci,
    'Split-half r': r_split,
    'B/W SD ratio': between_sd / within_sd,
    'PCI sign stab': np.mean(pci_sign_stability),
}
bars = ax.barh(range(len(summary_data)), list(summary_data.values()),
               color=['#3498DB', '#2ECC71', '#E74C3C', '#F39C12', '#9B59B6'], alpha=0.8, edgecolor='black')
ax.set_yticks(range(len(summary_data)))
ax.set_yticklabels(list(summary_data.keys()), fontsize=10)
ax.set_xlabel('Value', fontsize=11)
ax.set_title('Reliability Summary', fontsize=12, fontweight='bold')
for i, v in enumerate(summary_data.values()):
    ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')

plt.suptitle(f'Within-Subject Reliability Analysis — N={n_subjects}, K={k} windows',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/split_half_reliability.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure saved: outputs/split_half_reliability.png")

import json
results = {
    'n_subjects': n_subjects,
    'n_windows': k,
    'icc_ratio': float(icc_ratio),
    'icc_pci': float(icc_pci),
    'within_subject_sd': float(within_sd),
    'between_subject_sd': float(between_sd),
    'split_half_r': float(r_split),
    'split_half_p': float(p_split),
    'mean_pci_sign_stability': float(np.mean(pci_sign_stability)),
}
with open('outputs/split_half_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Results saved: outputs/split_half_results.json")
