import numpy as np
import pandas as pd
import os
import glob
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
N_PERTURBATIONS = 500

STANDARD_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),
}

np.random.seed(42)

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

def generate_perturbed_bands(jitter_sd=0.5):
    j = np.random.normal(0, jitter_sd, 4)
    theta_lo = 4 + j[0]
    alpha_lo = 8 + j[1]
    beta_lo = 13 + j[2]
    gamma_lo = 30 + j[3]

    theta_lo = np.clip(theta_lo, 3, 5)
    alpha_lo = np.clip(alpha_lo, 7, 9)
    beta_lo = np.clip(beta_lo, 12, 15)
    gamma_lo = np.clip(gamma_lo, 28, 32)

    return {
        'delta': (0.5, theta_lo),
        'theta': (theta_lo, alpha_lo),
        'alpha': (alpha_lo, beta_lo),
        'beta': (beta_lo, gamma_lo),
        'gamma': (gamma_lo, 45),
    }

mat_files = sorted(glob.glob('alpha_s[0-9][0-9].mat')) + sorted(glob.glob('alpha_subj_[0-9][0-9].mat'))
mat_files = mat_files[:35]
n_subjects = len(mat_files)
print(f"Loading {n_subjects} subjects...")

all_data = []
all_psd = []
for fpath in mat_files:
    mat = loadmat(fpath)
    data = mat['SIGNAL'].astype(np.float64).T
    data = preprocess(data, FS)
    all_data.append(data)
    nperseg = min(int(4 * FS), data.shape[-1])
    psds = []
    for ch in range(data.shape[0]):
        f, p = welch(data[ch], fs=FS, nperseg=nperseg)
        psds.append(p)
    all_psd.append((f, np.array(psds)))

print(f"All subjects loaded. Running {N_PERTURBATIONS} boundary perturbations...\n")

standard_pci = np.zeros(n_subjects)
standard_ratio = np.zeros(n_subjects)
for s in range(n_subjects):
    freqs, psds = all_psd[s]
    theta_c = np.mean([spectral_centroid(freqs, psds[ch], 4, 8) for ch in range(psds.shape[0])])
    alpha_c = np.mean([spectral_centroid(freqs, psds[ch], 8, 13) for ch in range(psds.shape[0])])
    ratio = alpha_c / theta_c
    standard_ratio[s] = ratio
    standard_pci[s] = compute_pci(ratio)

standard_class = (standard_pci > 0).astype(int)

perturbed_pci = np.zeros((N_PERTURBATIONS, n_subjects))
perturbed_ratio = np.zeros((N_PERTURBATIONS, n_subjects))
perturbed_class = np.zeros((N_PERTURBATIONS, n_subjects))

for p in range(N_PERTURBATIONS):
    if (p + 1) % 100 == 0:
        print(f"  Perturbation {p+1}/{N_PERTURBATIONS}")

    bands = generate_perturbed_bands(jitter_sd=0.5)
    for s in range(n_subjects):
        freqs, psds = all_psd[s]
        theta_c = np.mean([spectral_centroid(freqs, psds[ch], bands['theta'][0], bands['theta'][1]) for ch in range(psds.shape[0])])
        alpha_c = np.mean([spectral_centroid(freqs, psds[ch], bands['alpha'][0], bands['alpha'][1]) for ch in range(psds.shape[0])])
        ratio = alpha_c / theta_c
        pci = compute_pci(ratio)
        perturbed_ratio[p, s] = ratio
        perturbed_pci[p, s] = pci
        perturbed_class[p, s] = 1 if pci > 0 else 0

classification_stability = np.zeros(n_subjects)
for s in range(n_subjects):
    agreement = np.mean(perturbed_class[:, s] == standard_class[s])
    classification_stability[s] = agreement

ratio_cv = np.zeros(n_subjects)
for s in range(n_subjects):
    ratio_cv[s] = np.std(perturbed_ratio[:, s]) / np.mean(perturbed_ratio[:, s])

pci_rate_per_perm = np.mean(perturbed_class, axis=1)

print("\n" + "="*70)
print("BOUNDARY-JITTER ROBUSTNESS RESULTS")
print(f"N = {n_subjects} subjects, {N_PERTURBATIONS} perturbations (jitter SD = 0.5 Hz)")
print("="*70)

print(f"\n--- Classification Stability ---")
print(f"  Median stability: {np.median(classification_stability):.4f}")
print(f"  Mean stability:   {np.mean(classification_stability):.4f}")
print(f"  Min stability:    {np.min(classification_stability):.4f}")
print(f"  Max stability:    {np.max(classification_stability):.4f}")
print(f"  Subjects with stability >= 0.80: {np.sum(classification_stability >= 0.80)}/{n_subjects}")
print(f"  Subjects with stability >= 0.90: {np.sum(classification_stability >= 0.90)}/{n_subjects}")

print(f"\n--- Ratio Coefficient of Variation ---")
print(f"  Median CV: {np.median(ratio_cv)*100:.2f}%")
print(f"  Mean CV:   {np.mean(ratio_cv)*100:.2f}%")
print(f"  Max CV:    {np.max(ratio_cv)*100:.2f}%")

print(f"\n--- PCI Rate Across Perturbations ---")
print(f"  Standard band PCI>0 rate: {np.mean(standard_class)*100:.1f}%")
print(f"  Mean perturbed PCI>0 rate: {np.mean(pci_rate_per_perm)*100:.1f}% (SD={np.std(pci_rate_per_perm)*100:.1f}%)")
print(f"  Min perturbed PCI>0 rate:  {np.min(pci_rate_per_perm)*100:.1f}%")
print(f"  Max perturbed PCI>0 rate:  {np.max(pci_rate_per_perm)*100:.1f}%")

print(f"\n--- Perturbed Ratio Stats ---")
mean_ratios = np.mean(perturbed_ratio, axis=0)
print(f"  Grand mean alpha/theta: {np.mean(mean_ratios):.4f} (SD={np.std(mean_ratios):.4f})")
print(f"  Standard band mean:     {np.mean(standard_ratio):.4f}")

print(f"\n--- Per-Subject Table ---")
subj_names = [os.path.basename(f).replace('.mat', '').replace('alpha_s', 'S') for f in mat_files]
for s in range(n_subjects):
    print(f"  {subj_names[s]:>4s}: ratio={standard_ratio[s]:.4f}, PCI={standard_pci[s]:.2f}, "
          f"stability={classification_stability[s]:.3f}, ratio_CV={ratio_cv[s]*100:.2f}%")

os.makedirs('outputs', exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

ax = axes[0, 0]
ax.hist(classification_stability, bins=15, color='#2ECC71', alpha=0.7, edgecolor='black')
ax.axvline(np.median(classification_stability), color='red', linewidth=2, linestyle='--',
           label=f'median = {np.median(classification_stability):.3f}')
ax.set_xlabel('Classification Stability', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('PCI Classification Stability\n(agreement with standard bands)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)

ax = axes[0, 1]
ax.hist(pci_rate_per_perm * 100, bins=25, color='#3498DB', alpha=0.7, edgecolor='black')
ax.axvline(np.mean(standard_class) * 100, color='red', linewidth=2, linestyle='--',
           label=f'standard = {np.mean(standard_class)*100:.1f}%')
ax.set_xlabel('% Phi-Organized (PCI > 0)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'PCI>0 Rate Across {N_PERTURBATIONS} Perturbations', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)

ax = axes[0, 2]
all_perturbed_ratios = perturbed_ratio.flatten()
ax.hist(all_perturbed_ratios, bins=50, color='#E74C3C', alpha=0.5, density=True, label='Perturbed')
ax.hist(standard_ratio, bins=12, color='#2ECC71', alpha=0.5, density=True, label='Standard')
ax.axvline(PHI, color='gold', linewidth=2.5, linestyle='--', label=f'phi = {PHI:.4f}')
ax.set_xlabel('Alpha/Theta Ratio', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Ratio Distribution:\nStandard vs Perturbed', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

ax = axes[1, 0]
colors = ['#2ECC71' if cs >= 0.8 else '#E74C3C' for cs in classification_stability]
ax.bar(range(n_subjects), classification_stability, color=colors, alpha=0.8, edgecolor='black')
ax.axhline(0.8, color='gray', linewidth=1, linestyle=':', label='80% threshold')
ax.set_xticks(range(n_subjects))
ax.set_xticklabels(subj_names, rotation=45, fontsize=8)
ax.set_ylabel('Stability', fontsize=12)
ax.set_title('Per-Subject Classification Stability', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

ax = axes[1, 1]
ax.boxplot([perturbed_ratio[:, s] for s in range(n_subjects)], positions=range(n_subjects), widths=0.6,
           patch_artist=True, boxprops=dict(facecolor='#3498DB', alpha=0.5))
ax.scatter(range(n_subjects), standard_ratio, c='red', s=40, zorder=5, label='Standard')
ax.axhline(PHI, color='gold', linewidth=2, linestyle='--', label=f'phi')
ax.set_xticks(range(n_subjects))
ax.set_xticklabels(subj_names, rotation=45, fontsize=8)
ax.set_ylabel('Alpha/Theta Ratio', fontsize=12)
ax.set_title('Per-Subject Ratio Distributions\n(across boundary perturbations)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')

ax = axes[1, 2]
ax.scatter(standard_ratio, classification_stability, c='steelblue', s=70, alpha=0.7, edgecolors='black')
ax.axhline(0.8, color='gray', linewidth=1, linestyle=':')
ax.axvline(PHI, color='gold', linewidth=2, linestyle='--', alpha=0.5)
ax.set_xlabel('Standard Alpha/Theta Ratio', fontsize=12)
ax.set_ylabel('Classification Stability', fontsize=12)
ax.set_title('Ratio vs Stability', fontsize=12, fontweight='bold')

plt.suptitle(f'Boundary-Jitter Robustness Analysis\nN={n_subjects}, {N_PERTURBATIONS} perturbations, jitter SD=0.5 Hz',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/boundary_jitter_robustness.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure saved: outputs/boundary_jitter_robustness.png")

results = {
    'n_subjects': n_subjects,
    'n_perturbations': N_PERTURBATIONS,
    'jitter_sd_hz': 0.5,
    'median_classification_stability': float(np.median(classification_stability)),
    'mean_classification_stability': float(np.mean(classification_stability)),
    'subjects_stability_ge_80': int(np.sum(classification_stability >= 0.80)),
    'subjects_stability_ge_90': int(np.sum(classification_stability >= 0.90)),
    'mean_ratio_cv_pct': float(np.mean(ratio_cv) * 100),
    'standard_pci_rate': float(np.mean(standard_class) * 100),
    'mean_perturbed_pci_rate': float(np.mean(pci_rate_per_perm) * 100),
    'grand_mean_perturbed_ratio': float(np.mean(mean_ratios)),
}
import json
with open('outputs/boundary_jitter_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Results saved: outputs/boundary_jitter_results.json")
