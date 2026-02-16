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
N_SURROGATES = 100

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

def phase_randomize_shared(data):
    n_ch, n_samp = data.shape
    surrogate = np.zeros_like(data)
    random_phases = np.exp(2j * np.pi * np.random.random(n_samp // 2 + 1))
    random_phases[0] = 1
    if n_samp % 2 == 0:
        random_phases[-1] = np.sign(random_phases[-1].real)
    for ch in range(n_ch):
        fft_vals = np.fft.rfft(data[ch])
        amplitudes = np.abs(fft_vals)
        surrogate[ch] = np.fft.irfft(amplitudes * random_phases, n=n_samp)
    return surrogate

def compute_ratio_pci(data, fs):
    nperseg = min(int(4 * fs), data.shape[1])
    avg_psd = np.zeros(nperseg // 2 + 1)
    for ch in range(data.shape[0]):
        freqs, psd = welch(data[ch], fs=fs, nperseg=nperseg)
        avg_psd += psd
    avg_psd /= data.shape[0]

    theta_c = spectral_centroid(freqs, avg_psd, 4, 8)
    alpha_c = spectral_centroid(freqs, avg_psd, 8, 13)
    ratio = alpha_c / theta_c
    pci = compute_pci(ratio)
    return ratio, pci

mat_files = sorted(glob.glob('alpha_s[0-9][0-9].mat')) + sorted(glob.glob('alpha_subj_[0-9][0-9].mat'))
mat_files = mat_files[:35]
n_subjects = len(mat_files)
print(f"Loading {n_subjects} subjects...")

all_data = []
observed_ratios = np.zeros(n_subjects)
observed_pcis = np.zeros(n_subjects)

for i, fpath in enumerate(mat_files):
    mat = loadmat(fpath)
    data = mat['SIGNAL'].astype(np.float64).T
    data = preprocess(data, FS)
    max_samples = min(data.shape[1], 60 * FS)
    data = data[:, :max_samples]
    all_data.append(data)
    ratio, pci = compute_ratio_pci(data, FS)
    observed_ratios[i] = ratio
    observed_pcis[i] = pci

observed_phi_error = np.abs(observed_ratios - PHI)
observed_mean_phi_error = np.mean(observed_phi_error)
observed_pci_rate = np.mean(observed_pcis > 0) * 100

print(f"Observed: mean ratio={np.mean(observed_ratios):.4f}, mean phi-error={observed_mean_phi_error:.4f}, PCI>0={observed_pci_rate:.1f}%")
print(f"\nRunning {N_SURROGATES} phase-randomized surrogates...\n")

surrogate_mean_errors = np.zeros(N_SURROGATES)
surrogate_pci_rates = np.zeros(N_SURROGATES)
surrogate_mean_ratios = np.zeros(N_SURROGATES)
all_surr_ratios_flat = []
all_surr_pcis_flat = []

for s_idx in range(N_SURROGATES):
    if (s_idx + 1) % 25 == 0:
        print(f"  Surrogate {s_idx+1}/{N_SURROGATES}")

    surr_ratios = np.zeros(n_subjects)
    surr_pcis = np.zeros(n_subjects)
    for subj in range(n_subjects):
        surr_data = phase_randomize_shared(all_data[subj])
        ratio, pci = compute_ratio_pci(surr_data, FS)
        surr_ratios[subj] = ratio
        surr_pcis[subj] = pci
        all_surr_ratios_flat.append(ratio)
        all_surr_pcis_flat.append(pci)

    surrogate_mean_errors[s_idx] = np.mean(np.abs(surr_ratios - PHI))
    surrogate_pci_rates[s_idx] = np.mean(surr_pcis > 0) * 100
    surrogate_mean_ratios[s_idx] = np.mean(surr_ratios)

p_phi_error = np.mean(surrogate_mean_errors <= observed_mean_phi_error)
z_phi_error = (observed_mean_phi_error - np.mean(surrogate_mean_errors)) / np.std(surrogate_mean_errors)

p_pci_rate = np.mean(surrogate_pci_rates >= observed_pci_rate)

print("\n" + "="*70)
print("PHASE-RANDOMIZED SURROGATE NULL MODEL RESULTS")
print(f"N = {n_subjects} subjects, {N_SURROGATES} surrogates")
print("="*70)

print(f"\n--- Phi-Proximity Test ---")
print(f"  Observed mean |ratio - phi|:  {observed_mean_phi_error:.4f}")
print(f"  Surrogate mean |ratio - phi|: {np.mean(surrogate_mean_errors):.4f} (SD={np.std(surrogate_mean_errors):.4f})")
print(f"  Z-score: {z_phi_error:.4f}")
print(f"  p-value (observed <= surrogates): {p_phi_error:.4f}")
if p_phi_error < 0.05:
    print(f"  ** SIGNIFICANT: Real EEG ratios are closer to phi than surrogates (p={p_phi_error:.4f})")
else:
    print(f"  Not significant at p<0.05 (observed ratios not closer to phi than surrogates)")

print(f"\n--- PCI Rate Test ---")
print(f"  Observed PCI>0 rate: {observed_pci_rate:.1f}%")
print(f"  Surrogate PCI>0 rate: {np.mean(surrogate_pci_rates):.1f}% (SD={np.std(surrogate_pci_rates):.1f}%)")
print(f"  p-value (observed >= surrogates): {p_pci_rate:.4f}")

print(f"\n--- Ratio Distribution ---")
print(f"  Observed mean ratio:  {np.mean(observed_ratios):.4f} (SD={np.std(observed_ratios):.4f})")
print(f"  Surrogate mean ratio: {np.mean(surrogate_mean_ratios):.4f} (SD={np.std(surrogate_mean_ratios):.4f})")

os.makedirs('outputs', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

ax = axes[0, 0]
ax.hist(surrogate_mean_errors, bins=25, color='#3498DB', alpha=0.7, edgecolor='black', label='Surrogates')
ax.axvline(observed_mean_phi_error, color='red', linewidth=2.5, linestyle='-', label=f'Observed = {observed_mean_phi_error:.4f}')
ax.set_xlabel('Mean |Ratio - Phi|', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'Phi-Proximity: Observed vs Surrogates\nZ = {z_phi_error:.2f}, p = {p_phi_error:.4f}', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)

ax = axes[0, 1]
ax.hist(surrogate_pci_rates, bins=25, color='#2ECC71', alpha=0.7, edgecolor='black', label='Surrogates')
ax.axvline(observed_pci_rate, color='red', linewidth=2.5, linestyle='-', label=f'Observed = {observed_pci_rate:.1f}%')
ax.set_xlabel('% Subjects with PCI > 0', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'PCI>0 Rate: Observed vs Surrogates\np = {p_pci_rate:.4f}', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)

ax = axes[1, 0]
ax.hist(surrogate_mean_ratios, bins=25, color='#F39C12', alpha=0.7, edgecolor='black', label='Surrogate means')
ax.axvline(np.mean(observed_ratios), color='red', linewidth=2.5, linestyle='-', label=f'Observed = {np.mean(observed_ratios):.4f}')
ax.axvline(PHI, color='gold', linewidth=2, linestyle='--', label=f'phi = {PHI:.4f}')
ax.set_xlabel('Mean Alpha/Theta Ratio', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Mean Ratio Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)

ax = axes[1, 1]
ax.scatter(all_surr_ratios_flat[:500], all_surr_pcis_flat[:500], c='#3498DB', s=10, alpha=0.15, label='Surrogates')
ax.scatter(observed_ratios, observed_pcis, c='red', s=80, alpha=0.8, edgecolors='black', label='Observed', zorder=5)
ax.axhline(0, color='gray', linestyle=':', linewidth=1)
ax.axvline(PHI, color='gold', linewidth=2, linestyle='--', alpha=0.5)
ax.set_xlabel('Alpha/Theta Ratio', fontsize=12)
ax.set_ylabel('PCI', fontsize=12)
ax.set_title('Observed vs Surrogate: Ratio-PCI Space', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)

plt.suptitle(f'Phase-Randomized Surrogate Null Model — N={n_subjects}, {N_SURROGATES} surrogates',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/surrogate_null_model.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure saved: outputs/surrogate_null_model.png")

import json
results = {
    'n_subjects': n_subjects,
    'n_surrogates': N_SURROGATES,
    'observed_mean_phi_error': float(observed_mean_phi_error),
    'surrogate_mean_phi_error': float(np.mean(surrogate_mean_errors)),
    'z_score': float(z_phi_error),
    'p_phi_proximity': float(p_phi_error),
    'observed_pci_rate': float(observed_pci_rate),
    'surrogate_pci_rate': float(np.mean(surrogate_pci_rates)),
    'p_pci_rate': float(p_pci_rate),
    'observed_mean_ratio': float(np.mean(observed_ratios)),
    'surrogate_mean_ratio': float(np.mean(surrogate_mean_ratios)),
}
with open('outputs/surrogate_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Results saved: outputs/surrogate_results.json")
