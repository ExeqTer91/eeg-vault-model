import numpy as np
import pandas as pd
import os
import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.stats import pearsonr, spearmanr, ttest_1samp, ttest_rel, wilcoxon, norm
from scipy.optimize import minimize

mne.set_log_level('ERROR')

PHI = 1.6180339887
E_MINUS_1 = np.e - 1
HARMONIC = 2.0
FS_ALPHA = 512
BP_LOW = 1.0
BP_HIGH = 45.0
NOTCH_FREQ = 50
TARGET_N = 35

os.makedirs('outputs', exist_ok=True)

def preprocess_mat(data, fs):
    data = data - np.mean(data, axis=0, keepdims=True)
    nyq = fs / 2.0
    b, a = butter(4, [BP_LOW / nyq, BP_HIGH / nyq], btype='band')
    for ch in range(data.shape[0]):
        data[ch] = filtfilt(b, a, data[ch])
    b_n, a_n = iirnotch(NOTCH_FREQ, Q=30, fs=fs)
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

def compute_ratio_from_psd(freqs, avg_psd):
    theta_c = spectral_centroid(freqs, avg_psd, 4, 8)
    alpha_c = spectral_centroid(freqs, avg_psd, 8, 13)
    ratio = alpha_c / theta_c
    pci = compute_pci(ratio)
    return ratio, pci, theta_c, alpha_c

def load_all_subjects():
    subjects = []
    mat_files = sorted(glob.glob('alpha_s[0-9][0-9].mat'))
    for fpath in mat_files:
        subj = os.path.basename(fpath).replace('.mat', '')
        mat = loadmat(fpath)
        data = mat['SIGNAL'].astype(np.float64).T
        data = preprocess_mat(data, FS_ALPHA)
        subjects.append({'name': subj, 'data': data, 'fs': FS_ALPHA, 'source': 'alpha_waves'})

    print(f"Loaded {len(subjects)} subjects from alpha-waves dataset")

    n_needed = TARGET_N - len(subjects)
    if n_needed > 0:
        print(f"Loading {n_needed} additional subjects from PhysioNet EEGBCI...")
        mne.set_config('MNE_DATASETS_EEGBCI_PATH', '/home/runner/mne_data')
        loaded = 0
        for subj_id in range(1, 110):
            if loaded >= n_needed:
                break
            try:
                fnames = mne.datasets.eegbci.load_data(subj_id, [1], update_path=True)
                raw = mne.io.read_raw_edf(fnames[0], preload=True)
                mne.datasets.eegbci.standardize(raw)
                raw.filter(BP_LOW, BP_HIGH, fir_design='firwin')
                eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
                data = raw.get_data(picks=eeg_picks)
                fs = raw.info['sfreq']
                subjects.append({'name': f'EEGBCI_S{subj_id:03d}', 'data': data, 'fs': fs, 'source': 'eegbci'})
                loaded += 1
            except Exception as e:
                print(f"  Skip EEGBCI S{subj_id}: {str(e)[:60]}")

        print(f"Loaded {loaded} EEGBCI subjects")

    print(f"\nTotal subjects: {len(subjects)}\n")
    return subjects

def compute_subject_psd(subj):
    data = subj['data']
    fs = subj['fs']
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

print("="*70)
print(f"LOADING {TARGET_N} SUBJECTS FOR ALL EXPERIMENTS")
print("="*70)

all_subjects = load_all_subjects()
n = len(all_subjects)

all_freqs = []
all_psds = []
all_ratios = np.zeros(n)
all_pcis = np.zeros(n)
all_theta = np.zeros(n)
all_alpha = np.zeros(n)

for i, subj in enumerate(all_subjects):
    freqs, avg_psd = compute_subject_psd(subj)
    all_freqs.append(freqs)
    all_psds.append(avg_psd)
    ratio, pci, theta_c, alpha_c = compute_ratio_from_psd(freqs, avg_psd)
    all_ratios[i] = ratio
    all_pcis[i] = pci
    all_theta[i] = theta_c
    all_alpha[i] = alpha_c

subj_names = [s['name'] for s in all_subjects]

print("\n" + "="*70)
print("TEST 1: BATCH CENTROID ANALYSIS")
print(f"N = {n}")
print("="*70)

at_mean = np.mean(all_ratios)
at_err_pct = abs(at_mean - PHI) / PHI * 100
t_stat, p_val = ttest_1samp(all_ratios, PHI)
pci_positive = np.sum(all_pcis > 0)

print(f"  Mean alpha/theta ratio = {at_mean:.4f}")
print(f"  Phi                    = {PHI:.4f}")
print(f"  Error                  = {at_err_pct:.2f}%")
print(f"  t-test vs phi: t={t_stat:.4f}, p={p_val:.6f}")
print(f"  PCI > 0 (phi-organized): {pci_positive}/{n} ({pci_positive/n*100:.1f}%)")
print(f"  Mean PCI = {np.mean(all_pcis):.4f} (SD={np.std(all_pcis):.4f})")

batch_results = {
    'n_subjects': n,
    'mean_ratio': float(at_mean),
    'phi_error_pct': float(at_err_pct),
    't_stat': float(t_stat),
    'p_value': float(p_val),
    'pci_positive': int(pci_positive),
    'pci_positive_pct': float(pci_positive/n*100),
    'mean_pci': float(np.mean(all_pcis)),
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ax = axes[0]
ax.hist(all_ratios, bins=15, color='#E74C3C', alpha=0.7, edgecolor='black')
ax.axvline(PHI, color='gold', linewidth=2.5, linestyle='--', label=f'phi = {PHI:.4f}')
ax.axvline(at_mean, color='blue', linewidth=2, linestyle='-', label=f'mean = {at_mean:.4f}')
ax.set_xlabel('Alpha/Theta Centroid Ratio')
ax.set_ylabel('Count')
ax.set_title(f'Alpha/Theta Ratio Distribution (N={n})', fontweight='bold')
ax.legend()

ax = axes[1]
pci_colors = ['#2ECC71' if p > 0 else '#E74C3C' for p in all_pcis]
ax.bar(range(n), all_pcis, color=pci_colors, alpha=0.8, edgecolor='black')
ax.axhline(0, color='black', linewidth=1)
ax.set_ylabel('PCI')
ax.set_title(f'PCI per Subject ({pci_positive}/{n} phi-organized)', fontweight='bold')

ax = axes[2]
ax.scatter(all_theta, all_alpha, c='steelblue', s=60, alpha=0.7, edgecolors='black')
r_ta, p_ta = pearsonr(all_theta, all_alpha)
ax.set_xlabel('Theta Centroid (Hz)')
ax.set_ylabel('Alpha Centroid (Hz)')
ax.set_title(f'Theta vs Alpha Centroids (r={r_ta:.3f})', fontweight='bold')

plt.suptitle(f'Batch Centroid Analysis — N={n}, Mean Ratio={at_mean:.4f}, Phi Error={at_err_pct:.1f}%',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/batch_analysis_summary.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*70)
print("TEST 2: BOUNDARY-JITTER ROBUSTNESS")
print(f"N = {n}, 500 perturbations")
print("="*70)

N_PERTURBATIONS = 500
np.random.seed(42)

standard_pci = all_pcis.copy()
standard_ratio = all_ratios.copy()
standard_class = (standard_pci > 0).astype(int)

def generate_perturbed_bands(jitter_sd=0.5):
    j = np.random.normal(0, jitter_sd, 4)
    theta_lo = np.clip(4 + j[0], 3, 5)
    alpha_lo = np.clip(8 + j[1], 7, 9)
    beta_lo = np.clip(13 + j[2], 12, 15)
    gamma_lo = np.clip(30 + j[3], 28, 32)
    return {
        'theta': (theta_lo, alpha_lo),
        'alpha': (alpha_lo, beta_lo),
    }

all_ch_psds = []
for i, subj in enumerate(all_subjects):
    data = subj['data']
    fs = subj['fs']
    nperseg = min(int(4 * fs), data.shape[1])
    ch_psds = []
    for ch in range(data.shape[0]):
        freqs, psd = welch(data[ch], fs=fs, nperseg=nperseg)
        ch_psds.append(psd)
    all_ch_psds.append((freqs, np.array(ch_psds)))

perturbed_pci = np.zeros((N_PERTURBATIONS, n))
perturbed_ratio = np.zeros((N_PERTURBATIONS, n))
perturbed_class = np.zeros((N_PERTURBATIONS, n))

for p in range(N_PERTURBATIONS):
    if (p + 1) % 100 == 0:
        print(f"  Perturbation {p+1}/{N_PERTURBATIONS}")
    bands = generate_perturbed_bands(jitter_sd=0.5)
    for s in range(n):
        freqs, psds = all_ch_psds[s]
        theta_c = np.mean([spectral_centroid(freqs, psds[ch], bands['theta'][0], bands['theta'][1]) for ch in range(psds.shape[0])])
        alpha_c = np.mean([spectral_centroid(freqs, psds[ch], bands['alpha'][0], bands['alpha'][1]) for ch in range(psds.shape[0])])
        ratio = alpha_c / theta_c
        pci = compute_pci(ratio)
        perturbed_ratio[p, s] = ratio
        perturbed_pci[p, s] = pci
        perturbed_class[p, s] = 1 if pci > 0 else 0

classification_stability = np.array([np.mean(perturbed_class[:, s] == standard_class[s]) for s in range(n)])
ratio_cv = np.array([np.std(perturbed_ratio[:, s]) / np.mean(perturbed_ratio[:, s]) for s in range(n)])
pci_rate_per_perm = np.mean(perturbed_class, axis=1)

print(f"  Median classification stability: {np.median(classification_stability):.4f}")
print(f"  Mean classification stability:   {np.mean(classification_stability):.4f}")
print(f"  Subjects stability >= 80%: {np.sum(classification_stability >= 0.80)}/{n}")
print(f"  Subjects stability >= 90%: {np.sum(classification_stability >= 0.90)}/{n}")
print(f"  Median ratio CV: {np.median(ratio_cv)*100:.2f}%")
print(f"  Standard PCI>0 rate: {np.mean(standard_class)*100:.1f}%")
print(f"  Mean perturbed PCI>0 rate: {np.mean(pci_rate_per_perm)*100:.1f}% (SD={np.std(pci_rate_per_perm)*100:.1f}%)")

jitter_results = {
    'n_subjects': n,
    'n_perturbations': N_PERTURBATIONS,
    'median_classification_stability': float(np.median(classification_stability)),
    'mean_classification_stability': float(np.mean(classification_stability)),
    'subjects_stability_ge_80': int(np.sum(classification_stability >= 0.80)),
    'subjects_stability_ge_90': int(np.sum(classification_stability >= 0.90)),
    'mean_ratio_cv_pct': float(np.mean(ratio_cv) * 100),
    'standard_pci_rate': float(np.mean(standard_class) * 100),
    'mean_perturbed_pci_rate': float(np.mean(pci_rate_per_perm) * 100),
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ax = axes[0]
ax.hist(classification_stability, bins=15, color='#2ECC71', alpha=0.7, edgecolor='black')
ax.axvline(np.median(classification_stability), color='red', linewidth=2, linestyle='--',
           label=f'median = {np.median(classification_stability):.3f}')
ax.set_xlabel('Classification Stability')
ax.set_ylabel('Count')
ax.set_title('PCI Classification Stability', fontweight='bold')
ax.legend()

ax = axes[1]
ax.hist(pci_rate_per_perm * 100, bins=25, color='#3498DB', alpha=0.7, edgecolor='black')
ax.axvline(np.mean(standard_class) * 100, color='red', linewidth=2, linestyle='--',
           label=f'standard = {np.mean(standard_class)*100:.1f}%')
ax.set_xlabel('% Phi-Organized (PCI > 0)')
ax.set_ylabel('Count')
ax.set_title(f'PCI>0 Rate Across {N_PERTURBATIONS} Perturbations', fontweight='bold')
ax.legend()

ax = axes[2]
all_perturbed_flat = perturbed_ratio.flatten()
ax.hist(all_perturbed_flat, bins=50, color='#E74C3C', alpha=0.5, density=True, label='Perturbed')
ax.hist(standard_ratio, bins=15, color='#2ECC71', alpha=0.5, density=True, label='Standard')
ax.axvline(PHI, color='gold', linewidth=2.5, linestyle='--', label=f'phi')
ax.set_xlabel('Alpha/Theta Ratio')
ax.set_ylabel('Density')
ax.set_title('Ratio: Standard vs Perturbed', fontweight='bold')
ax.legend()

plt.suptitle(f'Boundary-Jitter Robustness — N={n}, {N_PERTURBATIONS} perturbations', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/boundary_jitter_robustness.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*70)
print("TEST 3: SPLIT-HALF RELIABILITY")
print(f"N = {n}")
print("="*70)

N_WINDOWS = 10

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
        theta_cs = []
        alpha_cs = []
        for ch in range(segment.shape[0]):
            freqs, psd = welch(segment[ch], fs=fs, nperseg=nperseg)
            theta_cs.append(spectral_centroid(freqs, psd, 4, 8))
            alpha_cs.append(spectral_centroid(freqs, psd, 8, 13))
        theta_c = np.mean(theta_cs)
        alpha_c = np.mean(alpha_cs)
        ratio = alpha_c / theta_c
        pci = compute_pci(ratio)
        ratios.append(ratio)
        pcis.append(pci)
    return np.array(ratios), np.array(pcis)

all_win_ratios = []
all_win_pcis = []
min_windows = N_WINDOWS

for subj in all_subjects:
    ratios_w, pcis_w = compute_window_ratios(subj['data'], subj['fs'], N_WINDOWS)
    all_win_ratios.append(ratios_w)
    all_win_pcis.append(pcis_w)
    min_windows = min(min_windows, len(ratios_w))

k = min_windows
ratio_matrix = np.zeros((n, k))
pci_matrix = np.zeros((n, k))
for s in range(n):
    ratio_matrix[s] = all_win_ratios[s][:k]
    pci_matrix[s] = all_win_pcis[s][:k]

def icc_oneway(data_matrix):
    n_s = data_matrix.shape[0]
    k_w = data_matrix.shape[1]
    grand_mean = np.mean(data_matrix)
    subject_means = np.mean(data_matrix, axis=1)
    ss_between = k_w * np.sum((subject_means - grand_mean)**2)
    ss_within = np.sum((data_matrix - subject_means[:, None])**2)
    ms_between = ss_between / (n_s - 1)
    ms_within = ss_within / (n_s * (k_w - 1))
    icc = (ms_between - ms_within) / (ms_between + (k_w - 1) * ms_within)
    return icc

icc_ratio = icc_oneway(ratio_matrix)
icc_pci = icc_oneway(pci_matrix)

within_sd = np.mean([np.std(ratio_matrix[s]) for s in range(n)])
between_sd = np.std(np.mean(ratio_matrix, axis=1))

pci_sign_stability = np.zeros(n)
for s in range(n):
    majority_sign = np.sign(np.median(pci_matrix[s]))
    if majority_sign == 0:
        majority_sign = 1
    pci_sign_stability[s] = np.mean(np.sign(pci_matrix[s]) == majority_sign)

first_half = ratio_matrix[:, :k//2]
second_half = ratio_matrix[:, k//2:k]
first_means = np.mean(first_half, axis=1)
second_means = np.mean(second_half, axis=1)
r_split, p_split = pearsonr(first_means, second_means)

print(f"  ICC (alpha/theta ratio): {icc_ratio:.4f}")
print(f"  ICC (PCI):               {icc_pci:.4f}")
print(f"  Within-subject SD:  {within_sd:.4f}")
print(f"  Between-subject SD: {between_sd:.4f}")
print(f"  B/W ratio: {between_sd/within_sd:.2f}")
print(f"  Split-half r: {r_split:.4f} (p={p_split:.4e})")
print(f"  Mean PCI sign stability: {np.mean(pci_sign_stability):.4f}")
print(f"  Subjects with >= 80% sign stability: {np.sum(pci_sign_stability >= 0.8)}/{n}")

split_results = {
    'n_subjects': n,
    'n_windows': k,
    'icc_ratio': float(icc_ratio),
    'icc_pci': float(icc_pci),
    'within_subject_sd': float(within_sd),
    'between_subject_sd': float(between_sd),
    'split_half_r': float(r_split),
    'split_half_p': float(p_split),
    'mean_pci_sign_stability': float(np.mean(pci_sign_stability)),
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ax = axes[0]
bp_plot = ax.boxplot([ratio_matrix[s] for s in range(n)], positions=range(n), widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor='#3498DB', alpha=0.5))
ax.axhline(PHI, color='gold', linewidth=2, linestyle='--', label=f'phi')
ax.set_ylabel('Alpha/Theta Ratio')
ax.set_title(f'Within-Subject Variability (ICC={icc_ratio:.3f})', fontweight='bold')
ax.legend()

ax = axes[1]
ax.scatter(first_means, second_means, c='steelblue', s=50, alpha=0.7, edgecolors='black')
lim = [min(first_means.min(), second_means.min()) - 0.05, max(first_means.max(), second_means.max()) + 0.05]
ax.plot(lim, lim, 'k--', alpha=0.3)
ax.set_xlabel('First Half Mean Ratio')
ax.set_ylabel('Second Half Mean Ratio')
ax.set_title(f'Split-Half Reliability (r={r_split:.3f})', fontweight='bold')

ax = axes[2]
colors_sign = ['#2ECC71' if p >= 0.8 else '#E74C3C' for p in pci_sign_stability]
ax.bar(range(n), pci_sign_stability, color=colors_sign, alpha=0.8, edgecolor='black')
ax.axhline(0.8, color='gray', linestyle=':', linewidth=1)
ax.set_ylabel('PCI Sign Stability')
ax.set_title('PCI Sign Stability per Subject', fontweight='bold')

plt.suptitle(f'Split-Half Reliability — N={n}, K={k} windows', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/split_half_reliability.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*70)
print("TEST 4: SURROGATE NULL MODEL")
print(f"N = {n}, 100 surrogates")
print("="*70)

N_SURROGATES = 100
np.random.seed(42)

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

def compute_ratio_pci_from_data(data, fs):
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

cropped_data = []
for subj in all_subjects:
    data = subj['data']
    fs = subj['fs']
    max_samples = min(data.shape[1], int(60 * fs))
    cropped_data.append((data[:, :max_samples], fs))

observed_phi_error = np.abs(all_ratios - PHI)
observed_mean_phi_error = np.mean(observed_phi_error)
observed_pci_rate = np.mean(all_pcis > 0) * 100

surrogate_mean_errors = np.zeros(N_SURROGATES)
surrogate_pci_rates = np.zeros(N_SURROGATES)
surrogate_mean_ratios = np.zeros(N_SURROGATES)

for s_idx in range(N_SURROGATES):
    if (s_idx + 1) % 25 == 0:
        print(f"  Surrogate {s_idx+1}/{N_SURROGATES}")
    surr_ratios = np.zeros(n)
    surr_pcis = np.zeros(n)
    for subj_i in range(n):
        data_c, fs_c = cropped_data[subj_i]
        surr_data = phase_randomize_shared(data_c)
        ratio, pci = compute_ratio_pci_from_data(surr_data, fs_c)
        surr_ratios[subj_i] = ratio
        surr_pcis[subj_i] = pci
    surrogate_mean_errors[s_idx] = np.mean(np.abs(surr_ratios - PHI))
    surrogate_pci_rates[s_idx] = np.mean(surr_pcis > 0) * 100
    surrogate_mean_ratios[s_idx] = np.mean(surr_ratios)

p_phi_error = np.mean(surrogate_mean_errors <= observed_mean_phi_error)
z_phi_error = (observed_mean_phi_error - np.mean(surrogate_mean_errors)) / np.std(surrogate_mean_errors)
p_pci_rate = np.mean(surrogate_pci_rates >= observed_pci_rate)

print(f"  Observed mean |ratio - phi|:  {observed_mean_phi_error:.4f}")
print(f"  Surrogate mean |ratio - phi|: {np.mean(surrogate_mean_errors):.4f} (SD={np.std(surrogate_mean_errors):.4f})")
print(f"  Z-score: {z_phi_error:.4f}")
print(f"  p-value (phi proximity): {p_phi_error:.4f}")
print(f"  Observed PCI>0 rate: {observed_pci_rate:.1f}%")
print(f"  Surrogate PCI>0 rate: {np.mean(surrogate_pci_rates):.1f}% (SD={np.std(surrogate_pci_rates):.1f}%)")
print(f"  p-value (PCI rate): {p_pci_rate:.4f}")

if p_phi_error < 0.05:
    print(f"  ** SIGNIFICANT: Real EEG closer to phi than surrogates")
else:
    print(f"  Not significant: ratios not closer to phi than surrogates")

surrogate_results = {
    'n_subjects': n,
    'n_surrogates': N_SURROGATES,
    'observed_mean_phi_error': float(observed_mean_phi_error),
    'surrogate_mean_phi_error': float(np.mean(surrogate_mean_errors)),
    'z_score': float(z_phi_error),
    'p_phi_proximity': float(p_phi_error),
    'observed_pci_rate': float(observed_pci_rate),
    'surrogate_pci_rate': float(np.mean(surrogate_pci_rates)),
    'p_pci_rate': float(p_pci_rate),
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ax = axes[0]
ax.hist(surrogate_mean_errors, bins=25, color='#3498DB', alpha=0.7, edgecolor='black', label='Surrogates')
ax.axvline(observed_mean_phi_error, color='red', linewidth=2.5, linestyle='-', label=f'Observed')
ax.set_xlabel('Mean |Ratio - Phi|')
ax.set_ylabel('Count')
ax.set_title(f'Phi-Proximity Test\nZ={z_phi_error:.2f}, p={p_phi_error:.4f}', fontweight='bold')
ax.legend()

ax = axes[1]
ax.hist(surrogate_pci_rates, bins=25, color='#2ECC71', alpha=0.7, edgecolor='black', label='Surrogates')
ax.axvline(observed_pci_rate, color='red', linewidth=2.5, linestyle='-', label=f'Observed = {observed_pci_rate:.1f}%')
ax.set_xlabel('% Subjects with PCI > 0')
ax.set_ylabel('Count')
ax.set_title(f'PCI>0 Rate Test\np={p_pci_rate:.4f}', fontweight='bold')
ax.legend()

ax = axes[2]
ax.hist(surrogate_mean_ratios, bins=25, color='#F39C12', alpha=0.7, edgecolor='black', label='Surrogate means')
ax.axvline(np.mean(all_ratios), color='red', linewidth=2.5, linestyle='-', label=f'Observed')
ax.axvline(PHI, color='gold', linewidth=2, linestyle='--', label=f'phi')
ax.set_xlabel('Mean Alpha/Theta Ratio')
ax.set_ylabel('Count')
ax.set_title('Mean Ratio Distribution', fontweight='bold')
ax.legend()

plt.suptitle(f'Surrogate Null Model — N={n}, {N_SURROGATES} surrogates', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/surrogate_null_model.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*70)
print("TEST 5: BAYESIAN MODEL COMPARISON")
print(f"N = {n}")
print("="*70)

def neg_log_lik_single(params, data, fixed_mu=None):
    if fixed_mu is not None:
        mu = fixed_mu
        sigma = max(params[0], 1e-6)
    else:
        mu = params[0]
        sigma = max(params[1], 1e-6)
    return -np.sum(norm.logpdf(data, loc=mu, scale=sigma))

def neg_log_lik_mixture2(params, data):
    w1 = 1 / (1 + np.exp(-params[0]))
    mu1 = params[1]
    s1 = max(np.exp(params[2]), 1e-6)
    mu2 = params[3]
    s2 = max(np.exp(params[4]), 1e-6)
    p1 = w1 * norm.pdf(data, loc=mu1, scale=s1)
    p2 = (1 - w1) * norm.pdf(data, loc=mu2, scale=s2)
    return -np.sum(np.log(p1 + p2 + 1e-300))

def neg_log_lik_mixture3(params, data):
    e1 = np.exp(params[0])
    e2 = np.exp(params[1])
    total = 1 + e1 + e2
    w1 = e1 / total
    w2 = e2 / total
    w3 = 1 / total
    mu1 = params[2]
    s1 = max(np.exp(params[3]), 1e-6)
    mu2 = params[4]
    s2 = max(np.exp(params[5]), 1e-6)
    mu3 = params[6]
    s3 = max(np.exp(params[7]), 1e-6)
    p = w1 * norm.pdf(data, loc=mu1, scale=s1) + w2 * norm.pdf(data, loc=mu2, scale=s2) + w3 * norm.pdf(data, loc=mu3, scale=s3)
    return -np.sum(np.log(p + 1e-300))

models = {}
for name, mu_fixed in [('phi', PHI), ('e-1', E_MINUS_1), ('2:1', HARMONIC)]:
    res = minimize(neg_log_lik_single, x0=[0.1], args=(all_ratios, mu_fixed), method='Nelder-Mead')
    nll = res.fun
    k_p = 1
    bic = k_p * np.log(n) + 2 * nll
    aic = 2 * k_p + 2 * nll
    models[f'Single_{name}'] = {'nll': nll, 'bic': bic, 'aic': aic, 'k': k_p, 'mu': mu_fixed, 'sigma': res.x[0]}

res_free = minimize(neg_log_lik_single, x0=[np.mean(all_ratios), 0.1], args=(all_ratios, None), method='Nelder-Mead')
nll_free = res_free.fun
k_free = 2
bic_free = k_free * np.log(n) + 2 * nll_free
models['Single_free'] = {'nll': nll_free, 'bic': bic_free, 'aic': 2 * k_free + 2 * nll_free, 'k': k_free}

for name, mu_init in [('phi+harm', (PHI, HARMONIC)), ('phi+e-1', (PHI, E_MINUS_1))]:
    x0 = [0.0, mu_init[0], np.log(0.05), mu_init[1], np.log(0.05)]
    res2 = minimize(neg_log_lik_mixture2, x0=x0, args=(all_ratios,), method='Nelder-Mead',
                    options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
    nll2 = res2.fun
    k2 = 5
    bic2 = k2 * np.log(n) + 2 * nll2
    w1 = 1 / (1 + np.exp(-res2.x[0]))
    models[f'Mix2_{name}'] = {'nll': nll2, 'bic': bic2, 'aic': 2 * k2 + 2 * nll2, 'k': k2,
                              'w1': w1, 'mu1': res2.x[1], 's1': np.exp(res2.x[2]),
                              'mu2': res2.x[3], 's2': np.exp(res2.x[4])}

x0_free2 = [0.0, PHI, np.log(0.05), np.mean(all_ratios), np.log(0.1)]
res_free2 = minimize(neg_log_lik_mixture2, x0=x0_free2, args=(all_ratios,), method='Nelder-Mead', options={'maxiter': 10000})
nll_f2 = res_free2.fun
k_f2 = 5
models['Mix2_free'] = {'nll': nll_f2, 'bic': k_f2 * np.log(n) + 2 * nll_f2, 'aic': 2 * k_f2 + 2 * nll_f2, 'k': k_f2,
                       'w1': 1 / (1 + np.exp(-res_free2.x[0])), 'mu1': res_free2.x[1], 's1': np.exp(res_free2.x[2]),
                       'mu2': res_free2.x[3], 's2': np.exp(res_free2.x[4])}

x0_3 = [0.0, 0.0, PHI, np.log(0.05), E_MINUS_1, np.log(0.05), HARMONIC, np.log(0.05)]
res3 = minimize(neg_log_lik_mixture3, x0=x0_3, args=(all_ratios,), method='Nelder-Mead', options={'maxiter': 20000})
nll3 = res3.fun
k3 = 8
models['Mix3_phi_e1_harm'] = {'nll': nll3, 'bic': k3 * np.log(n) + 2 * nll3, 'aic': 2 * k3 + 2 * nll3, 'k': k3}

sorted_models = sorted(models.items(), key=lambda x: x[1]['bic'])
best_bic = sorted_models[0][1]['bic']
best_name = sorted_models[0][0]

print(f"\n{'Model':<25s} {'k':>3s} {'NLL':>8s} {'BIC':>8s} {'dBIC':>8s}")
print("-" * 52)
for mname, m in sorted_models:
    dbic = m['bic'] - best_bic
    print(f"{mname:<25s} {m['k']:>3d} {m['nll']:>8.2f} {m['bic']:>8.2f} {dbic:>8.2f}")
print(f"\nBest model: {best_name} (BIC = {best_bic:.2f})")

bayesian_results = {
    'n_subjects': n,
    'best_model': best_name,
    'best_bic': float(best_bic),
    'models': {mname: {'bic': float(m['bic']), 'nll': float(m['nll']), 'k': m['k']}
               for mname, m in models.items()},
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
x_range = np.linspace(all_ratios.min() - 0.1, all_ratios.max() + 0.1, 200)
ax.hist(all_ratios, bins=15, density=True, alpha=0.5, color='gray', edgecolor='black', label='Data')
for mname, mu, color in [('phi', PHI, '#E74C3C'), ('e-1', E_MINUS_1, '#3498DB'), ('2:1', HARMONIC, '#2ECC71')]:
    m = models[f'Single_{mname}']
    pdf = norm.pdf(x_range, loc=m['mu'], scale=m['sigma'])
    ax.plot(x_range, pdf, color=color, linewidth=2, label=f'{mname} (BIC={m["bic"]:.1f})')
ax.set_xlabel('Alpha/Theta Ratio')
ax.set_ylabel('Density')
ax.set_title('Single-Attractor Models', fontweight='bold')
ax.legend()

ax = axes[1]
model_names_sorted = [mn for mn, _ in sorted_models]
bic_vals = [m['bic'] for _, m in sorted_models]
colors_bic = ['#2ECC71' if i == 0 else '#3498DB' for i in range(len(model_names_sorted))]
ax.barh(range(len(model_names_sorted)), bic_vals, color=colors_bic, alpha=0.8, edgecolor='black')
ax.set_yticks(range(len(model_names_sorted)))
ax.set_yticklabels(model_names_sorted, fontsize=9)
ax.set_xlabel('BIC')
ax.set_title('Model Comparison (BIC)', fontweight='bold')
ax.invert_yaxis()

plt.suptitle(f'Bayesian Model Comparison — N={n}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/bayesian_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

all_results = {
    'batch_analysis': batch_results,
    'boundary_jitter': jitter_results,
    'split_half': split_results,
    'surrogate': surrogate_results,
    'bayesian': bayesian_results,
}

with open('outputs/all_n35_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n" + "="*70)
print("ALL TESTS COMPLETE — Results saved to outputs/all_n35_results.json")
print("="*70)
