#!/usr/bin/env python3
"""Job 2: Downsampling experiment on OpenNeuro ds003969"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import welch, resample
import os
import mne
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

PHI = (1 + np.sqrt(5)) / 2
E_MINUS_1 = np.e - 1
OUTPUT_DIR = 'outputs/publication_figures'

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'font.family': 'serif',
})

data_dir = 'ds003969'
sub_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
print(f"Found {len(sub_dirs)} subjects")

target_fs = [1024, 512, 256, 160]

def compute_ratio_from_raw(raw_data, sfreq):
    freqs, psd = welch(raw_data, fs=sfreq, nperseg=min(int(2 * sfreq), raw_data.shape[1]))
    psd_mean = psd.mean(axis=0)

    theta_mask = (freqs >= 4) & (freqs <= 8)
    alpha_mask = (freqs >= 8) & (freqs <= 13)

    if np.sum(psd_mean[theta_mask]) < 1e-20 or np.sum(psd_mean[alpha_mask]) < 1e-20:
        return None

    theta_centroid = np.sum(freqs[theta_mask] * psd_mean[theta_mask]) / np.sum(psd_mean[theta_mask])
    alpha_centroid = np.sum(freqs[alpha_mask] * psd_mean[alpha_mask]) / np.sum(psd_mean[alpha_mask])

    if theta_centroid < 1e-6:
        return None
    return alpha_centroid / theta_centroid

all_results = []
processed = 0

for sub_dir in sub_dirs:
    eeg_dir = os.path.join(data_dir, sub_dir, 'eeg')
    if not os.path.exists(eeg_dir):
        continue

    bdf_files = [f for f in os.listdir(eeg_dir) if f.endswith('.bdf') and 'think1' in f]
    if not bdf_files:
        bdf_files = [f for f in os.listdir(eeg_dir) if f.endswith('.bdf')]
    if not bdf_files:
        continue

    bdf_path = os.path.join(eeg_dir, bdf_files[0])
    try:
        raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
        native_fs = raw.info['sfreq']

        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        if len(eeg_picks) == 0:
            continue
        data = raw.get_data(picks=eeg_picks)

        max_samples = int(10 * native_fs)
        if data.shape[1] > max_samples:
            data = data[:, :max_samples]

        subject_ratios = {}
        for fs_target in target_fs:
            if fs_target > native_fs:
                subject_ratios[fs_target] = None
                continue
            if fs_target == native_fs:
                data_fs = data
            else:
                n_samples_new = int(data.shape[1] * fs_target / native_fs)
                data_fs = resample(data, n_samples_new, axis=1)

            ratio = compute_ratio_from_raw(data_fs, fs_target)
            subject_ratios[fs_target] = ratio

        all_results.append({
            'subject': sub_dir,
            'native_fs': float(native_fs),
            'ratios': {str(k): v for k, v in subject_ratios.items()}
        })
        processed += 1
        if processed % 5 == 0:
            print(f"  Processed {processed} subjects")

    except Exception as e:
        print(f"  Error {sub_dir}: {e}")
        continue

    if processed >= 20:
        break

print(f"Total processed: {processed}")

mean_by_fs = {}
std_by_fs = {}
for fs in target_fs:
    vals = [r['ratios'][str(fs)] for r in all_results if r['ratios'].get(str(fs)) is not None]
    if vals:
        mean_by_fs[fs] = float(np.mean(vals))
        std_by_fs[fs] = float(np.std(vals))

paired_data = {fs: [] for fs in target_fs}
for r in all_results:
    if all(r['ratios'].get(str(fs)) is not None for fs in target_fs):
        for fs in target_fs:
            paired_data[fs].append(r['ratios'][str(fs)])

available_fs = [fs for fs in target_fs if len(paired_data[fs]) > 3]
min_len = min(len(paired_data[fs]) for fs in available_fs) if available_fs else 0
print(f"Paired subjects across all rates: {min_len}")

if len(available_fs) >= 3 and min_len >= 5:
    from scipy.stats import friedmanchisquare
    trimmed = {fs: paired_data[fs][:min_len] for fs in available_fs}
    try:
        stat, rm_p = friedmanchisquare(*[trimmed[fs] for fs in available_fs])
    except:
        stat, rm_p = np.nan, np.nan
elif len(available_fs) == 2 and min_len >= 5:
    stat, rm_p = stats.wilcoxon(paired_data[available_fs[0]][:min_len], paired_data[available_fs[1]][:min_len])
else:
    stat, rm_p = np.nan, np.nan

within_drifts = []
ref_fs = max(available_fs) if available_fs else 1024
low_fs = min(available_fs) if available_fs else 160
for r in all_results:
    r_high = r['ratios'].get(str(ref_fs))
    r_low = r['ratios'].get(str(low_fs))
    if r_high is not None and r_low is not None:
        within_drifts.append(r_low - r_high)

mean_drift = float(np.mean(within_drifts)) if within_drifts else 0

if not np.isnan(rm_p) and rm_p < 0.05 and abs(mean_drift) > 0.02:
    conclusion = 'methodological artifact (significant drift detected)'
elif abs(mean_drift) < 0.001:
    conclusion = 'no evidence of sampling-rate-induced drift; between-dataset differences are biological'
else:
    conclusion = 'inconclusive (drift present but not statistically significant)'

ds_results = {
    'n_subjects': processed,
    'mean_ratio_by_fs': {str(k): v for k, v in mean_by_fs.items()},
    'std_ratio_by_fs': {str(k): v for k, v in std_by_fs.items()},
    'within_subject_drift': float(mean_drift),
    'rm_anova_p': float(rm_p) if not np.isnan(rm_p) else None,
    'rm_anova_statistic': float(stat) if not np.isnan(stat) else None,
    'conclusion': conclusion,
    'limitations': f'N={processed} subjects, 10s epochs; full dataset may differ',
    'available_sampling_rates': available_fs,
    'n_paired': min_len
}

with open('outputs/downsampling_results.json', 'w') as f:
    json.dump(ds_results, f, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
if available_fs and min_len > 0:
    for r in all_results[:50]:
        fs_vals = []
        ratio_vals = []
        for fs in sorted(available_fs):
            v = r['ratios'].get(str(fs))
            if v is not None:
                fs_vals.append(fs)
                ratio_vals.append(v)
        if len(fs_vals) >= 2:
            ax.plot(fs_vals, ratio_vals, '-o', color='lightsteelblue', alpha=0.3, markersize=3)

    means = [mean_by_fs.get(fs, np.nan) for fs in sorted(available_fs)]
    ax.plot(sorted(available_fs), means, 'ko-', lw=2, markersize=8, label='Group mean')
    ax.axhline(E_MINUS_1, color='steelblue', ls='--', lw=1.5, alpha=0.7, label=f'e−1 = {E_MINUS_1:.3f}')
    ax.set_xlabel('Sampling Rate (Hz)')
    ax.set_ylabel('α/θ Centroid Ratio')
    ax.set_title(f'Within-Subject Drift (N={min_len} paired)')
    ax.legend(fontsize=9)

ax2 = axes[1]
if within_drifts:
    ax2.hist(within_drifts, bins=25, color='steelblue', alpha=0.7, edgecolor='white')
    ax2.axvline(0, color='black', ls='-', lw=1)
    ax2.axvline(mean_drift, color='red', ls='--', lw=2, label=f'Mean drift = {mean_drift:.4f}')
    ax2.set_xlabel(f'Ratio drift ({low_fs}→{ref_fs} Hz)')
    ax2.set_ylabel('Count')
    p_str = f'{rm_p:.4f}' if not np.isnan(rm_p) else 'N/A'
    ax2.set_title(f'Distribution of within-subject drift\np = {p_str}')
    ax2.legend(fontsize=9)

conclusion = ds_results['conclusion'].upper()
fig.suptitle(f'Figure 12: Downsampling Experiment (OpenNeuro ds003969)\n'
             f'Conclusion: {conclusion}',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig12_downsampling.png')
plt.close()

print(f"\nDone! Mean drift: {mean_drift:.4f}")
print(f"  Conclusion: {ds_results['conclusion']}")
print(f"  p = {rm_p}")
print(json.dumps(ds_results, indent=2))
