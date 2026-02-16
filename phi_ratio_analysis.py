#!/usr/bin/env python3
"""
PHI-RATIO ANALYSIS ON EEG DATA
==============================
Analyzing whether frequency band ratios converge to golden ratio φ ≈ 1.618

Based on hypothesis that optimal neural dynamics exhibit φ-ratio relationships
between adjacent frequency bands.
"""

import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from collections import defaultdict

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

PHI = 1.618033988749895

FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

BAND_ORDER = ['delta', 'theta', 'alpha', 'beta', 'gamma']

def load_mat_file(filepath):
    """Load .mat file and return data structure"""
    try:
        data = sio.loadmat(filepath)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def explore_mat_structure(data, filename):
    """Explore the structure of a .mat file"""
    print(f"\n{'='*60}")
    print(f"FILE: {filename}")
    print(f"{'='*60}")
    
    for key in data.keys():
        if not key.startswith('__'):
            val = data[key]
            if isinstance(val, np.ndarray):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
            else:
                print(f"  {key}: type={type(val)}")

def compute_psd(eeg_data, fs=128, nperseg=256):
    """Compute Power Spectral Density using Welch's method"""
    if len(eeg_data.shape) == 1:
        freqs, psd = signal.welch(eeg_data, fs=fs, nperseg=min(nperseg, len(eeg_data)//2))
    else:
        freqs, psd = signal.welch(eeg_data, fs=fs, nperseg=min(nperseg, eeg_data.shape[-1]//2), axis=-1)
    return freqs, psd

def find_peak_frequency(freqs, psd, freq_range):
    """Find the peak frequency within a specified range"""
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not mask.any():
        return np.nan
    
    band_freqs = freqs[mask]
    band_psd = psd[mask] if len(psd.shape) == 1 else psd[..., mask]
    
    if len(band_psd.shape) == 1:
        peak_idx = np.argmax(band_psd)
        return band_freqs[peak_idx]
    else:
        mean_psd = np.mean(band_psd, axis=tuple(range(len(band_psd.shape)-1)))
        peak_idx = np.argmax(mean_psd)
        return band_freqs[peak_idx]

def extract_band_peaks(freqs, psd):
    """Extract peak frequencies for all bands"""
    peaks = {}
    for band_name in BAND_ORDER:
        freq_range = FREQ_BANDS[band_name]
        peaks[band_name] = find_peak_frequency(freqs, psd, freq_range)
    return peaks

def compute_phi_ratios(peaks):
    """Compute ratios between adjacent frequency band peaks"""
    ratios = {}
    for i in range(len(BAND_ORDER) - 1):
        lower_band = BAND_ORDER[i]
        upper_band = BAND_ORDER[i + 1]
        
        if peaks[lower_band] > 0 and peaks[upper_band] > 0:
            ratio = peaks[upper_band] / peaks[lower_band]
            ratio_name = f"{upper_band}/{lower_band}"
            ratios[ratio_name] = ratio
    
    return ratios

def test_phi_convergence(ratios_list, ratio_name):
    """Test if a set of ratios converges to phi"""
    valid_ratios = [r for r in ratios_list if not np.isnan(r) and r > 0]
    
    if len(valid_ratios) < 3:
        return None
    
    mean_ratio = np.mean(valid_ratios)
    std_ratio = np.std(valid_ratios)
    
    t_stat, p_value = stats.ttest_1samp(valid_ratios, PHI)
    
    cohens_d = (mean_ratio - PHI) / std_ratio if std_ratio > 0 else np.inf
    
    distance_from_phi = abs(mean_ratio - PHI)
    
    bootstrap_means = []
    for _ in range(1000):
        sample = np.random.choice(valid_ratios, size=len(valid_ratios), replace=True)
        bootstrap_means.append(np.mean(sample))
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    
    return {
        'ratio_name': ratio_name,
        'n': len(valid_ratios),
        'mean': mean_ratio,
        'std': std_ratio,
        'median': np.median(valid_ratios),
        'distance_from_phi': distance_from_phi,
        'percent_error_from_phi': (distance_from_phi / PHI) * 100,
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'phi_in_ci': ci_lower <= PHI <= ci_upper
    }

def analyze_eeg_file(filepath, fs=128):
    """Analyze a single EEG file for phi-ratios"""
    data = load_mat_file(filepath)
    if data is None:
        return None
    
    eeg_data = None
    for key in data.keys():
        if not key.startswith('__'):
            val = data[key]
            if isinstance(val, np.ndarray) and val.size > 100:
                eeg_data = val
                break
    
    if eeg_data is None:
        return None
    
    if eeg_data.dtype == object:
        try:
            eeg_data = np.vstack([np.array(x).flatten() for x in eeg_data.flatten()])
        except:
            return None
    
    eeg_data = np.array(eeg_data, dtype=np.float64)
    
    if np.isnan(eeg_data).all():
        return None
    
    eeg_data = np.nan_to_num(eeg_data, nan=0.0)
    
    if len(eeg_data.shape) == 1:
        eeg_data = eeg_data.reshape(1, -1)
    
    freqs, psd = compute_psd(eeg_data, fs=fs)
    peaks = extract_band_peaks(freqs, psd)
    ratios = compute_phi_ratios(peaks)
    
    return {
        'peaks': peaks,
        'ratios': ratios,
        'shape': eeg_data.shape
    }

def main():
    print("="*70)
    print("PHI-RATIO ANALYSIS ON EEG FREQUENCY BANDS")
    print("Testing hypothesis: Adjacent band ratios → φ ≈ 1.618")
    print("="*70)
    
    mat_files = glob.glob("alpha_s*.mat") + glob.glob("subject_*.mat")
    mat_files = list(set(mat_files))
    
    print(f"\nFound {len(mat_files)} EEG data files")
    
    if len(mat_files) == 0:
        print("No .mat files found!")
        return
    
    print("\n" + "-"*50)
    print("EXPLORING DATA STRUCTURE")
    print("-"*50)
    sample_data = load_mat_file(mat_files[0])
    if sample_data:
        explore_mat_structure(sample_data, mat_files[0])
    
    all_ratios = defaultdict(list)
    all_peaks = defaultdict(list)
    successful_files = 0
    
    print("\n" + "-"*50)
    print("PROCESSING FILES")
    print("-"*50)
    
    for filepath in mat_files:
        result = analyze_eeg_file(filepath)
        if result:
            successful_files += 1
            for band, peak in result['peaks'].items():
                if not np.isnan(peak):
                    all_peaks[band].append(peak)
            for ratio_name, ratio in result['ratios'].items():
                all_ratios[ratio_name].append(ratio)
            print(f"  ✓ {filepath}: shape={result['shape']}")
        else:
            print(f"  ✗ {filepath}: Could not process")
    
    print(f"\nSuccessfully processed: {successful_files}/{len(mat_files)} files")
    
    print("\n" + "="*70)
    print("PEAK FREQUENCIES BY BAND")
    print("="*70)
    
    for band in BAND_ORDER:
        peaks = all_peaks[band]
        if len(peaks) > 0:
            print(f"\n{band.upper()} ({FREQ_BANDS[band][0]}-{FREQ_BANDS[band][1]} Hz):")
            print(f"  Mean peak: {np.mean(peaks):.2f} Hz")
            print(f"  Std:       {np.std(peaks):.2f} Hz")
            print(f"  Range:     {np.min(peaks):.2f} - {np.max(peaks):.2f} Hz")
    
    print("\n" + "="*70)
    print("PHI-RATIO STATISTICAL ANALYSIS")
    print("="*70)
    print(f"\nGolden Ratio (φ) = {PHI:.6f}")
    print("-"*70)
    
    results_df = []
    
    for ratio_name in all_ratios.keys():
        ratios = all_ratios[ratio_name]
        result = test_phi_convergence(ratios, ratio_name)
        
        if result:
            results_df.append(result)
            print(f"\n{ratio_name.upper()} RATIO:")
            print(f"  N samples:       {result['n']}")
            print(f"  Mean ratio:      {result['mean']:.4f}")
            print(f"  Median ratio:    {result['median']:.4f}")
            print(f"  Std:             {result['std']:.4f}")
            print(f"  95% CI:          [{result['ci_95_lower']:.4f}, {result['ci_95_upper']:.4f}]")
            print(f"  Distance from φ: {result['distance_from_phi']:.4f} ({result['percent_error_from_phi']:.1f}%)")
            print(f"  t-statistic:     {result['t_stat']:.3f}")
            print(f"  p-value:         {result['p_value']:.6f}")
            print(f"  Cohen's d:       {result['cohens_d']:.3f}")
            print(f"  φ in 95% CI:     {'YES ✓' if result['phi_in_ci'] else 'NO'}")
            
            if result['phi_in_ci']:
                print(f"  → φ is WITHIN confidence interval")
            elif result['mean'] < PHI:
                print(f"  → Ratio is LOWER than φ")
            else:
                print(f"  → Ratio is HIGHER than φ")
    
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    
    if results_df:
        df = pd.DataFrame(results_df)
        summary_cols = ['ratio_name', 'n', 'mean', 'distance_from_phi', 'p_value', 'cohens_d', 'phi_in_ci']
        print("\n", df[summary_cols].to_string(index=False))
        
        df.to_csv('phi_ratio_results.csv', index=False)
        print("\nResults saved to: phi_ratio_results.csv")
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    ax1 = axes[0, 0]
    ratio_names = list(all_ratios.keys())
    ratio_data = [all_ratios[r] for r in ratio_names]
    bp = ax1.boxplot(ratio_data, labels=ratio_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax1.axhline(y=PHI, color='gold', linestyle='--', linewidth=2, label=f'φ = {PHI:.3f}')
    ax1.axhline(y=2.0, color='gray', linestyle=':', alpha=0.5, label='2.0 (harmonic)')
    ax1.axhline(y=1.5, color='gray', linestyle=':', alpha=0.5, label='1.5 (3:2)')
    ax1.set_ylabel('Frequency Ratio')
    ax1.set_xlabel('Adjacent Band Pairs')
    ax1.set_title('Distribution of Frequency Band Ratios')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    if results_df:
        df_sorted = pd.DataFrame(results_df).sort_values('distance_from_phi')
        colors = ['green' if r['phi_in_ci'] else 'red' for _, r in df_sorted.iterrows()]
        bars = ax2.barh(df_sorted['ratio_name'], df_sorted['distance_from_phi'], color=colors, alpha=0.7)
        ax2.set_xlabel('Distance from φ')
        ax2.set_title('Distance from Golden Ratio by Band Pair')
        ax2.axvline(x=0.1, color='orange', linestyle='--', alpha=0.7, label='10% threshold')
        ax2.legend()
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            ax2.text(row['distance_from_phi'] + 0.01, i, f"{row['percent_error_from_phi']:.1f}%", va='center')
    
    ax3 = axes[1, 0]
    for ratio_name, ratios in all_ratios.items():
        valid_ratios = [r for r in ratios if not np.isnan(r)]
        if len(valid_ratios) > 5:
            if HAS_SEABORN:
                sns.kdeplot(valid_ratios, ax=ax3, label=ratio_name, fill=True, alpha=0.3)
            else:
                ax3.hist(valid_ratios, bins=15, alpha=0.4, label=ratio_name, density=True)
    ax3.axvline(x=PHI, color='gold', linestyle='--', linewidth=2, label=f'φ = {PHI:.3f}')
    ax3.set_xlabel('Ratio Value')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution of Ratios vs. Golden Ratio')
    ax3.legend()
    ax3.set_xlim(0.5, 4)
    
    ax4 = axes[1, 1]
    band_names = BAND_ORDER
    band_peaks_means = [np.mean(all_peaks[b]) if all_peaks[b] else 0 for b in band_names]
    band_peaks_stds = [np.std(all_peaks[b]) if all_peaks[b] else 0 for b in band_names]
    
    x_pos = np.arange(len(band_names))
    ax4.bar(x_pos, band_peaks_means, yerr=band_peaks_stds, capsize=5, color='steelblue', alpha=0.7)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([b.upper() for b in band_names])
    ax4.set_ylabel('Peak Frequency (Hz)')
    ax4.set_xlabel('Frequency Band')
    ax4.set_title('Mean Peak Frequencies by Band')
    
    for i, (mean, std) in enumerate(zip(band_peaks_means, band_peaks_stds)):
        if mean > 0:
            ax4.text(i, mean + std + 1, f'{mean:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('phi_ratio_analysis_results.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to: phi_ratio_analysis_results.png")
    plt.close()
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if results_df:
        phi_consistent = sum(1 for r in results_df if r['phi_in_ci'])
        total = len(results_df)
        print(f"\n{phi_consistent}/{total} band ratios have φ within 95% CI")
        
        closest = min(results_df, key=lambda x: x['distance_from_phi'])
        print(f"\nClosest to φ: {closest['ratio_name']} (mean={closest['mean']:.4f}, distance={closest['distance_from_phi']:.4f})")
        
        significant = [r for r in results_df if r['p_value'] < 0.05]
        if significant:
            print(f"\nSignificantly different from φ (p<0.05):")
            for r in significant:
                print(f"  - {r['ratio_name']}: mean={r['mean']:.3f}, p={r['p_value']:.4f}")
        else:
            print("\nNo ratios significantly different from φ at p<0.05")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
