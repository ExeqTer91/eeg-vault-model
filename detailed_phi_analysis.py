#!/usr/bin/env python3
"""
DETAILED PHI-RATIO ANALYSIS ON EEG DATA
=======================================
Extended analysis with proper frequency band detection
"""

import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import glob
from collections import defaultdict

PHI = 1.618033988749895

FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

BAND_ORDER = ['delta', 'theta', 'alpha', 'beta', 'gamma']

def load_and_inspect_mat(filepath):
    """Load and inspect .mat file structure"""
    data = sio.loadmat(filepath)
    info = {}
    for key in data.keys():
        if not key.startswith('__'):
            val = data[key]
            if isinstance(val, np.ndarray):
                info[key] = {'shape': val.shape, 'dtype': str(val.dtype), 'min': float(np.nanmin(val)) if val.size > 0 else None, 'max': float(np.nanmax(val)) if val.size > 0 else None}
    return data, info

def compute_psd_detailed(eeg_data, fs=128):
    """Compute detailed PSD with proper parameters"""
    if len(eeg_data.shape) == 1:
        eeg_data = eeg_data.reshape(1, -1)
    
    nperseg = min(fs * 4, eeg_data.shape[-1] // 4)
    nperseg = max(nperseg, 128)
    
    freqs, psd = signal.welch(eeg_data, fs=fs, nperseg=nperseg, noverlap=nperseg//2, axis=-1)
    
    if len(psd.shape) > 1:
        psd = np.mean(psd, axis=0)
    
    return freqs, psd

def get_band_power_and_peak(freqs, psd, band_range):
    """Get band power and peak frequency for a given band"""
    mask = (freqs >= band_range[0]) & (freqs <= band_range[1])
    if not mask.any():
        return np.nan, np.nan
    
    band_freqs = freqs[mask]
    band_psd = psd[mask]
    
    if len(band_psd) == 0 or np.all(np.isnan(band_psd)):
        return np.nan, np.nan
    
    power = np.trapezoid(band_psd, band_freqs) if hasattr(np, 'trapezoid') else np.sum(band_psd * np.diff(np.append(band_freqs, band_freqs[-1])))
    peak_idx = np.argmax(band_psd)
    peak_freq = band_freqs[peak_idx]
    
    return power, peak_freq

def compute_center_of_gravity(freqs, psd, band_range):
    """Compute spectral center of gravity (weighted mean frequency)"""
    mask = (freqs >= band_range[0]) & (freqs <= band_range[1])
    if not mask.any():
        return np.nan
    
    band_freqs = freqs[mask]
    band_psd = psd[mask]
    
    if np.sum(band_psd) == 0:
        return np.nan
    
    cog = np.sum(band_freqs * band_psd) / np.sum(band_psd)
    return cog

def analyze_subject(filepath, fs=128):
    """Full analysis for one subject"""
    data = sio.loadmat(filepath)
    
    eeg_data = None
    for key in data.keys():
        if not key.startswith('__'):
            val = data[key]
            if isinstance(val, np.ndarray) and val.size > 1000:
                eeg_data = val.astype(np.float64)
                break
    
    if eeg_data is None:
        return None
    
    eeg_data = np.nan_to_num(eeg_data, nan=0.0)
    
    if len(eeg_data.shape) == 1:
        eeg_data = eeg_data.reshape(1, -1)
    elif eeg_data.shape[0] > eeg_data.shape[1]:
        eeg_data = eeg_data.T
    
    freqs, psd = compute_psd_detailed(eeg_data, fs=fs)
    
    results = {
        'powers': {},
        'peaks': {},
        'cog': {},
        'ratios_peak': {},
        'ratios_cog': {}
    }
    
    for band in BAND_ORDER:
        power, peak = get_band_power_and_peak(freqs, psd, FREQ_BANDS[band])
        cog = compute_center_of_gravity(freqs, psd, FREQ_BANDS[band])
        results['powers'][band] = power
        results['peaks'][band] = peak
        results['cog'][band] = cog
    
    for i in range(len(BAND_ORDER) - 1):
        lower = BAND_ORDER[i]
        upper = BAND_ORDER[i + 1]
        ratio_name = f"{upper}/{lower}"
        
        if results['peaks'][lower] > 0 and results['peaks'][upper] > 0:
            results['ratios_peak'][ratio_name] = results['peaks'][upper] / results['peaks'][lower]
        
        if results['cog'][lower] > 0 and results['cog'][upper] > 0:
            results['ratios_cog'][ratio_name] = results['cog'][upper] / results['cog'][lower]
    
    return results, freqs, psd

def main():
    print("="*80)
    print("DETAILED PHI-RATIO ANALYSIS ON EEG FREQUENCY BANDS")
    print("Testing: Do adjacent band ratios converge to φ ≈ 1.618?")
    print("="*80)
    
    mat_files = glob.glob("alpha_s*.mat") + glob.glob("subject_*.mat")
    mat_files = list(set(mat_files))
    print(f"\nFound {len(mat_files)} EEG files")
    
    print("\n" + "-"*60)
    print("INSPECTING DATA STRUCTURE")
    print("-"*60)
    
    data, info = load_and_inspect_mat(mat_files[0])
    for key, val in info.items():
        print(f"  {key}: shape={val['shape']}, dtype={val['dtype']}")
        print(f"         range=[{val['min']:.2f}, {val['max']:.2f}]")
    
    all_peaks = defaultdict(list)
    all_cog = defaultdict(list)
    all_powers = defaultdict(list)
    all_ratios_peak = defaultdict(list)
    all_ratios_cog = defaultdict(list)
    
    print("\n" + "-"*60)
    print("ANALYZING SUBJECTS")
    print("-"*60)
    
    sample_freqs = None
    sample_psds = []
    
    for filepath in mat_files[:5]:
        result = analyze_subject(filepath)
        if result:
            res, freqs, psd = result
            if sample_freqs is None:
                sample_freqs = freqs
            sample_psds.append(psd)
            
            for band in BAND_ORDER:
                if not np.isnan(res['peaks'][band]):
                    all_peaks[band].append(res['peaks'][band])
                if not np.isnan(res['cog'][band]):
                    all_cog[band].append(res['cog'][band])
                if not np.isnan(res['powers'][band]):
                    all_powers[band].append(res['powers'][band])
            
            for ratio_name, val in res['ratios_peak'].items():
                if not np.isnan(val):
                    all_ratios_peak[ratio_name].append(val)
            for ratio_name, val in res['ratios_cog'].items():
                if not np.isnan(val):
                    all_ratios_cog[ratio_name].append(val)
    
    for filepath in mat_files[5:]:
        result = analyze_subject(filepath)
        if result:
            res, _, _ = result
            for band in BAND_ORDER:
                if not np.isnan(res['peaks'][band]):
                    all_peaks[band].append(res['peaks'][band])
                if not np.isnan(res['cog'][band]):
                    all_cog[band].append(res['cog'][band])
                if not np.isnan(res['powers'][band]):
                    all_powers[band].append(res['powers'][band])
            for ratio_name, val in res['ratios_peak'].items():
                if not np.isnan(val):
                    all_ratios_peak[ratio_name].append(val)
            for ratio_name, val in res['ratios_cog'].items():
                if not np.isnan(val):
                    all_ratios_cog[ratio_name].append(val)
    
    print(f"  Processed {len(mat_files)} files")
    
    print("\n" + "="*80)
    print("FREQUENCY BAND ANALYSIS")
    print("="*80)
    
    print("\n{:<10} {:>12} {:>12} {:>12} {:>12}".format(
        "Band", "Peak Mean", "Peak Std", "CoG Mean", "CoG Std"))
    print("-"*60)
    
    for band in BAND_ORDER:
        peaks = all_peaks[band]
        cogs = all_cog[band]
        if peaks:
            print("{:<10} {:>12.2f} {:>12.2f} {:>12.2f} {:>12.2f}".format(
                band.upper(),
                np.mean(peaks), np.std(peaks),
                np.mean(cogs), np.std(cogs)
            ))
        else:
            print(f"{band.upper():<10} {'N/A':>12}")
    
    print("\n" + "="*80)
    print("PHI-RATIO ANALYSIS: PEAK FREQUENCY METHOD")
    print("="*80)
    print(f"Target: φ = {PHI:.6f}")
    
    peak_results = []
    for ratio_name in ['theta/delta', 'alpha/theta', 'beta/alpha', 'gamma/beta']:
        if ratio_name in all_ratios_peak and len(all_ratios_peak[ratio_name]) >= 3:
            vals = all_ratios_peak[ratio_name]
            mean = np.mean(vals)
            std = np.std(vals)
            t_stat, p_val = stats.ttest_1samp(vals, PHI)
            
            boot_means = [np.mean(np.random.choice(vals, len(vals), replace=True)) for _ in range(1000)]
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
            phi_in_ci = ci_low <= PHI <= ci_high
            
            distance = abs(mean - PHI)
            pct_error = (distance / PHI) * 100
            
            peak_results.append({
                'ratio': ratio_name,
                'n': len(vals),
                'mean': mean,
                'std': std,
                'distance': distance,
                'pct_error': pct_error,
                'p_value': p_val,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'phi_in_ci': phi_in_ci
            })
            
            print(f"\n{ratio_name.upper()}:")
            print(f"  N = {len(vals)}")
            print(f"  Mean = {mean:.4f} ± {std:.4f}")
            print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
            print(f"  Distance from φ: {distance:.4f} ({pct_error:.1f}%)")
            print(f"  p-value: {p_val:.6f}")
            print(f"  φ in CI: {'YES ✓' if phi_in_ci else 'NO'}")
    
    print("\n" + "="*80)
    print("PHI-RATIO ANALYSIS: CENTER-OF-GRAVITY METHOD")
    print("="*80)
    
    cog_results = []
    for ratio_name in ['theta/delta', 'alpha/theta', 'beta/alpha', 'gamma/beta']:
        if ratio_name in all_ratios_cog and len(all_ratios_cog[ratio_name]) >= 3:
            vals = all_ratios_cog[ratio_name]
            mean = np.mean(vals)
            std = np.std(vals)
            t_stat, p_val = stats.ttest_1samp(vals, PHI)
            
            boot_means = [np.mean(np.random.choice(vals, len(vals), replace=True)) for _ in range(1000)]
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
            phi_in_ci = ci_low <= PHI <= ci_high
            
            distance = abs(mean - PHI)
            pct_error = (distance / PHI) * 100
            
            cog_results.append({
                'ratio': ratio_name,
                'n': len(vals),
                'mean': mean,
                'std': std,
                'distance': distance,
                'pct_error': pct_error,
                'p_value': p_val,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'phi_in_ci': phi_in_ci
            })
            
            print(f"\n{ratio_name.upper()}:")
            print(f"  N = {len(vals)}")
            print(f"  Mean = {mean:.4f} ± {std:.4f}")
            print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
            print(f"  Distance from φ: {distance:.4f} ({pct_error:.1f}%)")
            print(f"  p-value: {p_val:.6f}")
            print(f"  φ in CI: {'YES ✓' if phi_in_ci else 'NO'}")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    ax1 = axes[0, 0]
    if sample_freqs is not None and sample_psds:
        mean_psd = np.mean(sample_psds, axis=0)
        ax1.semilogy(sample_freqs, mean_psd, 'b-', linewidth=2)
        for band, (low, high) in FREQ_BANDS.items():
            ax1.axvspan(low, high, alpha=0.2, label=band)
        ax1.set_xlim(0, 50)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power (log scale)')
        ax1.set_title('Mean Power Spectral Density')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    band_names = [b.upper() for b in BAND_ORDER]
    peak_means = [np.mean(all_peaks[b]) if all_peaks[b] else 0 for b in BAND_ORDER]
    peak_stds = [np.std(all_peaks[b]) if all_peaks[b] else 0 for b in BAND_ORDER]
    x = np.arange(len(band_names))
    ax2.bar(x, peak_means, yerr=peak_stds, capsize=5, color='steelblue', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(band_names)
    ax2.set_ylabel('Peak Frequency (Hz)')
    ax2.set_title('Peak Frequencies by Band')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[0, 2]
    if peak_results:
        ratios = [r['ratio'] for r in peak_results]
        means = [r['mean'] for r in peak_results]
        stds = [r['std'] for r in peak_results]
        colors = ['green' if r['phi_in_ci'] else 'red' for r in peak_results]
        x = np.arange(len(ratios))
        ax3.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        ax3.axhline(y=PHI, color='gold', linestyle='--', linewidth=2, label=f'φ = {PHI:.3f}')
        ax3.set_xticks(x)
        ax3.set_xticklabels(ratios, rotation=45)
        ax3.set_ylabel('Ratio')
        ax3.set_title('Peak Frequency Ratios vs φ')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 0]
    if cog_results:
        ratios = [r['ratio'] for r in cog_results]
        means = [r['mean'] for r in cog_results]
        stds = [r['std'] for r in cog_results]
        colors = ['green' if r['phi_in_ci'] else 'red' for r in cog_results]
        x = np.arange(len(ratios))
        ax4.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        ax4.axhline(y=PHI, color='gold', linestyle='--', linewidth=2, label=f'φ = {PHI:.3f}')
        ax4.set_xticks(x)
        ax4.set_xticklabels(ratios, rotation=45)
        ax4.set_ylabel('Ratio')
        ax4.set_title('Center-of-Gravity Ratios vs φ')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    ax5 = axes[1, 1]
    for ratio_name, vals in all_ratios_cog.items():
        if len(vals) > 3:
            ax5.hist(vals, bins=15, alpha=0.4, label=ratio_name, density=True)
    ax5.axvline(x=PHI, color='gold', linestyle='--', linewidth=3, label=f'φ = {PHI:.3f}')
    ax5.axvline(x=2.0, color='gray', linestyle=':', linewidth=2, label='2.0')
    ax5.axvline(x=1.5, color='gray', linestyle=':', linewidth=2, label='1.5')
    ax5.set_xlabel('Ratio Value')
    ax5.set_ylabel('Density')
    ax5.set_title('Distribution of CoG Ratios')
    ax5.legend(fontsize=8)
    ax5.set_xlim(0.5, 4)
    
    ax6 = axes[1, 2]
    if peak_results:
        df = pd.DataFrame(peak_results)
        df['method'] = 'Peak'
    if cog_results:
        df2 = pd.DataFrame(cog_results)
        df2['method'] = 'CoG'
        if 'df' in dir():
            df = pd.concat([df, df2])
        else:
            df = df2
    if 'df' in dir():
        for method in ['Peak', 'CoG']:
            subset = df[df['method'] == method]
            marker = 'o' if method == 'Peak' else 's'
            ax6.scatter(subset['distance'], subset['p_value'], 
                       label=method, marker=marker, s=100, alpha=0.7)
            for _, row in subset.iterrows():
                ax6.annotate(row['ratio'], (row['distance'], row['p_value']),
                            fontsize=8, alpha=0.8)
        ax6.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='p=0.05')
        ax6.set_xlabel('Distance from φ')
        ax6.set_ylabel('p-value')
        ax6.set_title('Statistical Significance vs Distance from φ')
        ax6.legend()
        ax6.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('detailed_phi_analysis_results.png', dpi=150, bbox_inches='tight')
    print("\n\nVisualization saved to: detailed_phi_analysis_results.png")
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    all_results = []
    for r in peak_results:
        all_results.append({**r, 'method': 'Peak'})
    for r in cog_results:
        all_results.append({**r, 'method': 'CoG'})
    
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n", df[['method', 'ratio', 'n', 'mean', 'distance', 'pct_error', 'p_value', 'phi_in_ci']].to_string(index=False))
        df.to_csv('detailed_phi_results.csv', index=False)
        print("\nResults saved to: detailed_phi_results.csv")
    
    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    
    total_peak = len(peak_results)
    phi_peak = sum(1 for r in peak_results if r['phi_in_ci'])
    total_cog = len(cog_results)
    phi_cog = sum(1 for r in cog_results if r['phi_in_ci'])
    
    print(f"\nPeak Frequency Method: {phi_peak}/{total_peak} ratios have φ within 95% CI")
    print(f"Center-of-Gravity Method: {phi_cog}/{total_cog} ratios have φ within 95% CI")
    
    if peak_results:
        closest_peak = min(peak_results, key=lambda x: x['distance'])
        print(f"\nClosest to φ (Peak): {closest_peak['ratio']} = {closest_peak['mean']:.4f} (error: {closest_peak['pct_error']:.1f}%)")
    
    if cog_results:
        closest_cog = min(cog_results, key=lambda x: x['distance'])
        print(f"Closest to φ (CoG): {closest_cog['ratio']} = {closest_cog['mean']:.4f} (error: {closest_cog['pct_error']:.1f}%)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
