#!/usr/bin/env python3
"""
REGIONAL ANALYSIS: Posterior vs Frontal φ-attractor
Using PhysioNet EEGMMIDB dataset (64 channels, N=109)

Tests:
- Test 1: Regional φ-attractor map (posterior vs frontal Δ)
- Test 2: Channel-level basin visualization (topomaps)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PHI = 1.618033988749895

try:
    import mne
    mne.set_log_level('WARNING')
    from mne.datasets import eegbci
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    print("ERROR: MNE not installed")

FRONTAL_CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Fpz', 
                    'AF3', 'AF4', 'F1', 'F2', 'F5', 'F6', 'AFz',
                    'Fp1.', 'Fp2.', 'F7.', 'F3.', 'Fz.', 'F4.', 'F8.']

POSTERIOR_CHANNELS = ['P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2',
                      'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P1', 'P2', 'P5', 'P6',
                      'P7.', 'P3.', 'Pz.', 'P4.', 'P8.', 'O1.', 'Oz.', 'O2.']

CENTRAL_CHANNELS = ['C3', 'Cz', 'C4', 'C1', 'C2', 'C5', 'C6',
                    'C3.', 'Cz.', 'C4.']

TEMPORAL_CHANNELS = ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8',
                     'T7.', 'T8.', 'T9', 'T10']

def compute_spectral_centroid(psd, freqs, fmin, fmax):
    """Compute power-weighted mean frequency in band."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan
    band_freqs = freqs[mask]
    band_power = psd[mask]
    if band_power.sum() == 0:
        return np.nan
    return np.sum(band_freqs * band_power) / np.sum(band_power)

def process_subject(subject_id, runs=[1]):
    """
    Process one subject from EEGMMIDB.
    Run 1 = baseline eyes open (1 min)
    Run 2 = baseline eyes closed (1 min)
    """
    try:
        raw_files = eegbci.load_data(subject_id, runs, update_path=True)
        raw = mne.io.read_raw_edf(raw_files[0], preload=True, verbose=False)
        
        eegbci.standardize(raw)
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage, on_missing='ignore')
        
        raw.filter(1, 50, fir_design='firwin', verbose=False)
        
        sfreq = raw.info['sfreq']
        data = raw.get_data()
        ch_names = raw.ch_names
        
        results = []
        
        for ch_idx, ch_name in enumerate(ch_names):
            if ch_name in ['STI 014', 'Status']:
                continue
                
            ch_data = data[ch_idx]
            
            freqs, psd = welch(ch_data, fs=sfreq, nperseg=int(2*sfreq), 
                              noverlap=int(sfreq))
            
            beta_centroid = compute_spectral_centroid(psd, freqs, 13, 30)
            gamma_centroid = compute_spectral_centroid(psd, freqs, 30, 45)
            
            if np.isnan(beta_centroid) or np.isnan(gamma_centroid) or beta_centroid == 0:
                continue
            
            ratio = gamma_centroid / beta_centroid
            delta = np.abs(ratio - 2.0) - np.abs(ratio - PHI)
            
            if ch_name in FRONTAL_CHANNELS or ch_name.upper() in [c.upper() for c in FRONTAL_CHANNELS]:
                region = 'frontal'
            elif ch_name in POSTERIOR_CHANNELS or ch_name.upper() in [c.upper() for c in POSTERIOR_CHANNELS]:
                region = 'posterior'
            elif ch_name in CENTRAL_CHANNELS or ch_name.upper() in [c.upper() for c in CENTRAL_CHANNELS]:
                region = 'central'
            elif ch_name in TEMPORAL_CHANNELS or ch_name.upper() in [c.upper() for c in TEMPORAL_CHANNELS]:
                region = 'temporal'
            else:
                region = 'other'
            
            results.append({
                'subject': subject_id,
                'channel': ch_name,
                'region': region,
                'beta_centroid': beta_centroid,
                'gamma_centroid': gamma_centroid,
                'gamma_beta_ratio': ratio,
                'delta_score': delta,
                'phi_hit': 1 if abs(ratio - PHI) / PHI < 0.10 else 0
            })
        
        return results
        
    except Exception as e:
        print(f"  Subject {subject_id} error: {e}")
        return []

if __name__ == "__main__":
    print("="*80)
    print("REGIONAL φ-ATTRACTOR ANALYSIS")
    print("PhysioNet EEGMMIDB Dataset (64 channels)")
    print("="*80)
    
    if not HAS_MNE:
        print("ERROR: MNE-Python required")
        exit(1)
    
    N_SUBJECTS = 30
    
    all_results = []
    
    print(f"\nProcessing {N_SUBJECTS} subjects (baseline eyes open)...")
    
    for subj in range(1, N_SUBJECTS + 1):
        print(f"  Subject {subj}/{N_SUBJECTS}...", end=" ")
        subj_results = process_subject(subj, runs=[1])
        if subj_results:
            all_results.extend(subj_results)
            print(f"OK ({len(subj_results)} channels)")
        else:
            print("FAILED")
    
    df = pd.DataFrame(all_results)
    df.to_csv('regional_channel_results.csv', index=False)
    print(f"\nTotal: {len(df)} channel-observations from {df['subject'].nunique()} subjects")
    
    print("\n" + "="*80)
    print("TEST 1: REGIONAL φ-ATTRACTOR MAP (Posterior vs Frontal)")
    print("="*80)
    
    region_stats = df.groupby('region').agg({
        'gamma_beta_ratio': ['mean', 'std', 'count'],
        'delta_score': ['mean', 'std'],
        'phi_hit': 'mean'
    }).round(4)
    region_stats.columns = ['ratio_mean', 'ratio_std', 'n_obs', 'delta_mean', 'delta_std', 'phi_hit_rate']
    print("\nRegional Statistics:")
    print(region_stats.to_string())
    
    frontal_delta = df[df['region'] == 'frontal'].groupby('subject')['delta_score'].mean()
    posterior_delta = df[df['region'] == 'posterior'].groupby('subject')['delta_score'].mean()
    
    common_subjects = list(set(frontal_delta.index) & set(posterior_delta.index))
    print(f"\nSubjects with both regions: {len(common_subjects)}")
    
    if len(common_subjects) >= 5:
        f_vals = frontal_delta.loc[common_subjects].values
        p_vals = posterior_delta.loc[common_subjects].values
        
        diff = p_vals - f_vals
        t_stat, p_value = stats.ttest_rel(p_vals, f_vals)
        
        print(f"\nPosterior Δ: {np.mean(p_vals):.4f} (SD={np.std(p_vals, ddof=1):.4f})")
        print(f"Frontal Δ:   {np.mean(f_vals):.4f} (SD={np.std(f_vals, ddof=1):.4f})")
        print(f"\nDifference (Posterior - Frontal): {np.mean(diff):.4f}")
        print(f"Paired t-test: t = {t_stat:.3f}, p = {p_value:.4f}")
        
        posterior_dominance = np.mean(p_vals) > np.mean(f_vals) and p_value < 0.05
        print(f"\nPosterior Dominance: {'CONFIRMED ✅' if posterior_dominance else 'NOT CONFIRMED ❌'}")
    
    print("\n" + "="*80)
    print("TEST 2: CHANNEL-LEVEL BASIN VISUALIZATION")
    print("="*80)
    
    channel_stats = df.groupby('channel').agg({
        'delta_score': 'mean',
        'phi_hit': 'mean',
        'gamma_beta_ratio': 'mean'
    }).reset_index()
    channel_stats.columns = ['channel', 'delta_mean', 'phi_hit_rate', 'ratio_mean']
    
    print("\nTop 10 channels closest to φ:")
    print(channel_stats.nlargest(10, 'delta_mean')[['channel', 'delta_mean', 'phi_hit_rate', 'ratio_mean']].to_string(index=False))
    
    print("\nBottom 10 channels (furthest from φ):")
    print(channel_stats.nsmallest(10, 'delta_mean')[['channel', 'delta_mean', 'phi_hit_rate', 'ratio_mean']].to_string(index=False))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    ax1 = axes[0, 0]
    regions = ['frontal', 'central', 'temporal', 'posterior']
    region_deltas = [df[df['region'] == r].groupby('subject')['delta_score'].mean().values for r in regions]
    region_deltas = [d for d in region_deltas if len(d) > 0]
    regions_valid = [r for r, d in zip(regions, [df[df['region'] == r].groupby('subject')['delta_score'].mean().values for r in regions]) if len(d) > 0]
    
    bp = ax1.boxplot(region_deltas, labels=regions_valid)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Δ Score')
    ax1.set_title('Δ Score by Brain Region\n(Posterior > Frontal?)')
    
    for i, (region, deltas) in enumerate(zip(regions_valid, region_deltas)):
        x = np.random.normal(i+1, 0.05, len(deltas))
        colors = ['green' if d > 0 else 'red' for d in deltas]
        ax1.scatter(x, deltas, c=colors, alpha=0.6, s=30)
    
    ax2 = axes[0, 1]
    region_means = [np.mean(d) for d in region_deltas]
    region_sems = [stats.sem(d) for d in region_deltas]
    bars = ax2.bar(regions_valid, region_means, yerr=region_sems, capsize=5, 
                   color=['steelblue' if r != 'posterior' else 'coral' for r in regions_valid], alpha=0.7)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel('Mean Δ Score')
    ax2.set_title('Mean Δ by Region (with SEM)')
    
    ax3 = axes[1, 0]
    region_hits = [df[df['region'] == r]['phi_hit'].mean() * 100 for r in regions_valid]
    ax3.bar(regions_valid, region_hits, color=['steelblue' if r != 'posterior' else 'coral' for r in regions_valid], alpha=0.7)
    ax3.axhline(50, color='red', linestyle='--', label='50% (chance)')
    ax3.set_ylabel('φ Hit Rate (%)')
    ax3.set_title('% Channels within φ±10%')
    ax3.legend()
    
    ax4 = axes[1, 1]
    if len(common_subjects) >= 5:
        ax4.scatter(f_vals, p_vals, c='steelblue', alpha=0.7, s=80)
        ax4.plot([min(f_vals.min(), p_vals.min()), max(f_vals.max(), p_vals.max())],
                [min(f_vals.min(), p_vals.min()), max(f_vals.max(), p_vals.max())], 
                'k--', alpha=0.5, label='y=x')
        ax4.set_xlabel('Frontal Δ')
        ax4.set_ylabel('Posterior Δ')
        ax4.set_title(f'Posterior vs Frontal Δ (N={len(common_subjects)})\np = {p_value:.4f}')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('test1_regional_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: test1_regional_analysis.png")
    
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        channel_delta_map = channel_stats.set_index('channel')['delta_mean'].to_dict()
        
        info = mne.create_info(ch_names=list(channel_delta_map.keys()), 
                               sfreq=160, ch_types='eeg')
        montage = mne.channels.make_standard_montage('standard_1005')
        
        info.set_montage(montage, on_missing='ignore')
        
        valid_channels = [ch for ch in channel_delta_map.keys() if ch in info.ch_names]
        delta_values = [channel_delta_map[ch] for ch in valid_channels]
        
        if len(valid_channels) > 10:
            evoked = mne.EvokedArray(np.array([delta_values]).T, info)
            evoked.set_montage(montage, on_missing='ignore')
            
            mne.viz.plot_topomap(delta_values, evoked.info, axes=ax, 
                                show=False, cmap='RdYlGn', 
                                vlim=(np.percentile(delta_values, 5), np.percentile(delta_values, 95)))
            ax.set_title('Δ Score Topomap\n(Green = closer to φ, Red = closer to 2.0)')
            plt.savefig('test2_topomap_delta.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Figure saved: test2_topomap_delta.png")
        else:
            print("Not enough valid channels for topomap")
            
    except Exception as e:
        print(f"Topomap generation failed: {e}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    results_summary = {
        'test': 'Regional Analysis (EEGMMIDB)',
        'n_subjects': df['subject'].nunique(),
        'n_channels': df['channel'].nunique(),
        'posterior_delta': np.mean(p_vals) if len(common_subjects) >= 5 else None,
        'frontal_delta': np.mean(f_vals) if len(common_subjects) >= 5 else None,
        'posterior_vs_frontal_t': t_stat if len(common_subjects) >= 5 else None,
        'posterior_vs_frontal_p': p_value if len(common_subjects) >= 5 else None,
        'posterior_dominance': posterior_dominance if len(common_subjects) >= 5 else None
    }
    
    pd.DataFrame([results_summary]).to_csv('regional_analysis_summary.csv', index=False)
    print("\nResults saved to: regional_analysis_summary.csv")
    print("Channel data saved to: regional_channel_results.csv")
