#!/usr/bin/env python3
"""
COMPLETE PHI-RATIO ANALYSIS ON GAMEEMO DATASET
===============================================
28 subjects, 14 EEG channels, 4 games (boring, calm, horror, funny)
Emotiv EPOC+ device, 128 Hz sampling rate

Games by arousal level:
- G1: Boring (LOW arousal)
- G2: Calm (LOW arousal)  
- G3: Horror (HIGH arousal)
- G4: Funny (HIGH arousal)

All analyses from the task list:
A. Arousal comparison (LOW vs HIGH)
B. Correlation analysis (phi-distance vs arousal)
C. Valence comparison (positive vs negative)
D. Subject-level analysis
E. Channel/Region topographic analysis
F. Frequency band deep dive
G. Time-course analysis (early vs late)
H. Individual Alpha Frequency (IAF)
I. Cross-frequency coupling (simplified)
J. Aperiodic parameters (1/f slope)
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
import warnings
warnings.filterwarnings('ignore')

PHI = 1.618033988749895
FS = 128

FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}
BAND_ORDER = ['delta', 'theta', 'alpha', 'beta', 'gamma']

CHANNEL_NAMES = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
REGIONS = {
    'frontal': ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'FC5', 'FC6'],
    'temporal': ['T7', 'T8'],
    'parietal': ['P7', 'P8'],
    'occipital': ['O1', 'O2']
}

GAMES = {
    'G1': {'name': 'Boring', 'arousal': 'low', 'valence': 'negative'},
    'G2': {'name': 'Calm', 'arousal': 'low', 'valence': 'positive'},
    'G3': {'name': 'Horror', 'arousal': 'high', 'valence': 'negative'},
    'G4': {'name': 'Funny', 'arousal': 'high', 'valence': 'positive'}
}

BASE_PATH = "gameemo_data/Database for Emotion Recognition System Based on EEG Signals and Various Computer Games - GAMEEMO/GAMEEMO"

def load_gameemo_mat(subject_id, game_id):
    """Load a single GAMEEMO .mat file"""
    subj_folder = f"(S{subject_id:02d})"
    filename = f"S{subject_id:02d}{game_id}AllChannels.mat"
    filepath = os.path.join(BASE_PATH, subj_folder, "Preprocessed EEG Data", ".mat format", filename)
    
    if not os.path.exists(filepath):
        return None
    
    try:
        data = sio.loadmat(filepath)
        for key in data.keys():
            if not key.startswith('__'):
                eeg = data[key]
                if isinstance(eeg, np.ndarray) and eeg.size > 1000:
                    return eeg.astype(np.float64)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return None

def compute_psd(eeg_data, fs=128, nperseg=512):
    """Compute PSD using Welch's method"""
    nperseg = min(nperseg, eeg_data.shape[-1] // 4)
    nperseg = max(nperseg, 128)
    freqs, psd = signal.welch(eeg_data, fs=fs, nperseg=nperseg, noverlap=nperseg//2, axis=-1)
    return freqs, psd

def get_band_power(freqs, psd, band_range):
    """Get band power"""
    mask = (freqs >= band_range[0]) & (freqs <= band_range[1])
    if not mask.any() or len(psd.shape) == 0:
        return np.nan
    
    if len(psd.shape) == 1:
        band_psd = psd[mask]
    else:
        band_psd = psd[..., mask]
    
    if band_psd.size == 0:
        return np.nan
    
    return np.mean(band_psd)

def get_peak_frequency(freqs, psd, band_range):
    """Get peak frequency in band"""
    mask = (freqs >= band_range[0]) & (freqs <= band_range[1])
    if not mask.any():
        return np.nan
    
    band_freqs = freqs[mask]
    if len(psd.shape) == 1:
        band_psd = psd[mask]
    else:
        band_psd = np.mean(psd[..., mask], axis=tuple(range(len(psd.shape)-1)))
    
    if band_psd.size == 0:
        return np.nan
    
    peak_idx = np.argmax(band_psd)
    return band_freqs[peak_idx]

def get_cog_frequency(freqs, psd, band_range):
    """Get center-of-gravity frequency in band"""
    mask = (freqs >= band_range[0]) & (freqs <= band_range[1])
    if not mask.any():
        return np.nan
    
    band_freqs = freqs[mask]
    if len(psd.shape) == 1:
        band_psd = psd[mask]
    else:
        band_psd = np.mean(psd[..., mask], axis=tuple(range(len(psd.shape)-1)))
    
    if np.sum(band_psd) == 0:
        return np.nan
    
    return np.sum(band_freqs * band_psd) / np.sum(band_psd)

def compute_1f_slope(freqs, psd, freq_range=(2, 40)):
    """Compute 1/f aperiodic slope"""
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not mask.any():
        return np.nan
    
    log_freqs = np.log10(freqs[mask])
    if len(psd.shape) == 1:
        log_psd = np.log10(psd[mask] + 1e-10)
    else:
        mean_psd = np.mean(psd[..., mask], axis=tuple(range(len(psd.shape)-1)))
        log_psd = np.log10(mean_psd + 1e-10)
    
    try:
        slope, intercept = np.polyfit(log_freqs, log_psd, 1)
        return -slope
    except:
        return np.nan

def compute_phi_ratios(peaks_or_cog):
    """Compute ratios between adjacent bands"""
    ratios = {}
    for i in range(len(BAND_ORDER) - 1):
        lower = BAND_ORDER[i]
        upper = BAND_ORDER[i + 1]
        if peaks_or_cog[lower] > 0 and peaks_or_cog[upper] > 0:
            ratios[f"{upper}/{lower}"] = peaks_or_cog[upper] / peaks_or_cog[lower]
        else:
            ratios[f"{upper}/{lower}"] = np.nan
    return ratios

def phi_distance(ratio):
    """Calculate distance from phi"""
    if np.isnan(ratio):
        return np.nan
    return abs(ratio - PHI)

def analyze_trial(eeg_data, fs=128):
    """Complete analysis of one trial"""
    if eeg_data is None:
        return None
    
    eeg_data = np.nan_to_num(eeg_data, nan=0.0)
    
    if len(eeg_data.shape) == 1:
        eeg_data = eeg_data.reshape(1, -1)
    elif eeg_data.shape[0] > eeg_data.shape[1]:
        eeg_data = eeg_data.T
    
    freqs, psd = compute_psd(eeg_data, fs=fs)
    
    if len(psd.shape) > 1:
        mean_psd = np.mean(psd, axis=0)
    else:
        mean_psd = psd
    
    results = {
        'powers': {},
        'peaks': {},
        'cog': {},
        'ratios_peak': {},
        'ratios_cog': {},
        'slope_1f': compute_1f_slope(freqs, psd)
    }
    
    for band in BAND_ORDER:
        results['powers'][band] = get_band_power(freqs, mean_psd, FREQ_BANDS[band])
        results['peaks'][band] = get_peak_frequency(freqs, mean_psd, FREQ_BANDS[band])
        results['cog'][band] = get_cog_frequency(freqs, mean_psd, FREQ_BANDS[band])
    
    results['ratios_peak'] = compute_phi_ratios(results['peaks'])
    results['ratios_cog'] = compute_phi_ratios(results['cog'])
    
    for key, val in results['ratios_cog'].items():
        results[f'phi_dist_{key}'] = phi_distance(val)
    
    results['iaf'] = results['peaks']['alpha']
    results['itf'] = results['peaks']['theta']
    if results['iaf'] > 0 and results['itf'] > 0:
        results['iaf_itf_ratio'] = results['iaf'] / results['itf']
    else:
        results['iaf_itf_ratio'] = np.nan
    
    return results

def analyze_by_channel(eeg_data, fs=128):
    """Analyze each channel separately"""
    if eeg_data is None:
        return None
    
    eeg_data = np.nan_to_num(eeg_data, nan=0.0)
    if len(eeg_data.shape) == 1:
        eeg_data = eeg_data.reshape(1, -1)
    elif eeg_data.shape[0] > eeg_data.shape[1]:
        eeg_data = eeg_data.T
    
    n_channels = min(eeg_data.shape[0], len(CHANNEL_NAMES))
    channel_results = {}
    
    for ch_idx in range(n_channels):
        ch_data = eeg_data[ch_idx, :]
        freqs, psd = compute_psd(ch_data, fs=fs)
        
        peaks = {band: get_peak_frequency(freqs, psd, FREQ_BANDS[band]) for band in BAND_ORDER}
        cog = {band: get_cog_frequency(freqs, psd, FREQ_BANDS[band]) for band in BAND_ORDER}
        ratios = compute_phi_ratios(cog)
        
        channel_results[CHANNEL_NAMES[ch_idx]] = {
            'peaks': peaks,
            'cog': cog,
            'ratios': ratios,
            'mean_phi_dist': np.nanmean([phi_distance(r) for r in ratios.values()])
        }
    
    return channel_results

def analyze_time_course(eeg_data, fs=128, segment_sec=60):
    """Analyze early vs late portions of recording"""
    if eeg_data is None:
        return None
    
    eeg_data = np.nan_to_num(eeg_data, nan=0.0)
    if len(eeg_data.shape) == 1:
        eeg_data = eeg_data.reshape(1, -1)
    elif eeg_data.shape[0] > eeg_data.shape[1]:
        eeg_data = eeg_data.T
    
    samples_per_segment = segment_sec * fs
    n_samples = eeg_data.shape[1]
    
    if n_samples < 2 * samples_per_segment:
        return None
    
    early_data = eeg_data[:, :samples_per_segment]
    late_data = eeg_data[:, -samples_per_segment:]
    
    early_results = analyze_trial(early_data, fs)
    late_results = analyze_trial(late_data, fs)
    
    return {'early': early_results, 'late': late_results}

def main():
    print("="*80)
    print("COMPLETE PHI-RATIO ANALYSIS ON GAMEEMO DATASET")
    print("28 subjects × 4 games × 14 channels × 5 minutes each")
    print("="*80)
    
    all_trials = []
    all_channels = []
    all_time_course = []
    
    print("\n" + "-"*60)
    print("LOADING AND ANALYZING DATA")
    print("-"*60)
    
    for subj_id in range(1, 29):
        for game_id in ['G1', 'G2', 'G3', 'G4']:
            eeg = load_gameemo_mat(subj_id, game_id)
            
            if eeg is not None:
                result = analyze_trial(eeg)
                if result:
                    trial_data = {
                        'subject': subj_id,
                        'game': game_id,
                        'game_name': GAMES[game_id]['name'],
                        'arousal': GAMES[game_id]['arousal'],
                        'valence': GAMES[game_id]['valence'],
                        'arousal_level': 1 if GAMES[game_id]['arousal'] == 'low' else 5,
                        'valence_level': 1 if GAMES[game_id]['valence'] == 'negative' else 5,
                    }
                    
                    for band in BAND_ORDER:
                        trial_data[f'power_{band}'] = result['powers'][band]
                        trial_data[f'peak_{band}'] = result['peaks'][band]
                        trial_data[f'cog_{band}'] = result['cog'][band]
                    
                    for ratio_name, ratio_val in result['ratios_cog'].items():
                        trial_data[f'ratio_{ratio_name}'] = ratio_val
                        trial_data[f'phi_dist_{ratio_name}'] = phi_distance(ratio_val)
                    
                    trial_data['slope_1f'] = result['slope_1f']
                    trial_data['iaf'] = result['iaf']
                    trial_data['itf'] = result['itf']
                    trial_data['iaf_itf_ratio'] = result['iaf_itf_ratio']
                    
                    all_trials.append(trial_data)
                
                ch_results = analyze_by_channel(eeg)
                if ch_results:
                    for ch_name, ch_data in ch_results.items():
                        ch_record = {
                            'subject': subj_id,
                            'game': game_id,
                            'arousal': GAMES[game_id]['arousal'],
                            'channel': ch_name,
                        }
                        for region, channels in REGIONS.items():
                            if ch_name in channels:
                                ch_record['region'] = region
                                break
                        for ratio_name, ratio_val in ch_data['ratios'].items():
                            ch_record[f'ratio_{ratio_name}'] = ratio_val
                        ch_record['mean_phi_dist'] = ch_data['mean_phi_dist']
                        all_channels.append(ch_record)
                
                tc_results = analyze_time_course(eeg)
                if tc_results:
                    for period in ['early', 'late']:
                        tc_record = {
                            'subject': subj_id,
                            'game': game_id,
                            'arousal': GAMES[game_id]['arousal'],
                            'period': period,
                        }
                        for ratio_name, ratio_val in tc_results[period]['ratios_cog'].items():
                            tc_record[f'ratio_{ratio_name}'] = ratio_val
                        all_time_course.append(tc_record)
        
        if subj_id % 7 == 0:
            print(f"  Processed {subj_id}/28 subjects...")
    
    print(f"  Total trials analyzed: {len(all_trials)}")
    
    df = pd.DataFrame(all_trials)
    df_channels = pd.DataFrame(all_channels)
    df_time = pd.DataFrame(all_time_course)
    
    df.to_csv('gameemo_all_trials.csv', index=False)
    print("\nSaved: gameemo_all_trials.csv")
    
    print("\n" + "="*80)
    print("A. AROUSAL COMPARISON (LOW vs HIGH)")
    print("="*80)
    
    low_arousal = df[df['arousal'] == 'low']
    high_arousal = df[df['arousal'] == 'high']
    
    ratio_cols = [c for c in df.columns if c.startswith('ratio_')]
    arousal_results = []
    
    for col in ratio_cols:
        low_vals = low_arousal[col].dropna()
        high_vals = high_arousal[col].dropna()
        
        if len(low_vals) > 3 and len(high_vals) > 3:
            t_stat, p_val = stats.ttest_ind(low_vals, high_vals)
            cohens_d = (low_vals.mean() - high_vals.mean()) / np.sqrt((low_vals.std()**2 + high_vals.std()**2) / 2)
            
            result = {
                'ratio': col.replace('ratio_', ''),
                'low_mean': low_vals.mean(),
                'low_std': low_vals.std(),
                'high_mean': high_vals.mean(),
                'high_std': high_vals.std(),
                'low_phi_dist': abs(low_vals.mean() - PHI),
                'high_phi_dist': abs(high_vals.mean() - PHI),
                't_stat': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'closer_to_phi': 'LOW' if abs(low_vals.mean() - PHI) < abs(high_vals.mean() - PHI) else 'HIGH'
            }
            arousal_results.append(result)
            
            print(f"\n{col.replace('ratio_', '').upper()}:")
            print(f"  LOW arousal:  {low_vals.mean():.4f} ± {low_vals.std():.4f}")
            print(f"  HIGH arousal: {high_vals.mean():.4f} ± {high_vals.std():.4f}")
            print(f"  t = {t_stat:.3f}, p = {p_val:.4f}, d = {cohens_d:.3f}")
            print(f"  Closer to φ: {result['closer_to_phi']}")
    
    df_arousal = pd.DataFrame(arousal_results)
    
    print("\n" + "="*80)
    print("B. CORRELATION: PHI-DISTANCE vs AROUSAL")
    print("="*80)
    
    phi_dist_cols = [c for c in df.columns if c.startswith('phi_dist_')]
    corr_results = []
    
    for col in phi_dist_cols:
        valid_data = df[[col, 'arousal_level']].dropna()
        if len(valid_data) > 10:
            r_pearson, p_pearson = stats.pearsonr(valid_data[col], valid_data['arousal_level'])
            r_spearman, p_spearman = stats.spearmanr(valid_data[col], valid_data['arousal_level'])
            
            result = {
                'metric': col,
                'pearson_r': r_pearson,
                'pearson_p': p_pearson,
                'spearman_r': r_spearman,
                'spearman_p': p_spearman
            }
            corr_results.append(result)
            
            print(f"\n{col}:")
            print(f"  Pearson r = {r_pearson:.3f}, p = {p_pearson:.4f}")
            print(f"  Spearman ρ = {r_spearman:.3f}, p = {p_spearman:.4f}")
    
    df_corr = pd.DataFrame(corr_results)
    
    print("\n" + "="*80)
    print("C. VALENCE COMPARISON (POSITIVE vs NEGATIVE)")
    print("="*80)
    
    pos_valence = df[df['valence'] == 'positive']
    neg_valence = df[df['valence'] == 'negative']
    
    valence_results = []
    for col in ratio_cols:
        pos_vals = pos_valence[col].dropna()
        neg_vals = neg_valence[col].dropna()
        
        if len(pos_vals) > 3 and len(neg_vals) > 3:
            t_stat, p_val = stats.ttest_ind(pos_vals, neg_vals)
            cohens_d = (pos_vals.mean() - neg_vals.mean()) / np.sqrt((pos_vals.std()**2 + neg_vals.std()**2) / 2)
            
            result = {
                'ratio': col.replace('ratio_', ''),
                'positive_mean': pos_vals.mean(),
                'negative_mean': neg_vals.mean(),
                't_stat': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d
            }
            valence_results.append(result)
            
            print(f"\n{col.replace('ratio_', '').upper()}:")
            print(f"  POSITIVE: {pos_vals.mean():.4f} ± {pos_vals.std():.4f}")
            print(f"  NEGATIVE: {neg_vals.mean():.4f} ± {neg_vals.std():.4f}")
            print(f"  t = {t_stat:.3f}, p = {p_val:.4f}, d = {cohens_d:.3f}")
    
    print("\n" + "="*80)
    print("D. SUBJECT-LEVEL ANALYSIS")
    print("="*80)
    
    subject_means = df.groupby('subject').agg({
        **{col: 'mean' for col in ratio_cols},
        **{col: 'mean' for col in phi_dist_cols},
        'iaf': 'mean',
        'iaf_itf_ratio': 'mean'
    }).reset_index()
    
    subject_means.to_csv('gameemo_subject_means.csv', index=False)
    print("Saved: gameemo_subject_means.csv")
    
    for col in ['ratio_alpha/theta', 'iaf_itf_ratio']:
        if col in subject_means.columns:
            vals = subject_means[col].dropna()
            phi_subjects = vals[(vals > PHI - 0.2) & (vals < PHI + 0.2)]
            print(f"\n{col}: {len(phi_subjects)}/{len(vals)} subjects have ratio within ±0.2 of φ")
    
    print("\n" + "="*80)
    print("E. CHANNEL/REGION TOPOGRAPHIC ANALYSIS")
    print("="*80)
    
    if 'region' in df_channels.columns:
        region_means = df_channels.groupby(['region', 'arousal'])['mean_phi_dist'].mean().unstack()
        print("\nMean phi-distance by region and arousal:")
        print(region_means)
    
    print("\n" + "="*80)
    print("F. FREQUENCY BAND DEEP DIVE")
    print("="*80)
    
    test_constants = {'phi': PHI, '2.0': 2.0, '1.5': 1.5, '1.0': 1.0}
    band_tests = []
    
    for col in ratio_cols:
        vals = df[col].dropna()
        if len(vals) > 10:
            ratio_name = col.replace('ratio_', '')
            print(f"\n{ratio_name.upper()}:")
            print(f"  Mean = {vals.mean():.4f}, Median = {vals.median():.4f}, Std = {vals.std():.4f}")
            
            best_fit = None
            best_dist = float('inf')
            
            for const_name, const_val in test_constants.items():
                t_stat, p_val = stats.ttest_1samp(vals, const_val)
                distance = abs(vals.mean() - const_val)
                
                if distance < best_dist:
                    best_dist = distance
                    best_fit = const_name
                
                band_tests.append({
                    'ratio': ratio_name,
                    'constant': const_name,
                    'constant_value': const_val,
                    'distance': distance,
                    't_stat': t_stat,
                    'p_value': p_val
                })
                
                print(f"  vs {const_name} ({const_val}): distance={distance:.4f}, p={p_val:.4f}")
            
            print(f"  BEST FIT: {best_fit} (distance = {best_dist:.4f})")
    
    print("\n" + "="*80)
    print("G. TIME-COURSE ANALYSIS (EARLY vs LATE)")
    print("="*80)
    
    if len(df_time) > 0:
        early_df = df_time[df_time['period'] == 'early']
        late_df = df_time[df_time['period'] == 'late']
        
        for col in [c for c in df_time.columns if c.startswith('ratio_')]:
            early_vals = early_df[col].dropna()
            late_vals = late_df[col].dropna()
            
            if len(early_vals) > 3 and len(late_vals) > 3:
                t_stat, p_val = stats.ttest_rel(early_vals[:min(len(early_vals), len(late_vals))], 
                                                late_vals[:min(len(early_vals), len(late_vals))])
                print(f"\n{col.replace('ratio_', '').upper()}:")
                print(f"  EARLY: {early_vals.mean():.4f} ± {early_vals.std():.4f}")
                print(f"  LATE:  {late_vals.mean():.4f} ± {late_vals.std():.4f}")
                print(f"  Paired t = {t_stat:.3f}, p = {p_val:.4f}")
    
    print("\n" + "="*80)
    print("H. INDIVIDUAL ALPHA FREQUENCY (IAF) ANALYSIS")
    print("="*80)
    
    iaf_vals = df['iaf'].dropna()
    itf_vals = df['itf'].dropna()
    iaf_itf = df['iaf_itf_ratio'].dropna()
    
    print(f"\nIAF: {iaf_vals.mean():.2f} ± {iaf_vals.std():.2f} Hz")
    print(f"ITF: {itf_vals.mean():.2f} ± {itf_vals.std():.2f} Hz")
    print(f"IAF/ITF ratio: {iaf_itf.mean():.4f} ± {iaf_itf.std():.4f}")
    print(f"Distance from φ: {abs(iaf_itf.mean() - PHI):.4f} ({100*abs(iaf_itf.mean() - PHI)/PHI:.1f}%)")
    
    t_stat, p_val = stats.ttest_1samp(iaf_itf, PHI)
    print(f"t-test vs φ: t = {t_stat:.3f}, p = {p_val:.4f}")
    
    print("\n" + "="*80)
    print("J. APERIODIC (1/f) ANALYSIS")
    print("="*80)
    
    slope_vals = df['slope_1f'].dropna()
    print(f"\n1/f slope: {slope_vals.mean():.3f} ± {slope_vals.std():.3f}")
    
    r, p = stats.pearsonr(df['slope_1f'].dropna(), df.loc[df['slope_1f'].notna(), 'arousal_level'])
    print(f"Correlation with arousal: r = {r:.3f}, p = {p:.4f}")
    
    for col in phi_dist_cols[:2]:
        valid = df[[col, 'slope_1f']].dropna()
        if len(valid) > 10:
            r, p = stats.pearsonr(valid[col], valid['slope_1f'])
            print(f"Correlation {col} with 1/f slope: r = {r:.3f}, p = {p:.4f}")
    
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    ax = axes[0, 0]
    data_to_plot = []
    labels = []
    for col in ratio_cols[:4]:
        data_to_plot.append([low_arousal[col].dropna(), high_arousal[col].dropna()])
        labels.append(col.replace('ratio_', ''))
    
    positions = []
    for i, (low_data, high_data) in enumerate(data_to_plot):
        bp1 = ax.boxplot([low_data], positions=[i*3], widths=0.8, patch_artist=True)
        bp2 = ax.boxplot([high_data], positions=[i*3+1], widths=0.8, patch_artist=True)
        bp1['boxes'][0].set_facecolor('lightblue')
        bp2['boxes'][0].set_facecolor('salmon')
        positions.extend([i*3, i*3+1])
    
    ax.axhline(y=PHI, color='gold', linestyle='--', linewidth=2, label=f'φ = {PHI:.3f}')
    ax.set_xticks([i*3+0.5 for i in range(4)])
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Ratio')
    ax.set_title('Fig 1: Arousal Comparison')
    ax.legend(['φ', 'Low Arousal', 'High Arousal'])
    
    ax = axes[0, 1]
    scatter_col = 'phi_dist_alpha/theta' if 'phi_dist_alpha/theta' in df.columns else phi_dist_cols[0]
    valid_scatter = df[[scatter_col, 'arousal_level']].dropna()
    ax.scatter(valid_scatter['arousal_level'], valid_scatter[scatter_col], alpha=0.5)
    z = np.polyfit(valid_scatter['arousal_level'], valid_scatter[scatter_col], 1)
    p = np.poly1d(z)
    ax.plot([1, 5], [p(1), p(5)], 'r-', linewidth=2)
    ax.set_xlabel('Arousal Level')
    ax.set_ylabel('Phi Distance')
    ax.set_title('Fig 2: Phi-Distance vs Arousal')
    
    ax = axes[0, 2]
    for col in ratio_cols[:4]:
        vals = df[col].dropna()
        ax.hist(vals, bins=20, alpha=0.4, label=col.replace('ratio_', ''), density=True)
    ax.axvline(x=PHI, color='gold', linestyle='--', linewidth=2, label=f'φ = {PHI:.3f}')
    ax.axvline(x=2.0, color='gray', linestyle=':', label='2.0')
    ax.set_xlabel('Ratio')
    ax.set_ylabel('Density')
    ax.set_title('Fig 3: Ratio Distributions')
    ax.legend(fontsize=8)
    ax.set_xlim(0.5, 4)
    
    ax = axes[0, 3]
    if 'region' in df_channels.columns:
        region_phi = df_channels.groupby('region')['mean_phi_dist'].mean()
        regions = list(region_phi.index)
        values = list(region_phi.values)
        colors = plt.cm.viridis(np.linspace(0, 1, len(regions)))
        ax.barh(regions, values, color=colors)
        ax.set_xlabel('Mean Phi Distance')
        ax.set_title('Fig 4: Topographic Phi-Distance')
    
    ax = axes[1, 0]
    if len(df_time) > 0:
        early_means = df_time[df_time['period'] == 'early'].mean(numeric_only=True)
        late_means = df_time[df_time['period'] == 'late'].mean(numeric_only=True)
        ratio_names = [c.replace('ratio_', '') for c in df_time.columns if c.startswith('ratio_')]
        x = np.arange(len(ratio_names))
        width = 0.35
        ax.bar(x - width/2, [early_means.get(f'ratio_{r}', 0) for r in ratio_names], width, label='Early', color='lightgreen')
        ax.bar(x + width/2, [late_means.get(f'ratio_{r}', 0) for r in ratio_names], width, label='Late', color='darkgreen')
        ax.axhline(y=PHI, color='gold', linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(ratio_names, rotation=45)
        ax.set_ylabel('Ratio')
        ax.set_title('Fig 5: Early vs Late')
        ax.legend()
    
    ax = axes[1, 1]
    iaf_itf_vals = df['iaf_itf_ratio'].dropna()
    ax.hist(iaf_itf_vals, bins=20, color='steelblue', alpha=0.7, density=True)
    ax.axvline(x=PHI, color='gold', linestyle='--', linewidth=2, label=f'φ = {PHI:.3f}')
    ax.axvline(x=iaf_itf_vals.mean(), color='red', linestyle='-', linewidth=2, label=f'Mean = {iaf_itf_vals.mean():.3f}')
    ax.set_xlabel('IAF/ITF Ratio')
    ax.set_ylabel('Density')
    ax.set_title('Fig 6: IAF/ITF Ratio Distribution')
    ax.legend()
    
    ax = axes[1, 2]
    ax.text(0.5, 0.5, 'CFC Analysis\n(requires more computation)', 
            ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.set_title('Fig 7: Cross-Frequency Coupling')
    ax.axis('off')
    
    ax = axes[1, 3]
    valid_slope = df[['slope_1f', 'arousal_level']].dropna()
    ax.scatter(valid_slope['arousal_level'], valid_slope['slope_1f'], alpha=0.5)
    z = np.polyfit(valid_slope['arousal_level'], valid_slope['slope_1f'], 1)
    p = np.poly1d(z)
    ax.plot([1, 5], [p(1), p(5)], 'r-', linewidth=2)
    ax.set_xlabel('Arousal Level')
    ax.set_ylabel('1/f Slope')
    ax.set_title('Fig 8: Aperiodic Slope vs Arousal')
    
    plt.tight_layout()
    plt.savefig('gameemo_all_figures.png', dpi=150, bbox_inches='tight')
    print("Saved: gameemo_all_figures.png")
    plt.close()
    
    all_stats = []
    for r in arousal_results:
        all_stats.append({'test': 'arousal_comparison', 'variable': r['ratio'], **r})
    for r in corr_results:
        all_stats.append({'test': 'correlation', 'variable': r['metric'], **r})
    for r in valence_results:
        all_stats.append({'test': 'valence_comparison', 'variable': r['ratio'], **r})
    
    pd.DataFrame(all_stats).to_csv('gameemo_statistical_tests.csv', index=False)
    print("Saved: gameemo_statistical_tests.csv")
    
    pd.DataFrame(corr_results).to_csv('gameemo_correlations.csv', index=False)
    print("Saved: gameemo_correlations.csv")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nDataset: GAMEEMO (28 subjects × 4 games = {len(df)} trials)")
    print(f"Arousal comparison (LOW vs HIGH): {len(arousal_results)} ratio tests")
    
    sig_arousal = [r for r in arousal_results if r['p_value'] < 0.05]
    print(f"  Significant differences (p<0.05): {len(sig_arousal)}/{len(arousal_results)}")
    
    phi_closer_low = [r for r in arousal_results if r['closer_to_phi'] == 'LOW']
    print(f"  Ratios closer to φ in LOW arousal: {len(phi_closer_low)}/{len(arousal_results)}")
    
    print(f"\nIAF/ITF ratio: {iaf_itf.mean():.4f} (φ = {PHI:.4f}, distance = {abs(iaf_itf.mean() - PHI):.4f})")
    
    print("\n" + "="*80)
    print("FILES GENERATED")
    print("="*80)
    print("  gameemo_all_trials.csv")
    print("  gameemo_subject_means.csv")
    print("  gameemo_statistical_tests.csv")
    print("  gameemo_correlations.csv")
    print("  gameemo_all_figures.png")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
