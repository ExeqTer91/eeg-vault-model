#!/usr/bin/env python3
"""
FAST PHI-RATIO ANALYSIS ON GAMEEMO DATASET - OPTIMIZED VERSION
"""

import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
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

GAMES = {
    'G1': {'name': 'Boring', 'arousal': 'low', 'valence': 'negative'},
    'G2': {'name': 'Calm', 'arousal': 'low', 'valence': 'positive'},
    'G3': {'name': 'Horror', 'arousal': 'high', 'valence': 'negative'},
    'G4': {'name': 'Funny', 'arousal': 'high', 'valence': 'positive'}
}

BASE_PATH = "gameemo_data/Database for Emotion Recognition System Based on EEG Signals and Various Computer Games - GAMEEMO/GAMEEMO"

def load_mat(subj_id, game_id):
    subj_folder = f"(S{subj_id:02d})"
    filename = f"S{subj_id:02d}{game_id}AllChannels.mat"
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
    except:
        pass
    return None

def analyze_trial(eeg):
    if eeg is None:
        return None
    
    eeg = np.nan_to_num(eeg, nan=0.0)
    if len(eeg.shape) == 1:
        eeg = eeg.reshape(1, -1)
    elif eeg.shape[0] > eeg.shape[1]:
        eeg = eeg.T
    
    nperseg = min(256, eeg.shape[-1] // 4)
    nperseg = max(nperseg, 64)
    freqs, psd = signal.welch(eeg, fs=FS, nperseg=nperseg, axis=-1)
    mean_psd = np.mean(psd, axis=0) if len(psd.shape) > 1 else psd
    
    peaks = {}
    cog = {}
    for band, (low, high) in FREQ_BANDS.items():
        mask = (freqs >= low) & (freqs <= high)
        if mask.any():
            band_freqs = freqs[mask]
            band_psd = mean_psd[mask]
            peaks[band] = band_freqs[np.argmax(band_psd)]
            cog[band] = np.sum(band_freqs * band_psd) / (np.sum(band_psd) + 1e-10)
        else:
            peaks[band] = np.nan
            cog[band] = np.nan
    
    ratios = {}
    for i in range(len(BAND_ORDER) - 1):
        lower, upper = BAND_ORDER[i], BAND_ORDER[i+1]
        if cog[lower] > 0 and cog[upper] > 0:
            ratios[f"{upper}/{lower}"] = cog[upper] / cog[lower]
        else:
            ratios[f"{upper}/{lower}"] = np.nan
    
    log_freqs = np.log10(freqs[1:40])
    log_psd = np.log10(mean_psd[1:40] + 1e-10)
    try:
        slope = -np.polyfit(log_freqs, log_psd, 1)[0]
    except:
        slope = np.nan
    
    return {
        'peaks': peaks, 'cog': cog, 'ratios': ratios, 'slope': slope,
        'iaf': peaks['alpha'], 'itf': peaks['theta'],
        'iaf_itf': peaks['alpha'] / peaks['theta'] if peaks['theta'] > 0 and peaks['alpha'] > 0 else np.nan
    }

def main():
    print("="*70)
    print("GAMEEMO PHI-RATIO ANALYSIS (OPTIMIZED)")
    print("="*70)
    
    all_trials = []
    
    for subj_id in range(1, 29):
        for game_id in ['G1', 'G2', 'G3', 'G4']:
            eeg = load_mat(subj_id, game_id)
            result = analyze_trial(eeg)
            
            if result:
                trial = {
                    'subject': subj_id,
                    'game': game_id,
                    'arousal': GAMES[game_id]['arousal'],
                    'valence': GAMES[game_id]['valence'],
                    'arousal_num': 1 if GAMES[game_id]['arousal'] == 'low' else 5,
                }
                for band in BAND_ORDER:
                    trial[f'cog_{band}'] = result['cog'][band]
                for ratio_name, ratio_val in result['ratios'].items():
                    trial[f'ratio_{ratio_name}'] = ratio_val
                    trial[f'phi_dist_{ratio_name}'] = abs(ratio_val - PHI) if not np.isnan(ratio_val) else np.nan
                
                trial['slope'] = result['slope']
                trial['iaf'] = result['iaf']
                trial['itf'] = result['itf']
                trial['iaf_itf'] = result['iaf_itf']
                all_trials.append(trial)
        
        if subj_id % 7 == 0:
            print(f"  Processed {subj_id}/28 subjects...")
    
    df = pd.DataFrame(all_trials)
    df.to_csv('gameemo_all_trials.csv', index=False)
    print(f"\nTotal trials: {len(df)}")
    print("Saved: gameemo_all_trials.csv")
    
    print("\n" + "="*70)
    print("A. AROUSAL COMPARISON (LOW vs HIGH)")
    print("="*70)
    
    low = df[df['arousal'] == 'low']
    high = df[df['arousal'] == 'high']
    
    ratio_cols = [c for c in df.columns if c.startswith('ratio_')]
    arousal_results = []
    
    for col in ratio_cols:
        low_vals = low[col].dropna()
        high_vals = high[col].dropna()
        
        if len(low_vals) > 3 and len(high_vals) > 3:
            t, p = stats.ttest_ind(low_vals, high_vals)
            d = (low_vals.mean() - high_vals.mean()) / np.sqrt((low_vals.var() + high_vals.var()) / 2)
            
            closer = 'LOW' if abs(low_vals.mean() - PHI) < abs(high_vals.mean() - PHI) else 'HIGH'
            
            arousal_results.append({
                'ratio': col.replace('ratio_', ''),
                'low_mean': low_vals.mean(),
                'high_mean': high_vals.mean(),
                'low_phi_dist': abs(low_vals.mean() - PHI),
                'high_phi_dist': abs(high_vals.mean() - PHI),
                't': t, 'p': p, 'd': d, 'closer_to_phi': closer
            })
            
            sig = '*' if p < 0.05 else ''
            print(f"\n{col.replace('ratio_', '').upper()}:{sig}")
            print(f"  LOW:  {low_vals.mean():.4f} | HIGH: {high_vals.mean():.4f}")
            print(f"  t={t:.2f}, p={p:.4f}, d={d:.2f} | Closer to φ: {closer}")
    
    print("\n" + "="*70)
    print("B. CORRELATION: PHI-DISTANCE vs AROUSAL")
    print("="*70)
    
    phi_cols = [c for c in df.columns if c.startswith('phi_dist_')]
    for col in phi_cols:
        valid = df[[col, 'arousal_num']].dropna()
        if len(valid) > 10:
            r, p = stats.pearsonr(valid[col], valid['arousal_num'])
            sig = '*' if p < 0.05 else ''
            print(f"{col}: r={r:.3f}, p={p:.4f}{sig}")
    
    print("\n" + "="*70)
    print("C. VALENCE COMPARISON")
    print("="*70)
    
    pos = df[df['valence'] == 'positive']
    neg = df[df['valence'] == 'negative']
    
    for col in ratio_cols:
        pos_vals = pos[col].dropna()
        neg_vals = neg[col].dropna()
        if len(pos_vals) > 3 and len(neg_vals) > 3:
            t, p = stats.ttest_ind(pos_vals, neg_vals)
            sig = '*' if p < 0.05 else ''
            print(f"{col.replace('ratio_', '')}: pos={pos_vals.mean():.3f}, neg={neg_vals.mean():.3f}, p={p:.4f}{sig}")
    
    print("\n" + "="*70)
    print("F. FREQUENCY BAND DEEP DIVE")
    print("="*70)
    
    constants = {'φ': PHI, '2.0': 2.0, '1.5': 1.5}
    
    for col in ratio_cols:
        vals = df[col].dropna()
        if len(vals) > 10:
            print(f"\n{col.replace('ratio_', '').upper()}: mean={vals.mean():.4f}")
            best_const = None
            best_dist = float('inf')
            for name, val in constants.items():
                dist = abs(vals.mean() - val)
                if dist < best_dist:
                    best_dist = dist
                    best_const = name
                t, p = stats.ttest_1samp(vals, val)
                sig = '' if p < 0.05 else ' (not sig different)'
                print(f"  vs {name}: dist={dist:.4f}, p={p:.4f}{sig}")
            print(f"  → BEST FIT: {best_const}")
    
    print("\n" + "="*70)
    print("H. IAF/ITF RATIO ANALYSIS")
    print("="*70)
    
    iaf_itf = df['iaf_itf'].dropna()
    print(f"\nIAF/ITF ratio: {iaf_itf.mean():.4f} ± {iaf_itf.std():.4f}")
    print(f"φ = {PHI:.4f}")
    print(f"Distance from φ: {abs(iaf_itf.mean() - PHI):.4f} ({100*abs(iaf_itf.mean() - PHI)/PHI:.1f}%)")
    t, p = stats.ttest_1samp(iaf_itf, PHI)
    print(f"t-test vs φ: t={t:.2f}, p={p:.4f}")
    
    print("\n" + "="*70)
    print("J. APERIODIC SLOPE vs AROUSAL")
    print("="*70)
    
    valid = df[['slope', 'arousal_num']].dropna()
    r, p = stats.pearsonr(valid['slope'], valid['arousal_num'])
    print(f"1/f slope vs arousal: r={r:.3f}, p={p:.4f}")
    
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    ax = axes[0, 0]
    for i, col in enumerate(ratio_cols[:4]):
        low_vals = low[col].dropna()
        high_vals = high[col].dropna()
        bp1 = ax.boxplot([low_vals], positions=[i*2], widths=0.6, patch_artist=True)
        bp2 = ax.boxplot([high_vals], positions=[i*2+0.7], widths=0.6, patch_artist=True)
        bp1['boxes'][0].set_facecolor('lightblue')
        bp2['boxes'][0].set_facecolor('salmon')
    ax.axhline(y=PHI, color='gold', linestyle='--', linewidth=2)
    ax.set_xticks([i*2+0.35 for i in range(4)])
    ax.set_xticklabels([c.replace('ratio_', '') for c in ratio_cols[:4]], rotation=45)
    ax.set_title('Fig 1: LOW vs HIGH Arousal')
    ax.set_ylabel('Ratio')
    
    ax = axes[0, 1]
    scatter_col = phi_cols[1] if len(phi_cols) > 1 else phi_cols[0]
    valid = df[[scatter_col, 'arousal_num']].dropna()
    ax.scatter(valid['arousal_num'] + np.random.normal(0, 0.1, len(valid)), valid[scatter_col], alpha=0.3)
    z = np.polyfit(valid['arousal_num'], valid[scatter_col], 1)
    ax.plot([1, 5], [np.poly1d(z)(1), np.poly1d(z)(5)], 'r-', linewidth=2)
    ax.set_xlabel('Arousal Level')
    ax.set_ylabel('Phi Distance')
    ax.set_title('Fig 2: Phi-Distance vs Arousal')
    
    ax = axes[0, 2]
    for col in ratio_cols[:4]:
        ax.hist(df[col].dropna(), bins=20, alpha=0.3, label=col.replace('ratio_', ''), density=True)
    ax.axvline(x=PHI, color='gold', linestyle='--', linewidth=2, label='φ')
    ax.axvline(x=2.0, color='gray', linestyle=':', label='2.0')
    ax.set_xlim(0.5, 4)
    ax.legend(fontsize=7)
    ax.set_title('Fig 3: Ratio Distributions')
    
    ax = axes[0, 3]
    means = [abs(df[col].mean() - PHI) for col in ratio_cols]
    ax.barh([c.replace('ratio_', '') for c in ratio_cols], means, color='steelblue')
    ax.set_xlabel('Distance from φ')
    ax.set_title('Fig 4: Mean Phi-Distance by Ratio')
    
    ax = axes[1, 0]
    by_game = df.groupby('game')['ratio_alpha/theta'].mean()
    colors = ['blue' if GAMES[g]['arousal'] == 'low' else 'red' for g in by_game.index]
    ax.bar(by_game.index, by_game.values, color=colors, alpha=0.7)
    ax.axhline(y=PHI, color='gold', linestyle='--')
    ax.set_title('Fig 5: Alpha/Theta by Game')
    ax.set_ylabel('Ratio')
    
    ax = axes[1, 1]
    ax.hist(iaf_itf, bins=20, color='steelblue', alpha=0.7, density=True)
    ax.axvline(x=PHI, color='gold', linestyle='--', linewidth=2, label=f'φ={PHI:.2f}')
    ax.axvline(x=iaf_itf.mean(), color='red', linestyle='-', linewidth=2, label=f'mean={iaf_itf.mean():.2f}')
    ax.legend()
    ax.set_title('Fig 6: IAF/ITF Distribution')
    
    ax = axes[1, 2]
    valid = df[['slope', 'arousal_num']].dropna()
    ax.scatter(valid['arousal_num'] + np.random.normal(0, 0.1, len(valid)), valid['slope'], alpha=0.3)
    ax.set_xlabel('Arousal Level')
    ax.set_ylabel('1/f Slope')
    ax.set_title('Fig 8: Aperiodic Slope vs Arousal')
    
    ax = axes[1, 3]
    summary_text = f"""SUMMARY RESULTS

Trials analyzed: {len(df)}
Subjects: 28
Games: 4 (2 low, 2 high arousal)

AROUSAL EFFECT:
"""
    for r in arousal_results:
        sig = '*' if r['p'] < 0.05 else ''
        summary_text += f"  {r['ratio']}: closer in {r['closer_to_phi']}{sig}\n"
    
    summary_text += f"\nIAF/ITF: {iaf_itf.mean():.3f} (φ={PHI:.3f})"
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace')
    ax.axis('off')
    ax.set_title('Summary')
    
    plt.tight_layout()
    plt.savefig('gameemo_analysis_figures.png', dpi=150, bbox_inches='tight')
    print("Saved: gameemo_analysis_figures.png")
    plt.close()
    
    pd.DataFrame(arousal_results).to_csv('gameemo_arousal_results.csv', index=False)
    print("Saved: gameemo_arousal_results.csv")
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    phi_closer_low = sum(1 for r in arousal_results if r['closer_to_phi'] == 'LOW')
    sig_results = sum(1 for r in arousal_results if r['p'] < 0.05)
    
    print(f"\nTested {len(arousal_results)} frequency ratios:")
    print(f"  - {phi_closer_low}/{len(arousal_results)} closer to φ in LOW arousal")
    print(f"  - {sig_results}/{len(arousal_results)} significantly different between conditions")
    print(f"\nIAF/ITF ratio: {iaf_itf.mean():.4f} ({100*abs(iaf_itf.mean() - PHI)/PHI:.1f}% from φ)")
    
    if phi_closer_low > len(arousal_results) / 2:
        print("\n→ HYPOTHESIS SUPPORTED: Low arousal shows more φ-organization")
    else:
        print("\n→ HYPOTHESIS NOT SUPPORTED: No clear arousal-φ relationship")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
