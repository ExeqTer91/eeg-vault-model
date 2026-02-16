#!/usr/bin/env python3
"""
PHASE MODEL ANALYSIS: 6-7 Discrete States for Inner Awareness
Testing "broad inner awareness ↔ laser focus" hypothesis

Uses PhysioNet EEGMMIDB with:
- Baseline eyes-open (Run 1)
- Baseline eyes-closed (Run 2)
- Task conditions (Runs 3+)

Approaches:
A) Feature extraction per epoch per channel
B) State discovery: HMM, GMM/k-means, change-point
C) State interpretation: broad vs laser proxies
D) Hypothesis tests: H_phase, H_access, H_topography_shift
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from collections import Counter
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

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: sklearn not installed - using simplified methods")

try:
    from hmmlearn import hmm
    HAS_HMM = True
except ImportError:
    HAS_HMM = False
    print("WARNING: hmmlearn not installed - skipping HMM")

FRONTAL = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'AF3', 'AF4', 'AF7', 'AF8']
CENTRAL = ['C3', 'Cz', 'C4', 'FC1', 'FC2', 'FC5', 'FC6']
TEMPORAL = ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8']
POSTERIOR = ['P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2', 'PO3', 'PO4', 'PO7', 'PO8', 'POz']

def get_region(ch_name):
    ch_upper = ch_name.upper().replace('.', '')
    for ch in FRONTAL:
        if ch.upper() == ch_upper:
            return 'frontal'
    for ch in CENTRAL:
        if ch.upper() == ch_upper:
            return 'central'
    for ch in TEMPORAL:
        if ch.upper() == ch_upper:
            return 'temporal'
    for ch in POSTERIOR:
        if ch.upper() == ch_upper:
            return 'posterior'
    return 'other'

def compute_spectral_features(data, sfreq):
    """Compute spectral features for one epoch."""
    freqs, psd = welch(data, fs=sfreq, nperseg=min(int(2*sfreq), len(data)), 
                       noverlap=int(sfreq/2) if len(data) > sfreq else 0)
    
    def band_power(fmin, fmax):
        mask = (freqs >= fmin) & (freqs <= fmax)
        return np.mean(psd[mask]) if np.any(mask) else 0
    
    def band_centroid(fmin, fmax):
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask) or psd[mask].sum() == 0:
            return (fmin + fmax) / 2
        return np.sum(freqs[mask] * psd[mask]) / np.sum(psd[mask])
    
    delta_power = band_power(1, 4)
    theta_power = band_power(4, 8)
    alpha_power = band_power(8, 13)
    beta_power = band_power(13, 30)
    gamma_power = band_power(30, 45)
    
    beta_cf = band_centroid(13, 30)
    gamma_cf = band_centroid(30, 45)
    
    log_freqs = np.log10(freqs[1:20])
    log_psd = np.log10(psd[1:20] + 1e-20)
    if len(log_freqs) > 1:
        slope, intercept = np.polyfit(log_freqs, log_psd, 1)
        aperiodic_exponent = -slope
    else:
        aperiodic_exponent = 1.0
    
    if beta_cf > 0:
        r = gamma_cf / beta_cf
    else:
        r = 1.5
    
    delta_score = np.abs(r - 2.0) - np.abs(r - PHI)
    hit_phi = 1 if abs(r - PHI) / PHI < 0.10 else 0
    
    return {
        'delta_power': delta_power,
        'theta_power': theta_power,
        'alpha_power': alpha_power,
        'beta_power': beta_power,
        'gamma_power': gamma_power,
        'beta_cf': beta_cf,
        'gamma_cf': gamma_cf,
        'aperiodic_exponent': aperiodic_exponent,
        'r': r,
        'delta_score': delta_score,
        'hit_phi': hit_phi
    }

def process_subject_epochs(subject_id, runs, epoch_duration=2.0):
    """Process subject with epoching."""
    all_epochs = []
    
    condition_map = {1: 'eyes_open', 2: 'eyes_closed', 3: 'task1', 4: 'task2', 
                     5: 'task3', 6: 'task4', 7: 'task5', 8: 'task6'}
    
    for run in runs:
        try:
            raw_files = eegbci.load_data(subject_id, [run], update_path=True)
            raw = mne.io.read_raw_edf(raw_files[0], preload=True, verbose=False)
            
            eegbci.standardize(raw)
            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage, on_missing='ignore')
            
            raw.filter(1, 50, fir_design='firwin', verbose=False)
            
            sfreq = raw.info['sfreq']
            data = raw.get_data()
            ch_names = raw.ch_names
            
            epoch_samples = int(epoch_duration * sfreq)
            n_epochs = data.shape[1] // epoch_samples
            
            condition = condition_map.get(run, f'run{run}')
            
            for epoch_idx in range(n_epochs):
                start = epoch_idx * epoch_samples
                end = start + epoch_samples
                
                epoch_data = data[:, start:end]
                
                region_features = {r: [] for r in ['frontal', 'central', 'temporal', 'posterior']}
                
                for ch_idx, ch_name in enumerate(ch_names):
                    if ch_name in ['STI 014', 'Status']:
                        continue
                    
                    region = get_region(ch_name)
                    if region == 'other':
                        continue
                    
                    features = compute_spectral_features(epoch_data[ch_idx], sfreq)
                    region_features[region].append(features)
                
                for region, feat_list in region_features.items():
                    if not feat_list:
                        continue
                    
                    agg = {}
                    for key in feat_list[0].keys():
                        values = [f[key] for f in feat_list]
                        agg[key] = np.mean(values)
                    
                    all_epochs.append({
                        'subject': subject_id,
                        'condition': condition,
                        'epoch_id': epoch_idx,
                        'run': run,
                        'region': region,
                        **agg
                    })
                    
        except Exception as e:
            print(f"  Run {run} error: {e}")
            continue
    
    return all_epochs

if __name__ == "__main__":
    print("="*80)
    print("PHASE MODEL ANALYSIS: 6-7 Discrete States")
    print("Testing broad inner awareness ↔ laser focus hypothesis")
    print("="*80)
    
    if not HAS_MNE:
        print("ERROR: MNE-Python required")
        exit(1)
    
    N_SUBJECTS = 15
    RUNS = [1, 2, 3, 4]
    EPOCH_DURATION = 2.0
    
    all_data = []
    
    print(f"\n[A] FEATURE EXTRACTION")
    print(f"    {N_SUBJECTS} subjects, runs {RUNS}, {EPOCH_DURATION}s epochs")
    
    for subj in range(1, N_SUBJECTS + 1):
        print(f"  Subject {subj}/{N_SUBJECTS}...", end=" ")
        epochs = process_subject_epochs(subj, RUNS, EPOCH_DURATION)
        if epochs:
            all_data.extend(epochs)
            print(f"OK ({len(epochs)} region-epochs)")
        else:
            print("FAILED")
    
    df = pd.DataFrame(all_data)
    df.to_csv('epoch_features_full.csv', index=False)
    print(f"\n  Total: {len(df)} region-epochs from {df['subject'].nunique()} subjects")
    print(f"  Saved: epoch_features_full.csv")
    
    print("\n" + "="*80)
    print("[B] STATE DISCOVERY")
    print("="*80)
    
    feature_cols = ['r', 'delta_score', 'aperiodic_exponent', 'alpha_power', 'beta_power']
    
    df_complete = df.dropna(subset=feature_cols)
    X = df_complete[feature_cols].values
    
    if HAS_SKLEARN:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    
    k_range = range(2, 11)
    results = {'k': [], 'method': [], 'score': [], 'silhouette': []}
    
    print("\n[B1] Gaussian Mixture Models (GMM)")
    gmm_bic = []
    gmm_aic = []
    gmm_silhouette = []
    
    if HAS_SKLEARN:
        for k in k_range:
            gmm = GaussianMixture(n_components=k, random_state=42, n_init=5, max_iter=200)
            gmm.fit(X_scaled)
            bic = gmm.bic(X_scaled)
            aic = gmm.aic(X_scaled)
            labels = gmm.predict(X_scaled)
            sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0
            
            gmm_bic.append(bic)
            gmm_aic.append(aic)
            gmm_silhouette.append(sil)
            
            results['k'].append(k)
            results['method'].append('GMM')
            results['score'].append(-bic)
            results['silhouette'].append(sil)
            
            print(f"  k={k}: BIC={bic:.0f}, AIC={aic:.0f}, silhouette={sil:.3f}")
        
        best_k_gmm = k_range[np.argmin(gmm_bic)]
        print(f"  Best k (BIC): {best_k_gmm}")
    
    print("\n[B2] K-Means Clustering")
    kmeans_inertia = []
    kmeans_silhouette = []
    
    if HAS_SKLEARN:
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertia = km.inertia_
            sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0
            
            kmeans_inertia.append(inertia)
            kmeans_silhouette.append(sil)
            
            results['k'].append(k)
            results['method'].append('KMeans')
            results['score'].append(-inertia)
            results['silhouette'].append(sil)
            
            print(f"  k={k}: inertia={inertia:.0f}, silhouette={sil:.3f}")
        
        best_k_kmeans = k_range[np.argmax(kmeans_silhouette)]
        print(f"  Best k (silhouette): {best_k_kmeans}")
    
    print("\n[B3] HMM State Discovery")
    if HAS_HMM:
        hmm_scores = []
        hmm_bic = []
        
        for k in k_range:
            try:
                model = hmm.GaussianHMM(n_components=k, covariance_type='diag', 
                                        n_iter=100, random_state=42)
                model.fit(X_scaled)
                score = model.score(X_scaled)
                n_params = k * (k - 1) + k * len(feature_cols) * 2
                bic = -2 * score + n_params * np.log(len(X_scaled))
                
                hmm_scores.append(score)
                hmm_bic.append(bic)
                
                results['k'].append(k)
                results['method'].append('HMM')
                results['score'].append(score)
                results['silhouette'].append(0)
                
                print(f"  k={k}: log-likelihood={score:.0f}, BIC={bic:.0f}")
            except Exception as e:
                print(f"  k={k}: FAILED - {e}")
                hmm_scores.append(-np.inf)
                hmm_bic.append(np.inf)
        
        best_k_hmm = k_range[np.argmin(hmm_bic)]
        print(f"  Best k (BIC): {best_k_hmm}")
    else:
        best_k_hmm = best_k_gmm if HAS_SKLEARN else 4
        print("  HMM skipped (hmmlearn not installed)")
    
    best_k_consensus = int(np.median([best_k_gmm, best_k_kmeans, best_k_hmm]) if HAS_SKLEARN else 4)
    print(f"\n  CONSENSUS BEST k: {best_k_consensus}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    if HAS_SKLEARN:
        ax1 = axes[0, 0]
        ax1.plot(list(k_range), gmm_bic, 'b-o', label='GMM BIC')
        ax1.axvline(best_k_gmm, color='blue', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Number of States (k)')
        ax1.set_ylabel('BIC')
        ax1.set_title('GMM Model Selection')
        ax1.legend()
        
        ax2 = axes[0, 1]
        ax2.plot(list(k_range), gmm_silhouette, 'g-o', label='GMM')
        ax2.plot(list(k_range), kmeans_silhouette, 'r-o', label='K-Means')
        ax2.axvline(best_k_kmeans, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Number of States (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Clustering Quality')
        ax2.legend()
    
    if HAS_HMM:
        ax3 = axes[1, 0]
        ax3.plot(list(k_range), hmm_bic, 'm-o', label='HMM BIC')
        ax3.axvline(best_k_hmm, color='magenta', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Number of States (k)')
        ax3.set_ylabel('BIC')
        ax3.set_title('HMM Model Selection')
        ax3.legend()
    
    ax4 = axes[1, 1]
    methods = ['GMM', 'K-Means', 'HMM']
    best_ks = [best_k_gmm if HAS_SKLEARN else 0, 
               best_k_kmeans if HAS_SKLEARN else 0, 
               best_k_hmm]
    colors = ['blue', 'red', 'magenta']
    ax4.bar(methods, best_ks, color=colors, alpha=0.7)
    ax4.axhline(6, color='gold', linestyle='--', linewidth=2, label='k=6 (hypothesis)')
    ax4.axhline(7, color='orange', linestyle='--', linewidth=2, label='k=7 (hypothesis)')
    ax4.set_ylabel('Best k')
    ax4.set_title(f'Consensus: k = {best_k_consensus}')
    ax4.legend()
    ax4.set_ylim(0, 10)
    
    plt.tight_layout()
    plt.savefig('state_model_selection.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved: state_model_selection.png")
    
    print("\n" + "="*80)
    print("[C] STATE INTERPRETATION")
    print("="*80)
    
    if HAS_SKLEARN:
        final_gmm = GaussianMixture(n_components=best_k_consensus, random_state=42, n_init=10)
        df_complete['state'] = final_gmm.fit_predict(X_scaled)
    else:
        df_complete['state'] = 0
    
    print(f"\n  Using {best_k_consensus} states from GMM...")
    
    state_stats = df_complete.groupby('state').agg({
        'r': ['mean', 'std'],
        'delta_score': ['mean', 'std'],
        'hit_phi': 'mean',
        'aperiodic_exponent': 'mean',
        'alpha_power': 'mean',
        'beta_power': 'mean',
        'gamma_power': 'mean'
    }).round(4)
    state_stats.columns = ['r_mean', 'r_std', 'delta_mean', 'delta_std', 'hit_phi_rate',
                          'exponent', 'alpha', 'beta', 'gamma']
    
    print("\n  Per-State Statistics:")
    print(state_stats.to_string())
    
    state_region = pd.crosstab(df_complete['state'], df_complete['region'], normalize='index')
    print("\n  State x Region distribution:")
    print(state_region.round(3).to_string())
    
    state_condition = pd.crosstab(df_complete['state'], df_complete['condition'], normalize='index')
    print("\n  State x Condition distribution:")
    print(state_condition.round(3).to_string())
    
    print("\n  State Interpretation (heuristic):")
    for state in range(best_k_consensus):
        row = state_stats.loc[state]
        alpha = row['alpha']
        beta = row['beta']
        delta = row['delta_mean']
        
        if alpha > state_stats['alpha'].median() and delta > 0:
            label = "BROAD INNER (high alpha, φ-like)"
        elif beta > state_stats['beta'].median() and alpha < state_stats['alpha'].median():
            label = "LASER FOCUS (high beta, low alpha)"
        elif delta > 0.1:
            label = "STRONG φ-ATTRACTOR"
        elif delta < -0.1:
            label = "2:1 HARMONIC REGIME"
        else:
            label = "TRANSITIONAL/MIXED"
        
        print(f"    State {state}: {label} (Δ={delta:.3f}, α={alpha:.2e}, β={beta:.2e})")
    
    if HAS_SKLEARN and len(df_complete) > 100:
        trans_counts = np.zeros((best_k_consensus, best_k_consensus))
        
        for subj in df_complete['subject'].unique():
            subj_data = df_complete[df_complete['subject'] == subj].sort_values(['run', 'epoch_id'])
            states = subj_data['state'].values
            
            for i in range(len(states) - 1):
                trans_counts[states[i], states[i+1]] += 1
        
        trans_probs = trans_counts / (trans_counts.sum(axis=1, keepdims=True) + 1e-10)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(trans_probs, cmap='Blues')
        ax.set_xlabel('To State')
        ax.set_ylabel('From State')
        ax.set_title('State Transition Probabilities')
        ax.set_xticks(range(best_k_consensus))
        ax.set_yticks(range(best_k_consensus))
        plt.colorbar(im, ax=ax)
        
        for i in range(best_k_consensus):
            for j in range(best_k_consensus):
                ax.text(j, i, f'{trans_probs[i,j]:.2f}', ha='center', va='center',
                       color='white' if trans_probs[i,j] > 0.5 else 'black')
        
        plt.tight_layout()
        plt.savefig('state_transition_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n  Figure saved: state_transition_matrix.png")
    
    print("\n" + "="*80)
    print("[D] HYPOTHESIS TESTS")
    print("="*80)
    
    print("\n  H_phase: Do discrete phases (6-7 states) exist?")
    phase_confirmed = 5 <= best_k_consensus <= 8
    print(f"    Best k = {best_k_consensus}")
    print(f"    Result: {'SUPPORTED ✓' if phase_confirmed else 'NOT SUPPORTED (k outside 5-8 range)'}")
    
    print("\n  H_access: φ-like attractor stronger in 'inner' conditions?")
    
    conditions_present = df_complete['condition'].unique()
    print(f"    Conditions: {list(conditions_present)}")
    
    cond_stats = df_complete.groupby('condition').agg({
        'delta_score': ['mean', 'std'],
        'hit_phi': 'mean',
        'r': 'mean'
    }).round(4)
    cond_stats.columns = ['delta_mean', 'delta_std', 'hit_phi', 'r_mean']
    print(f"\n    Per-condition stats:")
    print(cond_stats.to_string())
    
    if 'eyes_open' in conditions_present and 'eyes_closed' in conditions_present:
        eo = df_complete[df_complete['condition'] == 'eyes_open']['delta_score']
        ec = df_complete[df_complete['condition'] == 'eyes_closed']['delta_score']
        t_stat, p_val = stats.ttest_ind(ec, eo)
        print(f"\n    Eyes-closed vs Eyes-open: t = {t_stat:.3f}, p = {p_val:.4f}")
        access_confirmed = ec.mean() > eo.mean() and p_val < 0.1
        print(f"    Result: {'SUPPORTED ✓' if access_confirmed else 'NOT SUPPORTED'}")
    else:
        access_confirmed = False
    
    print("\n  H_topography_shift: State-dependent anterior-posterior reversal?")
    
    state_region_delta = df_complete.groupby(['state', 'region'])['delta_score'].mean().unstack()
    
    if 'frontal' in state_region_delta.columns and 'posterior' in state_region_delta.columns:
        state_region_delta['front_minus_post'] = state_region_delta['frontal'] - state_region_delta['posterior']
        print(f"\n    Frontal - Posterior Δ by state:")
        print(state_region_delta['front_minus_post'].to_string())
        
        has_reversal = (state_region_delta['front_minus_post'].max() > 0.05 and 
                       state_region_delta['front_minus_post'].min() < -0.05)
        print(f"\n    Result: {'SUPPORTED ✓' if has_reversal else 'NOT SUPPORTED'}")
    else:
        has_reversal = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    state_means = df_complete.groupby('state')['delta_score'].mean()
    colors = ['green' if d > 0 else 'red' for d in state_means]
    ax1.bar(range(best_k_consensus), state_means, color=colors, alpha=0.7)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_xlabel('State')
    ax1.set_ylabel('Mean Δ Score')
    ax1.set_title('Δ Score by Discovered State')
    
    ax2 = axes[0, 1]
    hit_rates = df_complete.groupby('state')['hit_phi'].mean() * 100
    ax2.bar(range(best_k_consensus), hit_rates, color='steelblue', alpha=0.7)
    ax2.axhline(50, color='red', linestyle='--')
    ax2.set_xlabel('State')
    ax2.set_ylabel('φ Hit Rate (%)')
    ax2.set_title('% within φ±10% by State')
    
    ax3 = axes[1, 0]
    if len(conditions_present) > 1:
        cond_order = ['eyes_open', 'eyes_closed', 'task1', 'task2', 'task3', 'task4']
        cond_order = [c for c in cond_order if c in conditions_present]
        
        for state in range(min(5, best_k_consensus)):
            state_cond = df_complete[df_complete['state'] == state]
            props = [len(state_cond[state_cond['condition'] == c]) / len(state_cond) 
                    for c in cond_order]
            ax3.plot(range(len(cond_order)), props, 'o-', label=f'State {state}')
        
        ax3.set_xticks(range(len(cond_order)))
        ax3.set_xticklabels([c.replace('_', '\n') for c in cond_order])
        ax3.set_ylabel('Proportion of epochs')
        ax3.set_title('State Distribution by Condition')
        ax3.legend()
    
    ax4 = axes[1, 1]
    if 'frontal' in state_region_delta.columns and 'posterior' in state_region_delta.columns:
        x = range(best_k_consensus)
        width = 0.35
        ax4.bar([i - width/2 for i in x], state_region_delta['frontal'], width, label='Frontal', color='steelblue')
        ax4.bar([i + width/2 for i in x], state_region_delta['posterior'], width, label='Posterior', color='coral')
        ax4.axhline(0, color='black', linewidth=0.5)
        ax4.set_xlabel('State')
        ax4.set_ylabel('Mean Δ')
        ax4.set_title('Regional Δ by State (Topography Shift?)')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('state_by_condition.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved: state_by_condition.png")
    
    print("\n" + "="*80)
    print("[E] GENERATING REPORT")
    print("="*80)
    
    with open('phase_report.md', 'w') as f:
        f.write("# PHASE MODEL ANALYSIS REPORT\n\n")
        f.write("## Testing: 6-7 Discrete States for Inner Awareness\n\n")
        f.write(f"**Dataset**: PhysioNet EEGMMIDB (N={df['subject'].nunique()} subjects)\n")
        f.write(f"**Epochs**: {len(df)} region-epochs ({EPOCH_DURATION}s windows)\n")
        f.write(f"**Conditions**: {', '.join(df['condition'].unique())}\n\n")
        
        f.write("---\n\n")
        f.write("## Summary of Findings\n\n")
        
        f.write(f"### Best Number of States: **k = {best_k_consensus}**\n\n")
        f.write("| Method | Best k |\n")
        f.write("|--------|--------|\n")
        f.write(f"| GMM (BIC) | {best_k_gmm} |\n")
        f.write(f"| K-Means (silhouette) | {best_k_kmeans} |\n")
        f.write(f"| HMM (BIC) | {best_k_hmm} |\n")
        f.write(f"| **Consensus** | **{best_k_consensus}** |\n\n")
        
        f.write("---\n\n")
        f.write("## Hypothesis Tests\n\n")
        
        f.write("### H_phase: Do 6-7 discrete phases exist?\n\n")
        f.write(f"- Best k = {best_k_consensus}\n")
        f.write(f"- **Result**: {'✅ SUPPORTED' if phase_confirmed else '❌ NOT SUPPORTED'}\n\n")
        
        f.write("### H_access: φ-like attractor stronger in 'inner' conditions?\n\n")
        f.write(f"- **Result**: {'✅ SUPPORTED' if access_confirmed else '❌ NOT SUPPORTED'}\n\n")
        
        f.write("### H_topography_shift: State-dependent anterior-posterior reversal?\n\n")
        f.write(f"- **Result**: {'✅ SUPPORTED' if has_reversal else '❌ NOT SUPPORTED'}\n\n")
        
        f.write("---\n\n")
        f.write("## Per-State Statistics\n\n")
        f.write("| State | Mean Δ | φ Hit% | α Power | β Power | Interpretation |\n")
        f.write("|-------|--------|--------|---------|---------|----------------|\n")
        
        for state in range(best_k_consensus):
            row = state_stats.loc[state]
            delta = row['delta_mean']
            hit = row['hit_phi_rate'] * 100
            alpha = row['alpha']
            beta = row['beta']
            
            if alpha > state_stats['alpha'].median() and delta > 0:
                label = "Broad Inner"
            elif beta > state_stats['beta'].median():
                label = "Laser Focus"
            elif delta > 0.1:
                label = "φ-Attractor"
            else:
                label = "Transitional"
            
            f.write(f"| {state} | {delta:.3f} | {hit:.1f}% | {alpha:.2e} | {beta:.2e} | {label} |\n")
        
        f.write("\n---\n\n")
        f.write("## Figures Generated\n\n")
        f.write("- `state_model_selection.png` - k vs BIC/silhouette\n")
        f.write("- `state_transition_matrix.png` - State transition probabilities\n")
        f.write("- `state_by_condition.png` - State distribution and regional patterns\n")
        
        f.write("\n---\n\n")
        f.write("## Conclusion\n\n")
        
        if phase_confirmed:
            f.write(f"The data supports the existence of **{best_k_consensus} discrete states** ")
            f.write("in EEG dynamics, consistent with the 6-7 phase hypothesis.\n\n")
        else:
            f.write(f"The optimal number of states (k={best_k_consensus}) does not strongly support ")
            f.write("the 6-7 phase hypothesis. ")
            if best_k_consensus < 5:
                f.write("Fewer states may be sufficient to describe the data.\n\n")
            else:
                f.write("More states may be needed or the phase structure is less discrete.\n\n")
        
        f.write("**Treat φ-like Δ as an order parameter candidate; ")
        f.write("search for discontinuous regime shifts ('clicks') rather than smooth trends.**\n")
    
    print("\n  Report saved: phase_report.md")
    
    df_complete.to_csv('epoch_features_with_states.csv', index=False)
    print("  Data saved: epoch_features_with_states.csv")
    
    pd.DataFrame(results).to_csv('state_model_selection_results.csv', index=False)
    print("  Results saved: state_model_selection_results.csv")
    
    print("\n" + "="*80)
    print("PHASE MODEL ANALYSIS COMPLETE")
    print("="*80)
