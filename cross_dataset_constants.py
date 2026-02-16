#!/usr/bin/env python3
"""
Cross-Dataset Constant Checks: 2x2 Validation
Using GAMEEMO + PhysioNet EEGMMIDB datasets
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

PHI = 1.618033988749895
EULER_GAMMA = 0.5772156649015329
OMEGA = 0.5671432904097838

print("="*80)
print("CROSS-DATASET CONSTANT CHECKS (2x2 Validation)")
print("="*80)

# Load datasets
print("\n[LOADING DATASETS]")

# Dataset 1: PhysioNet EEGMMIDB (epoch-level)
df_physio = pd.read_csv('epoch_features_with_states.csv')
df_physio['dataset'] = 'PhysioNet'
print(f"  PhysioNet: {len(df_physio)} epochs, {df_physio['subject'].nunique()} subjects")

# Dataset 2: GAMEEMO (trial-level)
df_gameemo = pd.read_csv('gameemo_all_trials.csv')
df_gameemo['dataset'] = 'GAMEEMO'
print(f"  GAMEEMO: {len(df_gameemo)} trials, {df_gameemo['subject'].nunique()} subjects")

# Check common columns
print("\n  PhysioNet cols:", list(df_physio.columns[:10]))
print("  GAMEEMO cols:", list(df_gameemo.columns[:10]))

# =============================================================================
# PREPARE BOTH DATASETS WITH COMMON FEATURES
# =============================================================================
print("\n" + "="*80)
print("PREPARING COMMON FEATURE SPACE")
print("="*80)

# PhysioNet already has: r, delta_score, state, etc.
# GAMEEMO needs: compute r and delta_score if not present

if 'r' not in df_gameemo.columns:
    # Check what columns GAMEEMO has
    print("\n  GAMEEMO columns:", list(df_gameemo.columns))
    
# Let's check GAMEEMO structure
gameemo_cols = df_gameemo.columns.tolist()
print(f"\n  GAMEEMO has {len(gameemo_cols)} columns")

# Look for ratio/phi columns
ratio_cols = [c for c in gameemo_cols if 'ratio' in c.lower() or 'phi' in c.lower() or 'delta' in c.lower()]
print(f"  Ratio-related cols: {ratio_cols}")

# For cross-dataset, we'll use what's available
# PhysioNet: use existing features
# GAMEEMO: adapt to available

def compute_dataset_stats(df, name, r_col='r', delta_col='delta_score'):
    """Compute key statistics for a dataset"""
    stats_dict = {}
    
    if r_col in df.columns:
        r_vals = df[r_col].dropna().values
        stats_dict['r_mean'] = np.mean(r_vals)
        stats_dict['r_std'] = np.std(r_vals)
        stats_dict['r_median'] = np.median(r_vals)
        stats_dict['phi_proximity'] = np.mean(np.abs(r_vals - PHI) < np.abs(r_vals - 2.0))
    
    if delta_col in df.columns:
        d_vals = df[delta_col].dropna().values
        stats_dict['delta_mean'] = np.mean(d_vals)
        stats_dict['delta_std'] = np.std(d_vals)
        
        # Entropy
        hist, _ = np.histogram(d_vals, bins=20, density=True)
        hist = hist[hist > 0]
        bin_width = (d_vals.max() - d_vals.min()) / 20
        stats_dict['entropy'] = -np.sum(hist * np.log(hist) * bin_width)
    
    return stats_dict

# =============================================================================
# COMPUTE STATS FOR BOTH DATASETS
# =============================================================================
print("\n" + "="*80)
print("DATASET COMPARISON")
print("="*80)

physio_stats = compute_dataset_stats(df_physio, 'PhysioNet')
print("\n  PhysioNet:")
for k, v in physio_stats.items():
    print(f"    {k}: {v:.4f}")

# For GAMEEMO, check available columns
if 'gamma_beta_ratio' in df_gameemo.columns:
    gameemo_r_col = 'gamma_beta_ratio'
elif 'ratio' in df_gameemo.columns:
    gameemo_r_col = 'ratio'
else:
    # Compute from power bands if available
    if 'gamma_power' in df_gameemo.columns and 'beta_power' in df_gameemo.columns:
        df_gameemo['r'] = df_gameemo['gamma_power'] / (df_gameemo['beta_power'] + 1e-10)
        gameemo_r_col = 'r'
    else:
        gameemo_r_col = None

if gameemo_r_col:
    # Compute delta_score for GAMEEMO
    r_vals = df_gameemo[gameemo_r_col].dropna().values
    df_gameemo.loc[df_gameemo[gameemo_r_col].notna(), 'delta_score'] = np.log(
        np.abs(r_vals - 2.0) / (np.abs(r_vals - PHI) + 1e-10)
    )
    
    gameemo_stats = compute_dataset_stats(df_gameemo, 'GAMEEMO', r_col=gameemo_r_col)
    print("\n  GAMEEMO:")
    for k, v in gameemo_stats.items():
        print(f"    {k}: {v:.4f}")
else:
    print("\n  GAMEEMO: No ratio column found, using aggregate data")
    # Use pre-computed stats from gameemo_subject_means.csv
    df_gameemo_means = pd.read_csv('gameemo_subject_means.csv')
    print(f"  GAMEEMO means: {len(df_gameemo_means)} subjects")
    print(f"  Columns: {list(df_gameemo_means.columns)}")
    gameemo_stats = {}

# =============================================================================
# 2x2 CROSS-VALIDATION: TRAIN ON ONE, TEST ON OTHER
# =============================================================================
print("\n" + "="*80)
print("2x2 CROSS-VALIDATION")
print("="*80)

# For this to work, we need comparable features
# Let's use state discovery on each dataset separately, then compare

print("\n[A] State Discovery on Each Dataset")

feature_cols = ['r', 'delta_score']

# PhysioNet states (already computed)
print(f"\n  PhysioNet: 6 states already computed")
physio_state_means = df_physio.groupby('state')['delta_score'].mean()
print(f"    State Δ means: {physio_state_means.to_dict()}")

# GAMEEMO states (compute fresh)
if 'delta_score' in df_gameemo.columns:
    df_gameemo_clean = df_gameemo.dropna(subset=['delta_score'])
    if len(df_gameemo_clean) > 100:
        X_gameemo = StandardScaler().fit_transform(df_gameemo_clean[['delta_score']].values)
        
        from sklearn.metrics import silhouette_score
        
        best_k = 2
        best_sil = -1
        for k in range(2, 10):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_gameemo)
            sil = silhouette_score(X_gameemo, labels)
            if sil > best_sil:
                best_sil = sil
                best_k = k
        
        km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        df_gameemo_clean['state'] = km_final.fit_predict(X_gameemo)
        
        print(f"\n  GAMEEMO: {best_k} states discovered (silhouette={best_sil:.3f})")
        gameemo_state_means = df_gameemo_clean.groupby('state')['delta_score'].mean()
        print(f"    State Δ means: {gameemo_state_means.to_dict()}")

# =============================================================================
# COMPARE PHI-PROXIMITY ACROSS DATASETS
# =============================================================================
print("\n" + "="*80)
print("[B] PHI-PROXIMITY COMPARISON")
print("="*80)

if 'phi_proximity' in physio_stats and 'phi_proximity' in gameemo_stats:
    print(f"\n  PhysioNet φ-proximity: {100*physio_stats['phi_proximity']:.1f}%")
    print(f"  GAMEEMO φ-proximity: {100*gameemo_stats['phi_proximity']:.1f}%")
    
    # Statistical comparison
    physio_hits = df_physio['r'].apply(lambda x: abs(x - PHI) < abs(x - 2.0) if pd.notna(x) else False)
    gameemo_hits = df_gameemo[gameemo_r_col].apply(lambda x: abs(x - PHI) < abs(x - 2.0) if pd.notna(x) else False) if gameemo_r_col else []
    
    if len(gameemo_hits) > 0:
        table = [[physio_hits.sum(), len(physio_hits) - physio_hits.sum()],
                 [gameemo_hits.sum(), len(gameemo_hits) - gameemo_hits.sum()]]
        chi2, p = stats.chi2_contingency(table)[:2]
        print(f"\n  χ² = {chi2:.2f}, p = {p:.4f}")
        print(f"  Datasets {'AGREE' if p > 0.05 else 'DIFFER'} on φ-proximity rate")

# =============================================================================
# ENTROPY COMPARISON (Euler γ check)
# =============================================================================
print("\n" + "="*80)
print("[C] ENTROPY COMPARISON (Euler γ)")
print("="*80)

if 'entropy' in physio_stats:
    print(f"\n  PhysioNet H = {physio_stats['entropy']:.4f}")
if 'entropy' in gameemo_stats:
    print(f"  GAMEEMO H = {gameemo_stats['entropy']:.4f}")
print(f"  Euler γ = {EULER_GAMMA:.4f}")

if 'entropy' in physio_stats and 'entropy' in gameemo_stats:
    mean_H = (physio_stats['entropy'] + gameemo_stats['entropy']) / 2
    print(f"\n  Cross-dataset mean H = {mean_H:.4f}")
    print(f"  |H - γ| = {abs(mean_H - EULER_GAMMA):.4f}")

# =============================================================================
# TRANSITION ANALYSIS (Omega check) - where applicable
# =============================================================================
print("\n" + "="*80)
print("[D] OMEGA CHECK (p ≈ e^{-p})")
print("="*80)

# PhysioNet already has transitions computed
print("\n  PhysioNet transition analysis:")
trans_counts = np.zeros((6, 6))
for subj in df_physio['subject'].unique():
    sd = df_physio[df_physio['subject'] == subj].sort_values(['run', 'epoch_id'])
    states = sd['state'].values
    for i in range(len(states) - 1):
        trans_counts[states[i], states[i+1]] += 1

trans_probs = trans_counts / (trans_counts.sum(axis=1, keepdims=True) + 1e-10)

# Find Omega-like state
omega_residuals = []
for s in range(6):
    p = trans_probs[s, s]  # self-loop
    residual = abs(p - np.exp(-p))
    omega_residuals.append((s, p, residual))

closest = min(omega_residuals, key=lambda x: x[2])
print(f"    Closest to Ω: State {closest[0]} (p={closest[1]:.3f}, residual={closest[2]:.3f})")

# =============================================================================
# SUMMARY: 2x2 CONCORDANCE
# =============================================================================
print("\n" + "="*80)
print("SUMMARY: CROSS-DATASET CONCORDANCE")
print("="*80)

results = {
    'PhysioNet_phi_pct': physio_stats.get('phi_proximity', 0) * 100,
    'PhysioNet_entropy': physio_stats.get('entropy', 0),
    'PhysioNet_delta_mean': physio_stats.get('delta_mean', 0),
    'GAMEEMO_phi_pct': gameemo_stats.get('phi_proximity', 0) * 100 if gameemo_stats else 0,
    'GAMEEMO_entropy': gameemo_stats.get('entropy', 0) if gameemo_stats else 0,
    'GAMEEMO_delta_mean': gameemo_stats.get('delta_mean', 0) if gameemo_stats else 0,
}

print("\n┌──────────────────────────────────────────────────────────────────┐")
print("│ METRIC               │ PhysioNet      │ GAMEEMO        │ AGREE? │")
print("├──────────────────────────────────────────────────────────────────┤")
print(f"│ φ-proximity (%)      │ {results['PhysioNet_phi_pct']:14.1f} │ {results['GAMEEMO_phi_pct']:14.1f} │        │")
print(f"│ Mean Δ               │ {results['PhysioNet_delta_mean']:14.4f} │ {results['GAMEEMO_delta_mean']:14.4f} │        │")
print(f"│ Entropy H            │ {results['PhysioNet_entropy']:14.4f} │ {results['GAMEEMO_entropy']:14.4f} │        │")
print("└──────────────────────────────────────────────────────────────────┘")

# Check if both show same direction
both_phi_positive = results['PhysioNet_phi_pct'] > 50 and results['GAMEEMO_phi_pct'] > 50
both_delta_negative = results['PhysioNet_delta_mean'] < 0 and results['GAMEEMO_delta_mean'] < 0

print(f"\n  Both datasets φ > 50%: {'✅ YES' if both_phi_positive else '❌ NO'}")
print(f"  Both datasets Δ < 0: {'✅ YES' if both_delta_negative else '❌ NO'}")

# Save results
with open('cross_dataset_report.md', 'w') as f:
    f.write("# Cross-Dataset Validation Report\n\n")
    f.write("## Datasets\n")
    f.write(f"- PhysioNet EEGMMIDB: {df_physio['subject'].nunique()} subjects, {len(df_physio)} epochs\n")
    f.write(f"- GAMEEMO: {df_gameemo['subject'].nunique()} subjects, {len(df_gameemo)} trials\n\n")
    f.write("## Results\n\n")
    f.write("| Metric | PhysioNet | GAMEEMO |\n")
    f.write("|--------|-----------|--------|\n")
    f.write(f"| φ-proximity | {results['PhysioNet_phi_pct']:.1f}% | {results['GAMEEMO_phi_pct']:.1f}% |\n")
    f.write(f"| Mean Δ | {results['PhysioNet_delta_mean']:.4f} | {results['GAMEEMO_delta_mean']:.4f} |\n")
    f.write(f"| Entropy | {results['PhysioNet_entropy']:.4f} | {results['GAMEEMO_entropy']:.4f} |\n\n")
    f.write("## Conclusions\n\n")
    f.write(f"- Cross-dataset φ-proximity agreement: {'YES' if both_phi_positive else 'NO'}\n")
    f.write(f"- Cross-dataset Δ direction agreement: {'YES' if both_delta_negative else 'NO'}\n")

print("\nReport: cross_dataset_report.md")
print("\nDONE")
