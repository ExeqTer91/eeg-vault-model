#!/usr/bin/env python3
"""
Cross-Dataset 2x2 Validation - Fixed version
Using ratio_gamma/beta from GAMEEMO
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

PHI = 1.618033988749895
EULER_GAMMA = 0.5772156649015329
OMEGA = 0.5671432904097838

print("="*80)
print("CROSS-DATASET 2x2 VALIDATION (Fixed)")
print("="*80)

# Load PhysioNet
df_physio = pd.read_csv('epoch_features_with_states.csv')
print(f"\nPhysioNet: {len(df_physio)} epochs, {df_physio['subject'].nunique()} subjects")

# Load GAMEEMO with correct ratio column
df_gameemo = pd.read_csv('gameemo_all_trials.csv')
print(f"GAMEEMO: {len(df_gameemo)} trials, {df_gameemo['subject'].nunique()} subjects")

# GAMEEMO already has ratio_gamma/beta
if 'ratio_gamma/beta' in df_gameemo.columns:
    df_gameemo['r'] = df_gameemo['ratio_gamma/beta']
    df_gameemo['delta_score'] = np.log(
        np.abs(df_gameemo['r'] - 2.0) / (np.abs(df_gameemo['r'] - PHI) + 1e-10)
    )
    print(f"  GAMEEMO r range: [{df_gameemo['r'].min():.3f}, {df_gameemo['r'].max():.3f}]")

# =============================================================================
# COMPUTE STATS FOR BOTH
# =============================================================================
print("\n" + "="*80)
print("DATASET STATISTICS")
print("="*80)

def get_stats(df, name):
    r = df['r'].dropna()
    d = df['delta_score'].dropna()
    
    phi_hits = (np.abs(r - PHI) < np.abs(r - 2.0)).mean()
    
    # Entropy
    hist, _ = np.histogram(d, bins=20, density=True)
    hist = hist[hist > 0]
    H = -np.sum(hist * np.log(hist) * (d.max() - d.min()) / 20)
    
    print(f"\n{name}:")
    print(f"  r: {r.mean():.4f} ± {r.std():.4f}")
    print(f"  Δ: {d.mean():.4f} ± {d.std():.4f}")
    print(f"  φ-proximity: {100*phi_hits:.1f}%")
    print(f"  Entropy H: {H:.4f}")
    
    return {'r_mean': r.mean(), 'r_std': r.std(),
            'delta_mean': d.mean(), 'delta_std': d.std(),
            'phi_pct': phi_hits, 'entropy': H, 'n': len(d)}

physio_stats = get_stats(df_physio, "PhysioNet")
gameemo_stats = get_stats(df_gameemo, "GAMEEMO")

# =============================================================================
# STATE DISCOVERY ON BOTH
# =============================================================================
print("\n" + "="*80)
print("STATE DISCOVERY")
print("="*80)

def discover_states(df, name):
    X = StandardScaler().fit_transform(df[['delta_score']].dropna().values)
    
    best_k, best_sil = 2, -1
    for k in range(2, 10):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        if sil > best_sil:
            best_sil = sil
            best_k = k
    
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km_final.fit_predict(X)
    
    print(f"\n{name}: k={best_k} (silhouette={best_sil:.3f})")
    
    # Count phi-like vs harmonic states
    df_temp = df.dropna(subset=['delta_score']).copy()
    df_temp['state'] = labels
    state_means = df_temp.groupby('state')['delta_score'].mean()
    
    phi_states = (state_means > 0).sum()
    harm_states = (state_means <= 0).sum()
    print(f"  φ-like states: {phi_states}, Harmonic states: {harm_states}")
    
    return best_k, phi_states, harm_states

physio_k, physio_phi_s, physio_harm_s = discover_states(df_physio, "PhysioNet")
gameemo_k, gameemo_phi_s, gameemo_harm_s = discover_states(df_gameemo, "GAMEEMO")

# =============================================================================
# 2x2 AGREEMENT TESTS
# =============================================================================
print("\n" + "="*80)
print("2x2 CONCORDANCE TESTS")
print("="*80)

print("\n[A] φ-proximity agreement")
# Chi-square test
physio_r = df_physio['r'].dropna()
gameemo_r = df_gameemo['r'].dropna()

physio_phi = (np.abs(physio_r - PHI) < np.abs(physio_r - 2.0)).sum()
physio_harm = len(physio_r) - physio_phi
gameemo_phi = (np.abs(gameemo_r - PHI) < np.abs(gameemo_r - 2.0)).sum()
gameemo_harm = len(gameemo_r) - gameemo_phi

table = [[physio_phi, physio_harm], [gameemo_phi, gameemo_harm]]
chi2, p_chi = stats.chi2_contingency(table)[:2]

print(f"  PhysioNet: {physio_phi}/{len(physio_r)} = {100*physio_phi/len(physio_r):.1f}%")
print(f"  GAMEEMO: {gameemo_phi}/{len(gameemo_r)} = {100*gameemo_phi/len(gameemo_r):.1f}%")
print(f"  χ² = {chi2:.2f}, p = {p_chi:.4f}")

phi_agree = p_chi > 0.05  # No significant difference = agreement
print(f"  Agreement: {'✅ YES' if phi_agree else '❌ DIFFERENT RATES'}")

print("\n[B] Δ mean agreement")
t_delta, p_delta = stats.ttest_ind(
    df_physio['delta_score'].dropna(),
    df_gameemo['delta_score'].dropna()
)
print(f"  PhysioNet Δ: {physio_stats['delta_mean']:.4f}")
print(f"  GAMEEMO Δ: {gameemo_stats['delta_mean']:.4f}")
print(f"  t = {t_delta:.2f}, p = {p_delta:.4f}")

# Same direction?
same_dir = (physio_stats['delta_mean'] < 0) == (gameemo_stats['delta_mean'] < 0)
print(f"  Same direction: {'✅ YES' if same_dir else '❌ NO'}")

print("\n[C] State count agreement")
k_diff = abs(physio_k - gameemo_k)
print(f"  PhysioNet k: {physio_k}")
print(f"  GAMEEMO k: {gameemo_k}")
print(f"  Difference: {k_diff}")
print(f"  Agreement (|diff| ≤ 2): {'✅ YES' if k_diff <= 2 else '❌ NO'}")

print("\n[D] Entropy agreement")
H_diff = abs(physio_stats['entropy'] - gameemo_stats['entropy'])
print(f"  PhysioNet H: {physio_stats['entropy']:.4f}")
print(f"  GAMEEMO H: {gameemo_stats['entropy']:.4f}")
print(f"  |diff|: {H_diff:.4f}")
print(f"  Both negative (concentrated): {'✅ YES' if physio_stats['entropy'] < 0 and gameemo_stats['entropy'] < 0 else '❌ NO'}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

agreements = [phi_agree, same_dir, k_diff <= 2]
n_agree = sum(agreements)

print("\n┌────────────────────────────────────────────────────────────────┐")
print("│ TEST                    │ PhysioNet │ GAMEEMO  │ AGREE? │")
print("├────────────────────────────────────────────────────────────────┤")
print(f"│ φ-proximity (%)         │ {100*physio_stats['phi_pct']:9.1f} │ {100*gameemo_stats['phi_pct']:8.1f} │   {'✅' if phi_agree else '❌'}   │")
print(f"│ Mean Δ                  │ {physio_stats['delta_mean']:9.4f} │ {gameemo_stats['delta_mean']:8.4f} │   {'✅' if same_dir else '❌'}   │")
print(f"│ State count k           │ {physio_k:9d} │ {gameemo_k:8d} │   {'✅' if k_diff <= 2 else '❌'}   │")
print("└────────────────────────────────────────────────────────────────┘")

print(f"\nOVERALL: {n_agree}/3 agreements")

if n_agree >= 2:
    print("\n✅ CROSS-DATASET REPLICATION SUPPORTED")
else:
    print("\n❌ CROSS-DATASET REPLICATION WEAK")

# Save report
with open('cross_dataset_report.md', 'w') as f:
    f.write("# Cross-Dataset 2x2 Validation\n\n")
    f.write("## Datasets\n")
    f.write(f"- **PhysioNet EEGMMIDB**: {physio_stats['n']} epochs\n")
    f.write(f"- **GAMEEMO**: {gameemo_stats['n']} trials\n\n")
    f.write("## Results\n\n")
    f.write("| Metric | PhysioNet | GAMEEMO | Agreement |\n")
    f.write("|--------|-----------|---------|----------|\n")
    f.write(f"| φ-proximity | {100*physio_stats['phi_pct']:.1f}% | {100*gameemo_stats['phi_pct']:.1f}% | {'✅' if phi_agree else '❌'} |\n")
    f.write(f"| Mean Δ | {physio_stats['delta_mean']:.4f} | {gameemo_stats['delta_mean']:.4f} | {'✅' if same_dir else '❌'} |\n")
    f.write(f"| State k | {physio_k} | {gameemo_k} | {'✅' if k_diff <= 2 else '❌'} |\n\n")
    f.write(f"## Conclusion\n\n")
    f.write(f"**{n_agree}/3 tests show cross-dataset agreement**\n")

print("\nReport: cross_dataset_report.md")
print("DONE")
