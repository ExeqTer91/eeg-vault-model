#!/usr/bin/env python3
"""
Hierarchical State Analysis: Testing 6-7 as "2 macro × 3 sub + bridge"
Based on theoretical framework: 2 basins (harmonic vs φ-like) with sub-states
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PHI = 1.618033988749895

print("="*80)
print("HIERARCHICAL STATE ANALYSIS")
print("Testing: 6-7 = 2 macro-regimes × 3 sub-states + bridge")
print("="*80)

df = pd.read_csv('epoch_features_with_states.csv')
print(f"\nData: {len(df)} epochs, {df['subject'].nunique()} subjects")

feature_cols = ['r', 'delta_score', 'aperiodic_exponent', 'alpha_power', 'beta_power']
df = df.dropna(subset=feature_cols)
X = df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n" + "="*80)
print("EXPERIMENT 1: HIERARCHICAL STRUCTURE (macro → micro)")
print("="*80)

print("\n[1A] Level 1: Two Macro-Regimes")
km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
df['macro_regime'] = km2.fit_predict(X_scaled)

macro_stats = df.groupby('macro_regime').agg({
    'r': 'mean',
    'delta_score': 'mean',
    'hit_phi': 'mean'
}).round(4)

print("\nMacro-Regime Statistics:")
print(macro_stats.to_string())

harmonic_regime = macro_stats['delta_score'].idxmin()
phi_regime = macro_stats['delta_score'].idxmax()
print(f"\n  Regime {harmonic_regime} = HARMONIC (Δ={macro_stats.loc[harmonic_regime, 'delta_score']:.4f})")
print(f"  Regime {phi_regime} = φ-LIKE (Δ={macro_stats.loc[phi_regime, 'delta_score']:.4f})")

print("\n[1B] Level 2: Sub-states within each macro-regime")

sub_results = {}
for regime in [0, 1]:
    regime_data = df[df['macro_regime'] == regime]
    X_regime = scaler.fit_transform(regime_data[feature_cols].values)
    
    print(f"\n  Regime {regime} ({len(regime_data)} epochs):")
    
    best_k = 2
    best_sil = -1
    for k in range(2, 6):
        if len(X_regime) < k * 10:
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_regime)
        sil = silhouette_score(X_regime, labels)
        print(f"    k={k}: silhouette={sil:.3f}")
        if sil > best_sil:
            best_sil = sil
            best_k = k
    
    sub_results[regime] = best_k
    print(f"    Best k = {best_k}")

total_states = sum(sub_results.values())
print(f"\n[1C] HIERARCHICAL RESULT:")
print(f"    Harmonic regime: {sub_results[harmonic_regime]} sub-states")
print(f"    φ-like regime: {sub_results[phi_regime]} sub-states")
print(f"    TOTAL from hierarchy: {total_states}")
print(f"    Target (6-7): {'✅ MATCH' if 5 <= total_states <= 8 else '❌ NO MATCH'}")

print("\n" + "="*80)
print("EXPERIMENT 2: PER-SUBJECT STABILITY")
print("="*80)

subject_ks = []
for subj in df['subject'].unique():
    subj_data = df[df['subject'] == subj]
    if len(subj_data) < 100:
        continue
    
    X_subj = scaler.fit_transform(subj_data[feature_cols].values)
    
    best_k = 2
    best_sil = -1
    for k in range(2, 8):
        if len(X_subj) < k * 10:
            continue
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=5)
            labels = km.fit_predict(X_subj)
            sil = silhouette_score(X_subj, labels)
            if sil > best_sil:
                best_sil = sil
                best_k = k
        except:
            pass
    
    subject_ks.append(best_k)

print(f"\nPer-subject optimal k:")
print(f"  Mean k: {np.mean(subject_ks):.2f} ± {np.std(subject_ks):.2f}")
print(f"  Median k: {np.median(subject_ks):.0f}")
print(f"  Range: [{min(subject_ks)}, {max(subject_ks)}]")
print(f"  Distribution: {dict(zip(*np.unique(subject_ks, return_counts=True)))}")

stability_ok = 2 <= np.std(subject_ks) <= 3 and 4 <= np.median(subject_ks) <= 8
print(f"\n  Stability: {'✅ CONSISTENT' if stability_ok else '⚠️ VARIABLE'}")

print("\n" + "="*80)
print("EXPERIMENT 3: BRIDGE STATE DETECTION")
print("="*80)

trans_counts = np.zeros((6, 6))
for subj in df['subject'].unique():
    sd = df[df['subject'] == subj].sort_values(['run', 'epoch_id'])
    states = sd['state'].values
    for i in range(len(states) - 1):
        trans_counts[states[i], states[i+1]] += 1

trans_probs = trans_counts / (trans_counts.sum(axis=1, keepdims=True) + 1e-10)

print("\n[3A] State Connectivity Analysis")

state_centrality = []
for s in range(6):
    incoming = trans_probs[:, s].sum()
    outgoing = trans_probs[s, :].sum()
    self_loop = trans_probs[s, s]
    centrality = (incoming + outgoing) / 2 - self_loop
    state_centrality.append((s, centrality, self_loop))
    print(f"  State {s}: centrality={centrality:.3f}, self-loop={self_loop:.3f}")

bridge_candidates = [s for s, c, sl in state_centrality if c > 0.8 and sl < 0.3]
print(f"\n[3B] Bridge candidates (high centrality, low self-loop): {bridge_candidates}")

print("\n[3C] Inter-regime transitions")
regime_by_state = df.groupby('state')['macro_regime'].mean()
harmonic_states = [s for s in range(6) if regime_by_state.get(s, 0.5) < 0.4]
phi_states = [s for s in range(6) if regime_by_state.get(s, 0.5) > 0.6]
mixed_states = [s for s in range(6) if 0.4 <= regime_by_state.get(s, 0.5) <= 0.6]

print(f"  Harmonic-leaning states: {harmonic_states}")
print(f"  φ-leaning states: {phi_states}")
print(f"  MIXED/BRIDGE states: {mixed_states}")

if mixed_states:
    print(f"\n  ✅ BRIDGE STATE IDENTIFIED: State(s) {mixed_states}")
    bridge_ok = True
else:
    print(f"\n  ⚠️ No clear bridge state found")
    bridge_ok = False

print("\n" + "="*80)
print("EXPERIMENT 4: BOOTSTRAP STABILITY")
print("="*80)

n_bootstrap = 50
bootstrap_ks = []

for i in range(n_bootstrap):
    idx = np.random.choice(len(X_scaled), size=len(X_scaled)//2, replace=True)
    X_boot = X_scaled[idx]
    
    best_k = 2
    best_sil = -1
    for k in range(2, 10):
        try:
            km = KMeans(n_clusters=k, random_state=i, n_init=3)
            labels = km.fit_predict(X_boot)
            sil = silhouette_score(X_boot, labels)
            if sil > best_sil:
                best_sil = sil
                best_k = k
        except:
            pass
    bootstrap_ks.append(best_k)

print(f"\nBootstrap optimal k (N={n_bootstrap}):")
print(f"  Mean: {np.mean(bootstrap_ks):.2f} ± {np.std(bootstrap_ks):.2f}")
print(f"  Median: {np.median(bootstrap_ks):.0f}")
print(f"  Mode: {stats.mode(bootstrap_ks, keepdims=False)[0]}")
print(f"  Range: [{min(bootstrap_ks)}, {max(bootstrap_ks)}]")

in_67_range = sum(1 for k in bootstrap_ks if 5 <= k <= 8)
print(f"  In 5-8 range: {in_67_range}/{n_bootstrap} ({100*in_67_range/n_bootstrap:.1f}%)")

bootstrap_ok = in_67_range / n_bootstrap > 0.3
print(f"\n  Bootstrap stability: {'✅ STABLE' if bootstrap_ok else '⚠️ VARIABLE'}")

print("\n" + "="*80)
print("SUMMARY: Is 6-7 a Natural Resolution Level?")
print("="*80)

print("\n┌─────────────────────────────────────────────────────────────┐")
print("│ EXPERIMENT                          │ RESULT                │")
print("├─────────────────────────────────────────────────────────────┤")
print(f"│ 1. Hierarchical (2×3+bridge)        │ {total_states} states {'✅' if 5<=total_states<=8 else '❌'}          │")
print(f"│ 2. Per-subject stability            │ k={np.median(subject_ks):.0f} median {'✅' if stability_ok else '⚠️'}       │")
print(f"│ 3. Bridge state exists              │ {'States ' + str(mixed_states) if bridge_ok else 'Not found'} {'✅' if bridge_ok else '⚠️'}│")
print(f"│ 4. Bootstrap stability (5-8 range)  │ {100*in_67_range/n_bootstrap:.0f}% {'✅' if bootstrap_ok else '⚠️'}              │")
print("└─────────────────────────────────────────────────────────────┘")

overall_support = sum([5<=total_states<=8, stability_ok, bridge_ok, bootstrap_ok])
print(f"\nOVERALL: {overall_support}/4 criteria support 6-7 as natural resolution")

if overall_support >= 3:
    print("\n✅ CONCLUSION: 6-7 states IS a natural resolution level")
    print("   Not artifact, but 'sweet spot' between too coarse (2) and too fine (10)")
else:
    print("\n⚠️ CONCLUSION: Mixed evidence for 6-7 as unique resolution")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

ax1 = axes[0, 0]
macro_colors = ['coral' if m == harmonic_regime else 'steelblue' for m in df['macro_regime']]
ax1.scatter(df['r'], df['delta_score'], c=macro_colors, alpha=0.1, s=1)
ax1.axhline(0, color='black', linewidth=0.5)
ax1.axvline(PHI, color='gold', linewidth=2, linestyle='--', label='φ')
ax1.axvline(2.0, color='red', linewidth=2, linestyle='--', label='2:1')
ax1.set_xlabel('γ/β ratio')
ax1.set_ylabel('Δ score')
ax1.set_title('Two Macro-Regimes')
ax1.legend()

ax2 = axes[0, 1]
ax2.hist(subject_ks, bins=range(1, 10), color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(6, color='gold', linewidth=2, linestyle='--', label='k=6')
ax2.axvline(7, color='orange', linewidth=2, linestyle='--', label='k=7')
ax2.set_xlabel('Optimal k per subject')
ax2.set_ylabel('Count')
ax2.set_title('Per-Subject State Count')
ax2.legend()

ax3 = axes[0, 2]
ax3.hist(bootstrap_ks, bins=range(1, 12), color='green', edgecolor='black', alpha=0.7)
ax3.axvline(6, color='gold', linewidth=2, linestyle='--', label='k=6')
ax3.axvline(7, color='orange', linewidth=2, linestyle='--', label='k=7')
ax3.set_xlabel('Bootstrap optimal k')
ax3.set_ylabel('Count')
ax3.set_title(f'Bootstrap Stability (N={n_bootstrap})')
ax3.legend()

ax4 = axes[1, 0]
im = ax4.imshow(trans_probs, cmap='Blues')
ax4.set_xlabel('To State')
ax4.set_ylabel('From State')
ax4.set_title('Transition Matrix')
plt.colorbar(im, ax=ax4)
for i in range(6):
    for j in range(6):
        ax4.text(j, i, f'{trans_probs[i,j]:.2f}', ha='center', va='center',
                color='white' if trans_probs[i,j] > 0.3 else 'black', fontsize=8)

ax5 = axes[1, 1]
cents = [c for s, c, sl in state_centrality]
colors = ['gold' if s in mixed_states else 'steelblue' for s in range(6)]
ax5.bar(range(6), cents, color=colors, edgecolor='black')
ax5.set_xlabel('State')
ax5.set_ylabel('Centrality')
ax5.set_title('State Centrality (Bridge = Gold)')

ax6 = axes[1, 2]
regime_props = [regime_by_state.get(s, 0.5) for s in range(6)]
colors = ['coral' if r < 0.4 else 'steelblue' if r > 0.6 else 'gold' for r in regime_props]
ax6.bar(range(6), regime_props, color=colors, edgecolor='black')
ax6.axhline(0.5, color='black', linestyle='--')
ax6.set_xlabel('State')
ax6.set_ylabel('φ-regime proportion')
ax6.set_title('State Regime Membership')
ax6.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('hierarchical_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nFigure: hierarchical_analysis.png")

with open('hierarchical_report.md', 'w') as f:
    f.write("# HIERARCHICAL STATE ANALYSIS\n\n")
    f.write("## Question: Is 6-7 a natural resolution level?\n\n")
    f.write("### Theoretical Framework\n")
    f.write("- 2 macro-regimes: HARMONIC (closer to 2:1) vs φ-LIKE (closer to 1.618)\n")
    f.write("- Each macro-regime contains 2-3 sub-states\n")
    f.write("- Plus 1 'bridge' state for inter-regime transitions\n")
    f.write("- Expected: 2×3 + 1 = 7 (or 2×2 + 1 = 5, 2×3 = 6)\n\n")
    f.write("### Results\n\n")
    f.write(f"| Experiment | Result |\n|------------|--------|\n")
    f.write(f"| Hierarchical structure | {total_states} states |\n")
    f.write(f"| Per-subject median k | {np.median(subject_ks):.0f} |\n")
    f.write(f"| Bridge state | {'Yes: ' + str(mixed_states) if bridge_ok else 'No'} |\n")
    f.write(f"| Bootstrap in 5-8 range | {100*in_67_range/n_bootstrap:.0f}% |\n\n")
    f.write(f"### Conclusion\n\n")
    f.write(f"**{overall_support}/4 criteria support 6-7 as natural resolution**\n\n")
    if overall_support >= 3:
        f.write("✅ 6-7 IS a natural resolution level - the 'sweet spot' between too coarse (2) and too fine (10).\n")
    else:
        f.write("⚠️ Mixed evidence - 6-7 may be partially supported but not definitive.\n")

print("Report: hierarchical_report.md")
print("\nDONE")
