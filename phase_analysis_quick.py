#!/usr/bin/env python3
"""
Quick Phase Analysis on pre-extracted epoch features
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PHI = 1.618033988749895

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn import hmm
    HAS_HMM = True
except:
    HAS_HMM = False

print("="*80)
print("PHASE MODEL ANALYSIS (Quick Version)")
print("="*80)

df = pd.read_csv('epoch_features_full.csv')
print(f"\nLoaded: {len(df)} region-epochs, {df['subject'].nunique()} subjects")
print(f"Conditions: {list(df['condition'].unique())}")

feature_cols = ['r', 'delta_score', 'aperiodic_exponent', 'alpha_power', 'beta_power']
df_complete = df.dropna(subset=feature_cols)
X = df_complete[feature_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n[B] STATE DISCOVERY")
print("="*60)

k_range = range(2, 11)

print("\n[B1] Gaussian Mixture Models (GMM)")
gmm_bic = []
gmm_sil = []
for k in k_range:
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=5, max_iter=100)
    gmm.fit(X_scaled)
    bic = gmm.bic(X_scaled)
    labels = gmm.predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    gmm_bic.append(bic)
    gmm_sil.append(sil)
    print(f"  k={k}: BIC={bic:.0f}, silhouette={sil:.3f}")

best_k_gmm = list(k_range)[np.argmin(gmm_bic)]
print(f"  Best k (BIC): {best_k_gmm}")

print("\n[B2] K-Means Clustering")
km_sil = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    km_sil.append(sil)
    print(f"  k={k}: silhouette={sil:.3f}")

best_k_km = list(k_range)[np.argmax(km_sil)]
print(f"  Best k (silhouette): {best_k_km}")

print("\n[B3] HMM State Discovery")
if HAS_HMM:
    hmm_bic = []
    for k in k_range:
        try:
            model = hmm.GaussianHMM(n_components=k, covariance_type='diag', 
                                    n_iter=50, random_state=42)
            model.fit(X_scaled[:5000])
            score = model.score(X_scaled[:5000])
            n_params = k * (k - 1) + k * len(feature_cols) * 2
            bic = -2 * score + n_params * np.log(5000)
            hmm_bic.append(bic)
            print(f"  k={k}: BIC={bic:.0f}")
        except:
            hmm_bic.append(np.inf)
    best_k_hmm = list(k_range)[np.argmin(hmm_bic)]
    print(f"  Best k (BIC): {best_k_hmm}")
else:
    best_k_hmm = best_k_km
    print("  HMM skipped")

best_k = int(np.median([best_k_gmm, best_k_km, best_k_hmm]))
print(f"\n  CONSENSUS: k = {best_k}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax1 = axes[0]
ax1.plot(list(k_range), gmm_bic, 'b-o')
ax1.axvline(best_k_gmm, color='blue', linestyle='--', alpha=0.5)
ax1.axvline(6, color='gold', linestyle=':', linewidth=2, label='k=6')
ax1.axvline(7, color='orange', linestyle=':', linewidth=2, label='k=7')
ax1.set_xlabel('k')
ax1.set_ylabel('BIC')
ax1.set_title('GMM Model Selection')
ax1.legend()

ax2 = axes[1]
ax2.plot(list(k_range), gmm_sil, 'g-o', label='GMM')
ax2.plot(list(k_range), km_sil, 'r-o', label='K-Means')
ax2.axvline(best_k_km, color='red', linestyle='--', alpha=0.5)
ax2.set_xlabel('k')
ax2.set_ylabel('Silhouette')
ax2.set_title('Clustering Quality')
ax2.legend()

ax3 = axes[2]
methods = ['GMM\n(BIC)', 'K-Means\n(sil)', 'HMM\n(BIC)']
best_ks = [best_k_gmm, best_k_km, best_k_hmm]
colors = ['blue', 'red', 'magenta']
bars = ax3.bar(methods, best_ks, color=colors, alpha=0.7)
ax3.axhline(6, color='gold', linestyle='--', linewidth=2)
ax3.axhline(7, color='orange', linestyle='--', linewidth=2)
ax3.set_ylabel('Best k')
ax3.set_title(f'Consensus: k = {best_k}')
ax3.set_ylim(0, 12)

plt.tight_layout()
plt.savefig('state_model_selection.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  Figure: state_model_selection.png")

print("\n[C] STATE INTERPRETATION")
print("="*60)

final_gmm = GaussianMixture(n_components=best_k, random_state=42, n_init=10)
df_complete['state'] = final_gmm.fit_predict(X_scaled)

state_stats = df_complete.groupby('state').agg({
    'r': 'mean',
    'delta_score': 'mean',
    'hit_phi': 'mean',
    'aperiodic_exponent': 'mean',
    'alpha_power': 'mean',
    'beta_power': 'mean',
    'gamma_power': 'mean'
}).round(4)

print("\nPer-State Statistics:")
print(state_stats.to_string())

state_region = pd.crosstab(df_complete['state'], df_complete['region'], normalize='index')
print("\nState x Region:")
print(state_region.round(3).to_string())

state_condition = pd.crosstab(df_complete['state'], df_complete['condition'], normalize='index')
print("\nState x Condition:")
print(state_condition.round(3).to_string())

print("\n[D] HYPOTHESIS TESTS")
print("="*60)

print("\n  H_phase: k ~ 6-7?")
phase_ok = 5 <= best_k <= 8
print(f"    Best k = {best_k}: {'SUPPORTED' if phase_ok else 'NOT SUPPORTED'}")

print("\n  H_access: φ-like stronger in eyes-closed?")
if 'eyes_closed' in df_complete['condition'].values and 'eyes_open' in df_complete['condition'].values:
    eo_delta = df_complete[df_complete['condition'] == 'eyes_open']['delta_score']
    ec_delta = df_complete[df_complete['condition'] == 'eyes_closed']['delta_score']
    t, p = stats.ttest_ind(ec_delta, eo_delta)
    print(f"    Eyes-open Δ: {eo_delta.mean():.4f}")
    print(f"    Eyes-closed Δ: {ec_delta.mean():.4f}")
    print(f"    t = {t:.3f}, p = {p:.4f}")
    access_ok = ec_delta.mean() > eo_delta.mean()
    print(f"    Result: {'SUPPORTED' if access_ok else 'NOT SUPPORTED'}")
else:
    access_ok = False

print("\n  H_topography_shift: State-dependent reversal?")
state_region_delta = df_complete.groupby(['state', 'region'])['delta_score'].mean().unstack()
if 'frontal' in state_region_delta.columns and 'posterior' in state_region_delta.columns:
    diff = state_region_delta['frontal'] - state_region_delta['posterior']
    print(f"    Frontal - Posterior Δ by state:")
    for s, d in diff.items():
        print(f"      State {s}: {d:.4f}")
    reversal_ok = diff.max() > 0.02 and diff.min() < -0.02
    print(f"    Reversal: {'SUPPORTED' if reversal_ok else 'NOT SUPPORTED'}")
else:
    reversal_ok = False

trans_counts = np.zeros((best_k, best_k))
for subj in df_complete['subject'].unique():
    subj_data = df_complete[df_complete['subject'] == subj].sort_values(['run', 'epoch_id'])
    states = subj_data['state'].values
    for i in range(len(states) - 1):
        trans_counts[states[i], states[i+1]] += 1

trans_probs = trans_counts / (trans_counts.sum(axis=1, keepdims=True) + 1e-10)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
im = ax1.imshow(trans_probs, cmap='Blues')
ax1.set_xlabel('To State')
ax1.set_ylabel('From State')
ax1.set_title('State Transition Probabilities')
plt.colorbar(im, ax=ax1)
for i in range(best_k):
    for j in range(best_k):
        ax1.text(j, i, f'{trans_probs[i,j]:.2f}', ha='center', va='center',
                color='white' if trans_probs[i,j] > 0.3 else 'black', fontsize=8)

ax2 = axes[0, 1]
state_deltas = df_complete.groupby('state')['delta_score'].mean()
colors = ['green' if d > 0 else 'red' for d in state_deltas]
ax2.bar(range(best_k), state_deltas, color=colors, alpha=0.7)
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_xlabel('State')
ax2.set_ylabel('Mean Δ')
ax2.set_title('Δ Score by State')

ax3 = axes[1, 0]
for s in range(min(5, best_k)):
    s_data = df_complete[df_complete['state'] == s]
    conds = ['eyes_open', 'eyes_closed', 'task1', 'task2']
    conds = [c for c in conds if c in df_complete['condition'].unique()]
    props = [len(s_data[s_data['condition'] == c]) / len(s_data) for c in conds]
    ax3.plot(range(len(conds)), props, 'o-', label=f'State {s}')
ax3.set_xticks(range(len(conds)))
ax3.set_xticklabels([c.replace('_', '\n') for c in conds])
ax3.set_ylabel('Proportion')
ax3.set_title('State by Condition')
ax3.legend()

ax4 = axes[1, 1]
if 'frontal' in state_region_delta.columns and 'posterior' in state_region_delta.columns:
    x = range(best_k)
    width = 0.35
    ax4.bar([i - width/2 for i in x], state_region_delta['frontal'], width, label='Frontal', color='steelblue')
    ax4.bar([i + width/2 for i in x], state_region_delta['posterior'], width, label='Posterior', color='coral')
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.set_xlabel('State')
    ax4.set_ylabel('Mean Δ')
    ax4.set_title('Regional Δ by State')
    ax4.legend()

plt.tight_layout()
plt.savefig('state_by_condition.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  Figure: state_by_condition.png")

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(trans_probs, cmap='Blues')
ax.set_xlabel('To State')
ax.set_ylabel('From State')
ax.set_title(f'State Transition Matrix (k={best_k})')
plt.colorbar(im, ax=ax)
for i in range(best_k):
    for j in range(best_k):
        ax.text(j, i, f'{trans_probs[i,j]:.2f}', ha='center', va='center',
               color='white' if trans_probs[i,j] > 0.3 else 'black')
plt.tight_layout()
plt.savefig('state_transition_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Figure: state_transition_matrix.png")

with open('phase_report.md', 'w') as f:
    f.write("# PHASE MODEL ANALYSIS REPORT\n\n")
    f.write(f"**Dataset**: PhysioNet EEGMMIDB (N={df['subject'].nunique()} subjects)\n")
    f.write(f"**Epochs**: {len(df)} region-epochs\n\n")
    
    f.write("## Summary\n\n")
    f.write(f"### Best Number of States: **k = {best_k}**\n\n")
    f.write("| Method | Best k |\n|--------|--------|\n")
    f.write(f"| GMM (BIC) | {best_k_gmm} |\n")
    f.write(f"| K-Means (silhouette) | {best_k_km} |\n")
    f.write(f"| HMM (BIC) | {best_k_hmm} |\n\n")
    
    f.write("## Hypothesis Tests\n\n")
    f.write(f"- **H_phase (6-7 states)**: {'✅ SUPPORTED' if phase_ok else '❌ NOT SUPPORTED'} (k={best_k})\n")
    f.write(f"- **H_access (φ stronger in eyes-closed)**: {'✅ SUPPORTED' if access_ok else '❌ NOT SUPPORTED'}\n")
    f.write(f"- **H_topography_shift (reversal)**: {'✅ SUPPORTED' if reversal_ok else '❌ NOT SUPPORTED'}\n\n")
    
    f.write("## Per-State Statistics\n\n")
    f.write(state_stats.to_markdown() + "\n\n")
    
    f.write("## Figures\n")
    f.write("- `state_model_selection.png`\n")
    f.write("- `state_transition_matrix.png`\n")
    f.write("- `state_by_condition.png`\n")

print("\n  Report: phase_report.md")

df_complete.to_csv('epoch_features_with_states.csv', index=False)
print("  Data: epoch_features_with_states.csv")

print("\n" + "="*80)
print("PHASE ANALYSIS COMPLETE")
print("="*80)
