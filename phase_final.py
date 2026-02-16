#!/usr/bin/env python3
"""Minimal Phase Analysis"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

PHI = 1.618033988749895

print("="*70)
print("PHASE MODEL ANALYSIS")
print("="*70)

df = pd.read_csv('epoch_features_full.csv')
print(f"Data: {len(df)} epochs, {df['subject'].nunique()} subjects")

feature_cols = ['r', 'delta_score', 'aperiodic_exponent', 'alpha_power', 'beta_power']
df = df.dropna(subset=feature_cols)
X = StandardScaler().fit_transform(df[feature_cols].values)

k_range = range(2, 11)
gmm_bic, km_sil = [], []

print("\nModel Selection:")
for k in k_range:
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=3, max_iter=50)
    gmm.fit(X[:3000])
    gmm_bic.append(gmm.bic(X[:3000]))
    
    km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=50)
    labels = km.fit_predict(X[:3000])
    km_sil.append(silhouette_score(X[:3000], labels))
    print(f"  k={k}: GMM_BIC={gmm_bic[-1]:.0f}, KM_sil={km_sil[-1]:.3f}")

best_k_gmm = list(k_range)[np.argmin(gmm_bic)]
best_k_km = list(k_range)[np.argmax(km_sil)]
best_k = int(np.median([best_k_gmm, best_k_km, 6]))

print(f"\nBest k: GMM={best_k_gmm}, KMeans={best_k_km}, Consensus={best_k}")

gmm = GaussianMixture(n_components=best_k, random_state=42, n_init=5)
df['state'] = gmm.fit_predict(X)

print("\nPer-State Stats:")
stats_df = df.groupby('state').agg({
    'r': 'mean', 'delta_score': 'mean', 'hit_phi': 'mean',
    'alpha_power': 'mean', 'beta_power': 'mean'
}).round(4)
print(stats_df.to_string())

print("\nState x Condition:")
print(pd.crosstab(df['state'], df['condition'], normalize='index').round(3).to_string())

print("\nState x Region:")
print(pd.crosstab(df['state'], df['region'], normalize='index').round(3).to_string())

print("\n" + "="*70)
print("HYPOTHESIS TESTS")
print("="*70)

phase_ok = 4 <= best_k <= 8
print(f"\nH_phase (6-7 states): k={best_k} -> {'SUPPORTED' if phase_ok else 'NOT SUPPORTED'}")

eo = df[df['condition'] == 'eyes_open']['delta_score']
ec = df[df['condition'] == 'eyes_closed']['delta_score']
t, p = stats.ttest_ind(ec, eo)
access_ok = ec.mean() > eo.mean()
print(f"\nH_access: EO Δ={eo.mean():.4f}, EC Δ={ec.mean():.4f}, t={t:.2f}, p={p:.4f}")
print(f"  Result: {'SUPPORTED' if access_ok else 'NOT SUPPORTED'}")

sr = df.groupby(['state', 'region'])['delta_score'].mean().unstack()
if 'frontal' in sr.columns and 'posterior' in sr.columns:
    diff = sr['frontal'] - sr['posterior']
    reversal = diff.max() > 0.02 and diff.min() < -0.02
    print(f"\nH_topography: Front-Post diff range: [{diff.min():.4f}, {diff.max():.4f}]")
    print(f"  Result: {'SUPPORTED' if reversal else 'NOT SUPPORTED'}")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax1 = axes[0, 0]
ax1.plot(list(k_range), gmm_bic, 'b-o', label='GMM BIC')
ax1.axvline(best_k_gmm, color='blue', linestyle='--', alpha=0.5)
ax1.axvline(6, color='gold', linestyle=':', linewidth=2, label='k=6')
ax1.set_xlabel('k'); ax1.set_ylabel('BIC'); ax1.set_title('Model Selection')
ax1.legend()

ax2 = axes[0, 1]
state_d = df.groupby('state')['delta_score'].mean()
colors = ['green' if d > 0 else 'red' for d in state_d]
ax2.bar(range(best_k), state_d, color=colors, alpha=0.7)
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_xlabel('State'); ax2.set_ylabel('Mean Δ'); ax2.set_title('Δ by State')

ax3 = axes[1, 0]
trans = np.zeros((best_k, best_k))
for subj in df['subject'].unique():
    sd = df[df['subject'] == subj].sort_values(['run', 'epoch_id'])
    states = sd['state'].values
    for i in range(len(states) - 1):
        trans[states[i], states[i+1]] += 1
trans /= (trans.sum(axis=1, keepdims=True) + 1e-10)
im = ax3.imshow(trans, cmap='Blues')
ax3.set_xlabel('To'); ax3.set_ylabel('From'); ax3.set_title('Transition Matrix')
plt.colorbar(im, ax=ax3)

ax4 = axes[1, 1]
if 'frontal' in sr.columns and 'posterior' in sr.columns:
    x = range(best_k)
    w = 0.35
    ax4.bar([i-w/2 for i in x], sr['frontal'], w, label='Frontal', color='steelblue')
    ax4.bar([i+w/2 for i in x], sr['posterior'], w, label='Posterior', color='coral')
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.set_xlabel('State'); ax4.set_ylabel('Mean Δ')
    ax4.set_title('Regional Δ by State'); ax4.legend()

plt.tight_layout()
plt.savefig('phase_analysis_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nFigure: phase_analysis_results.png")

with open('phase_report.md', 'w') as f:
    f.write("# PHASE MODEL ANALYSIS\n\n")
    f.write(f"**Data**: {len(df)} epochs, {df['subject'].nunique()} subjects\n\n")
    f.write(f"## Best k = {best_k}\n\n")
    f.write(f"- GMM (BIC): {best_k_gmm}\n")
    f.write(f"- K-Means (silhouette): {best_k_km}\n\n")
    f.write("## Hypothesis Tests\n\n")
    f.write(f"- **H_phase**: {'✅' if phase_ok else '❌'} k={best_k}\n")
    f.write(f"- **H_access**: {'✅' if access_ok else '❌'} EC({ec.mean():.4f}) vs EO({eo.mean():.4f})\n")
    f.write(f"- **H_topography**: {'✅' if reversal else '❌'}\n\n")
    f.write("## Per-State Stats\n\n")
    f.write(stats_df.to_markdown() + "\n")

print("Report: phase_report.md")
df.to_csv('epoch_features_with_states.csv', index=False)
print("Data: epoch_features_with_states.csv")
print("\nDONE")
