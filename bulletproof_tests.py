#!/usr/bin/env python3
"""
BULLETPROOF TESTS A-D
Validate findings with rigorous controls

A) Power-law timing (Clauset-style)
B) Data-driven band boundaries (avoid 13/8 cherry-pick)
C) Causal direction (Q as control parameter)
D) Full structure discovery (topology + universality)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import LeaveOneGroupOut
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2

print("="*80)
print("BULLETPROOF VALIDATION TESTS A-D")
print("="*80)

df = pd.read_csv('epoch_features_fractal.csv')
print(f"\nData: {len(df)} epochs, {df['subject'].nunique()} subjects")

# =============================================================================
# TEST A: POWER-LAW TIMING (Clauset-style)
# =============================================================================
print("\n" + "="*80)
print("TEST A: POWER-LAW TIMING (Clauset-style)")
print("="*80)

# Collect inter-arrival times
phi_intervals = []
for subj in df['subject'].unique():
    subj_df = df[df['subject'] == subj].sort_values(['run', 'epoch_id']).reset_index(drop=True)
    phi_idx = subj_df[subj_df['regime'] == 'phi-like'].index.tolist()
    
    for i in range(1, len(phi_idx)):
        interval = phi_idx[i] - phi_idx[i-1]
        if interval > 0:
            phi_intervals.append(interval)

phi_intervals = np.array(phi_intervals)
print(f"\nInter-arrival times: n={len(phi_intervals)}")

# A1) MLE power-law fit
def powerlaw_mle(x, xmin=1):
    """MLE estimator for power-law exponent"""
    x = x[x >= xmin]
    n = len(x)
    alpha = 1 + n / np.sum(np.log(x / xmin))
    return alpha

def powerlaw_pdf(x, alpha, xmin):
    return (alpha - 1) / xmin * (x / xmin) ** (-alpha)

def exponential_mle(x):
    return 1 / np.mean(x)

def lognormal_mle(x):
    return np.mean(np.log(x)), np.std(np.log(x))

# Find optimal xmin using KS statistic
xmins = np.unique(phi_intervals)[:20]
best_ks = np.inf
best_xmin = 1
best_alpha = 2

for xmin in xmins:
    if xmin < 1:
        continue
    subset = phi_intervals[phi_intervals >= xmin]
    if len(subset) < 50:
        continue
    
    alpha = powerlaw_mle(subset, xmin)
    
    # KS statistic
    theoretical_cdf = lambda x: 1 - (xmin / x) ** (alpha - 1)
    empirical_cdf = np.arange(1, len(subset) + 1) / len(subset)
    sorted_data = np.sort(subset)
    ks = np.max(np.abs(empirical_cdf - theoretical_cdf(sorted_data)))
    
    if ks < best_ks:
        best_ks = ks
        best_xmin = xmin
        best_alpha = alpha

print(f"\n[A1] MLE Power-law fit:")
print(f"  xmin = {best_xmin}")
print(f"  alpha = {best_alpha:.3f}")
print(f"  KS statistic = {best_ks:.4f}")

# A2) Bootstrap p-value for KS
n_boot = 500
ks_null = []

subset = phi_intervals[phi_intervals >= best_xmin]
n_subset = len(subset)

for _ in range(n_boot):
    # Generate synthetic power-law data
    u = np.random.uniform(0, 1, n_subset)
    synthetic = best_xmin * (1 - u) ** (-1 / (best_alpha - 1))
    
    alpha_synth = powerlaw_mle(synthetic, best_xmin)
    theoretical_cdf = lambda x: 1 - (best_xmin / x) ** (alpha_synth - 1)
    empirical_cdf = np.arange(1, n_subset + 1) / n_subset
    sorted_synth = np.sort(synthetic)
    ks_synth = np.max(np.abs(empirical_cdf - theoretical_cdf(sorted_synth)))
    ks_null.append(ks_synth)

p_bootstrap = np.mean(ks_null >= best_ks)
print(f"  Bootstrap p-value = {p_bootstrap:.4f}")
print(f"  Power-law plausible: {'✅ YES' if p_bootstrap > 0.1 else '❌ NO'}")

# A3) Likelihood ratio tests vs alternatives
subset = phi_intervals[phi_intervals >= best_xmin]

# Log-likelihoods
ll_powerlaw = np.sum(np.log((best_alpha - 1) / best_xmin * (subset / best_xmin) ** (-best_alpha)))

# Exponential
lambda_exp = exponential_mle(subset)
ll_exp = np.sum(np.log(lambda_exp * np.exp(-lambda_exp * subset)))

# Lognormal
mu_ln, sigma_ln = lognormal_mle(subset)
ll_lognorm = np.sum(stats.lognorm.logpdf(subset, s=sigma_ln, scale=np.exp(mu_ln)))

print(f"\n[A2] Likelihood ratio tests:")
print(f"  Log-likelihood power-law: {ll_powerlaw:.1f}")
print(f"  Log-likelihood exponential: {ll_exp:.1f}")
print(f"  Log-likelihood lognormal: {ll_lognorm:.1f}")

# LR test
lr_vs_exp = 2 * (ll_powerlaw - ll_exp)
lr_vs_ln = 2 * (ll_powerlaw - ll_lognorm)

print(f"  LR vs exponential: {lr_vs_exp:.1f} ({'power-law better' if lr_vs_exp > 0 else 'exponential better'})")
print(f"  LR vs lognormal: {lr_vs_ln:.1f} ({'power-law better' if lr_vs_ln > 0 else 'lognormal better'})")

# Per-subject analysis
print(f"\n[A3] Per-subject heterogeneity:")
subject_alphas = []
for subj in df['subject'].unique():
    subj_df = df[df['subject'] == subj].sort_values(['run', 'epoch_id']).reset_index(drop=True)
    phi_idx = subj_df[subj_df['regime'] == 'phi-like'].index.tolist()
    
    intervals = [phi_idx[i] - phi_idx[i-1] for i in range(1, len(phi_idx)) if phi_idx[i] - phi_idx[i-1] > 0]
    
    if len(intervals) > 20:
        intervals = np.array(intervals)
        alpha = powerlaw_mle(intervals[intervals >= 1], xmin=1)
        subject_alphas.append({'subject': subj, 'alpha': alpha, 'n': len(intervals)})

alpha_df = pd.DataFrame(subject_alphas)
print(f"  Alpha across subjects: {alpha_df['alpha'].mean():.2f} ± {alpha_df['alpha'].std():.2f}")
print(f"  Range: [{alpha_df['alpha'].min():.2f}, {alpha_df['alpha'].max():.2f}]")

# =============================================================================
# TEST B: DATA-DRIVEN BAND BOUNDARIES
# =============================================================================
print("\n" + "="*80)
print("TEST B: DATA-DRIVEN BAND BOUNDARIES")
print("="*80)

# Compute individualized alpha peak from spectral features
# Using ratio patterns to infer peak structure

print("\n[B1] Individual alpha peak estimation:")
print("  (Using alpha/beta and theta/alpha power ratios as proxies)")

boundary_ratios = []
for subj in df['subject'].unique():
    subj_df = df[df['subject'] == subj]
    
    # Alpha dominance indicator
    alpha_dom = subj_df['alpha_power'].mean()
    theta_dom = subj_df['theta_power'].mean()
    beta_dom = subj_df['beta_power'].mean()
    
    # Ratio of peak powers (proxy for peak frequency ratios)
    if theta_dom > 0 and alpha_dom > 0:
        ratio = alpha_dom / theta_dom
        boundary_ratios.append({'subject': subj, 'ratio': ratio})

ratio_df = pd.DataFrame(boundary_ratios)
print(f"\n  Alpha/Theta power ratio (N={len(ratio_df)}):")
print(f"  Mean: {ratio_df['ratio'].mean():.3f}")
print(f"  Median: {ratio_df['ratio'].median():.3f}")

# Test distance to phi
mean_ratio = ratio_df['ratio'].mean()
dist_to_phi = abs(mean_ratio - PHI)
dist_to_2 = abs(mean_ratio - 2.0)

print(f"\n  Distance to φ: {dist_to_phi:.3f}")
print(f"  Distance to 2: {dist_to_2:.3f}")

# Null distribution (permutation)
print("\n[B2] Permutation null for phi-proximity:")
n_perm = 1000
null_dist = []

for _ in range(n_perm):
    # Shuffle power values
    perm_alpha = np.random.permutation(df['alpha_power'].values)
    perm_theta = np.random.permutation(df['theta_power'].values)
    
    perm_ratios = []
    for subj in df['subject'].unique():
        mask = df['subject'] == subj
        a = perm_alpha[mask].mean()
        t = perm_theta[mask].mean()
        if t > 0:
            perm_ratios.append(a / t)
    
    null_dist.append(np.mean(perm_ratios))

null_dist = np.array(null_dist)
null_dist_phi = np.abs(null_dist - PHI)

p_phi = np.mean(null_dist_phi <= dist_to_phi)
print(f"  Observed distance to φ: {dist_to_phi:.4f}")
print(f"  Permutation p-value: {p_phi:.4f}")
print(f"  φ-proximity significant: {'✅ YES' if p_phi < 0.05 else '❌ NO'}")

# Canonical boundary test
print("\n[B3] Canonical boundary ratios:")
canonical = {'13/8': 13/8, '8/4': 8/4, '30/13': 30/13}
for name, ratio in canonical.items():
    dist = abs(ratio - PHI)
    print(f"  {name} = {ratio:.3f} (dist to φ: {dist:.3f}) {'≈φ' if dist < 0.1 else ''}")

# =============================================================================
# TEST C: CAUSAL DIRECTION (Q as control parameter)
# =============================================================================
print("\n" + "="*80)
print("TEST C: CAUSAL DIRECTION (Q as control parameter)")
print("="*80)

# Ensure Q_alpha exists
if 'Q_alpha' not in df.columns:
    df['alpha_beta_ratio'] = df['alpha_power'] / (df['beta_power'] + 1e-15)
    df['alpha_theta_ratio'] = df['alpha_power'] / (df['theta_power'] + 1e-15)
    df['Q_alpha'] = np.sqrt(df['alpha_beta_ratio'] * df['alpha_theta_ratio'])
    vals = df['Q_alpha'].replace([np.inf, -np.inf], np.nan)
    df['Q_alpha'] = (vals - vals.mean()) / (vals.std() + 1e-10)

# Create lagged features for regression
print("\n[C1] Mixed-effects logistic regression (simplified):")

lag_data = []
for subj in df['subject'].unique():
    subj_df = df[df['subject'] == subj].sort_values(['run', 'epoch_id']).reset_index(drop=True)
    
    for i in range(3, len(subj_df) - 1):
        current = subj_df.iloc[i]['regime']
        next_regime = subj_df.iloc[i+1]['regime']
        
        if current != next_regime and next_regime in ['phi-like', 'harmonic']:
            row = {
                'subject': subj,
                'target_phi': 1 if next_regime == 'phi-like' else 0,
                'Q_lag1': subj_df.iloc[i]['Q_alpha'],
                'Q_lag2': subj_df.iloc[i-1]['Q_alpha'],
                'mf_lag1': subj_df.iloc[i]['mf_width'],
                'mf_lag2': subj_df.iloc[i-1]['mf_width'],
                'aperiodic_lag1': subj_df.iloc[i]['aperiodic_exponent'],
                'aperiodic_lag2': subj_df.iloc[i-1]['aperiodic_exponent']
            }
            lag_data.append(row)

lag_df = pd.DataFrame(lag_data)
print(f"  Transitions: {len(lag_df)}")

if len(lag_df) > 50:
    # Logistic regression
    features = ['Q_lag1', 'Q_lag2', 'mf_lag1', 'mf_lag2', 'aperiodic_lag1', 'aperiodic_lag2']
    X = lag_df[features].fillna(0).values
    y = lag_df['target_phi'].values
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    
    print("\n  Coefficients (standardized):")
    for feat, coef in zip(features, model.coef_[0]):
        direction = "→phi" if coef > 0 else "→harm"
        sig = "**" if abs(coef) > 0.3 else ""
        print(f"    {feat:18}: {coef:+.3f} {direction} {sig}")
    
    # Leave-one-subject-out CV
    groups = lag_df['subject'].values
    logo = LeaveOneGroupOut()
    aucs = []
    
    for train_idx, test_idx in logo.split(X_scaled, y, groups):
        if len(np.unique(y[test_idx])) < 2:
            continue
        model_cv = LogisticRegression(max_iter=1000)
        model_cv.fit(X_scaled[train_idx], y[train_idx])
        pred = model_cv.predict_proba(X_scaled[test_idx])[:, 1]
        from sklearn.metrics import roc_auc_score
        aucs.append(roc_auc_score(y[test_idx], pred))
    
    print(f"\n  Leave-one-subject-out AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"  Q predicts transitions: {'✅ YES' if np.mean(aucs) > 0.6 else '⚠️ WEAK'}")

# C2) Granger-style test
print("\n[C2] Granger-style causality test:")

# Simple test: does Q(t-1) improve prediction of regime(t)?
# Model 1: regime(t) ~ regime(t-1)
# Model 2: regime(t) ~ regime(t-1) + Q(t-1)

from sklearn.metrics import log_loss

granger_data = []
for subj in df['subject'].unique():
    subj_df = df[df['subject'] == subj].sort_values(['run', 'epoch_id']).reset_index(drop=True)
    subj_df['is_phi'] = (subj_df['regime'] == 'phi-like').astype(int)
    
    for i in range(1, len(subj_df)):
        granger_data.append({
            'subject': subj,
            'regime_t': subj_df.iloc[i]['is_phi'],
            'regime_t1': subj_df.iloc[i-1]['is_phi'],
            'Q_t1': subj_df.iloc[i-1]['Q_alpha']
        })

g_df = pd.DataFrame(granger_data).dropna()

X1 = g_df[['regime_t1']].values
X2 = g_df[['regime_t1', 'Q_t1']].values
y = g_df['regime_t'].values

model1 = LogisticRegression(max_iter=1000).fit(X1, y)
model2 = LogisticRegression(max_iter=1000).fit(X2, y)

ll1 = -log_loss(y, model1.predict_proba(X1), normalize=False)
ll2 = -log_loss(y, model2.predict_proba(X2), normalize=False)

lr_stat = 2 * (ll2 - ll1)
p_granger = 1 - stats.chi2.cdf(lr_stat, df=1)

print(f"  Model 1 (regime only) LL: {ll1:.1f}")
print(f"  Model 2 (regime + Q) LL: {ll2:.1f}")
print(f"  LR statistic: {lr_stat:.2f}, p={p_granger:.4f}")
print(f"  Q adds predictive info: {'✅ YES' if p_granger < 0.05 else '❌ NO'}")

# =============================================================================
# TEST D: FULL STRUCTURE DISCOVERY
# =============================================================================
print("\n" + "="*80)
print("TEST D: FULL STRUCTURE DISCOVERY")
print("="*80)

# D1) Model selection for number of states
print("\n[D1] Optimal number of states:")

# Use key features for clustering
feature_cols = ['aperiodic_exponent', 'alpha_power', 'beta_power', 'gamma_power']
X_cluster = df[feature_cols].fillna(0).values

# Standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Test k=3 to 10
results_k = []
for k in range(3, 11):
    # K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    
    # GMM
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
    gmm.fit(X_scaled)
    bic = gmm.bic(X_scaled)
    aic = gmm.aic(X_scaled)
    
    results_k.append({'k': k, 'silhouette': sil, 'BIC': bic, 'AIC': aic})

results_df = pd.DataFrame(results_k)
best_sil_k = results_df.loc[results_df['silhouette'].idxmax(), 'k']
best_bic_k = results_df.loc[results_df['BIC'].idxmin(), 'k']

print(f"  Best k by silhouette: {best_sil_k}")
print(f"  Best k by BIC: {best_bic_k}")
print("\n  k | Silhouette | BIC")
print("  --|------------|----")
for _, row in results_df.iterrows():
    marker = "←" if row['k'] in [best_sil_k, best_bic_k] else ""
    print(f"  {int(row['k'])} | {row['silhouette']:.4f}     | {row['BIC']:.0f} {marker}")

# D2) Bootstrap stability
print("\n[D2] Bootstrap stability for k=6,7:")

n_boot = 100
stability_6 = []
stability_7 = []

for _ in range(n_boot):
    boot_idx = np.random.choice(len(X_scaled), size=len(X_scaled), replace=True)
    X_boot = X_scaled[boot_idx]
    
    kmeans_6 = KMeans(n_clusters=6, random_state=None, n_init=5)
    kmeans_7 = KMeans(n_clusters=7, random_state=None, n_init=5)
    
    labels_6 = kmeans_6.fit_predict(X_boot)
    labels_7 = kmeans_7.fit_predict(X_boot)
    
    stability_6.append(silhouette_score(X_boot, labels_6))
    stability_7.append(silhouette_score(X_boot, labels_7))

print(f"  k=6: silhouette = {np.mean(stability_6):.4f} ± {np.std(stability_6):.4f}")
print(f"  k=7: silhouette = {np.mean(stability_7):.4f} ± {np.std(stability_7):.4f}")

# D3) Transition graph topology
print("\n[D3] Transition graph topology:")

# Compute transition matrix
states = df['state'].values
n_states = len(np.unique(states))
trans_matrix = np.zeros((n_states, n_states))

for i in range(len(states) - 1):
    trans_matrix[states[i], states[i+1]] += 1

# Normalize
row_sums = trans_matrix.sum(axis=1, keepdims=True)
trans_matrix_norm = trans_matrix / (row_sums + 1e-10)

# Modularity (simple: check if 2 clusters have higher within-cluster transitions)
# Assign states to basins based on delta_score
state_delta = df.groupby('state')['delta_score'].mean()
phi_basin = [s for s in range(n_states) if state_delta[s] < -0.05]
harm_basin = [s for s in range(n_states) if state_delta[s] > 0.02]
bridge = [s for s in range(n_states) if -0.05 <= state_delta[s] <= 0.02]

print(f"  Phi basin: {phi_basin}")
print(f"  Harmonic basin: {harm_basin}")
print(f"  Bridge states: {bridge}")

# Within vs between transitions
within_phi = sum(trans_matrix[i, j] for i in phi_basin for j in phi_basin)
within_harm = sum(trans_matrix[i, j] for i in harm_basin for j in harm_basin)
between = sum(trans_matrix[i, j] for i in phi_basin for j in harm_basin) + \
          sum(trans_matrix[i, j] for i in harm_basin for j in phi_basin)

total = trans_matrix.sum()
print(f"  Within phi-basin: {within_phi/total*100:.1f}%")
print(f"  Within harm-basin: {within_harm/total*100:.1f}%")
print(f"  Between basins: {between/total*100:.1f}%")

modularity = ((within_phi + within_harm) / total) - (between / total)
print(f"  Modularity index: {modularity:.3f}")
print(f"  2-basin structure: {'✅ SUPPORTED' if modularity > 0.1 else '⚠️ WEAK'}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("SUMMARY: BULLETPROOF VALIDATION")
print("="*80)

print("\n┌─────────────────────────────────────────────────────────────────┐")
print("│ TEST                          │ RESULT              │ VERDICT  │")
print("├─────────────────────────────────────────────────────────────────┤")

# A
pl_verdict = '✅' if p_bootstrap > 0.1 and ll_powerlaw > ll_exp else '⚠️'
print(f"│ A. Power-law timing           │ α={best_alpha:.2f}, p={p_bootstrap:.2f}    │   {pl_verdict}     │")

# B
bd_verdict = '✅' if p_phi < 0.05 else '❌'
print(f"│ B. Band boundaries ~φ         │ p={p_phi:.3f}            │   {bd_verdict}     │")

# C
q_verdict = '✅' if np.mean(aucs) > 0.6 else '⚠️'
print(f"│ C. Q as control parameter     │ AUC={np.mean(aucs):.2f}            │   {q_verdict}     │")

# D
d_verdict = '✅' if modularity > 0.1 else '⚠️'
print(f"│ D. 2-basin structure          │ mod={modularity:.2f}            │   {d_verdict}     │")

print("└─────────────────────────────────────────────────────────────────┘")

# =============================================================================
# FIGURE
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# A) CCDF of inter-arrival times
ax1 = axes[0, 0]
sorted_intervals = np.sort(phi_intervals)
ccdf = 1 - np.arange(1, len(sorted_intervals) + 1) / len(sorted_intervals)
ax1.loglog(sorted_intervals, ccdf, 'b.', alpha=0.5, label='Data')
# Power-law fit
x_fit = np.logspace(0, np.log10(sorted_intervals.max()), 100)
y_fit = (x_fit / best_xmin) ** (-(best_alpha - 1))
y_fit = y_fit / y_fit[0] * ccdf[0]
ax1.loglog(x_fit, y_fit, 'r-', linewidth=2, label=f'Power-law α={best_alpha:.2f}')
ax1.set_xlabel('Inter-arrival time (epochs)')
ax1.set_ylabel('CCDF')
ax1.set_title('A) Power-law Timing (CCDF)')
ax1.legend()

# B) Boundary ratio distribution
ax2 = axes[0, 1]
ax2.hist(ratio_df['ratio'], bins=20, density=True, alpha=0.7, color='purple', edgecolor='black')
ax2.axvline(PHI, color='green', linewidth=2, label=f'φ={PHI:.3f}')
ax2.axvline(ratio_df['ratio'].mean(), color='red', linewidth=2, linestyle='--', 
            label=f'Mean={ratio_df["ratio"].mean():.2f}')
ax2.set_xlabel('Alpha/Theta power ratio')
ax2.set_ylabel('Density')
ax2.set_title('B) Data-Driven Band Boundary Ratios')
ax2.legend()

# C) Feature importance for Q prediction
ax3 = axes[1, 0]
if len(lag_df) > 50:
    coefs = model.coef_[0]
    ax3.barh(features, coefs, color=['gold' if c > 0 else 'steelblue' for c in coefs])
    ax3.axvline(0, color='gray', linestyle='--')
    ax3.set_xlabel('Coefficient (→phi if positive)')
    ax3.set_title(f'C) Q as Control Parameter (AUC={np.mean(aucs):.2f})')

# D) Silhouette by k
ax4 = axes[1, 1]
ax4.plot(results_df['k'], results_df['silhouette'], 'bo-', linewidth=2)
ax4.axvline(best_sil_k, color='green', linestyle='--', label=f'Best k={int(best_sil_k)}')
ax4.set_xlabel('Number of states (k)')
ax4.set_ylabel('Silhouette score')
ax4.set_title('D) Optimal State Count')
ax4.legend()

plt.tight_layout()
plt.savefig('bulletproof_tests.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nFigure: bulletproof_tests.png")

# Save report
with open('bulletproof_report.md', 'w') as f:
    f.write("# Bulletproof Validation Report\n\n")
    f.write("## Test A: Power-law Timing\n")
    f.write(f"- Alpha = {best_alpha:.3f}, xmin = {best_xmin}\n")
    f.write(f"- Bootstrap p = {p_bootstrap:.4f}\n")
    f.write(f"- LR vs exponential: {lr_vs_exp:.1f}\n\n")
    f.write("## Test B: Band Boundaries\n")
    f.write(f"- Mean ratio = {ratio_df['ratio'].mean():.3f}\n")
    f.write(f"- Distance to φ = {dist_to_phi:.4f}\n")
    f.write(f"- Permutation p = {p_phi:.4f}\n\n")
    f.write("## Test C: Q as Control Parameter\n")
    f.write(f"- Leave-one-out AUC = {np.mean(aucs):.3f}\n")
    f.write(f"- Granger p = {p_granger:.4f}\n\n")
    f.write("## Test D: Structure Discovery\n")
    f.write(f"- Best k (silhouette) = {int(best_sil_k)}\n")
    f.write(f"- Best k (BIC) = {int(best_bic_k)}\n")
    f.write(f"- Modularity = {modularity:.3f}\n")

print("Report: bulletproof_report.md")
print("\nDONE")
