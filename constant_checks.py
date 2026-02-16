#!/usr/bin/env python3
"""
Three Mathematical Constant Checks:
1. Gamma-band check: Is bridge membership predicted by gamma amplitude/peakiness after 1/f?
2. Euler gamma constant check: Do entropic measures converge to γ ≈ 0.5772?
3. Omega constant check: Does p ≈ e^{-p} (Ω ≈ 0.5671) appear in transition behavior?
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import entr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

PHI = 1.618033988749895
EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni constant
OMEGA = 0.5671432904097838  # Lambert W(1) = Ω, satisfies Ω = e^{-Ω}

print("="*80)
print("MATHEMATICAL CONSTANT CHECKS")
print("="*80)

df = pd.read_csv('epoch_features_with_states.csv')
print(f"\nData: {len(df)} epochs, {df['subject'].nunique()} subjects, 6 states")

# =============================================================================
# CHECK 1: GAMMA-BAND PREDICTS BRIDGE MEMBERSHIP
# =============================================================================
print("\n" + "="*80)
print("CHECK 1: GAMMA-BAND → BRIDGE MEMBERSHIP")
print("Is bridge membership best predicted by gamma amplitude/peakiness after 1/f removal?")
print("="*80)

df['is_bridge'] = (df['state'] == 5).astype(int)

gamma_features = ['gamma_power', 'gamma_cf']  
control_features = ['alpha_power', 'beta_power', 'aperiodic_exponent']
all_features = gamma_features + control_features

df_clean = df.dropna(subset=all_features + ['is_bridge'])
X = df_clean[all_features].values
y = df_clean['is_bridge'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n[1A] Logistic Regression: Bridge ~ features")

model_full = LogisticRegression(max_iter=1000, random_state=42)
model_full.fit(X_scaled, y)

coefs = dict(zip(all_features, model_full.coef_[0]))
print("\n  Feature coefficients (standardized):")
for f, c in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"    {f}: {c:.4f}")

print("\n[1B] Comparison: Gamma-only vs Control-only models")

from sklearn.model_selection import cross_val_score

X_gamma = scaler.fit_transform(df_clean[gamma_features].values)
X_control = scaler.fit_transform(df_clean[control_features].values)

cv_gamma = cross_val_score(LogisticRegression(max_iter=1000), X_gamma, y, cv=5, scoring='roc_auc')
cv_control = cross_val_score(LogisticRegression(max_iter=1000), X_control, y, cv=5, scoring='roc_auc')
cv_full = cross_val_score(LogisticRegression(max_iter=1000), X_scaled, y, cv=5, scoring='roc_auc')

print(f"\n  Gamma-only AUC: {cv_gamma.mean():.3f} ± {cv_gamma.std():.3f}")
print(f"  Control-only AUC: {cv_control.mean():.3f} ± {cv_control.std():.3f}")
print(f"  Full model AUC: {cv_full.mean():.3f} ± {cv_full.std():.3f}")

gamma_better = cv_gamma.mean() > cv_control.mean()
print(f"\n  Result: {'✅ GAMMA BEST PREDICTOR' if gamma_better else '❌ Control features better'}")

gamma_unique = cv_full.mean() - cv_control.mean()
print(f"  Gamma unique contribution: +{gamma_unique:.3f} AUC")

# =============================================================================
# CHECK 2: EULER-MASCHERONI GAMMA (γ ≈ 0.5772) IN ENTROPY
# =============================================================================
print("\n" + "="*80)
print("CHECK 2: EULER-MASCHERONI CONSTANT (γ ≈ 0.5772)")
print("Do entropic/complexity measures converge to γ or simple transforms?")
print("="*80)

print("\n[2A] Per-state entropy of Δ distribution")

state_entropies = {}
for s in range(6):
    state_data = df[df['state'] == s]['delta_score'].values
    hist, _ = np.histogram(state_data, bins=20, density=True)
    hist = hist[hist > 0]
    h = -np.sum(hist * np.log(hist) * (state_data.max() - state_data.min()) / 20)
    state_entropies[s] = h
    print(f"  State {s}: H = {h:.4f}")

mean_H = np.mean(list(state_entropies.values()))
print(f"\n  Mean H = {mean_H:.4f}")
print(f"  Euler γ = {EULER_GAMMA:.4f}")
print(f"  Ratio H/γ = {mean_H/EULER_GAMMA:.4f}")
print(f"  Diff |H - γ| = {abs(mean_H - EULER_GAMMA):.4f}")

print("\n[2B] Normalized ratio distribution entropy")

r_vals = df['r'].dropna().values
r_normalized = (r_vals - r_vals.min()) / (r_vals.max() - r_vals.min())
hist_r, _ = np.histogram(r_normalized, bins=50, density=True)
hist_r = hist_r[hist_r > 0]
H_ratio = -np.sum(hist_r * np.log(hist_r) * 1/50)
print(f"  H(ratio) = {H_ratio:.4f}")

print("\n[2C] Cross-subject entropy stability")

subj_H = []
for subj in df['subject'].unique():
    subj_data = df[df['subject'] == subj]['delta_score'].values
    if len(subj_data) > 50:
        hist, _ = np.histogram(subj_data, bins=15, density=True)
        hist = hist[hist > 0]
        h = -np.sum(hist * np.log(hist) * (subj_data.max() - subj_data.min()) / 15)
        subj_H.append(h)

print(f"  Per-subject H: {np.mean(subj_H):.4f} ± {np.std(subj_H):.4f}")
print(f"  CV = {np.std(subj_H)/np.mean(subj_H):.3f}")

print("\n[2D] Test against surrogate nulls")

n_surr = 100
surr_H = []
for _ in range(n_surr):
    surr = np.random.permutation(df['delta_score'].dropna().values)
    hist, _ = np.histogram(surr, bins=20, density=True)
    hist = hist[hist > 0]
    h = -np.sum(hist * np.log(hist) * (surr.max() - surr.min()) / 20)
    surr_H.append(h)

real_H = mean_H
z_score = (real_H - np.mean(surr_H)) / np.std(surr_H)
p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))

print(f"\n  Real H = {real_H:.4f}")
print(f"  Surrogate H = {np.mean(surr_H):.4f} ± {np.std(surr_H):.4f}")
print(f"  z = {z_score:.2f}, p = {p_val:.4f}")

gamma_close = abs(mean_H - EULER_GAMMA) < 0.2
print(f"\n  Result: {'✅ CLOSE TO γ' if gamma_close else '❌ Not close to γ'} (|diff| = {abs(mean_H - EULER_GAMMA):.4f})")

# =============================================================================
# CHECK 3: OMEGA CONSTANT (Ω ≈ 0.5671) IN TRANSITIONS
# =============================================================================
print("\n" + "="*80)
print("CHECK 3: OMEGA CONSTANT (Ω ≈ 0.5671)")
print("Does p ≈ e^{-p} appear in transition/stationary behavior?")
print("="*80)

print("\n[3A] Transition matrix analysis")

trans_counts = np.zeros((6, 6))
for subj in df['subject'].unique():
    sd = df[df['subject'] == subj].sort_values(['run', 'epoch_id'])
    states = sd['state'].values
    for i in range(len(states) - 1):
        trans_counts[states[i], states[i+1]] += 1

trans_probs = trans_counts / (trans_counts.sum(axis=1, keepdims=True) + 1e-10)

print("  Transition matrix diagonal (self-loop probabilities):")
for s in range(6):
    p_self = trans_probs[s, s]
    omega_pred = np.exp(-p_self)
    match = abs(p_self - omega_pred) < 0.15
    print(f"    State {s}: p={p_self:.3f}, e^{{-p}}={omega_pred:.3f}, |diff|={abs(p_self-omega_pred):.3f} {'✓' if match else ''}")

print("\n[3B] Stationary distribution")

eigenvalues, eigenvectors = np.linalg.eig(trans_probs.T)
stationary_idx = np.argmin(np.abs(eigenvalues - 1))
stationary = np.real(eigenvectors[:, stationary_idx])
stationary = stationary / stationary.sum()

print("  Stationary distribution π:")
for s in range(6):
    print(f"    State {s}: π = {stationary[s]:.4f}")

print("\n[3C] Testing p ≈ e^{-p} (Ω fixed-point)")

omega_residuals = []
for s in range(6):
    p = stationary[s]
    if p > 0.01:
        residual = abs(p - np.exp(-p))
        omega_residuals.append((s, p, residual))
        print(f"    State {s}: p={p:.4f}, e^{{-p}}={np.exp(-p):.4f}, residual={residual:.4f}")

closest = min(omega_residuals, key=lambda x: x[2])
print(f"\n  Closest to Ω fixed-point: State {closest[0]} (residual = {closest[2]:.4f})")

print("\n[3D] Bootstrap stability of Ω-like state")

n_boot = 50
omega_states = []
for _ in range(n_boot):
    idx = np.random.choice(len(df), size=len(df), replace=True)
    boot_df = df.iloc[idx]
    
    boot_trans = np.zeros((6, 6))
    for subj in boot_df['subject'].unique():
        sd = boot_df[boot_df['subject'] == subj].sort_values(['run', 'epoch_id'])
        states = sd['state'].values
        for i in range(len(states) - 1):
            if states[i] < 6 and states[i+1] < 6:
                boot_trans[states[i], states[i+1]] += 1
    
    boot_probs = boot_trans / (boot_trans.sum(axis=1, keepdims=True) + 1e-10)
    
    try:
        ev, evec = np.linalg.eig(boot_probs.T)
        stat_idx = np.argmin(np.abs(ev - 1))
        stat = np.real(evec[:, stat_idx])
        stat = stat / stat.sum()
        
        best_s = 0
        best_res = 1e10
        for s in range(6):
            if stat[s] > 0.01:
                res = abs(stat[s] - np.exp(-stat[s]))
                if res < best_res:
                    best_res = res
                    best_s = s
        omega_states.append(best_s)
    except:
        pass

if omega_states:
    omega_mode = stats.mode(omega_states, keepdims=False)[0]
    omega_freq = sum(1 for s in omega_states if s == omega_mode) / len(omega_states)
    print(f"  Most frequent Ω-like state: {omega_mode} ({100*omega_freq:.0f}% of bootstraps)")

omega_found = closest[2] < 0.15
print(f"\n  Result: {'✅ Ω FIXED-POINT FOUND' if omega_found else '❌ No clear Ω relation'}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("SUMMARY: MATHEMATICAL CONSTANT CHECKS")
print("="*80)

print("\n┌────────────────────────────────────────────────────────────────────┐")
print("│ CHECK                              │ RESULT                        │")
print("├────────────────────────────────────────────────────────────────────┤")
print(f"│ 1. Gamma-band → Bridge             │ {'✅ SUPPORTED' if gamma_better else '❌ NOT SUPPORTED':30} │")
print(f"│    (AUC gamma={cv_gamma.mean():.3f} vs ctrl={cv_control.mean():.3f})   │                               │")
print("├────────────────────────────────────────────────────────────────────┤")
print(f"│ 2. Euler γ in entropy              │ {'✅ CLOSE' if gamma_close else '❌ NOT CLOSE':30} │")
print(f"│    (H={mean_H:.4f}, γ={EULER_GAMMA:.4f}, diff={abs(mean_H-EULER_GAMMA):.4f}) │                               │")
print("├────────────────────────────────────────────────────────────────────┤")
print(f"│ 3. Ω fixed-point in transitions    │ {'✅ FOUND' if omega_found else '❌ NOT FOUND':30} │")
print(f"│    (State {closest[0]}: residual={closest[2]:.4f})         │                               │")
print("└────────────────────────────────────────────────────────────────────┘")

with open('constant_checks_report.md', 'w') as f:
    f.write("# Mathematical Constant Checks Report\n\n")
    f.write("## Summary\n\n")
    f.write("| Check | Constant | Result | Evidence |\n")
    f.write("|-------|----------|--------|----------|\n")
    f.write(f"| 1. Gamma-band → Bridge | - | {'✅' if gamma_better else '❌'} | AUC: {cv_gamma.mean():.3f} vs {cv_control.mean():.3f} |\n")
    f.write(f"| 2. Euler γ in entropy | γ ≈ 0.577 | {'✅' if gamma_close else '❌'} | H = {mean_H:.4f} |\n")
    f.write(f"| 3. Ω in transitions | Ω ≈ 0.567 | {'✅' if omega_found else '❌'} | State {closest[0]}, residual = {closest[2]:.4f} |\n")
    f.write("\n## Details\n\n")
    f.write("### Check 1: Gamma-band predicts bridge membership\n\n")
    f.write(f"- Gamma-only model AUC: {cv_gamma.mean():.3f} ± {cv_gamma.std():.3f}\n")
    f.write(f"- Control-only model AUC: {cv_control.mean():.3f} ± {cv_control.std():.3f}\n")
    f.write(f"- Gamma unique contribution: +{gamma_unique:.3f}\n\n")
    f.write("### Check 2: Euler-Mascheroni γ\n\n")
    f.write(f"- Mean state entropy H = {mean_H:.4f}\n")
    f.write(f"- Euler γ = {EULER_GAMMA:.4f}\n")
    f.write(f"- Difference: {abs(mean_H - EULER_GAMMA):.4f}\n\n")
    f.write("### Check 3: Omega constant\n\n")
    f.write(f"- Closest state to Ω fixed-point: State {closest[0]}\n")
    f.write(f"- p = {closest[1]:.4f}, e^{{-p}} = {np.exp(-closest[1]):.4f}\n")
    f.write(f"- Residual: {closest[2]:.4f}\n")
    if omega_states:
        f.write(f"- Bootstrap stability: State {omega_mode} in {100*omega_freq:.0f}% of resamples\n")

print("\nReport: constant_checks_report.md")
print("\nDONE")
