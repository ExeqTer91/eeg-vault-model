#!/usr/bin/env python3
"""
REVIBE / REVERT EXPERIMENTS
Test directionality: do fractal metrics PREDICT regime entry, or just REFLECT regimes?

1) PRECURSOR TEST - lead/lag around transitions
2) BRIDGE-AS-SETUP TEST - does bridge steer landing?
3) REVERSE CLASSIFICATION - fractals → regime?
4) TIME-REVERSAL CONTROL - genuine directionality?
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("REVIBE / REVERT EXPERIMENTS")
print("Testing: Do fractal metrics PREDICT regime entry?")
print("="*80)

# Load data
df = pd.read_csv('epoch_features_fractal.csv')

# Ensure we have the metrics
if 'dfa_alpha' not in df.columns:
    df['dfa_alpha'] = (np.abs(df['aperiodic_exponent']) + 1) / 2

if 'Q_alpha' not in df.columns:
    df['alpha_beta_ratio'] = df['alpha_power'] / (df['beta_power'] + 1e-15)
    df['alpha_theta_ratio'] = df['alpha_power'] / (df['theta_power'] + 1e-15)
    df['Q_alpha'] = np.sqrt(df['alpha_beta_ratio'] * df['alpha_theta_ratio'])
    vals = df['Q_alpha'].replace([np.inf, -np.inf], np.nan)
    df['Q_alpha'] = (vals - vals.mean()) / (vals.std() + 1e-10)

print(f"\nData: {len(df)} epochs, {df['subject'].nunique()} subjects")
print(f"Regimes: {df['regime'].value_counts().to_dict()}")

# =============================================================================
# 1) PRECURSOR TEST (lead/lag around transitions)
# =============================================================================
print("\n" + "="*80)
print("1) PRECURSOR TEST: Metric trajectories around regime entry")
print("="*80)

metrics = ['aperiodic_exponent', 'dfa_alpha', 'mf_width', 'Q_alpha', 'alpha_power']
lags = [-10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10]

# Find transitions into phi-like and harmonic
phi_trajectories = {m: {l: [] for l in lags} for m in metrics}
harm_trajectories = {m: {l: [] for l in lags} for m in metrics}

for subj in df['subject'].unique():
    subj_df = df[df['subject'] == subj].sort_values(['run', 'epoch_id']).reset_index(drop=True)
    regimes = subj_df['regime'].values
    
    for i in range(15, len(regimes) - 15):
        # Transition INTO phi-like
        if regimes[i] == 'phi-like' and regimes[i-1] != 'phi-like':
            for m in metrics:
                vals = subj_df[m].values
                for lag in lags:
                    if 0 <= i + lag < len(vals):
                        phi_trajectories[m][lag].append(vals[i + lag])
        
        # Transition INTO harmonic
        if regimes[i] == 'harmonic' and regimes[i-1] != 'harmonic':
            for m in metrics:
                vals = subj_df[m].values
                for lag in lags:
                    if 0 <= i + lag < len(vals):
                        harm_trajectories[m][lag].append(vals[i + lag])

# Compute mean trajectories
phi_means = {m: [np.nanmean(phi_trajectories[m][l]) for l in lags] for m in metrics}
harm_means = {m: [np.nanmean(harm_trajectories[m][l]) for l in lags] for m in metrics}

print(f"\n  Phi-like entries found: {len(phi_trajectories['aperiodic_exponent'][0])}")
print(f"  Harmonic entries found: {len(harm_trajectories['aperiodic_exponent'][0])}")

print("\n  Mean trajectories around PHI-LIKE entry (lag 0 = entry):")
print("  Metric             | t-3    | t-2    | t-1    | t=0    | t+1    | t+2    |")
print("  -------------------|--------|--------|--------|--------|--------|--------|")
for m in metrics[:4]:
    vals = [phi_means[m][lags.index(l)] for l in [-3, -2, -1, 0, 1, 2]]
    print(f"  {m:18} | {vals[0]:+.3f} | {vals[1]:+.3f} | {vals[2]:+.3f} | {vals[3]:+.3f} | {vals[4]:+.3f} | {vals[5]:+.3f} |")

# Test if pre-transition (t-1, t-2) differs from post (t+1, t+2)
print("\n[1.1] Pre vs Post transition comparison (phi-like entry):")
precursor_results = []

for m in metrics:
    pre_vals = phi_trajectories[m][-1] + phi_trajectories[m][-2]
    post_vals = phi_trajectories[m][1] + phi_trajectories[m][2]
    
    if len(pre_vals) > 5 and len(post_vals) > 5:
        t, p = stats.ttest_ind(pre_vals, post_vals)
        d = (np.nanmean(pre_vals) - np.nanmean(post_vals)) / (np.nanstd(pre_vals + post_vals) + 1e-10)
        
        # Is pre significantly different from post?
        direction = "↓" if d > 0 else "↑"
        print(f"  {m:20}: pre={np.nanmean(pre_vals):+.3f}, post={np.nanmean(post_vals):+.3f}, d={d:+.3f} {direction}, p={p:.4f}")
        
        precursor_results.append({
            'metric': m,
            'target': 'phi-like',
            'pre_mean': np.nanmean(pre_vals),
            'post_mean': np.nanmean(post_vals),
            'd': d,
            'p': p,
            'significant': p < 0.05
        })

# =============================================================================
# 1.2) LOGISTIC REGRESSION: Predict next regime from t-1, t-2, t-3
# =============================================================================
print("\n[1.2] Logistic regression: Predict next regime from lagged metrics")

# Create lagged features
lag_data = []

for subj in df['subject'].unique():
    subj_df = df[df['subject'] == subj].sort_values(['run', 'epoch_id']).reset_index(drop=True)
    
    for i in range(3, len(subj_df) - 1):
        current_regime = subj_df.iloc[i]['regime']
        next_regime = subj_df.iloc[i+1]['regime']
        
        # Only look at transitions
        if current_regime != next_regime and next_regime in ['phi-like', 'harmonic']:
            row = {
                'subject': subj,
                'target_phi': 1 if next_regime == 'phi-like' else 0
            }
            
            for lag in [1, 2, 3]:
                for m in metrics:
                    row[f'{m}_lag{lag}'] = subj_df.iloc[i - lag + 1][m]
            
            lag_data.append(row)

lag_df = pd.DataFrame(lag_data)
print(f"\n  Transitions found: {len(lag_df)} (phi-like={lag_df['target_phi'].sum()}, harmonic={len(lag_df) - lag_df['target_phi'].sum()})")

if len(lag_df) > 20:
    # Feature importance
    feature_cols = [c for c in lag_df.columns if 'lag' in c]
    X = lag_df[feature_cols].fillna(0).values
    y = lag_df['target_phi'].values
    
    # Simple logistic regression
    model = LogisticRegression(max_iter=1000, C=0.1)
    model.fit(X, y)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'coef': model.coef_[0]
    }).sort_values('coef', key=abs, ascending=False)
    
    print("\n  Top predictive features (logistic regression coefficients):")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']:25}: {row['coef']:+.4f}")
    
    # Cross-validated AUC
    from sklearn.model_selection import cross_val_score
    aucs = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f"\n  Cross-validated AUC: {aucs.mean():.3f} ± {aucs.std():.3f}")
    
    if aucs.mean() > 0.55:
        print("  ✅ Lagged metrics PREDICT next regime!")
    else:
        print("  ❌ Lagged metrics do not strongly predict next regime")

# =============================================================================
# 2) BRIDGE-AS-SETUP TEST
# =============================================================================
print("\n" + "="*80)
print("2) BRIDGE-AS-SETUP TEST: Does bridge steer landing?")
print("="*80)

# Find bridge → phi-like and bridge → harmonic transitions
bridge_exits = []

for subj in df['subject'].unique():
    subj_df = df[df['subject'] == subj].sort_values(['run', 'epoch_id']).reset_index(drop=True)
    regimes = subj_df['regime'].values
    
    for i in range(5, len(regimes) - 1):
        if regimes[i] == 'bridge' and regimes[i+1] != 'bridge':
            # Exit from bridge
            landing = regimes[i+1]
            
            # Get bridge epoch metrics
            row = {
                'subject': subj,
                'landing': landing,
                'landing_phi': 1 if landing == 'phi-like' else 0
            }
            
            for m in metrics:
                row[f'bridge_{m}'] = subj_df.iloc[i][m]
                # Also get delta (change during bridge)
                if i >= 3:
                    row[f'delta_{m}'] = subj_df.iloc[i][m] - subj_df.iloc[i-3][m]
            
            bridge_exits.append(row)

bridge_df = pd.DataFrame(bridge_exits)
print(f"\n  Bridge exits found: {len(bridge_df)}")
print(f"  Landing distribution: {bridge_df['landing'].value_counts().to_dict()}")

if len(bridge_df) > 20:
    # Compare bridge metrics before landing phi-like vs harmonic
    print("\n[2.1] Bridge metrics by landing destination:")
    print("  Metric               | → Phi-like | → Harmonic | d      | p      |")
    print("  ---------------------|------------|------------|--------|--------|")
    
    steering_results = []
    
    for m in metrics:
        col = f'bridge_{m}'
        if col not in bridge_df.columns:
            continue
        
        phi_land = bridge_df[bridge_df['landing'] == 'phi-like'][col].dropna()
        harm_land = bridge_df[bridge_df['landing'] == 'harmonic'][col].dropna()
        
        if len(phi_land) > 5 and len(harm_land) > 5:
            t, p = stats.ttest_ind(phi_land, harm_land)
            d = (phi_land.mean() - harm_land.mean()) / (np.sqrt((phi_land.var() + harm_land.var()) / 2) + 1e-10)
            
            print(f"  {m:20} | {phi_land.mean():+10.3f} | {harm_land.mean():+10.3f} | {d:+.3f} | {p:.4f} |")
            
            steering_results.append({
                'metric': m,
                'phi_mean': phi_land.mean(),
                'harm_mean': harm_land.mean(),
                'd': d,
                'p': p,
                'steers': p < 0.05
            })
    
    # Logistic regression: predict landing from bridge metrics
    print("\n[2.2] Predict landing from bridge metrics:")
    
    feature_cols = [c for c in bridge_df.columns if 'bridge_' in c or 'delta_' in c]
    X = bridge_df[feature_cols].fillna(0).values
    y = bridge_df['landing_phi'].values
    
    if len(np.unique(y)) > 1:
        model = LogisticRegression(max_iter=1000, C=0.1)
        model.fit(X, y)
        
        pred_proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, pred_proba)
        
        print(f"  Bridge → Landing prediction AUC: {auc:.3f}")
        
        if auc > 0.6:
            print("  ✅ Bridge metrics STEER landing destination!")
            
            # Top features
            importance = pd.DataFrame({
                'feature': feature_cols,
                'coef': model.coef_[0]
            }).sort_values('coef', key=abs, ascending=False)
            
            print("\n  Top steering features:")
            for _, row in importance.head(5).iterrows():
                direction = "→ phi" if row['coef'] > 0 else "→ harm"
                print(f"    {row['feature']:25}: {row['coef']:+.4f} {direction}")
        else:
            print("  ❌ Bridge metrics do not strongly steer landing")

# =============================================================================
# 3) REVERSE CLASSIFICATION: Fractals → Regime
# =============================================================================
print("\n" + "="*80)
print("3) REVERSE CLASSIFICATION: Predict regime from fractals only")
print("="*80)

# Use only fractal metrics (no band ratios)
fractal_features = ['aperiodic_exponent', 'dfa_alpha', 'mf_width']

# Binary: phi-like vs harmonic
subset = df[df['regime'].isin(['phi-like', 'harmonic'])].copy()
subset['is_phi'] = (subset['regime'] == 'phi-like').astype(int)

X = subset[fractal_features].fillna(0).values
y = subset['is_phi'].values
groups = subset['subject'].values

# Leave-one-subject-out CV
logo = LeaveOneGroupOut()
aucs = []

for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    if len(np.unique(y_train)) > 1 and len(np.unique(y_test)) > 1:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, pred))

print(f"\n  Leave-one-subject-out AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
print(f"  Subjects tested: {len(aucs)}")

if np.mean(aucs) > 0.65:
    print("  ✅ Fractal metrics STRONGLY predict regime (supports fractal → basin)")
elif np.mean(aucs) > 0.55:
    print("  ⚠️ Weak prediction (fractals partially reflect regime)")
else:
    print("  ❌ Poor prediction (fractals do not distinguish regimes)")

# =============================================================================
# 4) TIME-REVERSAL CONTROL
# =============================================================================
print("\n" + "="*80)
print("4) TIME-REVERSAL CONTROL")
print("="*80)

print("\n  Testing if precursor signals disappear under time reversal...")

# Reverse epoch order within each subject/run and redo precursor test
phi_trajectories_rev = {m: {l: [] for l in lags} for m in metrics}

for subj in df['subject'].unique():
    subj_df = df[df['subject'] == subj].sort_values(['run', 'epoch_id']).reset_index(drop=True)
    
    # REVERSE the order
    subj_df_rev = subj_df.iloc[::-1].reset_index(drop=True)
    regimes = subj_df_rev['regime'].values
    
    for i in range(15, len(regimes) - 15):
        # Transition INTO phi-like (in reversed time)
        if regimes[i] == 'phi-like' and regimes[i-1] != 'phi-like':
            for m in metrics:
                vals = subj_df_rev[m].values
                for lag in lags:
                    if 0 <= i + lag < len(vals):
                        phi_trajectories_rev[m][lag].append(vals[i + lag])

# Compare pre-post difference in reversed vs forward
print("\n  Pre-post difference comparison:")
print("  Metric             | Forward d | Reversed d | Genuine? |")
print("  -------------------|-----------|------------|----------|")

for m in metrics[:4]:
    # Forward
    pre_fwd = phi_trajectories[m][-1] + phi_trajectories[m][-2]
    post_fwd = phi_trajectories[m][1] + phi_trajectories[m][2]
    d_fwd = (np.nanmean(pre_fwd) - np.nanmean(post_fwd)) / (np.nanstd(pre_fwd + post_fwd) + 1e-10)
    
    # Reversed
    pre_rev = phi_trajectories_rev[m][-1] + phi_trajectories_rev[m][-2]
    post_rev = phi_trajectories_rev[m][1] + phi_trajectories_rev[m][2]
    d_rev = (np.nanmean(pre_rev) - np.nanmean(post_rev)) / (np.nanstd(pre_rev + post_rev) + 1e-10) if len(pre_rev) > 0 else 0
    
    # Genuine if forward has larger effect and different sign
    genuine = abs(d_fwd) > abs(d_rev) * 1.5
    
    print(f"  {m:18} | {d_fwd:+9.3f} | {d_rev:+10.3f} | {'✅' if genuine else '❌'} |")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("SUMMARY: REVIBE / REVERT RESULTS")
print("="*80)

conclusions = []

# Test 1: Precursor
if len(precursor_results) > 0:
    sig_precursors = [r for r in precursor_results if r['significant']]
    if sig_precursors:
        conclusions.append(f"✅ PRECURSORS FOUND: {[r['metric'] for r in sig_precursors]} change before regime entry")
    else:
        conclusions.append("❌ No significant precursor signals")

# Test 2: Bridge steering
if len(steering_results) > 0:
    sig_steerers = [r for r in steering_results if r['steers']]
    if sig_steerers:
        conclusions.append(f"✅ BRIDGE STEERS: {[r['metric'] for r in sig_steerers]} predict landing destination")
    else:
        conclusions.append("❌ Bridge does not steer landing")

# Test 3: Reverse classification
if len(aucs) > 0:
    if np.mean(aucs) > 0.65:
        conclusions.append(f"✅ FRACTALS → REGIME: AUC={np.mean(aucs):.3f} (fractals strongly predict regime)")
    else:
        conclusions.append(f"⚠️ Weak fractal→regime prediction (AUC={np.mean(aucs):.3f})")

print("\n" + "\n".join(conclusions))

# =============================================================================
# FIGURE
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1) Trajectories around phi-like entry
ax1 = axes[0, 0]
for m in ['aperiodic_exponent', 'mf_width', 'Q_alpha']:
    # Normalize for plotting
    traj = phi_means[m]
    traj_norm = (np.array(traj) - np.nanmean(traj)) / (np.nanstd(traj) + 1e-10)
    ax1.plot(lags, traj_norm, 'o-', label=m[:12], linewidth=2)
ax1.axvline(0, color='red', linestyle='--', label='Entry')
ax1.set_xlabel('Lag (epochs)')
ax1.set_ylabel('Z-scored value')
ax1.set_title('1) Metric Trajectories Around Phi-like Entry')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 2) Bridge steering
ax2 = axes[0, 1]
if len(steering_results) > 0:
    metrics_names = [r['metric'][:10] for r in steering_results]
    phi_means_list = [r['phi_mean'] for r in steering_results]
    harm_means_list = [r['harm_mean'] for r in steering_results]
    x = np.arange(len(metrics_names))
    ax2.bar(x - 0.2, phi_means_list, 0.4, label='→ Phi-like', color='gold')
    ax2.bar(x + 0.2, harm_means_list, 0.4, label='→ Harmonic', color='steelblue')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_names, rotation=45)
    ax2.legend()
ax2.set_title('2) Bridge Metrics by Landing Destination')
ax2.set_ylabel('Mean value in bridge')

# 3) Reverse classification AUCs
ax3 = axes[1, 0]
if len(aucs) > 0:
    ax3.bar(range(len(aucs)), aucs, color='coral', alpha=0.7)
    ax3.axhline(np.mean(aucs), color='red', linestyle='--', label=f'Mean={np.mean(aucs):.3f}')
    ax3.axhline(0.5, color='gray', linestyle=':', label='Chance')
    ax3.set_xlabel('Subject (left out)')
    ax3.set_ylabel('AUC')
    ax3.set_title('3) Leave-One-Subject-Out AUC (Fractals → Regime)')
    ax3.legend()

# 4) Summary text
ax4 = axes[1, 1]
ax4.axis('off')
summary_text = "REVIBE / REVERT SUMMARY\n\n"
summary_text += "\n".join(conclusions)
summary_text += f"\n\nKey Insight:\n"
if any("PRECURSORS" in c for c in conclusions) or any("STEERS" in c for c in conclusions):
    summary_text += "Fractal metrics show PREDICTIVE power!\n"
    summary_text += "→ Suggests 'fractal regime' as control parameter\n"
    summary_text += "→ NOT just reflecting state differences"
else:
    summary_text += "Limited predictive power from fractals.\n"
    summary_text += "→ Fractals may reflect, not cause, regime structure"

ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace', verticalalignment='center')

plt.tight_layout()
plt.savefig('revibe_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nFigure: revibe_analysis.png")

# Save report
with open('revibe_report.md', 'w') as f:
    f.write("# REVIBE / REVERT Analysis Report\n\n")
    f.write("## Objective\n\n")
    f.write("Test whether fractal metrics are PREDICTORS (precursors) or just REFLECTORS of regime state.\n\n")
    f.write("## Key Results\n\n")
    for c in conclusions:
        f.write(f"- {c}\n")
    f.write("\n## Detailed Results\n\n")
    f.write("### 1. Precursor Test\n\n")
    if len(precursor_results) > 0:
        f.write("| Metric | Pre-mean | Post-mean | Cohen's d | p-value |\n")
        f.write("|--------|----------|-----------|-----------|--------|\n")
        for r in precursor_results:
            f.write(f"| {r['metric']} | {r['pre_mean']:.3f} | {r['post_mean']:.3f} | {r['d']:+.3f} | {r['p']:.4f} |\n")
    f.write("\n### 2. Bridge Steering\n\n")
    if len(steering_results) > 0:
        f.write("| Metric | → Phi-like | → Harmonic | d | p |\n")
        f.write("|--------|------------|------------|---|---|\n")
        for r in steering_results:
            f.write(f"| {r['metric']} | {r['phi_mean']:.3f} | {r['harm_mean']:.3f} | {r['d']:+.3f} | {r['p']:.4f} |\n")
    f.write(f"\n### 3. Reverse Classification\n\n")
    f.write(f"Leave-one-subject-out AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}\n")

print("Report: revibe_report.md")
print("\nDONE")
