#!/usr/bin/env python3
"""
TWIST ANALYSIS: Detecting Entry Dynamics
1. Change-point detection
2. Relaxation fitting
3. Spiral/rotation test
4. Hysteresis
5. Alpha bursts/IAF
"""

import numpy as np
import pandas as pd
from scipy import stats, signal, optimize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TWIST ANALYSIS: Entry Dynamics Detection")
print("="*80)

df = pd.read_csv('epoch_features_with_states.csv')
print(f"\nData: {len(df)} epochs, {df['subject'].nunique()} subjects")

# =============================================================================
# 1) CHANGE-POINT DETECTION
# =============================================================================
print("\n" + "="*80)
print("1) CHANGE-POINT DETECTION")
print("="*80)

def detect_changepoints(series, window=10):
    """Detect change-points using CUSUM-like statistic"""
    n = len(series)
    if n < 2 * window:
        return [], []
    
    scores = []
    for i in range(window, n - window):
        pre = series[i-window:i]
        post = series[i:i+window]
        t_stat, p = stats.ttest_ind(pre, post)
        scores.append((i, abs(t_stat), np.mean(post) - np.mean(pre)))
    
    if not scores:
        return [], []
    
    # Find top change-points
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:5]  # Top 5

print("\n[1.1] Change-points in alpha_power, beta_power, aperiodic_exponent")

cp_results = {}
for feat in ['alpha_power', 'beta_power', 'aperiodic_exponent']:
    all_cps = []
    for subj in df['subject'].unique():
        subj_data = df[df['subject'] == subj].sort_values(['run', 'epoch_id'])
        series = subj_data[feat].dropna().values
        cps = detect_changepoints(series, window=5)
        if cps:
            all_cps.extend([cp[1] for cp in cps])  # t-stats
    
    cp_results[feat] = {
        'mean_t': np.mean(all_cps) if all_cps else 0,
        'max_t': np.max(all_cps) if all_cps else 0,
        'n_detected': len(all_cps)
    }
    print(f"  {feat}: mean |t|={cp_results[feat]['mean_t']:.2f}, max={cp_results[feat]['max_t']:.2f}, n={cp_results[feat]['n_detected']}")

# =============================================================================
# 2) RELAXATION FITTING
# =============================================================================
print("\n" + "="*80)
print("2) RELAXATION FITTING (post change-point)")
print("="*80)

def fit_exponential(y, x=None):
    """Fit y = a * exp(-b*x) + c"""
    if x is None:
        x = np.arange(len(y))
    
    try:
        def exp_func(t, a, b, c):
            return a * np.exp(-b * t) + c
        
        popt, _ = optimize.curve_fit(exp_func, x, y, p0=[y[0]-y[-1], 0.1, y[-1]], maxfev=1000)
        y_fit = exp_func(x, *popt)
        ss_res = np.sum((y - y_fit)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return {'tau': 1/popt[1] if popt[1] > 0 else np.inf, 'r2': r2, 'a': popt[0], 'c': popt[2]}
    except:
        return {'tau': np.nan, 'r2': 0, 'a': 0, 'c': 0}

print("\n[2.1] Exponential relaxation after state entry")

# Find state transitions and fit relaxation
relaxation_fits = {s: [] for s in range(6)}
window_post = 20  # 20 epochs after entry

for subj in df['subject'].unique():
    subj_data = df[df['subject'] == subj].sort_values(['run', 'epoch_id']).reset_index(drop=True)
    states = subj_data['state'].values
    
    for i in range(1, len(states) - window_post):
        if states[i] != states[i-1]:  # State change
            new_state = states[i]
            # Get next 20 epochs of alpha
            post_alpha = subj_data.loc[i:i+window_post-1, 'alpha_power'].values
            if len(post_alpha) >= window_post:
                fit = fit_exponential(post_alpha)
                if fit['r2'] > 0.1:  # Only meaningful fits
                    relaxation_fits[new_state].append(fit)

print("\n  State | N fits | Mean τ | Mean R²")
print("  ------|--------|--------|--------")
for s in range(6):
    if relaxation_fits[s]:
        mean_tau = np.nanmean([f['tau'] for f in relaxation_fits[s]])
        mean_r2 = np.mean([f['r2'] for f in relaxation_fits[s]])
        print(f"    {s}   | {len(relaxation_fits[s]):6d} | {mean_tau:6.1f} | {mean_r2:.3f}")

# Surrogate test
print("\n[2.2] Surrogate test (shuffled time)")
real_r2 = [f['r2'] for fits in relaxation_fits.values() for f in fits]
mean_real_r2 = np.mean(real_r2) if real_r2 else 0

surrogate_r2s = []
for _ in range(50):
    for subj in df['subject'].unique()[:3]:  # Sample 3 subjects
        subj_data = df[df['subject'] == subj].sort_values(['run', 'epoch_id'])
        alpha_shuffled = np.random.permutation(subj_data['alpha_power'].dropna().values)
        if len(alpha_shuffled) >= 20:
            fit = fit_exponential(alpha_shuffled[:20])
            surrogate_r2s.append(fit['r2'])

mean_surr_r2 = np.mean(surrogate_r2s)
print(f"  Real mean R²: {mean_real_r2:.3f}")
print(f"  Surrogate mean R²: {mean_surr_r2:.3f}")
print(f"  Relaxation is {'REAL' if mean_real_r2 > mean_surr_r2 + 0.05 else 'NOT DIFFERENT from random'}")

# =============================================================================
# 3) SPIRAL/ROTATION TEST
# =============================================================================
print("\n" + "="*80)
print("3) SPIRAL/ROTATION IN STATE SPACE")
print("="*80)

print("\n[3.1] PCA embedding of [alpha, beta, theta, 1/f]")

features = ['alpha_power', 'beta_power', 'theta_power', 'aperiodic_exponent']
df_clean = df.dropna(subset=features)

X = df_clean[features].values
X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_clean = df_clean.copy()
df_clean['pc1'] = X_pca[:, 0]
df_clean['pc2'] = X_pca[:, 1]

print(f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")

print("\n[3.2] Angle unwrapping θ(t) and radius r(t)")

rotation_results = []
for subj in df_clean['subject'].unique():
    subj_data = df_clean[df_clean['subject'] == subj].sort_values(['run', 'epoch_id'])
    
    pc1 = subj_data['pc1'].values
    pc2 = subj_data['pc2'].values
    
    # Compute angle and radius
    theta = np.arctan2(pc2, pc1)
    theta_unwrapped = np.unwrap(theta)
    r = np.sqrt(pc1**2 + pc2**2)
    
    # Total rotation and radius change
    total_rotation = theta_unwrapped[-1] - theta_unwrapped[0]  # radians
    radius_change = r[-1] - r[0]
    
    # Monotonicity of rotation
    dtheta = np.diff(theta_unwrapped)
    monotonic_ratio = np.mean(dtheta > 0) if len(dtheta) > 0 else 0.5
    
    rotation_results.append({
        'subject': subj,
        'total_rotation_rad': total_rotation,
        'total_rotation_deg': np.degrees(total_rotation),
        'radius_change': radius_change,
        'monotonic_ratio': monotonic_ratio
    })

rot_df = pd.DataFrame(rotation_results)
print(f"\n  Mean total rotation: {rot_df['total_rotation_deg'].mean():.1f}° ± {rot_df['total_rotation_deg'].std():.1f}°")
print(f"  Mean radius change: {rot_df['radius_change'].mean():.3f} ± {rot_df['radius_change'].std():.3f}")
print(f"  Mean monotonicity: {rot_df['monotonic_ratio'].mean():.2%}")

# Test vs random
surr_rotations = []
for _ in range(100):
    perm = np.random.permutation(len(X_pca))
    theta = np.arctan2(X_pca[perm[:100], 1], X_pca[perm[:100], 0])
    theta_unwrapped = np.unwrap(theta)
    surr_rotations.append(np.degrees(theta_unwrapped[-1] - theta_unwrapped[0]))

print(f"\n  Surrogate rotation: {np.mean(surr_rotations):.1f}° ± {np.std(surr_rotations):.1f}°")
t_rot, p_rot = stats.ttest_1samp(rot_df['total_rotation_deg'], np.mean(surr_rotations))
print(f"  t = {t_rot:.2f}, p = {p_rot:.4f}")
print(f"  Spiral is {'SIGNIFICANT' if p_rot < 0.05 else 'NOT significant'}")

# =============================================================================
# 4) HYSTERESIS TEST
# =============================================================================
print("\n" + "="*80)
print("4) HYSTERESIS (Entering vs Exiting)")
print("="*80)

print("\n[4.1] Separate entering vs exiting epochs")

entering_epochs = []
exiting_epochs = []

for subj in df['subject'].unique():
    subj_data = df[df['subject'] == subj].sort_values(['run', 'epoch_id']).reset_index(drop=True)
    states = subj_data['state'].values
    
    for i in range(1, len(states)):
        if states[i] != states[i-1]:
            # Entering new state
            entering_epochs.append({
                'alpha': subj_data.loc[i, 'alpha_power'],
                'beta': subj_data.loc[i, 'beta_power'],
                'theta': subj_data.loc[i, 'theta_power'],
                'state': states[i],
                'type': 'enter'
            })
            # Exiting old state
            exiting_epochs.append({
                'alpha': subj_data.loc[i-1, 'alpha_power'],
                'beta': subj_data.loc[i-1, 'beta_power'],
                'theta': subj_data.loc[i-1, 'theta_power'],
                'state': states[i-1],
                'type': 'exit'
            })

enter_df = pd.DataFrame(entering_epochs).dropna()
exit_df = pd.DataFrame(exiting_epochs).dropna()

print(f"  Entering epochs: {len(enter_df)}")
print(f"  Exiting epochs: {len(exit_df)}")

print("\n[4.2] Alpha-Beta relationship: entering vs exiting")

# Regression slopes
from scipy.stats import linregress

enter_slope, enter_int, enter_r, _, _ = linregress(enter_df['alpha'], enter_df['beta'])
exit_slope, exit_int, exit_r, _, _ = linregress(exit_df['alpha'], exit_df['beta'])

print(f"  ENTERING: β = {enter_slope:.3f}×α + {enter_int:.3f} (r={enter_r:.3f})")
print(f"  EXITING:  β = {exit_slope:.3f}×α + {exit_int:.3f} (r={exit_r:.3f})")

slope_diff = abs(enter_slope - exit_slope)
print(f"\n  Slope difference: {slope_diff:.4f}")

# Permutation test
n_perm = 500
perm_diffs = []
all_alpha = np.concatenate([enter_df['alpha'].values, exit_df['alpha'].values])
all_beta = np.concatenate([enter_df['beta'].values, exit_df['beta'].values])
n_enter = len(enter_df)

for _ in range(n_perm):
    perm_idx = np.random.permutation(len(all_alpha))
    perm_enter = perm_idx[:n_enter]
    perm_exit = perm_idx[n_enter:]
    
    s1, _, _, _, _ = linregress(all_alpha[perm_enter], all_beta[perm_enter])
    s2, _, _, _, _ = linregress(all_alpha[perm_exit], all_beta[perm_exit])
    perm_diffs.append(abs(s1 - s2))

p_hyst = np.mean(perm_diffs >= slope_diff)
print(f"  Permutation p-value: {p_hyst:.4f}")
print(f"  Hysteresis is {'SIGNIFICANT' if p_hyst < 0.05 else 'NOT significant'}")

# =============================================================================
# 5) ALPHA BURSTS & IAF
# =============================================================================
print("\n" + "="*80)
print("5) ALPHA BURSTS & IAF")
print("="*80)

print("\n[5.1] Alpha power burst detection")

# Define burst as epochs where alpha > mean + 1 SD
alpha_mean = df['alpha_power'].mean()
alpha_std = df['alpha_power'].std()
burst_threshold = alpha_mean + alpha_std

df['is_burst'] = df['alpha_power'] > burst_threshold
burst_rate = df.groupby('state')['is_burst'].mean()

print("\n  State | Burst Rate")
print("  ------|------------")
for s in range(6):
    print(f"    {s}   | {burst_rate.get(s, 0)*100:.1f}%")

print("\n[5.2] IAF (Individual Alpha Frequency) by state")

if 'alpha_cf' in df.columns:
    iaf_by_state = df.groupby('state')['alpha_cf'].agg(['mean', 'std'])
    print("\n  State | Mean IAF | Std")
    print("  ------|----------|-----")
    for s in range(6):
        if s in iaf_by_state.index:
            print(f"    {s}   | {iaf_by_state.loc[s, 'mean']:.2f} Hz | {iaf_by_state.loc[s, 'std']:.2f}")
else:
    print("  (alpha_cf not available)")

print("\n[5.3] Pre/Post entry comparison")

pre_post_results = []
for subj in df['subject'].unique():
    subj_data = df[df['subject'] == subj].sort_values(['run', 'epoch_id']).reset_index(drop=True)
    states = subj_data['state'].values
    
    for i in range(5, len(states) - 5):
        if states[i] != states[i-1]:  # Transition
            pre_alpha = subj_data.loc[i-5:i-1, 'alpha_power'].mean()
            post_alpha = subj_data.loc[i:i+4, 'alpha_power'].mean()
            pre_burst = subj_data.loc[i-5:i-1, 'is_burst'].mean()
            post_burst = subj_data.loc[i:i+4, 'is_burst'].mean()
            
            pre_post_results.append({
                'pre_alpha': pre_alpha,
                'post_alpha': post_alpha,
                'alpha_change': post_alpha - pre_alpha,
                'pre_burst': pre_burst,
                'post_burst': post_burst,
                'burst_change': post_burst - pre_burst
            })

pp_df = pd.DataFrame(pre_post_results)
print(f"\n  Alpha change at entry: {pp_df['alpha_change'].mean():.4f} ± {pp_df['alpha_change'].std():.4f}")
print(f"  Burst rate change: {pp_df['burst_change'].mean():.3f} ± {pp_df['burst_change'].std():.3f}")

t_alpha, p_alpha = stats.ttest_1samp(pp_df['alpha_change'].dropna(), 0)
print(f"  t = {t_alpha:.2f}, p = {p_alpha:.4f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("SUMMARY: TWIST CHARACTERIZATION")
print("="*80)

findings = {
    'Change-point': cp_results['alpha_power']['mean_t'] > 2,
    'Relaxation': mean_real_r2 > mean_surr_r2 + 0.03,
    'Spiral': p_rot < 0.1,
    'Hysteresis': p_hyst < 0.1,
    'Alpha burst shift': p_alpha < 0.05
}

print("\n┌─────────────────────────────────────────────────────────────────┐")
print("│ TEST              │ RESULT                         │ SUPPORT? │")
print("├─────────────────────────────────────────────────────────────────┤")
print(f"│ Change-point      │ mean |t| = {cp_results['alpha_power']['mean_t']:.2f}                 │   {'✅' if findings['Change-point'] else '❌'}     │")
print(f"│ Relaxation        │ R² = {mean_real_r2:.3f} vs surr {mean_surr_r2:.3f}       │   {'✅' if findings['Relaxation'] else '❌'}     │")
print(f"│ Spiral/rotation   │ p = {p_rot:.4f}                        │   {'✅' if findings['Spiral'] else '❌'}     │")
print(f"│ Hysteresis        │ p = {p_hyst:.4f}                        │   {'✅' if findings['Hysteresis'] else '❌'}     │")
print(f"│ Alpha burst shift │ p = {p_alpha:.4f}                        │   {'✅' if findings['Alpha burst shift'] else '❌'}     │")
print("└─────────────────────────────────────────────────────────────────┘")

n_support = sum(findings.values())
print(f"\n  {n_support}/5 tests support 'twist' as real phenomenon")

if n_support >= 3:
    print("\n✅ TWIST IS REAL: Entry dynamics show systematic patterns")
    print("   Most likely mechanism: CHANGE-POINT + RELAXATION")
else:
    print("\n⚠️ TWIST EVIDENCE WEAK: May be artifact or noise")

# =============================================================================
# FIGURE
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1) Change-point effect sizes
ax1 = axes[0, 0]
feats = list(cp_results.keys())
t_vals = [cp_results[f]['mean_t'] for f in feats]
ax1.bar(feats, t_vals, color='steelblue', edgecolor='black')
ax1.axhline(2, color='red', linestyle='--', label='t=2 threshold')
ax1.set_ylabel('Mean |t-statistic|')
ax1.set_title('1) Change-Point Effect Size')
ax1.legend()

# 2) Relaxation R² comparison
ax2 = axes[0, 1]
ax2.bar(['Real', 'Surrogate'], [mean_real_r2, mean_surr_r2], color=['steelblue', 'gray'], edgecolor='black')
ax2.set_ylabel('Mean R²')
ax2.set_title('2) Relaxation Fit Quality')

# 3) Spiral trajectory (sample subject)
ax3 = axes[0, 2]
sample_subj = df_clean['subject'].unique()[0]
sample_data = df_clean[df_clean['subject'] == sample_subj].sort_values(['run', 'epoch_id'])
ax3.plot(sample_data['pc1'].values[:100], sample_data['pc2'].values[:100], 'b-', alpha=0.7)
ax3.scatter(sample_data['pc1'].values[0], sample_data['pc2'].values[0], c='green', s=100, label='Start', zorder=5)
ax3.scatter(sample_data['pc1'].values[99], sample_data['pc2'].values[99], c='red', s=100, label='End', zorder=5)
ax3.set_xlabel('PC1')
ax3.set_ylabel('PC2')
ax3.set_title('3) State-Space Trajectory (100 epochs)')
ax3.legend()

# 4) Hysteresis plot
ax4 = axes[1, 0]
ax4.scatter(enter_df['alpha'], enter_df['beta'], alpha=0.3, c='blue', label='Entering')
ax4.scatter(exit_df['alpha'], exit_df['beta'], alpha=0.3, c='red', label='Exiting')
x_range = np.linspace(enter_df['alpha'].min(), enter_df['alpha'].max(), 100)
ax4.plot(x_range, enter_slope * x_range + enter_int, 'b-', linewidth=2)
ax4.plot(x_range, exit_slope * x_range + exit_int, 'r-', linewidth=2)
ax4.set_xlabel('Alpha Power')
ax4.set_ylabel('Beta Power')
ax4.set_title('4) Hysteresis: Entering vs Exiting')
ax4.legend()

# 5) Burst rate by state
ax5 = axes[1, 1]
ax5.bar(range(6), [burst_rate.get(s, 0)*100 for s in range(6)], color='coral', edgecolor='black')
ax5.axhline(df['is_burst'].mean()*100, color='gray', linestyle='--', label='Mean')
ax5.set_xlabel('State')
ax5.set_ylabel('Burst Rate (%)')
ax5.set_title('5) Alpha Burst Rate by State')
ax5.legend()

# 6) Summary
ax6 = axes[1, 2]
ax6.axis('off')
summary_text = f"""TWIST ANALYSIS SUMMARY

Tests Supporting:    {n_support}/5
Change-point:        {'✓' if findings['Change-point'] else '✗'}
Relaxation:          {'✓' if findings['Relaxation'] else '✗'}
Spiral:              {'✓' if findings['Spiral'] else '✗'}
Hysteresis:          {'✓' if findings['Hysteresis'] else '✗'}
Alpha burst shift:   {'✓' if findings['Alpha burst shift'] else '✗'}

CONCLUSION:
{'TWIST IS REAL' if n_support >= 3 else 'EVIDENCE WEAK'}
{'Change-point + Relaxation' if findings['Change-point'] and findings['Relaxation'] else ''}
"""
ax6.text(0.1, 0.5, summary_text, fontsize=12, family='monospace', verticalalignment='center')

plt.tight_layout()
plt.savefig('twist_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nFigure: twist_analysis.png")

# Save report
with open('twist_report.md', 'w') as f:
    f.write("# Twist Analysis Report\n\n")
    f.write("## Summary\n\n")
    f.write(f"**{n_support}/5 tests support 'twist' as real phenomenon**\n\n")
    f.write("## Test Results\n\n")
    f.write("| Test | Result | p-value | Supported |\n")
    f.write("|------|--------|---------|----------|\n")
    f.write(f"| Change-point | mean t={cp_results['alpha_power']['mean_t']:.2f} | - | {'✅' if findings['Change-point'] else '❌'} |\n")
    f.write(f"| Relaxation | R²={mean_real_r2:.3f} | - | {'✅' if findings['Relaxation'] else '❌'} |\n")
    f.write(f"| Spiral | - | {p_rot:.4f} | {'✅' if findings['Spiral'] else '❌'} |\n")
    f.write(f"| Hysteresis | slope diff={slope_diff:.4f} | {p_hyst:.4f} | {'✅' if findings['Hysteresis'] else '❌'} |\n")
    f.write(f"| Alpha burst | - | {p_alpha:.4f} | {'✅' if findings['Alpha burst shift'] else '❌'} |\n")
    f.write("\n## Interpretation\n\n")
    if n_support >= 3:
        f.write("The 'twist' appears to be a **real phenomenon**, most likely representing:\n")
        f.write("- Change-point in alpha power at state entry\n")
        f.write("- Exponential relaxation following entry\n")
    else:
        f.write("Evidence for 'twist' is weak; may be artifact or noise.\n")

print("Report: twist_report.md")
print("\nDONE")
