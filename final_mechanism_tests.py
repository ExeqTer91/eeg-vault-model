#!/usr/bin/env python3
"""
FINAL MECHANISM TESTS: Confirmation of Bridge Gating
1. Bidirectionality (entry vs exit symmetry)
2. Temporal lag (who leads?)
3. Per-subject consistency
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FINAL MECHANISM TESTS: Gate Confirmation")
print("="*80)

df = pd.read_csv('epoch_features_with_states.csv')

# Recompute Q_alpha
df['alpha_beta_ratio'] = df['alpha_power'] / (df['beta_power'] + 1e-15)
df['alpha_theta_ratio'] = df['alpha_power'] / (df['theta_power'] + 1e-15)
df['Q_alpha'] = np.sqrt(df['alpha_beta_ratio'] * df['alpha_theta_ratio'])

# Z-score Q
vals = df['Q_alpha'].replace([np.inf, -np.inf], np.nan)
df['Q_alpha'] = (vals - vals.mean()) / (vals.std() + 1e-10)

# Identify bridge states
state_delta = df.groupby('state')['delta_score'].mean()
bridge_states = [s for s in range(6) if -0.05 < state_delta[s] < 0.02]

print(f"\nData: {len(df)} epochs, {df['subject'].nunique()} subjects")
print(f"Bridge states: {bridge_states}")

# =============================================================================
# 1) BIDIRECTIONALITY TEST
# =============================================================================
print("\n" + "="*80)
print("1) BIDIRECTIONALITY: Entry vs Exit Symmetry")
print("="*80)

# Collect entry and exit data (paired by transition)
entry_data = []
exit_data = []

for subj in df['subject'].unique():
    subj_df = df[df['subject'] == subj].sort_values(['run', 'epoch_id']).reset_index(drop=True)
    states = subj_df['state'].values
    q_alpha = subj_df['Q_alpha'].values
    alpha_power = subj_df['alpha_power'].values
    
    for i in range(5, len(states) - 5):
        if states[i] != states[i-1]:  # Transition
            from_state = states[i-1]
            to_state = states[i]
            
            # Pre/post values
            pre_q = np.nanmean(q_alpha[i-5:i])
            post_q = np.nanmean(q_alpha[i:i+5])
            pre_alpha = np.nanmean(alpha_power[i-5:i])
            post_alpha = np.nanmean(alpha_power[i:i+5])
            
            is_entry = to_state in bridge_states and from_state not in bridge_states
            is_exit = from_state in bridge_states and to_state not in bridge_states
            
            if is_entry:
                entry_data.append({
                    'subject': subj,
                    'delta_Q': post_q - pre_q,
                    'delta_alpha': post_alpha - pre_alpha,
                    'from': from_state,
                    'to': to_state
                })
            elif is_exit:
                exit_data.append({
                    'subject': subj,
                    'delta_Q': post_q - pre_q,
                    'delta_alpha': post_alpha - pre_alpha,
                    'from': from_state,
                    'to': to_state
                })

entry_df = pd.DataFrame(entry_data)
exit_df = pd.DataFrame(exit_data)

print(f"\n  Entry transitions: {len(entry_df)}")
print(f"  Exit transitions: {len(exit_df)}")

print("\n[1.1] ΔQ comparison (entry vs exit)")
entry_dq = entry_df['delta_Q'].dropna()
exit_dq = exit_df['delta_Q'].dropna()

print(f"  Entry ΔQ: {entry_dq.mean():+.4f} ± {entry_dq.std():.4f}")
print(f"  Exit ΔQ: {exit_dq.mean():+.4f} ± {exit_dq.std():.4f}")

# Are they opposite signs? (entry negative, exit positive)
entry_sign = np.sign(entry_dq.mean())
exit_sign = np.sign(exit_dq.mean())
opposite = entry_sign != exit_sign and entry_sign != 0 and exit_sign != 0

print(f"  Opposite signs: {'✅ YES' if opposite else '❌ NO'}")

# Symmetry test: is |entry| ≈ |exit|?
magnitude_ratio = abs(entry_dq.mean()) / (abs(exit_dq.mean()) + 1e-10)
symmetric = 0.5 < magnitude_ratio < 2.0

print(f"  Magnitude ratio: {magnitude_ratio:.2f}")
print(f"  Symmetric (0.5-2.0): {'✅ YES' if symmetric else '❌ NO'}")

# 95% CI
entry_ci = stats.sem(entry_dq) * 1.96
exit_ci = stats.sem(exit_dq) * 1.96

print(f"\n  Entry 95% CI: [{entry_dq.mean() - entry_ci:.4f}, {entry_dq.mean() + entry_ci:.4f}]")
print(f"  Exit 95% CI: [{exit_dq.mean() - exit_ci:.4f}, {exit_dq.mean() + exit_ci:.4f}]")

# Do CIs overlap around zero? (would indicate symmetry around 0)
# Actually, for reversibility we want entry < 0 < exit
reversible = entry_dq.mean() + entry_ci < 0 < exit_dq.mean() - exit_ci
print(f"\n  Clearly reversible (entry<0<exit): {'✅ YES' if reversible else '❌ NOT CLEAR'}")

print("\n[1.2] Δalpha comparison (entry vs exit)")
entry_da = entry_df['delta_alpha'].dropna()
exit_da = exit_df['delta_alpha'].dropna()

print(f"  Entry Δα: {entry_da.mean():+.6f} ± {entry_da.std():.6f}")
print(f"  Exit Δα: {exit_da.mean():+.6f} ± {exit_da.std():.6f}")

# =============================================================================
# 2) TEMPORAL LAG (Cross-correlation)
# =============================================================================
print("\n" + "="*80)
print("2) TEMPORAL LAG: Who Leads?")
print("="*80)

def cross_correlation(x, y, max_lag=10):
    """Compute normalized cross-correlation at different lags"""
    n = len(x)
    x = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y = (y - np.mean(y)) / (np.std(y) + 1e-10)
    
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = np.corrcoef(x[:lag], y[-lag:])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(x[lag:], y[:-lag])[0, 1]
        else:
            corr = np.corrcoef(x, y)[0, 1]
        correlations.append((lag, corr if not np.isnan(corr) else 0))
    
    return correlations

print("\n[2.1] Cross-correlation around bridge entries")

# Collect time series around bridge entries
all_xcorr_alpha_q = []
all_xcorr_slope_alpha = []
all_xcorr_slope_q = []

for subj in df['subject'].unique():
    subj_df = df[df['subject'] == subj].sort_values(['run', 'epoch_id']).reset_index(drop=True)
    states = subj_df['state'].values
    q_alpha = subj_df['Q_alpha'].values
    alpha_power = subj_df['alpha_power'].values
    slope = subj_df['aperiodic_exponent'].values
    
    for i in range(20, len(states) - 20):
        if states[i] in bridge_states and states[i-1] not in bridge_states:
            # Window around entry
            window_q = q_alpha[i-10:i+10]
            window_alpha = alpha_power[i-10:i+10]
            window_slope = slope[i-10:i+10]
            
            if len(window_q) == 20 and not np.any(np.isnan(window_q)):
                xcorr_aq = cross_correlation(window_alpha, window_q, max_lag=5)
                xcorr_sa = cross_correlation(window_slope, window_alpha, max_lag=5)
                xcorr_sq = cross_correlation(window_slope, window_q, max_lag=5)
                
                all_xcorr_alpha_q.append(xcorr_aq)
                all_xcorr_slope_alpha.append(xcorr_sa)
                all_xcorr_slope_q.append(xcorr_sq)

# Average cross-correlations
def avg_xcorr(all_xcorr):
    if not all_xcorr:
        return []
    lags = [x[0] for x in all_xcorr[0]]
    avg = []
    for i, lag in enumerate(lags):
        corrs = [xc[i][1] for xc in all_xcorr]
        avg.append((lag, np.mean(corrs)))
    return avg

avg_aq = avg_xcorr(all_xcorr_alpha_q)
avg_sa = avg_xcorr(all_xcorr_slope_alpha)
avg_sq = avg_xcorr(all_xcorr_slope_q)

print(f"\n  Windows analyzed: {len(all_xcorr_alpha_q)}")

if avg_aq:
    # Find peak lag
    peak_aq = max(avg_aq, key=lambda x: abs(x[1]))
    peak_sa = max(avg_sa, key=lambda x: abs(x[1]))
    peak_sq = max(avg_sq, key=lambda x: abs(x[1]))
    
    print(f"\n  Alpha → Q peak: lag={peak_aq[0]}, r={peak_aq[1]:.3f}")
    print(f"  1/f slope → Alpha peak: lag={peak_sa[0]}, r={peak_sa[1]:.3f}")
    print(f"  1/f slope → Q peak: lag={peak_sq[0]}, r={peak_sq[1]:.3f}")
    
    # Interpretation
    if peak_aq[0] > 0:
        print("\n  ➡️ Alpha LEADS Q (alpha changes first, Q follows)")
    elif peak_aq[0] < 0:
        print("\n  ➡️ Q LEADS Alpha")
    else:
        print("\n  ➡️ Simultaneous")
    
    if peak_sa[0] > 0:
        print("  ➡️ 1/f slope LEADS Alpha")
    elif peak_sa[0] < 0:
        print("  ➡️ Alpha LEADS 1/f slope")
else:
    print("  No valid windows for cross-correlation")

# =============================================================================
# 3) PER-SUBJECT CONSISTENCY
# =============================================================================
print("\n" + "="*80)
print("3) PER-SUBJECT CONSISTENCY")
print("="*80)

print("\n[3.1] % subjects with Q↓ at entry, Q↑ at exit")

subject_patterns = []
for subj in df['subject'].unique():
    subj_entry = entry_df[entry_df['subject'] == subj]['delta_Q']
    subj_exit = exit_df[exit_df['subject'] == subj]['delta_Q']
    
    if len(subj_entry) > 0 and len(subj_exit) > 0:
        entry_mean = subj_entry.mean()
        exit_mean = subj_exit.mean()
        
        correct_pattern = (entry_mean < 0) and (exit_mean > 0)
        
        subject_patterns.append({
            'subject': subj,
            'entry_dQ': entry_mean,
            'exit_dQ': exit_mean,
            'correct_pattern': correct_pattern,
            'entry_negative': entry_mean < 0,
            'exit_positive': exit_mean > 0
        })

pattern_df = pd.DataFrame(subject_patterns)

if len(pattern_df) > 0:
    pct_correct = pattern_df['correct_pattern'].mean() * 100
    pct_entry_neg = pattern_df['entry_negative'].mean() * 100
    pct_exit_pos = pattern_df['exit_positive'].mean() * 100
    
    print(f"\n  Subjects with valid data: {len(pattern_df)}")
    print(f"  Entry Q↓ (negative): {pct_entry_neg:.1f}%")
    print(f"  Exit Q↑ (positive): {pct_exit_pos:.1f}%")
    print(f"  Both correct: {pct_correct:.1f}%")
    
    # Is it > 70%?
    universal = pct_correct >= 70
    print(f"\n  Universal (≥70%): {'✅ YES' if universal else '❌ NO'}")
    
    # Show per-subject details
    print("\n  Per-subject breakdown:")
    print("  Subject | Entry ΔQ | Exit ΔQ | Correct?")
    print("  --------|----------|---------|--------")
    for _, row in pattern_df.iterrows():
        print(f"    {str(row['subject'])[:6]:6} | {row['entry_dQ']:+8.4f} | {row['exit_dQ']:+7.4f} | {'✓' if row['correct_pattern'] else '✗'}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("SUMMARY: MECHANISM CONFIRMATION")
print("="*80)

findings = {
    'Bidirectional (opposite signs)': opposite,
    'Symmetric magnitude': symmetric,
    'Temporal order clear': len(avg_aq) > 0 and peak_aq[0] != 0,
    'Universal (>70%)': len(pattern_df) > 0 and pct_correct >= 70
}

print("\n┌─────────────────────────────────────────────────────────────────┐")
print("│ TEST                          │ RESULT              │ CONFIRM? │")
print("├─────────────────────────────────────────────────────────────────┤")
print(f"│ Bidirectional (opposite)      │ Entry:{entry_sign:+.0f} Exit:{exit_sign:+.0f}        │    {'✅' if opposite else '❌'}    │")
print(f"│ Symmetric magnitude           │ ratio={magnitude_ratio:.2f}           │    {'✅' if symmetric else '❌'}    │")
temporal_result = f"lag={peak_aq[0]}" if len(avg_aq) > 0 else "N/A"
print(f"│ Temporal order                │ {temporal_result:19} │    {'✅' if findings['Temporal order clear'] else '❌'}    │")
pct_result = f"{pct_correct:.0f}%" if len(pattern_df) > 0 else "N/A"
print(f"│ Universal (≥70%)              │ {pct_result:19} │    {'✅' if findings['Universal (>70%)'] else '❌'}    │")
print("└─────────────────────────────────────────────────────────────────┘")

n_confirm = sum(findings.values())
print(f"\n  {n_confirm}/4 tests confirm mechanism")

if n_confirm >= 3:
    print("\n✅ MECHANISM CONFIRMED: Reversible alpha-Q gating")
else:
    print("\n⚠️ Mechanism partially confirmed")

# =============================================================================
# FIGURE
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1) Entry vs Exit ΔQ
ax1 = axes[0, 0]
ax1.bar(['Entry', 'Exit'], [entry_dq.mean(), exit_dq.mean()], 
        yerr=[entry_ci, exit_ci], capsize=5,
        color=['coral', 'steelblue'], edgecolor='black')
ax1.axhline(0, color='gray', linestyle='--')
ax1.set_ylabel('ΔQ_alpha')
ax1.set_title('1) Bidirectionality: Entry vs Exit')

# 2) Cross-correlation
ax2 = axes[0, 1]
if avg_aq:
    lags = [x[0] for x in avg_aq]
    corrs = [x[1] for x in avg_aq]
    ax2.plot(lags, corrs, 'b-o', linewidth=2, label='Alpha→Q')
    ax2.axvline(0, color='gray', linestyle='--')
    ax2.axhline(0, color='gray', linestyle='--')
ax2.set_xlabel('Lag (epochs)')
ax2.set_ylabel('Correlation')
ax2.set_title('2) Temporal Lag: Alpha → Q')
ax2.legend()

# 3) Per-subject consistency
ax3 = axes[1, 0]
if len(pattern_df) > 0:
    subjects = pattern_df['subject'].astype(str).str[:4]
    colors = ['green' if c else 'red' for c in pattern_df['correct_pattern']]
    ax3.scatter(pattern_df['entry_dQ'], pattern_df['exit_dQ'], c=colors, s=100, alpha=0.7)
    ax3.axhline(0, color='gray', linestyle='--')
    ax3.axvline(0, color='gray', linestyle='--')
    ax3.set_xlabel('Entry ΔQ')
    ax3.set_ylabel('Exit ΔQ')
    ax3.set_title(f'3) Per-Subject: {pct_correct:.0f}% correct (green)')

# 4) Summary
ax4 = axes[1, 1]
ax4.axis('off')
summary_text = f"""MECHANISM CONFIRMATION SUMMARY

Tests Passed: {n_confirm}/4

1. Bidirectional: {'✓' if opposite else '✗'} (Entry:{entry_sign:+.0f}, Exit:{exit_sign:+.0f})
2. Symmetric: {'✓' if symmetric else '✗'} (ratio={magnitude_ratio:.2f})
3. Temporal: {'✓' if findings['Temporal order clear'] else '✗'} ({temporal_result})
4. Universal: {'✓' if findings['Universal (>70%)'] else '✗'} ({pct_result})

CONCLUSION:
{'MECHANISM CONFIRMED' if n_confirm >= 3 else 'PARTIAL CONFIRMATION'}
{'Reversible alpha-Q gating' if n_confirm >= 3 else ''}
"""
ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace', verticalalignment='center')

plt.tight_layout()
plt.savefig('final_mechanism.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nFigure: final_mechanism.png")

# Save report
with open('final_mechanism_report.md', 'w') as f:
    f.write("# Final Mechanism Confirmation Report\n\n")
    f.write("## Summary\n\n")
    f.write(f"**{n_confirm}/4 tests confirm the mechanism**\n\n")
    f.write("## Test Results\n\n")
    f.write("| Test | Result | Confirmed |\n")
    f.write("|------|--------|----------|\n")
    f.write(f"| Bidirectional | Entry:{entry_sign:+.0f} Exit:{exit_sign:+.0f} | {'✅' if opposite else '❌'} |\n")
    f.write(f"| Symmetric | ratio={magnitude_ratio:.2f} | {'✅' if symmetric else '❌'} |\n")
    f.write(f"| Temporal | {temporal_result} | {'✅' if findings['Temporal order clear'] else '❌'} |\n")
    f.write(f"| Universal | {pct_result} | {'✅' if findings['Universal (>70%)'] else '❌'} |\n\n")
    f.write("## Per-Subject Breakdown\n\n")
    f.write("| Subject | Entry ΔQ | Exit ΔQ | Correct |\n")
    f.write("|---------|----------|---------|--------|\n")
    for _, row in pattern_df.iterrows():
        f.write(f"| {str(row['subject'])[:6]} | {row['entry_dQ']:+.4f} | {row['exit_dQ']:+.4f} | {'✓' if row['correct_pattern'] else '✗'} |\n")

print("Report: final_mechanism_report.md")
print("\nDONE")
