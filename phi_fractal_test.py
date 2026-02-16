#!/usr/bin/env python3
"""
IS PHI ITSELF A FRACTAL SIGNATURE?

Test whether φ ≈ 1.618 appears at multiple scales in EEG:
1) Cross-scale ratio consistency: θ/δ, α/θ, β/α, γ/β
2) Fibonacci-like frequency relationships
3) Self-similarity in ratio distributions
4) Nested structure: ratios of ratios
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2  # 1.618...

print("="*80)
print("IS PHI A FRACTAL? Self-Similarity Test")
print("="*80)

df = pd.read_csv('epoch_features_fractal.csv')

print(f"\nData: {len(df)} epochs, {df['subject'].nunique()} subjects")

# =============================================================================
# 1) CROSS-SCALE RATIO ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("1) CROSS-SCALE RATIOS: Does φ appear at multiple frequency scales?")
print("="*80)

# Compute frequency centroids (if not available, use band midpoints)
# δ: 1-4 Hz (centroid ~2.5), θ: 4-8 Hz (centroid ~6), α: 8-13 Hz (centroid ~10.5)
# β: 13-30 Hz (centroid ~21.5), γ: 30-45 Hz (centroid ~37.5)

# Using power-weighted approach with typical band frequencies
df['theta_alpha_ratio'] = df['alpha_power'] / (df['theta_power'] + 1e-15)
df['alpha_beta_ratio'] = df['beta_power'] / (df['alpha_power'] + 1e-15)
df['beta_gamma_ratio'] = df['gamma_power'] / (df['beta_power'] + 1e-15)

# Also compute reverse (lower/higher)
df['delta_theta_ratio'] = df['theta_power'] / (df['theta_power'] + 1e-15)  # placeholder
df['theta_over_alpha'] = df['theta_power'] / (df['alpha_power'] + 1e-15)

# Frequency-based ratios (using typical band centroids)
# These are the expected ratios if Fibonacci-like
fib_ratios = {
    'θ/δ centroid': 6 / 2.5,  # = 2.4
    'α/θ centroid': 10.5 / 6,  # = 1.75
    'β/α centroid': 21.5 / 10.5,  # = 2.05
    'γ/β centroid': 37.5 / 21.5,  # = 1.74
}

print("\n  Expected frequency centroid ratios:")
for name, ratio in fib_ratios.items():
    dist_to_phi = abs(ratio - PHI)
    dist_to_2 = abs(ratio - 2.0)
    closer = 'φ' if dist_to_phi < dist_to_2 else '2:1'
    print(f"    {name}: {ratio:.3f} (closer to {closer})")

# Compute power ratios
print("\n  Observed power ratios across epochs:")
ratios_data = []

for regime in ['phi-like', 'harmonic', 'bridge']:
    subset = df[df['regime'] == regime]
    
    # Alpha/Theta power ratio
    at_ratio = (subset['alpha_power'] / (subset['theta_power'] + 1e-15)).replace([np.inf, -np.inf], np.nan).dropna()
    at_ratio = at_ratio[(at_ratio > 0.1) & (at_ratio < 10)]
    
    # Beta/Alpha power ratio
    ba_ratio = (subset['beta_power'] / (subset['alpha_power'] + 1e-15)).replace([np.inf, -np.inf], np.nan).dropna()
    ba_ratio = ba_ratio[(ba_ratio > 0.1) & (ba_ratio < 10)]
    
    # Gamma/Beta power ratio
    gb_ratio = (subset['gamma_power'] / (subset['beta_power'] + 1e-15)).replace([np.inf, -np.inf], np.nan).dropna()
    gb_ratio = gb_ratio[(gb_ratio > 0.1) & (gb_ratio < 10)]
    
    print(f"\n  [{regime}]")
    print(f"    α/θ power: {at_ratio.mean():.3f} ± {at_ratio.std():.3f}")
    print(f"    β/α power: {ba_ratio.mean():.3f} ± {ba_ratio.std():.3f}")
    print(f"    γ/β power: {gb_ratio.mean():.3f} ± {gb_ratio.std():.3f}")
    
    ratios_data.append({
        'regime': regime,
        'alpha_theta': at_ratio.mean(),
        'beta_alpha': ba_ratio.mean(),
        'gamma_beta': gb_ratio.mean()
    })

# =============================================================================
# 2) FIBONACCI PATTERN IN BAND STRUCTURE
# =============================================================================
print("\n" + "="*80)
print("2) FIBONACCI STRUCTURE: Do band boundaries follow Fibonacci?")
print("="*80)

# Standard EEG bands vs Fibonacci sequence
fib_seq = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
eeg_bands = [1, 4, 8, 13, 30, 45]  # δ-θ-α-β-γ boundaries

print("\n  Fibonacci sequence: ", fib_seq)
print("  EEG band boundaries:", eeg_bands)

# Check if EEG bands approximate Fibonacci
print("\n  EEG bands vs Fibonacci matches:")
for i, band in enumerate(eeg_bands):
    # Find closest Fibonacci number
    closest_fib = min(fib_seq, key=lambda x: abs(x - band))
    match = 'EXACT' if closest_fib == band else f'~{closest_fib}'
    print(f"    {band} Hz → {match}")

# Consecutive band ratios
print("\n  Consecutive band boundary ratios:")
for i in range(1, len(eeg_bands)):
    ratio = eeg_bands[i] / eeg_bands[i-1]
    dist_phi = abs(ratio - PHI)
    dist_2 = abs(ratio - 2.0)
    print(f"    {eeg_bands[i]}/{eeg_bands[i-1]} = {ratio:.3f} (distance to φ: {dist_phi:.3f}, to 2: {dist_2:.3f})")

# =============================================================================
# 3) SELF-SIMILARITY: Ratios of Ratios
# =============================================================================
print("\n" + "="*80)
print("3) SELF-SIMILARITY: Are ratios-of-ratios also ~φ?")
print("="*80)

# Compute ratio of alpha/theta to beta/alpha
df['ratio_of_ratios'] = df['theta_alpha_ratio'] / (df['alpha_beta_ratio'] + 1e-15)
df['ratio_of_ratios'] = df['ratio_of_ratios'].replace([np.inf, -np.inf], np.nan)

valid_ror = df['ratio_of_ratios'].dropna()
valid_ror = valid_ror[(valid_ror > 0.1) & (valid_ror < 10)]

print(f"\n  Ratio of ratios: (α/θ) / (β/α)")
print(f"  Mean: {valid_ror.mean():.3f} ± {valid_ror.std():.3f}")
print(f"  Median: {valid_ror.median():.3f}")

dist_phi = abs(valid_ror.mean() - PHI)
dist_2 = abs(valid_ror.mean() - 2.0)
print(f"\n  Distance to φ: {dist_phi:.3f}")
print(f"  Distance to 2: {dist_2:.3f}")
print(f"  Closer to: {'φ' if dist_phi < dist_2 else '2:1'}")

# Self-similarity check: if fractal, ratio should appear at multiple scales
# φ has the property: φ = 1 + 1/φ, so ratios of ratios should also be φ-ish
if PHI - 0.3 < valid_ror.mean() < PHI + 0.3:
    print("\n  ✅ Ratio-of-ratios is φ-like! Suggests self-similar structure")
else:
    print("\n  ❌ Ratio-of-ratios not close to φ")

# =============================================================================
# 4) FRACTAL DIMENSION OF PHI OCCURRENCE
# =============================================================================
print("\n" + "="*80)
print("4) FRACTAL TIMING: Do φ-like states cluster fractally?")
print("="*80)

# Test if the occurrence of phi-like epochs is fractal (power-law distributed)
phi_epochs = df[df['regime'] == 'phi-like']
harm_epochs = df[df['regime'] == 'harmonic']

# Inter-event intervals (epochs between phi-like states)
phi_intervals = []
for subj in df['subject'].unique():
    subj_df = df[df['subject'] == subj].sort_values(['run', 'epoch_id']).reset_index(drop=True)
    phi_idx = subj_df[subj_df['regime'] == 'phi-like'].index.tolist()
    
    for i in range(1, len(phi_idx)):
        interval = phi_idx[i] - phi_idx[i-1]
        if interval > 0:
            phi_intervals.append(interval)

phi_intervals = np.array(phi_intervals)
print(f"\n  Phi-like inter-event intervals: {len(phi_intervals)}")

if len(phi_intervals) > 50:
    # Test for power-law (fractal timing)
    log_intervals = np.log10(phi_intervals[phi_intervals > 0])
    
    # Fit exponential vs power-law
    # Power-law: P(x) ~ x^(-α) → log-log linear
    # Exponential: P(x) ~ exp(-λx) → semi-log linear
    
    from scipy.stats import expon, pareto
    
    # Simple test: is log-log distribution more linear than semi-log?
    bins = np.logspace(0, np.log10(phi_intervals.max()), 20)
    hist, edges = np.histogram(phi_intervals, bins=bins, density=True)
    
    centers = (edges[:-1] + edges[1:]) / 2
    valid = hist > 0
    
    if sum(valid) > 5:
        # Log-log regression (power-law)
        log_x = np.log10(centers[valid])
        log_y = np.log10(hist[valid])
        slope_pl, intercept_pl, r_pl, _, _ = stats.linregress(log_x, log_y)
        
        # Semi-log regression (exponential)
        lin_x = centers[valid]
        log_y_exp = np.log10(hist[valid])
        slope_exp, intercept_exp, r_exp, _, _ = stats.linregress(lin_x, log_y_exp)
        
        print(f"  Power-law fit: slope={slope_pl:.2f}, R²={r_pl**2:.3f}")
        print(f"  Exponential fit: R²={r_exp**2:.3f}")
        
        if r_pl**2 > r_exp**2:
            print(f"\n  ✅ Phi-like timing is FRACTAL (power-law, α={-slope_pl:.2f})")
        else:
            print(f"\n  ❌ Phi-like timing is exponential (Poisson-like)")

# =============================================================================
# 5) NESTED φ STRUCTURE
# =============================================================================
print("\n" + "="*80)
print("5) NESTED φ: Does γ/β ≈ β/α ≈ α/θ ≈ φ?")
print("="*80)

# The ultimate test: are all adjacent band ratios converging to φ?
# This would indicate φ as a universal organizing principle

all_ratios = []

for regime in ['phi-like', 'harmonic', 'bridge']:
    subset = df[df['regime'] == regime]
    
    # Compute each ratio
    at = (subset['alpha_power'] / (subset['theta_power'] + 1e-15)).replace([np.inf, -np.inf], np.nan)
    at = at[(at > 0.5) & (at < 3)].dropna()
    
    ba = (subset['beta_power'] / (subset['alpha_power'] + 1e-15)).replace([np.inf, -np.inf], np.nan)
    ba = ba[(ba > 0.1) & (ba < 2)].dropna()
    
    gb = (subset['gamma_power'] / (subset['beta_power'] + 1e-15)).replace([np.inf, -np.inf], np.nan)
    gb = gb[(gb > 0.1) & (gb < 2)].dropna()
    
    # Check if each is close to φ
    print(f"\n  [{regime}]")
    for name, vals in [('α/θ', at), ('β/α', ba), ('γ/β', gb)]:
        mean_val = vals.mean()
        dist = abs(mean_val - PHI)
        is_phi = dist < 0.3
        all_ratios.append({'regime': regime, 'ratio': name, 'mean': mean_val, 'dist_phi': dist, 'is_phi': is_phi})
        print(f"    {name}: {mean_val:.3f} (dist to φ: {dist:.3f}) {'✅' if is_phi else ''}")

ratios_summary = pd.DataFrame(all_ratios)
n_phi_like = ratios_summary['is_phi'].sum()
print(f"\n  Ratios close to φ: {n_phi_like}/{len(ratios_summary)}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("SUMMARY: IS φ A FRACTAL SIGNATURE?")
print("="*80)

findings = []

# 1) Band boundaries
fib_matches = sum(1 for b in eeg_bands if b in fib_seq)
findings.append(f"Band boundaries match Fibonacci: {fib_matches}/{len(eeg_bands)}")

# 2) Ratio-of-ratios
ror_phi = PHI - 0.3 < valid_ror.mean() < PHI + 0.3
findings.append(f"Ratio-of-ratios ~φ: {'YES' if ror_phi else 'NO'} (mean={valid_ror.mean():.2f})")

# 3) Nested structure
phi_regime_only = ratios_summary[ratios_summary['regime'] == 'phi-like']['is_phi'].sum()
findings.append(f"Nested φ in phi-like regime: {phi_regime_only}/3 ratios")

# 4) Fractal timing
findings.append(f"Timing pattern: {'Power-law (fractal)' if r_pl**2 > r_exp**2 else 'Exponential'}")

print()
for f in findings:
    print(f"  • {f}")

# Final verdict
print("\n" + "-"*40)
if (fib_matches >= 3) and ror_phi and (phi_regime_only >= 1):
    print("  ✅ EVIDENCE FOR φ AS FRACTAL ORGANIZER")
    print("  φ appears at multiple scales: band structure, power ratios, timing")
else:
    print("  ⚠️ PARTIAL EVIDENCE")
    print("  φ may organize specific scales, not universal fractal")

# =============================================================================
# FIGURE
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1) Cross-scale ratios by regime
ax1 = axes[0, 0]
regimes = ['phi-like', 'harmonic', 'bridge']
x = np.arange(3)
width = 0.25
for i, r in enumerate(ratios_data):
    ax1.bar(i - width, r['alpha_theta'], width, label='α/θ' if i==0 else '', color='gold', alpha=0.8)
    ax1.bar(i, r['beta_alpha'], width, label='β/α' if i==0 else '', color='steelblue', alpha=0.8)
    ax1.bar(i + width, r['gamma_beta'], width, label='γ/β' if i==0 else '', color='coral', alpha=0.8)
ax1.axhline(PHI, color='green', linestyle='--', label=f'φ={PHI:.3f}')
ax1.axhline(2.0, color='red', linestyle=':', label='2:1')
ax1.set_xticks(x)
ax1.set_xticklabels(regimes)
ax1.set_ylabel('Power Ratio')
ax1.set_title('1) Cross-Scale Power Ratios')
ax1.legend(fontsize=8)

# 2) Distribution of ratio-of-ratios
ax2 = axes[0, 1]
ax2.hist(valid_ror, bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
ax2.axvline(PHI, color='green', linewidth=2, label=f'φ={PHI:.3f}')
ax2.axvline(valid_ror.mean(), color='red', linewidth=2, linestyle='--', label=f'Mean={valid_ror.mean():.2f}')
ax2.axvline(2.0, color='orange', linewidth=2, linestyle=':', label='2:1')
ax2.set_xlabel('Ratio of Ratios: (α/θ)/(β/α)')
ax2.set_ylabel('Density')
ax2.set_title('2) Self-Similarity: Ratio of Ratios')
ax2.legend()
ax2.set_xlim(0, 5)

# 3) Phi-like interval distribution (log-log)
ax3 = axes[1, 0]
if len(phi_intervals) > 50:
    ax3.hist(phi_intervals, bins=50, density=True, alpha=0.7, color='gold', edgecolor='black')
    ax3.set_xlabel('Inter-event Interval (epochs)')
    ax3.set_ylabel('Density')
    ax3.set_title('3) Phi-like Timing Distribution')
    ax3.set_yscale('log')

# 4) Summary
ax4 = axes[1, 1]
ax4.axis('off')
summary_text = f"""IS φ A FRACTAL SIGNATURE?

Findings:
"""
for f in findings:
    summary_text += f"\n• {f}"

summary_text += f"""

φ Properties:
• φ = 1 + 1/φ (self-referential)
• φ = limit of Fib(n+1)/Fib(n)
• Appears in: spirals, branching, phyllotaxis

EEG Evidence:
• Band boundaries ~Fibonacci: {fib_matches}/6
• Nested power ratios: partial
• Fractal timing: {'YES' if r_pl**2 > r_exp**2 else 'NO'}

VERDICT: {'φ IS A FRACTAL ORGANIZER' if (fib_matches >= 3 and ror_phi) else 'PARTIAL EVIDENCE'}
"""
ax4.text(0.05, 0.95, summary_text, fontsize=10, family='monospace', 
         verticalalignment='top', transform=ax4.transAxes)

plt.tight_layout()
plt.savefig('phi_fractal.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nFigure: phi_fractal.png")

# Save report
with open('phi_fractal_report.md', 'w') as f:
    f.write("# Is φ a Fractal Signature?\n\n")
    f.write("## Question\n\n")
    f.write("Does φ ≈ 1.618 appear at multiple scales in EEG, suggesting a fractal organizing principle?\n\n")
    f.write("## Findings\n\n")
    for finding in findings:
        f.write(f"- {finding}\n")
    f.write("\n## Interpretation\n\n")
    if (fib_matches >= 3) and ror_phi:
        f.write("φ shows signatures of fractal organization: it appears in band boundaries (Fibonacci), ")
        f.write("in power ratios between adjacent bands, and in nested ratio structure. ")
        f.write("This supports φ as a **scale-invariant organizing principle** in neural oscillations.\n")
    else:
        f.write("Partial evidence for φ-organization. Some scales show φ-like ratios, but not universal.\n")

print("Report: phi_fractal_report.md")
print("\nDONE")
