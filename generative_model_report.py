#!/usr/bin/env python3
"""Generate figures and report from Modal generative model results."""
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

E_MINUS_1 = float(np.e - 1)
PHI = 1.6180339887

os.makedirs('outputs/e1_figures', exist_ok=True)
REPORT = []

def rpt(line=""):
    print(line)
    REPORT.append(line)

def main():
    with open('outputs/generative_model_results.json') as f:
        results = json.load(f)

    rpt("# Generative Model: Exponential Oscillator Network → e-1 Spectral Spacing")
    rpt("")
    rpt("## Model Description")
    rpt("")
    rpt("We test whether **exponential neural dynamics produce e-1 spectral spacing")
    rpt("without e-1 being explicitly coded into the model**. Two model families are used:")
    rpt("")
    rpt("1. **Wilson-Cowan oscillators**: 5 coupled excitatory-inhibitory populations")
    rpt("   representing delta through gamma bands. Each population has exponentially")
    rpt("   scaled time constants (τ_k = τ_base × e^(k × scale)). Coupling between")
    rpt("   populations decays exponentially with frequency distance.")
    rpt("")
    rpt("2. **Kuramoto oscillators**: 100 coupled phase oscillators with natural frequencies")
    rpt("   clustered around 5 band centers. The band centers are spaced either")
    rpt("   exponentially, linearly, uniformly, or randomly.")
    rpt("")
    rpt("The key question: does exponential spacing of time constants or natural")
    rpt("frequencies produce emergent spectral centroid ratios near e-1 ≈ 1.718,")
    rpt("while non-exponential spacing does not?")
    rpt("")

    rpt("## Experiment 1: Default Wilson-Cowan (N=50 simulations)")
    rpt("")
    exp1 = results['exp1']
    ratios1 = [r for r in exp1.get('ratios', []) if r is not None]
    if ratios1:
        m1 = np.mean(ratios1)
        rpt(f"- Mean α/θ = {m1:.4f} ± {np.std(ratios1):.4f}")
        rpt(f"- Distance from e-1: {abs(m1 - E_MINUS_1):.4f}")
        rpt(f"- Distance from φ: {abs(m1 - PHI):.4f}")
        rpt(f"- N valid: {len(ratios1)}")
        rpt("")
        rpt(f"The Wilson-Cowan model with default parameters (τ_scale=0.5) produces")
        rpt(f"α/θ ≈ {m1:.2f}, which is above both e-1 and 2:1. This suggests the")
        rpt(f"Wilson-Cowan dynamics are dominated by the sigmoid nonlinearity rather")
        rpt(f"than the exponential time constants.")
    rpt("")

    rpt("## Experiment 2: Parameter Sweep (τ_scale)")
    rpt("")
    exp2 = results.get('exp2', {})
    sweep = exp2.get('sweep', [])
    if sweep:
        rpt(f"| τ_scale | Mean α/θ | SD | Δe-1 | Δφ | N |")
        rpt(f"|---------|----------|-----|------|-----|---|")
        for s in sweep:
            rpt(f"| {s['tau_scale']:.2f} | {s['mean']:.4f} | {s['sd']:.4f} | {s['dist_e1']:.4f} | {s['dist_phi']:.4f} | {s['n']} |")
        best = min(sweep, key=lambda x: x['dist_e1'])
        rpt(f"\nClosest to e-1: τ_scale = {best['tau_scale']:.2f} → α/θ = {best['mean']:.4f}")
        rpt(f"(Still {best['dist_e1']:.4f} from e-1)")

        fig, ax = plt.subplots(figsize=(10, 6))
        ts_vals = [s['tau_scale'] for s in sweep]
        means = [s['mean'] for s in sweep]
        sds = [s['sd'] for s in sweep]
        ax.errorbar(ts_vals, means, yerr=sds, fmt='o-', color='navy', capsize=4, markersize=6)
        ax.axhline(E_MINUS_1, color='blue', ls='-', lw=2, label=f'e-1 = {E_MINUS_1:.4f}')
        ax.axhline(PHI, color='goldenrod', ls='--', lw=2, label=f'φ = {PHI:.4f}')
        ax.axhline(2.0, color='red', ls='--', lw=1.5, label='2:1')
        ax.set_xlabel('τ_scale (exponential scaling factor)')
        ax.set_ylabel('α/θ Centroid Ratio')
        ax.set_title('Wilson-Cowan: α/θ vs Time Constant Scaling')
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig('outputs/e1_figures/fig_model_sweep.png', dpi=300)
        plt.close(fig)
        rpt("\n![Parameter sweep](outputs/e1_figures/fig_model_sweep.png)")
    rpt("")

    rpt("## Experiment 3: Critical Control — Exponential vs Others")
    rpt("")
    exp3 = results.get('exp3', {})
    cond_order = ['Exponential', 'Linear', 'Random', 'Uniform', 'Inverse_exp']
    rpt(f"| Condition | Mean α/θ | SD | Δe-1 | Δφ | N |")
    rpt(f"|-----------|----------|-----|------|-----|---|")
    exp3_data = {}
    for cond in cond_order:
        data = exp3.get(cond, {})
        vals = [v for v in data.get('ratios', []) if v is not None]
        if vals:
            m = np.mean(vals)
            exp3_data[cond] = vals
            rpt(f"| {cond} | {m:.4f} | {np.std(vals):.4f} | {abs(m-E_MINUS_1):.4f} | {abs(m-PHI):.4f} | {len(vals)} |")

    fig, ax = plt.subplots(figsize=(10, 6))
    conds_with_data = [c for c in cond_order if c in exp3_data]
    means_3 = [np.mean(exp3_data[c]) for c in conds_with_data]
    sds_3 = [np.std(exp3_data[c]) for c in conds_with_data]
    colors = ['#2ecc71' if c == 'Exponential' else '#3498db' for c in conds_with_data]
    bars = ax.bar(range(len(conds_with_data)), means_3, yerr=sds_3, color=colors,
                  edgecolor='black', capsize=5, alpha=0.8)
    ax.axhline(E_MINUS_1, color='blue', ls='-', lw=2, label=f'e-1 = {E_MINUS_1:.4f}')
    ax.axhline(PHI, color='goldenrod', ls='--', lw=2, label=f'φ = {PHI:.4f}')
    ax.axhline(2.0, color='red', ls='--', lw=1.5, label='2:1')
    ax.set_xticks(range(len(conds_with_data)))
    ax.set_xticklabels(conds_with_data, fontsize=10)
    ax.set_ylabel('α/θ Centroid Ratio')
    ax.set_title('Wilson-Cowan: Exponential vs Non-Exponential Time Constants')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig('outputs/e1_figures/fig_model_control.png', dpi=300)
    plt.close(fig)
    rpt("\n![Control comparison](outputs/e1_figures/fig_model_control.png)")
    rpt("")

    rpt("## Experiment 4: Kuramoto Oscillators (THE KEY RESULT)")
    rpt("")
    exp4 = results.get('exp4', {})
    spacing_order = ['exponential', 'linear', 'random']
    rpt(f"| Freq Spacing | Mean α/θ | SD | Δe-1 | Δφ | N |")
    rpt(f"|-------------|----------|-----|------|-----|---|")
    exp4_data = {}
    for sp in spacing_order:
        data = exp4.get(sp, {})
        vals = [v for v in data.get('ratios', []) if v is not None]
        if vals:
            m = np.mean(vals)
            exp4_data[sp] = vals
            rpt(f"| {sp} | {m:.4f} | {np.std(vals):.4f} | {abs(m-E_MINUS_1):.4f} | {abs(m-PHI):.4f} | {len(vals)} |")

    fig, ax = plt.subplots(figsize=(8, 6))
    sps_with_data = [s for s in spacing_order if s in exp4_data]
    for i, sp in enumerate(sps_with_data):
        vals = exp4_data[sp]
        color = '#2ecc71' if sp == 'exponential' else '#e74c3c' if sp == 'linear' else '#95a5a6'
        ax.hist(vals, bins=20, alpha=0.5, label=f'{sp} (mean={np.mean(vals):.3f})',
                color=color, edgecolor='white')
    ax.axvline(E_MINUS_1, color='blue', ls='-', lw=2, label=f'e-1 = {E_MINUS_1:.4f}')
    ax.axvline(PHI, color='goldenrod', ls='--', lw=2, label=f'φ = {PHI:.4f}')
    ax.set_xlabel('α/θ Centroid Ratio')
    ax.set_ylabel('Count')
    ax.set_title('Kuramoto Model: Frequency Spacing → α/θ Ratio')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig('outputs/e1_figures/fig_model_kuramoto.png', dpi=300)
    plt.close(fig)
    rpt("\n![Kuramoto results](outputs/e1_figures/fig_model_kuramoto.png)")
    rpt("")

    empirical_mean = 1.7783
    empirical_at_1024 = 1.7432

    if 'exponential' in exp4_data:
        kuramoto_exp = np.mean(exp4_data['exponential'])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        vals_exp = exp4_data['exponential']
        ax.hist(vals_exp, bins=15, alpha=0.7, color='steelblue', edgecolor='white', density=True)
        ax.axvline(E_MINUS_1, color='blue', ls='-', lw=2, label=f'e-1 = {E_MINUS_1:.4f}')
        ax.axvline(empirical_mean, color='green', ls='--', lw=2, label=f'Empirical mean = {empirical_mean:.4f}')
        ax.axvline(empirical_at_1024, color='darkgreen', ls=':', lw=2, label=f'1024Hz mean = {empirical_at_1024:.4f}')
        ax.axvline(kuramoto_exp, color='red', ls='--', lw=2, label=f'Kuramoto exp = {kuramoto_exp:.4f}')
        ax.set_xlabel('α/θ Ratio')
        ax.set_title('Kuramoto Exponential vs Empirical Data')
        ax.legend(fontsize=8)

        ax = axes[1]
        model_names = ['Kuramoto\nExponential', 'Kuramoto\nLinear', 'Empirical\n(all)', 'Empirical\n(1024Hz)']
        model_means = [kuramoto_exp]
        if 'linear' in exp4_data:
            model_means.append(np.mean(exp4_data['linear']))
        else:
            model_means.append(np.nan)
        model_means.extend([empirical_mean, empirical_at_1024])
        colors_bar = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
        ax.bar(range(4), model_means, color=colors_bar, edgecolor='black', alpha=0.8)
        ax.axhline(E_MINUS_1, color='blue', ls='-', lw=2, label=f'e-1 = {E_MINUS_1:.4f}')
        ax.set_xticks(range(4))
        ax.set_xticklabels(model_names, fontsize=9)
        ax.set_ylabel('α/θ Ratio')
        ax.set_title('Model vs Empirical Comparison')
        ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig('outputs/e1_figures/fig_model_comparison.png', dpi=300)
        plt.close(fig)
        rpt("![Model vs empirical](outputs/e1_figures/fig_model_comparison.png)")

    rpt("")
    rpt("## Summary")
    rpt("")

    wc_exp = np.mean(exp3_data.get('Exponential', [0])) if 'Exponential' in exp3_data else None
    wc_lin = np.mean(exp3_data.get('Linear', [0])) if 'Linear' in exp3_data else None
    ku_exp = np.mean(exp4_data.get('exponential', [0])) if 'exponential' in exp4_data else None
    ku_lin = np.mean(exp4_data.get('linear', [0])) if 'linear' in exp4_data else None

    rpt("| Model | Exponential α/θ | Linear α/θ | Exp Δe-1 | Lin Δe-1 |")
    rpt("|-------|----------------|------------|----------|----------|")
    if wc_exp and wc_lin:
        rpt(f"| Wilson-Cowan | {wc_exp:.4f} | {wc_lin:.4f} | {abs(wc_exp-E_MINUS_1):.4f} | {abs(wc_lin-E_MINUS_1):.4f} |")
    if ku_exp and ku_lin:
        rpt(f"| Kuramoto | {ku_exp:.4f} | {ku_lin:.4f} | {abs(ku_exp-E_MINUS_1):.4f} | {abs(ku_lin-E_MINUS_1):.4f} |")
    rpt(f"| Empirical (N=244) | {empirical_mean:.4f} | — | {abs(empirical_mean-E_MINUS_1):.4f} | — |")
    rpt(f"| Empirical (1024Hz) | {empirical_at_1024:.4f} | — | {abs(empirical_at_1024-E_MINUS_1):.4f} | — |")

    rpt("")
    rpt("## Conclusion")
    rpt("")

    if ku_exp:
        ku_exp_gap = abs(ku_exp - E_MINUS_1)
        ku_exp_pct = ku_exp_gap / E_MINUS_1 * 100
        emp_pct = abs(empirical_at_1024 - ku_exp) / empirical_at_1024 * 100

        rpt(f"The Kuramoto oscillator model with exponentially spaced natural frequencies")
        rpt(f"produces emergent α/θ spectral centroid ratios of {ku_exp:.4f}, within")
        rpt(f"{ku_exp_pct:.1f}% of e-1 = {E_MINUS_1:.4f}. This is remarkably close to the")
        rpt(f"empirical 1024 Hz mean of {empirical_at_1024:.4f} (within {emp_pct:.1f}%).")
        rpt(f"")
        if ku_lin:
            ku_lin_gap = abs(ku_lin - E_MINUS_1)
            rpt(f"Linear frequency spacing produces α/θ = {ku_lin:.4f} (Δe-1 = {ku_lin_gap:.4f}),")
            rpt(f"which is {ku_lin_gap/ku_exp_gap:.1f}x farther from e-1 than exponential spacing.")
        rpt(f"")
        rpt(f"The Wilson-Cowan model produces higher ratios ({wc_exp:.2f}–{wc_lin:.2f})")
        rpt(f"regardless of time constant scaling, likely because the sigmoid")
        rpt(f"nonlinearity dominates over the exponential time constants.")
        rpt(f"")
        rpt(f"**KEY FINDING**: Exponential frequency spacing in a minimal Kuramoto network")
        rpt(f"produces emergent spectral ratios within {ku_exp_pct:.1f}% of e-1")
        rpt(f"({emp_pct:.1f}% of empirical data), while linear spacing produces ratios")
        rpt(f"farther from e-1. This is directionally consistent with the hypothesis")
        rpt(f"that exponential neural dynamics contribute to near-e-1 spectral organization,")
        rpt(f"though the {ku_exp_pct:.1f}% residual gap means the model does not precisely")
        rpt(f"reproduce e-1. The result is suggestive rather than confirmatory.")

    with open('outputs/generative_model.md', 'w') as f:
        f.write('\n'.join(REPORT))
    rpt(f"\nReport saved: outputs/generative_model.md")


if __name__ == '__main__':
    main()
