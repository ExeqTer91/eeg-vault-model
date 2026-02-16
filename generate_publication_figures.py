#!/usr/bin/env python3
"""
Generate all 10 publication figures for Ursachi (2026) Preprint v3.1
"Corrugated Spectral Attractor Landscape in Human EEG"

Uses N=244 data from cached JSON results files.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.optimize import minimize
import os

PHI = (1 + np.sqrt(5)) / 2
E_MINUS_1 = np.e - 1
SQRT_PI = np.sqrt(np.pi)

OUTPUT_DIR = 'outputs/publication_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})


def load_all_subjects():
    subjects = []
    for fn in ['aw_cached_subjects.json', 'ds003969_cached_subjects.json', 'eegbci_modal_results.json']:
        with open(f'outputs/{fn}') as f:
            subjects.extend(json.load(f))
    return subjects


def get_alpha_theta_ratios(subjects):
    ratios = []
    for s in subjects:
        ar = s['adjacent_ratios']
        key = 'alpha/theta' if 'alpha/theta' in ar else 'alpha_theta'
        if key in ar and ar[key] is not None:
            ratios.append(ar[key])
    return np.array(ratios)


def get_all_adjacent_ratios(subjects):
    ratio_names_slash = ['theta/delta', 'alpha/theta', 'beta/alpha', 'gamma/beta']
    ratio_names_under = ['theta_delta', 'alpha_theta', 'beta_alpha', 'gamma_beta']
    canonical = ['theta_delta', 'alpha_theta', 'beta_alpha', 'gamma_beta']
    result = {name: [] for name in canonical}
    for s in subjects:
        ar = s['adjacent_ratios']
        for slash, under, canon in zip(ratio_names_slash, ratio_names_under, canonical):
            key = slash if slash in ar else under
            if key in ar and ar[key] is not None:
                result[canon].append(ar[key])
    return {k: np.array(v) for k, v in result.items()}


def vault_potential(x, x0, k_left, k_right):
    v = np.where(x < x0, 0.5 * k_left * (x - x0)**2, 0.5 * k_right * (x - x0)**2)
    return v


def corrugated_potential(x, x0, k_left, k_right, A_phi, A_e1, sigma=0.03):
    v = vault_potential(x, x0, k_left, k_right)
    v += A_phi * np.exp(-0.5 * ((x - PHI) / sigma)**2)
    v += A_e1 * np.exp(-0.5 * ((x - E_MINUS_1) / sigma)**2)
    return v


def boltzmann_pdf(x, V, T):
    log_p = -V / T
    log_p -= np.max(log_p)
    p = np.exp(log_p)
    dx = x[1] - x[0]
    trapz_fn = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    p /= trapz_fn(p, x)
    return p


def fit_skew_normal(data):
    from scipy.stats import skewnorm
    params = skewnorm.fit(data)
    return params


subjects = load_all_subjects()
ratios = get_alpha_theta_ratios(subjects)

with open('outputs/vault_model_results.json') as f:
    vault_res = json.load(f)
with open('outputs/geometric_obstruction_results.json') as f:
    obstruction_res = json.load(f)
with open('outputs/generative_model_results.json') as f:
    gen_res = json.load(f)


# ====================================================================
# FIGURE 1: Population distribution with corrugated potential overlay
# ====================================================================
def figure1():
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 1, height_ratios=[1, 1.2], hspace=0.3)

    vf = vault_res['vault_fit']
    x0, kl, kr = vf['x0'], vf['k_left'], vf['k_right']
    T = vf['T']

    cf = obstruction_res['corrugated_fit']
    A_phi = cf['A_phi']
    A_e1 = cf['A_e1']
    sigma = cf.get('sigma', 0.03)

    x_grid = np.linspace(1.3, 2.3, 1000)

    V_smooth = vault_potential(x_grid, x0, kl, kr)
    V_corr = corrugated_potential(x_grid, x0, kl, kr, A_phi, A_e1, sigma)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x_grid, V_corr, 'k-', lw=2, label='Corrugated potential V(x)')
    ax1.plot(x_grid, V_smooth, '--', color='gray', lw=1, alpha=0.6, label='Smooth vault')

    ax1.axvline(PHI, color='goldenrod', ls='--', lw=1.5, alpha=0.8)
    ax1.axvline(E_MINUS_1, color='steelblue', ls='--', lw=1.5, alpha=0.8)
    ax1.axvline(x0, color='black', ls=':', lw=1, alpha=0.6)
    ax1.axvline(2.0, color='red', ls='--', lw=1.5, alpha=0.8)

    y_top = ax1.get_ylim()[1]
    ax1.text(PHI, y_top * 0.9, 'φ\n(barrier)', ha='center', fontsize=8, color='goldenrod', fontweight='bold')
    ax1.text(E_MINUS_1, y_top * 0.75, 'e−1\n(trap)', ha='center', fontsize=8, color='steelblue', fontweight='bold')
    ax1.text(x0, y_top * 0.9, f'x₀={x0:.2f}\n(global min)', ha='center', fontsize=7, color='black')
    ax1.text(2.0, y_top * 0.9, '2:1\n(wall)', ha='center', fontsize=8, color='red', fontweight='bold')

    ax1.set_ylabel('Potential V(x)', fontsize=11)
    ax1.set_title('Corrugated Spectral Attractor Landscape (N = 244)', fontsize=13, fontweight='bold')
    ax1.set_xlim(1.35, 2.15)
    ax1.legend(fontsize=8, loc='upper left')

    ax2 = fig.add_subplot(gs[1])
    ax2.hist(ratios, bins=40, density=True, color='lightsteelblue', edgecolor='steelblue', alpha=0.7, label='Observed (N=244)')

    pdf_corr = boltzmann_pdf(x_grid, V_corr, T)
    ax2.plot(x_grid, pdf_corr, 'k-', lw=2, label='Corrugated model')

    from scipy.stats import skewnorm
    sn_params = skewnorm.fit(ratios)
    sn_pdf = skewnorm.pdf(x_grid, *sn_params)
    ax2.plot(x_grid, sn_pdf, 'r--', lw=1.5, alpha=0.7, label='Skew-normal (best AIC)')

    ax2.axvline(PHI, color='goldenrod', ls='--', lw=1.5, alpha=0.8)
    ax2.axvline(E_MINUS_1, color='steelblue', ls='--', lw=1.5, alpha=0.8)
    ax2.axvline(np.mean(ratios), color='black', ls='-', lw=1, alpha=0.5)
    ax2.axvline(2.0, color='red', ls='--', lw=1.5, alpha=0.8)

    ax2.set_xlabel('α/θ Centroid Ratio', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_xlim(1.35, 2.15)
    ax2.legend(fontsize=8)

    stats_text = (f'Mean = {np.mean(ratios):.3f} ± {np.std(ratios):.3f}\n'
                  f'Skewness = {float(vault_res["empirical"]["skew"]):.2f}\n'
                  f'Dip test p = {obstruction_res["dip_test"]["p_value"]:.3f}')
    ax2.text(0.98, 0.95, stats_text, transform=ax2.transAxes, fontsize=8,
             va='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.savefig(f'{OUTPUT_DIR}/fig1_population_corrugated.png')
    plt.close(fig)
    print("  Fig 1: Population distribution with corrugated potential")


# ====================================================================
# FIGURE 2: Corrugated landscape detail
# ====================================================================
def figure2():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    vf = vault_res['vault_fit']
    x0, kl, kr = vf['x0'], vf['k_left'], vf['k_right']
    T = vf['T']
    cf = obstruction_res['corrugated_fit']
    A_phi, A_e1 = cf['A_phi'], cf['A_e1']
    sigma = cf.get('sigma', 0.03)

    x_grid = np.linspace(1.4, 2.2, 1000)
    V_smooth = vault_potential(x_grid, x0, kl, kr)
    V_corr = corrugated_potential(x_grid, x0, kl, kr, A_phi, A_e1, sigma)

    ax = axes[0]
    ax.plot(x_grid, V_smooth, 'gray', lw=2, label='Smooth vault')
    ax.plot(x_grid, V_corr, 'k-', lw=2.5, label='Corrugated')
    ax.fill_between(x_grid, V_smooth, V_corr, where=V_corr < V_smooth,
                     color='steelblue', alpha=0.3, label='e−1 trap (A<0)')
    ax.fill_between(x_grid, V_smooth, V_corr, where=V_corr > V_smooth,
                     color='goldenrod', alpha=0.3, label='φ barrier (A>0)')

    ax.axvline(PHI, color='goldenrod', ls=':', lw=1)
    ax.axvline(E_MINUS_1, color='steelblue', ls=':', lw=1)
    ax.axvline(x0, color='black', ls=':', lw=1)

    ax.set_xlabel('α/θ Ratio')
    ax.set_ylabel('Potential V(x)')
    ax.set_title('Smooth vs. Corrugated Potential')
    ax.legend(fontsize=8)

    ax.annotate(f'A_φ = +{A_phi:.2f}\n(repulsive)', xy=(PHI, vault_potential(PHI, x0, kl, kr)),
                xytext=(PHI - 0.08, vault_potential(PHI, x0, kl, kr) + 1.5),
                fontsize=8, color='goldenrod', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='goldenrod'))
    ax.annotate(f'A_e−1 = {A_e1:.2f}\n(attractive)', xy=(E_MINUS_1, vault_potential(E_MINUS_1, x0, kl, kr)),
                xytext=(E_MINUS_1 + 0.05, vault_potential(E_MINUS_1, x0, kl, kr) + 2.0),
                fontsize=8, color='steelblue', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='steelblue'))

    ax2 = axes[1]
    pdf_smooth = boltzmann_pdf(x_grid, V_smooth, T)
    pdf_corr = boltzmann_pdf(x_grid, V_corr, T)

    ax2.plot(x_grid, pdf_smooth, 'gray', lw=2, label='Smooth vault PDF')
    ax2.plot(x_grid, pdf_corr, 'k-', lw=2.5, label='Corrugated PDF')
    ax2.hist(ratios, bins=40, density=True, color='lightsteelblue', edgecolor='steelblue', alpha=0.5)
    ax2.axvline(PHI, color='goldenrod', ls=':', lw=1)
    ax2.axvline(E_MINUS_1, color='steelblue', ls=':', lw=1)

    sim_mean = vault_res['simulation']['mean']
    sim_sd = vault_res['simulation']['sd']
    emp_mean = vault_res['empirical']['mean']
    emp_sd = vault_res['empirical']['sd']

    ax2.set_xlabel('α/θ Ratio')
    ax2.set_ylabel('Density')
    ax2.set_title('Boltzmann Distribution Fits')
    ax2.legend(fontsize=8)

    fit_text = (f'Simulation: μ={sim_mean:.3f}, σ={sim_sd:.3f}\n'
                f'Empirical:  μ={emp_mean:.3f}, σ={emp_sd:.3f}\n'
                f'Mean error: {abs(sim_mean - emp_mean):.4f}')
    ax2.text(0.98, 0.95, fit_text, transform=ax2.transAxes, fontsize=8,
             va='top', ha='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    fig.suptitle('Figure 2: Corrugated Potential Landscape Detail', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig2_corrugated_landscape.png')
    plt.close(fig)
    print("  Fig 2: Corrugated landscape detail")


# ====================================================================
# FIGURE 3: Basin classification
# ====================================================================
def figure3():
    fig, ax = plt.subplots(figsize=(9, 5))

    boundaries = [1.67, 1.75, 1.90]
    basin_names = ['φ-trapped', 'e−1 residents', 'Main well', '2:1-pressed']
    basin_colors = ['goldenrod', 'steelblue', 'forestgreen', 'firebrick']

    bins = np.linspace(1.35, 2.15, 45)
    for r in ratios:
        if r < boundaries[0]:
            idx = 0
        elif r < boundaries[1]:
            idx = 1
        elif r < boundaries[2]:
            idx = 2
        else:
            idx = 3
        ax.bar(r, 0, color=basin_colors[idx])

    for i, (name, color) in enumerate(zip(basin_names, basin_colors)):
        if i == 0:
            mask = ratios < boundaries[0]
        elif i == 1:
            mask = (ratios >= boundaries[0]) & (ratios < boundaries[1])
        elif i == 2:
            mask = (ratios >= boundaries[1]) & (ratios < boundaries[2])
        else:
            mask = ratios >= boundaries[2]

        subset = ratios[mask]
        pct = 100 * len(subset) / len(ratios)
        ax.hist(subset, bins=bins, color=color, alpha=0.7, edgecolor='white',
                label=f'{name} ({pct:.1f}%, N={len(subset)})')

    for b in boundaries:
        ax.axvline(b, color='black', ls=':', lw=1, alpha=0.5)

    ax.axvline(PHI, color='goldenrod', ls='--', lw=1.5, alpha=0.6)
    ax.axvline(E_MINUS_1, color='steelblue', ls='--', lw=1.5, alpha=0.6)
    ax.axvline(2.0, color='red', ls='--', lw=1.5, alpha=0.6)

    ax.text(PHI, ax.get_ylim()[1] * 0.95, 'φ', ha='center', fontsize=10, color='goldenrod', fontweight='bold')
    ax.text(E_MINUS_1, ax.get_ylim()[1] * 0.95, 'e−1', ha='center', fontsize=10, color='steelblue', fontweight='bold')
    ax.text(2.0, ax.get_ylim()[1] * 0.95, '2:1', ha='center', fontsize=10, color='red', fontweight='bold')

    ax.set_xlabel('α/θ Centroid Ratio', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Figure 3: Basin Classification (N = 244)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig3_basin_classification.png')
    plt.close(fig)
    print("  Fig 3: Basin classification")


# ====================================================================
# FIGURE 4: Temperature (SD) by sampling rate
# ====================================================================
def figure4():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fs_vals = []
    at_ratios = []
    for s in subjects:
        ar = s['adjacent_ratios']
        if 'alpha_theta' in ar and ar['alpha_theta'] is not None:
            fs_vals.append(s['fs'])
            at_ratios.append(ar['alpha_theta'])

    fs_arr = np.array(fs_vals)
    at_arr = np.array(at_ratios)

    unique_fs = sorted(np.unique(fs_arr))
    means = [np.mean(at_arr[fs_arr == fs]) for fs in unique_fs]
    sds = [np.std(at_arr[fs_arr == fs], ddof=1) for fs in unique_fs]
    ns = [np.sum(fs_arr == fs) for fs in unique_fs]

    ax = axes[0]
    colors_fs = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    for i, fs in enumerate(unique_fs):
        subset = at_arr[fs_arr == fs]
        c = colors_fs[i % len(colors_fs)]
        ax.scatter([fs] * len(subset), subset, alpha=0.3, s=20, color=c)
        ax.errorbar(fs, means[i], yerr=sds[i], fmt='o', color=c, markersize=10,
                     capsize=5, elinewidth=2, label=f'{int(fs)} Hz (N={ns[i]})')

    ax.axhline(E_MINUS_1, color='steelblue', ls='--', lw=1.5, alpha=0.7, label='e−1')
    ax.axhline(np.mean(at_arr), color='gray', ls=':', lw=1, alpha=0.5, label=f'Grand mean ({np.mean(at_arr):.3f})')

    ax.set_xlabel('Sampling Rate (Hz)', fontsize=11)
    ax.set_ylabel('α/θ Centroid Ratio', fontsize=11)
    ax.set_title('Centroid Ratio by Sampling Rate')
    ax.legend(fontsize=7, loc='upper right')

    ax2 = axes[1]
    ax2.bar(range(len(unique_fs)), sds, color=colors_fs[:len(unique_fs)], alpha=0.8, edgecolor='black')
    ax2.set_xticks(range(len(unique_fs)))
    ax2.set_xticklabels([f'{int(fs)} Hz\n(N={n})' for fs, n in zip(unique_fs, ns)], fontsize=9)
    ax2.set_ylabel('SD (Effective Temperature)', fontsize=11)
    ax2.set_title('Effective Temperature by Fs')

    for i, (sd, m) in enumerate(zip(sds, means)):
        ax2.text(i, sd + 0.002, f'μ={m:.3f}\nσ={sd:.3f}', ha='center', fontsize=7)

    fig.suptitle('Figure 4: Sampling Rate Effect on Centroid Estimation', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig4_temperature_sampling_rate.png')
    plt.close(fig)
    print("  Fig 4: Temperature by sampling rate")


# ====================================================================
# FIGURE 5: Generative model comparison (Kuramoto)
# ====================================================================
def figure5():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    exp3 = gen_res['exp3']
    model_names = list(exp3.keys())
    model_means = [np.mean(exp3[m]['ratios']) for m in model_names]
    model_sds = [np.std(exp3[m]['ratios']) for m in model_names]

    colors_model = ['steelblue', 'gray', 'orange', 'green', 'purple']
    ax = axes[0]
    bars = ax.bar(range(len(model_names)), model_means, yerr=model_sds,
                   color=colors_model[:len(model_names)], alpha=0.8,
                   edgecolor='black', capsize=5)

    ax.axhline(E_MINUS_1, color='steelblue', ls='--', lw=1.5, alpha=0.7, label='e−1 = 1.718')
    ax.axhline(np.mean(ratios), color='black', ls=':', lw=1, alpha=0.5, label=f'Empirical mean = {np.mean(ratios):.3f}')

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([m.replace('_', '\n') for m in model_names], fontsize=8)
    ax.set_ylabel('α/θ Ratio', fontsize=11)
    ax.set_title('Kuramoto Oscillator: Frequency Spacing')
    ax.legend(fontsize=8)

    for i, (m, s) in enumerate(zip(model_means, model_sds)):
        pct_from_e1 = abs(m - E_MINUS_1) / E_MINUS_1 * 100
        ax.text(i, m + s + 0.02, f'{m:.3f}\n({pct_from_e1:.1f}%)', ha='center', fontsize=7)

    exp1_ratios = gen_res['exp1']['ratios']
    ax2 = axes[1]
    ax2.hist(exp1_ratios, bins=20, density=True, color='steelblue', alpha=0.7, edgecolor='white',
             label=f'Kuramoto (N={len(exp1_ratios)} sims)')
    ax2.hist(ratios, bins=30, density=True, color='lightcoral', alpha=0.5, edgecolor='white',
             label=f'Empirical (N={len(ratios)})')
    ax2.axvline(E_MINUS_1, color='steelblue', ls='--', lw=1.5, label='e−1')
    ax2.axvline(np.mean(exp1_ratios), color='navy', ls='-', lw=1, label=f'Kuramoto mean = {np.mean(exp1_ratios):.3f}')

    ax2.set_xlabel('α/θ Ratio', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Kuramoto (Exp) vs. Empirical Distribution')
    ax2.legend(fontsize=8)

    fig.suptitle('Figure 5: Generative Model — Kuramoto Oscillators', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig5_generative_model.png')
    plt.close(fig)
    print("  Fig 5: Generative model comparison")


# ====================================================================
# FIGURE 6: Alternating attractor pattern
# ====================================================================
def figure6():
    fig, ax = plt.subplots(figsize=(9, 5))

    all_ratios = get_all_adjacent_ratios(subjects)
    pair_labels = ['θ/δ', 'α/θ', 'β/α', 'γ/β']
    pair_keys = ['theta_delta', 'alpha_theta', 'beta_alpha', 'gamma_beta']

    pair_means = [np.mean(all_ratios[k]) for k in pair_keys]
    pair_sds = [np.std(all_ratios[k]) for k in pair_keys]
    pair_sems = [np.std(all_ratios[k]) / np.sqrt(len(all_ratios[k])) for k in pair_keys]

    nearest = []
    for m in pair_means:
        d_phi = abs(m - PHI)
        d_e1 = abs(m - E_MINUS_1)
        d_2 = abs(m - 2.0)
        if d_phi < d_e1 and d_phi < d_2:
            nearest.append(('φ', 'goldenrod'))
        elif d_e1 < d_2:
            nearest.append(('e−1', 'steelblue'))
        else:
            nearest.append(('2:1', 'firebrick'))

    colors = [n[1] for n in nearest]
    bars = ax.bar(range(4), pair_means, yerr=pair_sems, color=colors, alpha=0.8,
                   edgecolor='black', capsize=5)

    ax.axhline(PHI, color='goldenrod', ls='--', lw=1.5, alpha=0.5, label=f'φ = {PHI:.3f}')
    ax.axhline(E_MINUS_1, color='steelblue', ls='--', lw=1.5, alpha=0.5, label=f'e−1 = {E_MINUS_1:.3f}')
    ax.axhline(2.0, color='firebrick', ls='--', lw=1.5, alpha=0.5, label='2:1 = 2.000')

    for i, (m, sem, n_info) in enumerate(zip(pair_means, pair_sems, nearest)):
        pct_err = abs(m - {'φ': PHI, 'e−1': E_MINUS_1, '2:1': 2.0}[n_info[0]]) / {'φ': PHI, 'e−1': E_MINUS_1, '2:1': 2.0}[n_info[0]] * 100
        ax.text(i, m + sem + 0.05, f'{m:.3f}\n≈ {n_info[0]} ({pct_err:.1f}%)',
                ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(range(4))
    ax.set_xticklabels(pair_labels, fontsize=12)
    ax.set_ylabel('Mean Centroid Ratio', fontsize=11)
    ax.set_title('Figure 6: Alternating Attractor Pattern Across Band Pairs (N = 244)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig6_alternating_attractors.png')
    plt.close(fig)
    print("  Fig 6: Alternating attractor pattern")


# ====================================================================
# FIGURE 7: AIC model comparison
# ====================================================================
def figure7():
    fig, ax = plt.subplots(figsize=(10, 5))

    mc = vault_res['model_comparison']
    model_names = list(mc.keys())
    delta_aics = [mc[m]['delta_aic'] for m in model_names]

    short_names = []
    for n in model_names:
        if 'φ' in n or 'phi' in n.lower():
            short_names.append('Gaussian\n@ φ')
        elif 'e-1' in n:
            short_names.append('Gaussian\n@ e−1')
        elif 'Vault' in n or 'asym' in n:
            short_names.append('Smooth\nVault (4p)')
        elif 'Skew' in n:
            short_names.append('Skew-\nnormal (3p)')
        elif 'GMM' in n:
            short_names.append('GMM\n2-comp (5p)')
        else:
            short_names.append(n)

    corr_daic = abs(obstruction_res['corrugated_vs_smooth_daic'])
    vault_daic = mc['Vault (asym well)']['delta_aic']
    corr_abs_daic = vault_daic - corr_daic

    all_names = short_names + ['Corrugated\nVault (6p)']
    all_daics = delta_aics + [corr_abs_daic]

    sort_idx = np.argsort(all_daics)
    all_names = [all_names[i] for i in sort_idx]
    all_daics = [all_daics[i] for i in sort_idx]

    colors_aic = []
    for n in all_names:
        if 'φ' in n:
            colors_aic.append('goldenrod')
        elif 'e−1' in n:
            colors_aic.append('steelblue')
        elif 'Corrugated' in n:
            colors_aic.append('darkgreen')
        elif 'Skew' in n:
            colors_aic.append('firebrick')
        elif 'Vault' in n or 'Smooth' in n:
            colors_aic.append('forestgreen')
        else:
            colors_aic.append('gray')

    bars = ax.barh(range(len(all_names)), all_daics, color=colors_aic, alpha=0.8, edgecolor='black')

    for i, (d, name) in enumerate(zip(all_daics, all_names)):
        ax.text(d + 2, i, f'ΔAIC = {d:.1f}', va='center', fontsize=8)

    ax.set_yticks(range(len(all_names)))
    ax.set_yticklabels(all_names, fontsize=9)
    ax.set_xlabel('ΔAIC (lower = better)', fontsize=11)
    ax.set_title('Figure 7: Model Comparison — AIC Rankings', fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', lw=0.5)

    ax.text(0.7, 0.85, 'Best overall: Skew-normal (3p)\nBest mechanistic: Corrugated (6p)',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig7_model_comparison_aic.png')
    plt.close(fig)
    print("  Fig 7: AIC model comparison")


# ====================================================================
# FIGURE 8: TOST equivalence plot
# ====================================================================
def figure8():
    fig, ax = plt.subplots(figsize=(8, 6))

    mean_r = np.mean(ratios)
    sem_r = stats.sem(ratios)
    n = len(ratios)

    bounds_list = [0.05, 0.10, 0.15]
    y_positions = [0, 1, 2]

    ax.errorbar(mean_r, 1, xerr=1.96 * sem_r, fmt='ko', markersize=10,
                 capsize=8, elinewidth=2, label=f'Mean = {mean_r:.4f} ± {1.96*sem_r:.4f}')

    colors_tost = ['red', 'orange', 'green']
    for i, (bound, y_off) in enumerate(zip(bounds_list, [0.3, 0.15, 0.0])):
        lower = E_MINUS_1 - bound
        upper = E_MINUS_1 + bound
        ax.fill_betweenx([0.5, 1.5], lower, upper, alpha=0.15, color=colors_tost[i])
        ax.axvline(lower, color=colors_tost[i], ls=':', lw=1, alpha=0.5)
        ax.axvline(upper, color=colors_tost[i], ls=':', lw=1, alpha=0.5)

        t_lower = (mean_r - lower) / (np.std(ratios, ddof=1) / np.sqrt(n))
        t_upper = (upper - mean_r) / (np.std(ratios, ddof=1) / np.sqrt(n))
        p_tost = max(1 - stats.t.cdf(t_lower, n - 1), 1 - stats.t.cdf(t_upper, n - 1))

        status = "PASS" if p_tost < 0.05 else "FAIL"
        ax.text(upper + 0.01, 1.3 - i * 0.15,
                f'±{bound:.2f}: p = {p_tost:.2e} ({status})',
                fontsize=8, color=colors_tost[i], fontweight='bold')

    ax.axvline(E_MINUS_1, color='steelblue', ls='--', lw=2, label=f'e−1 = {E_MINUS_1:.4f}')
    ax.axvline(PHI, color='goldenrod', ls='--', lw=2, label=f'φ = {PHI:.4f}')

    t_vs_e1 = (mean_r - E_MINUS_1) / (np.std(ratios, ddof=1) / np.sqrt(n))
    p_vs_e1 = 2 * (1 - stats.t.cdf(abs(t_vs_e1), n - 1))
    d_vs_e1 = (mean_r - E_MINUS_1) / np.std(ratios, ddof=1)

    t_vs_phi = (mean_r - PHI) / (np.std(ratios, ddof=1) / np.sqrt(n))
    p_vs_phi = 2 * (1 - stats.t.cdf(abs(t_vs_phi), n - 1))
    d_vs_phi = (mean_r - PHI) / np.std(ratios, ddof=1)

    stats_text = (f'vs e−1: d = {d_vs_e1:.3f}, p = {p_vs_e1:.2e}\n'
                  f'vs φ:   d = {d_vs_phi:.3f}, p = {p_vs_phi:.2e}\n'
                  f'ΔAIC(e−1 vs φ) = {vault_res["model_comparison"]["Gaussian @ e-1"]["delta_aic"] - vault_res["model_comparison"]["Gaussian @ φ"]["delta_aic"]:.1f}')
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.set_xlabel('α/θ Centroid Ratio', fontsize=11)
    ax.set_yticks([])
    ax.set_title('Figure 8: TOST Equivalence Testing vs. e−1', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(1.55, 1.95)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig8_tost_equivalence.png')
    plt.close(fig)
    print("  Fig 8: TOST equivalence")


# ====================================================================
# FIGURE 9: Cross-domain analysis
# ====================================================================
def figure9():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    all_ratios_dict = get_all_adjacent_ratios(subjects)

    constants = {
        'φ': PHI,
        'e−1': E_MINUS_1,
        '√π': SQRT_PI,
        '16/9': 16/9,
        '2:1': 2.0,
        '3:2': 1.5,
        '√3': np.sqrt(3),
        'ln(2)': np.log(2),
        '4/√5': 4/np.sqrt(5),
        '√e': np.sqrt(np.e),
        'π/2': np.pi/2,
    }

    ratio_types = {
        'power_α/power_β': [],
        'power_β/power_θ': [],
        'power_γ/power_α': [],
    }
    for s in subjects:
        p = s['powers']
        if all(b in p for b in ['alpha', 'beta', 'theta', 'gamma']):
            if p['beta'] > 0:
                ratio_types['power_α/power_β'].append(p['alpha'] / p['beta'])
            if p['theta'] > 0:
                ratio_types['power_β/power_θ'].append(p['beta'] / p['theta'])
            if p['alpha'] > 0:
                ratio_types['power_γ/power_α'].append(p['gamma'] / p['alpha'])

    all_cross = {}
    for rname, rvals in {**{k: list(v) for k, v in all_ratios_dict.items()}, **ratio_types}.items():
        if len(rvals) < 10:
            continue
        mean_val = np.mean(rvals)
        for cname, cval in constants.items():
            pct_err = abs(mean_val - cval) / cval * 100
            all_cross[f'{rname} ≈ {cname}'] = {
                'ratio': rname, 'constant': cname, 'mean': mean_val,
                'target': cval, 'pct_error': pct_err
            }

    sorted_matches = sorted(all_cross.values(), key=lambda x: x['pct_error'])[:15]

    ax = axes[0]
    names = [f"{m['ratio']}\n≈ {m['constant']}" for m in sorted_matches]
    pcts = [m['pct_error'] for m in sorted_matches]
    colors_cross = ['steelblue' if 'e−1' in m['constant'] else
                     'goldenrod' if 'φ' in m['constant'] else
                     'firebrick' if '2:1' in m['constant'] else 'gray'
                     for m in sorted_matches]

    bars = ax.barh(range(len(names)), pcts, color=colors_cross, alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('% Error from Constant', fontsize=11)
    ax.set_title('Top Cross-Domain Matches')
    ax.invert_yaxis()

    ax2 = axes[1]
    excess = obstruction_res['excess_at_constants']
    const_names_exc = list(excess.keys())
    z_scores = [excess[c]['z_excess'] for c in const_names_exc]
    residuals = [excess[c]['residual'] for c in const_names_exc]

    colors_exc = ['goldenrod' if 'φ' in c else
                   'steelblue' if 'e' in c else
                   'gray' for c in const_names_exc]

    ax2.bar(range(len(const_names_exc)), residuals, color=colors_exc, alpha=0.8, edgecolor='black')
    ax2.set_xticks(range(len(const_names_exc)))
    ax2.set_xticklabels(const_names_exc, fontsize=7, rotation=45, ha='right')
    ax2.set_ylabel('KDE Residual (observed − expected)', fontsize=10)
    ax2.set_title('Density Excess at Constants')
    ax2.axhline(0, color='black', lw=0.5)

    for i, (r, z) in enumerate(zip(residuals, z_scores)):
        ax2.text(i, r + 0.02 * np.sign(r), f'z={z:.2f}', ha='center', fontsize=7)

    fig.suptitle('Figure 9: Cross-Domain Analysis', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig9_cross_domain_analysis.png')
    plt.close(fig)
    print("  Fig 9: Cross-domain analysis")


# ====================================================================
# FIGURE 10: Surrogate comparison
# ====================================================================
def figure10():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    np.random.seed(42)
    n_surr = 10000
    n_sub = len(ratios)

    surrogate_means = []
    for _ in range(n_surr):
        fake = np.random.normal(np.mean(ratios), np.std(ratios), n_sub)
        surrogate_means.append(np.mean(fake))

    surrogate_means = np.array(surrogate_means)

    surr_d_e1 = np.abs(surrogate_means - E_MINUS_1)
    obs_d_e1 = abs(np.mean(ratios) - E_MINUS_1)
    p_surr = np.mean(surr_d_e1 <= obs_d_e1)

    ax = axes[0]
    ax.hist(surrogate_means, bins=80, density=True, color='lightgray', edgecolor='gray',
            alpha=0.7, label=f'Surrogates (N={n_surr})')
    ax.axvline(np.mean(ratios), color='black', lw=2, label=f'Observed mean = {np.mean(ratios):.4f}')
    ax.axvline(E_MINUS_1, color='steelblue', ls='--', lw=2, label=f'e−1 = {E_MINUS_1:.4f}')
    ax.axvline(PHI, color='goldenrod', ls='--', lw=2, label=f'φ = {PHI:.4f}')

    ax.set_xlabel('Surrogate Mean α/θ', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Null Distribution of Mean α/θ')
    ax.legend(fontsize=8)

    ax2 = axes[1]
    surr_distances_e1 = np.abs(surrogate_means - E_MINUS_1)
    surr_distances_phi = np.abs(surrogate_means - PHI)

    ax2.hist(surr_distances_e1, bins=60, density=True, color='steelblue', alpha=0.5,
             label='Distance from e−1')
    ax2.hist(surr_distances_phi, bins=60, density=True, color='goldenrod', alpha=0.5,
             label='Distance from φ')

    ax2.axvline(obs_d_e1, color='steelblue', lw=2, ls='-',
                 label=f'Observed |mean − e−1| = {obs_d_e1:.4f}')
    obs_d_phi = abs(np.mean(ratios) - PHI)
    ax2.axvline(obs_d_phi, color='goldenrod', lw=2, ls='-',
                 label=f'Observed |mean − φ| = {obs_d_phi:.4f}')

    ax2.set_xlabel('Distance from Constant', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Proximity Test')
    ax2.legend(fontsize=7)

    ax2.text(0.95, 0.95, f'p(closer to e−1) = {p_surr:.4f}\nφ is {obs_d_phi/obs_d_e1:.1f}× farther',
             transform=ax2.transAxes, fontsize=9, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    fig.suptitle('Figure 10: Surrogate Null Distribution (10,000 Permutations)', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig10_surrogate_comparison.png')
    plt.close(fig)
    print("  Fig 10: Surrogate comparison")


# ====================================================================
# RUN ALL
# ====================================================================
if __name__ == '__main__':
    print(f"\nGenerating 10 publication figures for Ursachi (2026) Preprint v3.1")
    print(f"N = {len(ratios)} subjects, output: {OUTPUT_DIR}/\n")

    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    figure7()
    figure8()
    figure9()
    figure10()

    print(f"\nAll 10 figures saved to {OUTPUT_DIR}/")
    print("Ready for manuscript insertion.")
