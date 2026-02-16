#!/usr/bin/env python3
"""
Additional analyses for Ursachi (2026) Preprint v3.2
Jobs 1, 3, 4 — computational analyses on cached N=244 data
Job 2 — downsampling experiment on OpenNeuro ds003969 raw EEG
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import os
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
E_MINUS_1 = np.e - 1

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

trapz_fn = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz


def load_ratios():
    subjects = []
    for fn in ['aw_cached_subjects.json', 'ds003969_cached_subjects.json', 'eegbci_modal_results.json']:
        with open(f'outputs/{fn}') as f:
            subjects.extend(json.load(f))
    ratios = []
    for s in subjects:
        ar = s['adjacent_ratios']
        key = 'alpha/theta' if 'alpha/theta' in ar else 'alpha_theta'
        if key in ar and ar[key] is not None:
            ratios.append(ar[key])
    return np.array(ratios), subjects


def vault_potential(x, x0, k_left, k_right):
    return np.where(x < x0, 0.5 * k_left * (x - x0)**2, 0.5 * k_right * (x - x0)**2)


def corrugated_potential_general(x, x0, k_left, k_right, A1, A2, loc1, loc2, sigma=0.03):
    v = vault_potential(x, x0, k_left, k_right)
    v += A1 * np.exp(-0.5 * ((x - loc1) / sigma)**2)
    v += A2 * np.exp(-0.5 * ((x - loc2) / sigma)**2)
    return v


def boltzmann_nll(params, data, x_grid, potential_fn):
    try:
        V = potential_fn(x_grid, *params[:-1])
        T = params[-1]
        if T <= 0:
            return 1e10
        log_p = -V / T
        log_p -= np.max(log_p)
        p = np.exp(log_p)
        p /= trapz_fn(p, x_grid)
        dx = x_grid[1] - x_grid[0]
        data_indices = np.searchsorted(x_grid, data).clip(0, len(x_grid) - 1)
        log_likelihood = np.sum(np.log(p[data_indices] + 1e-30))
        return -log_likelihood
    except:
        return 1e10


ratios, all_subjects = load_ratios()
print(f"Loaded N={len(ratios)} subjects")

with open('outputs/vault_model_results.json') as f:
    vault_res = json.load(f)
with open('outputs/geometric_obstruction_results.json') as f:
    obstruction_res = json.load(f)

vf = vault_res['vault_fit']
cf = obstruction_res['corrugated_fit']


# ====================================================================
# JOB 1: Permutation test on corrugated perturbation locations
# ====================================================================
def job1_permutation_test():
    print("\n" + "="*60)
    print("JOB 1: Permutation test on perturbation locations")
    print("="*60)

    x_grid = np.linspace(1.0, 2.5, 2000)
    x0, kl, kr = cf['x0'], cf['k_left'], cf['k_right']

    idx = np.searchsorted(x_grid, ratios).clip(0, len(x_grid) - 1)

    def fit_smooth_vault():
        def nll_s(params):
            x0p, klp, krp, Tp = params
            if Tp <= 0 or klp <= 0 or krp <= 0:
                return 1e10
            V = vault_potential(x_grid, x0p, klp, krp)
            log_p = -V / Tp
            log_p -= np.max(log_p)
            p = np.exp(log_p)
            p /= trapz_fn(p, x_grid)
            return -np.sum(np.log(p[idx] + 1e-30))
        res = minimize(nll_s, [x0, kl, kr, vf['T']], method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-6})
        return res.fun

    nll_smooth = fit_smooth_vault()
    aic_smooth = 2 * nll_smooth + 2 * 4

    def fit_corrugated_at_locations(loc1, loc2):
        bump1 = np.exp(-0.5 * ((x_grid - loc1) / 0.03)**2)
        bump2 = np.exp(-0.5 * ((x_grid - loc2) / 0.03)**2)

        def nll(params):
            A1, A2, T = params
            if T <= 0:
                return 1e10
            V = vault_potential(x_grid, x0, kl, kr) + A1 * bump1 + A2 * bump2
            log_p = -V / T
            log_p -= np.max(log_p)
            p = np.exp(log_p)
            p /= trapz_fn(p, x_grid)
            return -np.sum(np.log(p[idx] + 1e-30))

        res = minimize(nll, [0.5, -0.25, 1.0], method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-5})

        aic_corr = 2 * res.fun + 2 * 6
        return aic_corr - aic_smooth

    real_daic = fit_corrugated_at_locations(PHI, E_MINUS_1)
    print(f"Real model (φ, e-1): ΔAIC = {real_daic:.2f}")

    n_perm = 1000
    random_daics = []
    np.random.seed(42)
    for i in range(n_perm):
        loc1 = np.random.uniform(1.5, 2.0)
        loc2 = np.random.uniform(1.5, 2.0)
        while abs(loc1 - loc2) < 0.05:
            loc2 = np.random.uniform(1.5, 2.0)
        daic = fit_corrugated_at_locations(loc1, loc2)
        random_daics.append(daic)
        if (i + 1) % 100 == 0:
            print(f"  Permutation {i+1}/{n_perm}: median ΔAIC = {np.median(random_daics):.2f}")

    random_daics = np.array(random_daics)
    percentile = np.mean(random_daics >= real_daic) * 100
    p_value = np.mean(random_daics <= real_daic)

    results = {
        'real_delta_aic': float(real_daic),
        'random_delta_aic_mean': float(np.mean(random_daics)),
        'random_delta_aic_median': float(np.median(random_daics)),
        'random_delta_aic_std': float(np.std(random_daics)),
        'percentile_rank': float(100 - percentile),
        'p_value': float(p_value),
        'n_permutations': n_perm,
        'interpretation': f"Real φ/e-1 placement achieves ΔAIC={real_daic:.2f}, outperforming {100-percentile:.1f}% of random placements (p={p_value:.4f})"
    }

    with open('outputs/permutation_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(random_daics, bins=50, density=True, color='lightgray', edgecolor='gray', alpha=0.8,
            label=f'Random placements (N={n_perm})')
    ax.axvline(real_daic, color='red', lw=2.5, ls='-', label=f'Real (φ, e−1): ΔAIC = {real_daic:.2f}')
    ax.axvline(np.median(random_daics), color='gray', lw=1.5, ls='--', alpha=0.7,
               label=f'Median random: {np.median(random_daics):.2f}')

    ax.text(real_daic, ax.get_ylim()[1] * 0.85,
            f'  p = {p_value:.4f}\n  Rank: {100-percentile:.1f}th percentile',
            fontsize=9, color='red', fontweight='bold')

    ax.set_xlabel('ΔAIC (corrugated − smooth vault)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Figure 11: Permutation Test — Perturbation Location Specificity', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig11_permutation_test.png')
    plt.close()
    print(f"  Saved fig11. p = {p_value:.4f}")
    return results


# ====================================================================
# JOB 3: Kuramoto parameter sweep
# ====================================================================
def job3_kuramoto_sweep():
    print("\n" + "="*60)
    print("JOB 3: Kuramoto parameter sweep")
    print("="*60)

    def run_kuramoto(K, N_osc, noise_sigma, dt=0.002, T_total=5.0, fs_out=256):
        np.random.seed(hash((K, N_osc, noise_sigma)) % 2**31)
        f_min, f_max = 2.0, 45.0
        freqs = f_min * (f_max / f_min) ** (np.arange(N_osc) / (N_osc - 1))
        omega = 2 * np.pi * freqs

        n_steps = int(T_total / dt)
        theta = np.random.uniform(0, 2 * np.pi, N_osc)

        t_out = np.arange(0, T_total, 1.0 / fs_out)
        signal = np.zeros(len(t_out))
        out_idx = 0

        for step in range(n_steps):
            t = step * dt
            sin_diff = np.sin(theta[:, None] - theta[None, :])
            coupling = K / N_osc * np.sum(sin_diff, axis=1)
            noise = noise_sigma * np.sqrt(dt) * np.random.randn(N_osc)
            theta += (omega - coupling) * dt + noise

            if out_idx < len(t_out) and t >= t_out[out_idx]:
                signal[out_idx] = np.mean(np.cos(theta))
                out_idx += 1

        from scipy.signal import welch
        freqs_psd, psd = welch(signal, fs=fs_out, nperseg=min(512, len(signal)))

        theta_mask = (freqs_psd >= 4) & (freqs_psd <= 8)
        alpha_mask = (freqs_psd >= 8) & (freqs_psd <= 13)

        if np.sum(psd[theta_mask]) < 1e-20 or np.sum(psd[alpha_mask]) < 1e-20:
            return None

        theta_centroid = np.sum(freqs_psd[theta_mask] * psd[theta_mask]) / np.sum(psd[theta_mask])
        alpha_centroid = np.sum(freqs_psd[alpha_mask] * psd[alpha_mask]) / np.sum(psd[alpha_mask])

        if theta_centroid < 1e-6:
            return None

        return alpha_centroid / theta_centroid

    K_values = np.linspace(0.1, 5.0, 10)
    N_values = [10, 20, 50, 100, 200]
    sigma_values = [0.01, 0.05, 0.1, 0.5, 1.0]

    results_grid = []
    total = len(K_values) * len(N_values) * len(sigma_values)
    count = 0

    for K in K_values:
        for N_osc in N_values:
            for sigma in sigma_values:
                count += 1
                ratio = run_kuramoto(K, N_osc, sigma)
                if ratio is not None:
                    pct_from_e1 = abs(ratio - E_MINUS_1) / E_MINUS_1 * 100
                    results_grid.append({
                        'K': float(K), 'N': N_osc, 'sigma': float(sigma),
                        'ratio': float(ratio), 'pct_from_e1': float(pct_from_e1)
                    })
                if count % 50 == 0:
                    print(f"  {count}/{total} parameter combinations done")

    print(f"  Valid results: {len(results_grid)}/{total}")

    valid_ratios = [r['ratio'] for r in results_grid]
    pct_within_5 = np.mean([abs(r - E_MINUS_1) / E_MINUS_1 < 0.05 for r in valid_ratios]) * 100
    pct_within_10 = np.mean([abs(r - E_MINUS_1) / E_MINUS_1 < 0.10 for r in valid_ratios]) * 100
    pct_within_20 = np.mean([abs(r - E_MINUS_1) / E_MINUS_1 < 0.20 for r in valid_ratios]) * 100

    K_corr = np.corrcoef([r['K'] for r in results_grid], [r['ratio'] for r in results_grid])[0, 1]
    N_corr = np.corrcoef([r['N'] for r in results_grid], [r['ratio'] for r in results_grid])[0, 1]
    sig_corr = np.corrcoef([r['sigma'] for r in results_grid], [r['ratio'] for r in results_grid])[0, 1]

    sensitivity = {
        'K_correlation': float(K_corr),
        'N_correlation': float(N_corr),
        'sigma_correlation': float(sig_corr),
        'most_sensitive': max([('K', abs(K_corr)), ('N', abs(N_corr)), ('sigma', abs(sig_corr))], key=lambda x: x[1])[0]
    }

    sweep_results = {
        'grid_size': f"{len(K_values)} x {len(N_values)} x {len(sigma_values)} = {total}",
        'valid_results': len(results_grid),
        'mean_ratio': float(np.mean(valid_ratios)),
        'std_ratio': float(np.std(valid_ratios)),
        'pct_within_5_of_e1': float(pct_within_5),
        'pct_within_10_of_e1': float(pct_within_10),
        'pct_within_20_of_e1': float(pct_within_20),
        'sensitivity': sensitivity,
        'grid_results': results_grid
    }

    with open('outputs/kuramoto_sweep_results.json', 'w') as f:
        json.dump(sweep_results, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ni, N_osc in enumerate([20, 50, 100]):
        ax = axes[ni]
        subset = [r for r in results_grid if r['N'] == N_osc]
        if not subset:
            ax.set_title(f'N={N_osc} (no data)')
            continue

        K_vals = sorted(set(r['K'] for r in subset))
        sig_vals = sorted(set(r['sigma'] for r in subset))
        heatmap = np.full((len(sig_vals), len(K_vals)), np.nan)

        for r in subset:
            ki = K_vals.index(r['K'])
            si = sig_vals.index(r['sigma'])
            heatmap[si, ki] = r['ratio']

        im = ax.imshow(heatmap, cmap='RdYlBu_r', aspect='auto',
                       vmin=1.5, vmax=2.5, origin='lower',
                       extent=[K_vals[0], K_vals[-1], 0, len(sig_vals)])
        ax.set_yticks(np.arange(len(sig_vals)) + 0.5)
        ax.set_yticklabels([f'{s:.2f}' for s in sig_vals])
        ax.set_xlabel('Coupling K')
        ax.set_ylabel('Noise σ')
        ax.set_title(f'N = {N_osc} oscillators')

        for r in subset:
            ki = K_vals.index(r['K'])
            si = sig_vals.index(r['sigma'])
            if abs(r['ratio'] - E_MINUS_1) / E_MINUS_1 < 0.10:
                ax.plot(r['K'], si + 0.5, 'k*', markersize=6)

    fig.suptitle(f'Figure 13: Kuramoto Parameter Sweep — α/θ Ratio\n'
                 f'{pct_within_10:.1f}% within 10% of e−1, {pct_within_20:.1f}% within 20%',
                 fontsize=13, fontweight='bold')
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, label='α/θ Ratio')
    cbar.ax.axhline(E_MINUS_1, color='steelblue', lw=2, ls='--')
    cbar.ax.axhline(PHI, color='goldenrod', lw=2, ls='--')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig13_kuramoto_sweep.png')
    plt.close()
    print(f"  Saved fig13. Mean ratio: {np.mean(valid_ratios):.3f}, {pct_within_10:.1f}% within 10% of e-1")
    return sweep_results


# ====================================================================
# JOB 4: Trapezoidal potential test
# ====================================================================
def job4_trapezoidal_test():
    print("\n" + "="*60)
    print("JOB 4: Trapezoidal potential test (flat bottom vs parabolic)")
    print("="*60)

    x_grid = np.linspace(1.0, 2.5, 2000)
    x0 = cf['x0']
    kl, kr = cf['k_left'], cf['k_right']

    def trapezoidal_potential(x, x_left, x_right, k_l, k_r, A_phi, A_e1, V_shelf, sigma=0.03):
        v = np.zeros_like(x)
        left_mask = x < x_left
        right_mask = x > x_right
        shelf_mask = (~left_mask) & (~right_mask)

        v[left_mask] = 0.5 * k_l * (x[left_mask] - x_left)**2 + V_shelf
        v[right_mask] = 0.5 * k_r * (x[right_mask] - x_right)**2 + V_shelf
        v[shelf_mask] = V_shelf

        v += A_phi * np.exp(-0.5 * ((x - PHI) / sigma)**2)
        v += A_e1 * np.exp(-0.5 * ((x - E_MINUS_1) / sigma)**2)
        return v

    def nll_parabolic(params):
        x0p, klp, krp, Ap, Ae, Tp = params
        if Tp <= 0 or klp <= 0 or krp <= 0:
            return 1e10
        V = vault_potential(x_grid, x0p, klp, krp)
        V += Ap * np.exp(-0.5 * ((x_grid - PHI) / 0.03)**2)
        V += Ae * np.exp(-0.5 * ((x_grid - E_MINUS_1) / 0.03)**2)
        log_p = -V / Tp
        log_p -= np.max(log_p)
        p = np.exp(log_p)
        p /= trapz_fn(p, x_grid)
        idx = np.searchsorted(x_grid, ratios).clip(0, len(x_grid) - 1)
        return -np.sum(np.log(p[idx] + 1e-30))

    def nll_trapezoidal(params):
        x_left, x_right, k_l, k_r, Ap, Ae, V_shelf, Tp = params
        if Tp <= 0 or k_l <= 0 or k_r <= 0 or x_right <= x_left:
            return 1e10
        V = trapezoidal_potential(x_grid, x_left, x_right, k_l, k_r, Ap, Ae, V_shelf)
        log_p = -V / Tp
        log_p -= np.max(log_p)
        p = np.exp(log_p)
        p /= trapz_fn(p, x_grid)
        idx = np.searchsorted(x_grid, ratios).clip(0, len(x_grid) - 1)
        return -np.sum(np.log(p[idx] + 1e-30))

    best_parab = minimize(nll_parabolic,
                          [x0, kl, kr, cf['A_phi'], cf['A_e1'], cf['T']],
                          method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-8})

    best_trap = None
    best_trap_nll = 1e10
    for xl_init in [1.72, 1.75, 1.78]:
        for xr_init in [1.82, 1.85, 1.88]:
            try:
                res = minimize(nll_trapezoidal,
                               [xl_init, xr_init, 60, 140, 0.8, -0.25, 0.0, 1.0],
                               method='Nelder-Mead', options={'maxiter': 15000, 'xatol': 1e-8})
                if res.fun < best_trap_nll:
                    best_trap_nll = res.fun
                    best_trap = res
            except:
                pass

    n_parab = 6
    n_trap = 8
    aic_parab = 2 * best_parab.fun + 2 * n_parab
    aic_trap = 2 * best_trap_nll + 2 * n_trap
    bic_parab = 2 * best_parab.fun + n_parab * np.log(len(ratios))
    bic_trap = 2 * best_trap_nll + n_trap * np.log(len(ratios))

    from scipy.stats import skewnorm
    sn_params = skewnorm.fit(ratios)
    sn_nll = -np.sum(skewnorm.logpdf(ratios, *sn_params))
    aic_sn = 2 * sn_nll + 2 * 3
    bic_sn = 2 * sn_nll + 3 * np.log(len(ratios))

    main_well = ratios[(ratios >= 1.75) & (ratios <= 1.85)]
    from scipy.stats import kstest
    if len(main_well) > 5:
        ks_stat, ks_p = kstest(main_well, 'uniform',
                                args=(1.75, 0.10))
    else:
        ks_stat, ks_p = np.nan, np.nan

    from scipy.ndimage import gaussian_filter1d
    kde_x = np.linspace(1.3, 2.3, 500)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(ratios, bw_method=0.05)
    kde_y = kde(kde_x)
    log_density = np.log(kde_y + 1e-30)
    dx = kde_x[1] - kde_x[0]
    d2 = np.gradient(np.gradient(log_density, dx), dx)
    well_mask = (kde_x >= 1.75) & (kde_x <= 1.85)
    mean_d2 = float(np.mean(d2[well_mask]))

    trap_params = best_trap.x if best_trap else [1.75, 1.85, 60, 140, 0.8, -0.25, 0.0, 1.0]

    trap_results = {
        'aic_parabolic_corrugated': float(aic_parab),
        'aic_trapezoidal_corrugated': float(aic_trap),
        'aic_skew_normal': float(aic_sn),
        'bic_parabolic_corrugated': float(bic_parab),
        'bic_trapezoidal_corrugated': float(bic_trap),
        'bic_skew_normal': float(bic_sn),
        'delta_aic_trap_vs_parab': float(aic_trap - aic_parab),
        'delta_bic_trap_vs_parab': float(bic_trap - bic_parab),
        'nll_parabolic': float(best_parab.fun),
        'nll_trapezoidal': float(best_trap_nll),
        'second_derivative_1.75_1.85': mean_d2,
        'ks_uniform_main_well': {'statistic': float(ks_stat), 'p_value': float(ks_p), 'n_subjects': int(len(main_well))},
        'shelf_bounds': [float(trap_params[0]), float(trap_params[1])],
        'trapezoidal_params': {
            'x_left': float(trap_params[0]),
            'x_right': float(trap_params[1]),
            'k_left': float(trap_params[2]),
            'k_right': float(trap_params[3]),
            'A_phi': float(trap_params[4]),
            'A_e1': float(trap_params[5]),
            'V_shelf': float(trap_params[6]),
            'T': float(trap_params[7]),
        },
        'interpretation': 'Trapezoidal wins' if aic_trap < aic_parab else 'Parabolic wins (or comparable)'
    }

    with open('outputs/trapezoidal_test_results.json', 'w') as f:
        json.dump(trap_results, f, indent=2)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    parab_params = best_parab.x
    V_parab = vault_potential(x_grid, parab_params[0], parab_params[1], parab_params[2])
    V_parab += parab_params[3] * np.exp(-0.5 * ((x_grid - PHI) / 0.03)**2)
    V_parab += parab_params[4] * np.exp(-0.5 * ((x_grid - E_MINUS_1) / 0.03)**2)

    V_trap = trapezoidal_potential(x_grid, *trap_params)

    ax = axes[0]
    ax.plot(x_grid, V_parab, 'b-', lw=2, label=f'Parabolic corrugated (AIC={aic_parab:.1f})')
    ax.plot(x_grid, V_trap, 'r-', lw=2, label=f'Trapezoidal corrugated (AIC={aic_trap:.1f})')
    ax.axvline(PHI, color='goldenrod', ls='--', lw=1, alpha=0.6)
    ax.axvline(E_MINUS_1, color='steelblue', ls='--', lw=1, alpha=0.6)
    ax.axvline(2.0, color='red', ls='--', lw=1, alpha=0.4)
    ax.set_ylabel('Potential V(x)')
    ax.set_xlim(1.3, 2.2)
    ax.set_ylim(bottom=-1)
    ax.legend(fontsize=9)
    ax.set_title('Full Potential Comparison')

    delta_str = f'ΔAIC(trap−parab) = {aic_trap - aic_parab:.1f}'
    winner = 'Trapezoidal' if aic_trap < aic_parab else 'Parabolic'
    ax.text(0.02, 0.95, f'{delta_str}\n{winner} preferred',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2 = axes[1]
    zoom_mask = (x_grid >= 1.70) & (x_grid <= 1.90)
    ax2.plot(x_grid[zoom_mask], V_parab[zoom_mask], 'b-', lw=2, label='Parabolic')
    ax2.plot(x_grid[zoom_mask], V_trap[zoom_mask], 'r-', lw=2, label='Trapezoidal')

    ax2_twin = ax2.twinx()
    ax2_twin.hist(ratios[(ratios >= 1.70) & (ratios <= 1.90)], bins=20, density=True,
                  color='lightgray', alpha=0.5, edgecolor='gray')
    ax2_twin.set_ylabel('Data density', color='gray')

    ax2.set_xlabel('α/θ Centroid Ratio')
    ax2.set_ylabel('Potential V(x)')
    ax2.set_title(f'Zoom: Well Bottom [1.70, 1.90]\nd²(log density)/dx² = {mean_d2:.2f}, '
                  f'KS uniform p = {ks_p:.3f}')
    ax2.legend(fontsize=9)

    if float(trap_params[0]) > 1.70 and float(trap_params[1]) < 1.90:
        ax2.axvspan(float(trap_params[0]), float(trap_params[1]),
                    alpha=0.15, color='red', label='Fitted shelf')

    fig.suptitle('Figure 14: Trapezoidal vs. Parabolic Potential Well', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig14_trapezoidal.png')
    plt.close()
    print(f"  Saved fig14. ΔAIC(trap-parab) = {aic_trap - aic_parab:.1f}, d² = {mean_d2:.2f}")
    return trap_results


# ====================================================================
# JOB 2: Downsampling experiment on OpenNeuro ds003969
# ====================================================================
def job2_downsampling():
    print("\n" + "="*60)
    print("JOB 2: Downsampling experiment on OpenNeuro 1024 Hz data")
    print("="*60)

    import mne
    mne.set_log_level('ERROR')
    from scipy.signal import welch, resample

    data_dir = 'ds003969'
    sub_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    print(f"  Found {len(sub_dirs)} subjects in ds003969")

    target_fs = [1024, 512, 256, 160]

    def compute_ratio_from_raw(raw_data, sfreq):
        freqs, psd = welch(raw_data, fs=sfreq, nperseg=min(int(2 * sfreq), len(raw_data)))
        psd_mean = psd.mean(axis=0) if psd.ndim > 1 else psd

        theta_mask = (freqs >= 4) & (freqs <= 8)
        alpha_mask = (freqs >= 8) & (freqs <= 13)

        if np.sum(psd_mean[theta_mask]) < 1e-20 or np.sum(psd_mean[alpha_mask]) < 1e-20:
            return None

        theta_centroid = np.sum(freqs[theta_mask] * psd_mean[theta_mask]) / np.sum(psd_mean[theta_mask])
        alpha_centroid = np.sum(freqs[alpha_mask] * psd_mean[alpha_mask]) / np.sum(psd_mean[alpha_mask])

        if theta_centroid < 1e-6:
            return None
        return alpha_centroid / theta_centroid

    all_results = []
    processed = 0

    for sub_dir in sub_dirs:
        eeg_dir = os.path.join(data_dir, sub_dir, 'eeg')
        if not os.path.exists(eeg_dir):
            continue

        bdf_files = [f for f in os.listdir(eeg_dir) if f.endswith('.bdf') and 'think1' in f]
        if not bdf_files:
            bdf_files = [f for f in os.listdir(eeg_dir) if f.endswith('.bdf')]
        if not bdf_files:
            continue

        bdf_path = os.path.join(eeg_dir, bdf_files[0])
        try:
            raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
            native_fs = raw.info['sfreq']

            eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
            if len(eeg_picks) == 0:
                continue
            data = raw.get_data(picks=eeg_picks)

            subject_ratios = {}

            for fs_target in target_fs:
                if fs_target > native_fs:
                    subject_ratios[fs_target] = None
                    continue

                if fs_target == native_fs:
                    data_fs = data
                else:
                    n_samples_new = int(data.shape[1] * fs_target / native_fs)
                    data_fs = resample(data, n_samples_new, axis=1)

                ratio = compute_ratio_from_raw(data_fs, fs_target)
                subject_ratios[fs_target] = ratio

            all_results.append({
                'subject': sub_dir,
                'native_fs': float(native_fs),
                'ratios': {str(k): v for k, v in subject_ratios.items()}
            })
            processed += 1
            if processed % 10 == 0:
                print(f"  Processed {processed} subjects")

        except Exception as e:
            continue

        if processed >= 50:
            break

    print(f"  Total processed: {processed} subjects")

    mean_by_fs = {}
    std_by_fs = {}
    for fs in target_fs:
        vals = [r['ratios'][str(fs)] for r in all_results if r['ratios'].get(str(fs)) is not None]
        if vals:
            mean_by_fs[fs] = float(np.mean(vals))
            std_by_fs[fs] = float(np.std(vals))

    paired_data = {}
    for fs in target_fs:
        paired_data[fs] = []
    for r in all_results:
        if all(r['ratios'].get(str(fs)) is not None for fs in target_fs if fs <= r['native_fs']):
            for fs in target_fs:
                if r['ratios'].get(str(fs)) is not None:
                    paired_data[fs].append(r['ratios'][str(fs)])

    available_fs = [fs for fs in target_fs if len(paired_data[fs]) > 5]
    min_len = min(len(paired_data[fs]) for fs in available_fs) if available_fs else 0

    if len(available_fs) >= 2 and min_len >= 5:
        trimmed = {fs: paired_data[fs][:min_len] for fs in available_fs}
        from scipy.stats import friedmanchisquare
        try:
            if len(available_fs) >= 3:
                stat, rm_p = friedmanchisquare(*[trimmed[fs] for fs in available_fs])
            else:
                stat, rm_p = stats.wilcoxon(trimmed[available_fs[0]], trimmed[available_fs[1]])
                stat = float(stat)
            rm_p = float(rm_p)
        except:
            stat, rm_p = np.nan, np.nan
    else:
        stat, rm_p = np.nan, np.nan

    within_drifts = []
    ref_fs = max(available_fs) if available_fs else 1024
    low_fs = min(available_fs) if available_fs else 160
    for r in all_results:
        r_high = r['ratios'].get(str(ref_fs))
        r_low = r['ratios'].get(str(low_fs))
        if r_high is not None and r_low is not None:
            within_drifts.append(r_low - r_high)

    mean_drift = float(np.mean(within_drifts)) if within_drifts else 0
    is_artifact = abs(mean_drift) > 0.02 and rm_p < 0.05

    ds_results = {
        'n_subjects': processed,
        'mean_ratio_by_fs': {str(k): v for k, v in mean_by_fs.items()},
        'std_ratio_by_fs': {str(k): v for k, v in std_by_fs.items()},
        'within_subject_drift': float(mean_drift),
        'rm_anova_p': rm_p,
        'rm_anova_statistic': float(stat) if not np.isnan(stat) else None,
        'conclusion': 'methodological artifact' if is_artifact else 'biological difference',
        'available_sampling_rates': available_fs,
        'n_paired': min_len
    }

    with open('outputs/downsampling_results.json', 'w') as f:
        json.dump(ds_results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    if available_fs:
        for r in all_results[:50]:
            fs_vals = []
            ratio_vals = []
            for fs in sorted(available_fs):
                v = r['ratios'].get(str(fs))
                if v is not None:
                    fs_vals.append(fs)
                    ratio_vals.append(v)
            if len(fs_vals) >= 2:
                ax.plot(fs_vals, ratio_vals, '-o', color='lightsteelblue', alpha=0.3, markersize=3)

        means = [mean_by_fs.get(fs, np.nan) for fs in sorted(available_fs)]
        ax.plot(sorted(available_fs), means, 'ko-', lw=2, markersize=8, label='Group mean')
        ax.axhline(E_MINUS_1, color='steelblue', ls='--', lw=1.5, alpha=0.7, label=f'e−1 = {E_MINUS_1:.3f}')
        ax.set_xlabel('Sampling Rate (Hz)')
        ax.set_ylabel('α/θ Centroid Ratio')
        ax.set_title(f'Within-Subject Drift (N={min_len} paired)')
        ax.legend(fontsize=9)

    ax2 = axes[1]
    if within_drifts:
        ax2.hist(within_drifts, bins=25, color='steelblue', alpha=0.7, edgecolor='white')
        ax2.axvline(0, color='black', ls='-', lw=1)
        ax2.axvline(mean_drift, color='red', ls='--', lw=2, label=f'Mean drift = {mean_drift:.4f}')
        ax2.set_xlabel(f'Ratio drift ({low_fs}→{ref_fs} Hz)')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Distribution of within-subject drift\np = {rm_p:.4f}')
        ax2.legend(fontsize=9)

    conclusion = ds_results['conclusion'].upper()
    fig.suptitle(f'Figure 12: Downsampling Experiment (OpenNeuro ds003969)\n'
                 f'Conclusion: {conclusion}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig12_downsampling.png')
    plt.close()
    print(f"  Saved fig12. Mean drift: {mean_drift:.4f}, p = {rm_p}")
    return ds_results


# ====================================================================
# MAIN
# ====================================================================
if __name__ == '__main__':
    print("Running additional analyses for v3.2\n")

    r1 = job1_permutation_test()
    print(f"\nJob 1 complete: p = {r1['p_value']:.4f}")

    r4 = job4_trapezoidal_test()
    print(f"\nJob 4 complete: ΔAIC = {r4['delta_aic_trap_vs_parab']:.1f}")

    r3 = job3_kuramoto_sweep()
    print(f"\nJob 3 complete: {r3['pct_within_10_of_e1']:.1f}% within 10% of e-1")

    r2 = job2_downsampling()
    print(f"\nJob 2 complete: conclusion = {r2['conclusion']}")

    print("\n" + "="*60)
    print("ALL JOBS COMPLETE")
    print("="*60)
    print(f"  Job 1: Permutation test p = {r1['p_value']:.4f}")
    print(f"  Job 2: Downsampling: {r2['conclusion']}")
    print(f"  Job 3: Kuramoto: {r3['pct_within_10_of_e1']:.1f}% within 10%")
    print(f"  Job 4: Trapezoidal ΔAIC = {r4['delta_aic_trap_vs_parab']:.1f}")
    print(f"\nFigures saved: fig11-fig14 in {OUTPUT_DIR}/")
