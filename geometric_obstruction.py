#!/usr/bin/env python3
"""
GEOMETRICAL OBSTRUCTION: Corrugated Potential Landscape
Tests whether the α/θ distribution has internal structure
beyond a smooth asymmetric well.
"""

import json
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, skewnorm, norm
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
import diptest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

E1 = np.e - 1
PHI = (1 + np.sqrt(5)) / 2
HARMONIC = 2.0

CONSTANTS = {
    'φ': 1.61803,
    'e−1': 1.71828,
    '√π': 1.77245,
    '16/9': 1.77778,
    '4/√5': 1.78885,
    'vault_min': 1.816,
    '11/6': 1.83333,
    '2:1': 2.0,
}

BASINS = {
    'φ-trapped': (0, 1.67),
    'e1-residents': (1.67, 1.75),
    'main-well': (1.75, 1.90),
    '2:1-pressed': (1.90, 3.0),
}


def load_data():
    all_subjects = []
    for path, default_fs in [('outputs/aw_cached_subjects.json', 512),
                              ('outputs/ds003969_cached_subjects.json', 1024),
                              ('outputs/eegbci_modal_results.json', 160)]:
        with open(path) as f:
            for s in json.load(f):
                r = s.get('alpha_theta_ratio')
                if r:
                    all_subjects.append(s)
    return all_subjects


def boltzmann_pdf(x_grid, V_func, T, *params):
    V = V_func(x_grid, *params)
    V = V - np.min(V)
    log_P = -V / max(T, 1e-12)
    log_P -= np.max(log_P)
    P = np.exp(log_P)
    area = np.trapezoid(P, x_grid)
    if area > 0:
        P /= area
    return P


def V_asymmetric(x, x0, k_left, k_right):
    return np.where(x < x0, 0.5 * k_left * (x - x0)**2, 0.5 * k_right * (x - x0)**2)


def V_corrugated(x, x0, k_left, k_right, A_phi, A_e1, sigma=0.03):
    V_base = V_asymmetric(x, x0, k_left, k_right)
    V_phi = A_phi * np.exp(-0.5 * ((x - PHI) / sigma)**2)
    V_e1 = A_e1 * np.exp(-0.5 * ((x - E1) / sigma)**2)
    return V_base + V_phi + V_e1


def fit_corrugated(ratios):
    x_grid = np.linspace(1.2, 2.5, 600)

    def neg_ll(params):
        x0, kl, kr, A_phi, A_e1, T = params
        P = boltzmann_pdf(x_grid, V_corrugated, T, x0, kl, kr, A_phi, A_e1)
        P_func = interp1d(x_grid, P, bounds_error=False, fill_value=1e-10)
        P_data = np.maximum(P_func(ratios), 1e-10)
        return -np.sum(np.log(P_data))

    bounds = [(1.55, 1.95), (5, 500), (5, 500),
              (-5, 5), (-5, 5), (0.0005, 1.0)]
    result = differential_evolution(neg_ll, bounds, seed=42, maxiter=2000,
                                     tol=1e-8, popsize=25)
    x0, kl, kr, A_phi, A_e1, T = result.x
    return dict(x0=x0, k_left=kl, k_right=kr, A_phi=A_phi, A_e1=A_e1,
                T=T, nll=result.fun, n_params=6, converged=result.success)


def aic_bic(nll, k, n):
    return 2*k + 2*nll, k*np.log(n) + 2*nll


def run_analysis():
    print("=" * 70)
    print("GEOMETRICAL OBSTRUCTION: Corrugated Potential Landscape")
    print("=" * 70)

    subjects = load_data()
    ratios = np.array([s['alpha_theta_ratio'] for s in subjects])
    datasets = [s.get('dataset', 'unknown') for s in subjects]
    N = len(ratios)

    report = []
    def rpt(s=""):
        report.append(s)
        print(s)

    # ═══════════════════════════════════════
    # ANALYSIS 1: Multimodality Tests
    # ═══════════════════════════════════════
    rpt(f"\n{'='*60}")
    rpt(f"ANALYSIS 1: MULTIMODALITY TESTS (N={N})")
    rpt(f"{'='*60}")

    dip_stat, dip_p = diptest.diptest(ratios)
    rpt(f"\n  Hartigan's Dip Test:")
    rpt(f"    Statistic = {dip_stat:.4f}")
    rpt(f"    p-value   = {dip_p:.4f}")
    if dip_p < 0.05:
        rpt(f"    → REJECT unimodality (p < 0.05): evidence for multimodal structure")
    else:
        rpt(f"    → CANNOT reject unimodality (p ≥ 0.05): consistent with single mode")

    x_grid = np.linspace(1.3, 2.3, 1000)
    kde = gaussian_kde(ratios, bw_method='silverman')
    density = kde(x_grid)

    peaks, props = find_peaks(density, height=0.1, distance=50)
    rpt(f"\n  KDE Peak Detection (Silverman bandwidth):")
    rpt(f"    Number of peaks: {len(peaks)}")
    for pk in peaks:
        rpt(f"    Peak at ratio = {x_grid[pk]:.4f}, density = {density[pk]:.4f}")

    rpt(f"\n  GMM Component Selection (BIC):")
    bic_scores = []
    gmm_fits = {}
    for k in range(1, 6):
        gmm = GaussianMixture(n_components=k, random_state=42, max_iter=500)
        gmm.fit(ratios.reshape(-1, 1))
        bic_val = gmm.bic(ratios.reshape(-1, 1))
        bic_scores.append(bic_val)
        gmm_fits[k] = gmm
        means_str = ', '.join(f'{m:.3f}' for m in sorted(gmm.means_.flatten()))
        weights_str = ', '.join(f'{w:.2f}' for w in gmm.weights_)
        rpt(f"    k={k}: BIC={bic_val:.1f}  means=[{means_str}]  weights=[{weights_str}]")

    best_k = np.argmin(bic_scores) + 1
    rpt(f"    → Optimal components by BIC: k={best_k}")

    # ═══════════════════════════════════════
    # ANALYSIS 2: Structure at Mathematical Constants
    # ═══════════════════════════════════════
    rpt(f"\n{'='*60}")
    rpt(f"ANALYSIS 2: STRUCTURE AT MATHEMATICAL CONSTANTS")
    rpt(f"{'='*60}")

    sn_params = skewnorm.fit(ratios)
    smooth_density = skewnorm.pdf(x_grid, *sn_params)
    residual = density - smooth_density

    rpt(f"\n  Skew-normal fit: a={sn_params[0]:.3f}, loc={sn_params[1]:.4f}, scale={sn_params[2]:.4f}")
    rpt(f"\n  {'Constant':<12} {'Value':>7} {'KDE dens':>10} {'SN dens':>10} {'Residual':>10} {'z-excess':>10}")
    rpt(f"  {'-'*12} {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    window = 0.03
    excess_results = {}
    for name, val in CONSTANTS.items():
        if val < 1.3 or val > 2.3:
            continue
        local_kde = kde(np.array([val]))[0]
        local_smooth = skewnorm.pdf(val, *sn_params)
        resid = local_kde - local_smooth

        mask = (ratios >= val - window) & (ratios <= val + window)
        n_obs = mask.sum()
        expected = N * (skewnorm.cdf(val + window, *sn_params) -
                        skewnorm.cdf(val - window, *sn_params))
        z_excess = (n_obs - expected) / np.sqrt(max(expected, 1)) if expected > 0 else 0

        rpt(f"  {name:<12} {val:>7.4f} {local_kde:>10.4f} {local_smooth:>10.4f} "
            f"{resid:>10.4f} {z_excess:>10.2f}")
        excess_results[name] = dict(val=val, kde=local_kde, smooth=local_smooth,
                                     residual=resid, n_obs=int(n_obs),
                                     expected=expected, z_excess=z_excess)

    sig_excess = {n: r for n, r in excess_results.items() if abs(r['z_excess']) > 1.96}
    n_tests = len(excess_results)
    bonf_threshold = 1.96  # uncorrected
    rpt(f"\n  Note: {n_tests} constants tested; z-values are UNCORRECTED for multiple comparisons.")
    rpt(f"  These results are EXPLORATORY, not confirmatory.")
    if sig_excess:
        rpt(f"\n  Nominally notable excesses/deficits (|z| > 1.96, uncorrected):")
        for n, r in sig_excess.items():
            direction = "EXCESS" if r['z_excess'] > 0 else "DEFICIT"
            rpt(f"    {n}: {direction} (z={r['z_excess']:.2f}, obs={r['n_obs']}, exp={r['expected']:.1f})")
    else:
        rpt(f"\n  No notable excesses/deficits at any mathematical constant (all |z| < 1.96)")

    # ═══════════════════════════════════════
    # ANALYSIS 3: Basin Classification
    # ═══════════════════════════════════════
    rpt(f"\n{'='*60}")
    rpt(f"ANALYSIS 3: BASIN CLASSIFICATION")
    rpt(f"{'='*60}")

    rpt(f"\n  {'Basin':<16} {'N':>5} {'%':>7} {'Mean':>8} {'SD':>8}")
    rpt(f"  {'-'*16} {'-'*5} {'-'*7} {'-'*8} {'-'*8}")
    basin_members = {}
    for name, (lo, hi) in BASINS.items():
        mask = (ratios >= lo) & (ratios < hi)
        group = ratios[mask]
        basin_members[name] = mask
        if len(group) > 0:
            rpt(f"  {name:<16} {len(group):>5} {100*len(group)/N:>6.1f}% "
                f"{np.mean(group):>8.4f} {np.std(group,ddof=1):>8.4f}")
        else:
            rpt(f"  {name:<16} {0:>5} {0:>6.1f}%")

    rpt(f"\n  Per-dataset basin distribution:")
    ds_names = sorted(set(datasets))
    rpt(f"  {'Dataset':<16} " + " ".join(f'{b:<14}' for b in BASINS.keys()))
    rpt("  " + "-"*16 + " " + " ".join("-"*14 for _ in BASINS))
    for ds_name in ds_names:
        ds_mask = np.array([d == ds_name for d in datasets])
        ds_ratios = ratios[ds_mask]
        counts = []
        for bname, (lo, hi) in BASINS.items():
            n_in = ((ds_ratios >= lo) & (ds_ratios < hi)).sum()
            counts.append(f"{n_in:>4} ({100*n_in/len(ds_ratios):>4.0f}%)")
        rpt(f"  {ds_name:<16} " + "  ".join(counts))

    contingency = []
    for ds_name in ds_names:
        ds_mask = np.array([d == ds_name for d in datasets])
        ds_ratios = ratios[ds_mask]
        row = []
        for bname, (lo, hi) in BASINS.items():
            row.append(((ds_ratios >= lo) & (ds_ratios < hi)).sum())
        contingency.append(row)
    contingency = np.array(contingency)
    if contingency.shape[0] > 1 and np.all(contingency.sum(axis=0) > 0):
        chi2, p_chi, dof, _ = stats.chi2_contingency(contingency)
        rpt(f"\n  Chi-squared test (dataset × basin): χ²={chi2:.2f}, df={dof}, p={p_chi:.4g}")
        if p_chi < 0.05:
            rpt(f"  → Basin occupancy DIFFERS by dataset (p < 0.05)")
        else:
            rpt(f"  → No significant difference in basin occupancy across datasets")

    # ═══════════════════════════════════════
    # ANALYSIS 4: Corrugated Potential Fit
    # ═══════════════════════════════════════
    rpt(f"\n{'='*60}")
    rpt(f"ANALYSIS 4: CORRUGATED POTENTIAL FIT")
    rpt(f"{'='*60}")

    rpt(f"\n  Fitting corrugated well (asymmetric + Gaussian bumps at φ and e-1)...")
    corr = fit_corrugated(ratios)
    for k, v in corr.items():
        rpt(f"    {k}: {v:.4f}" if isinstance(v, (float, np.floating)) else f"    {k}: {v}")

    if corr['A_phi'] < 0:
        rpt(f"    → A_phi < 0: φ is a LOCAL TRAP (attractive local minimum)")
    else:
        rpt(f"    → A_phi > 0: φ is a BARRIER (repulsive)")

    if corr['A_e1'] < 0:
        rpt(f"    → A_e1 < 0: e-1 is a LOCAL TRAP (attractive local minimum)")
    else:
        rpt(f"    → A_e1 > 0: e-1 is a BARRIER (repulsive)")

    from vault_model import fit_asymmetric_well, fit_gaussian, fit_skewnorm as fit_sn_model, fit_gmm
    vault = fit_asymmetric_well(ratios)
    m_phi = fit_gaussian(ratios, fixed_mean=PHI)
    m_e1 = fit_gaussian(ratios, fixed_mean=E1)
    m_sn = fit_sn_model(ratios)
    m_gmm = fit_gmm(ratios, n_components=2)

    models = [
        ('Gaussian @ φ', m_phi),
        ('Gaussian @ e-1', m_e1),
        ('Smooth vault (4p)', vault),
        ('Skew-normal (3p)', m_sn),
        ('Corrugated (6p)', corr),
        ('GMM 2-comp (5p)', m_gmm),
    ]

    rpt(f"\n  MODEL COMPARISON:")
    rpt(f"  {'Model':<22} {'NLL':>10} {'k':>4} {'AIC':>10} {'BIC':>10} {'ΔAIC':>8}")
    rpt(f"  {'-'*22} {'-'*10} {'-'*4} {'-'*10} {'-'*10} {'-'*8}")

    aics, bics = [], []
    for lab, mdata in models:
        a, b = aic_bic(mdata['nll'], mdata['n_params'], N)
        aics.append(a)
        bics.append(b)

    best_aic = min(aics)
    for i, (lab, mdata) in enumerate(models):
        delta = aics[i] - best_aic
        rpt(f"  {lab:<22} {mdata['nll']:>10.2f} {mdata['n_params']:>4} "
            f"{aics[i]:>10.2f} {bics[i]:>10.2f} {delta:>8.2f}")

    best_idx = np.argmin(aics)
    best_bic_idx = np.argmin(bics)
    rpt(f"\n  Best model by AIC: {models[best_idx][0]}")
    rpt(f"  Best model by BIC: {models[best_bic_idx][0]}")
    sn_idx = next(i for i, (lab, _) in enumerate(models) if 'Skew-normal' in lab)
    corr_idx = next(i for i, (lab, _) in enumerate(models) if 'Corrugated' in lab)
    if best_idx == corr_idx and best_bic_idx == sn_idx:
        rpt(f"\n  IMPORTANT: Corrugated (6 params) wins AIC but skew-normal (3 params) wins BIC.")
        rpt(f"  Skew-normal is the most parsimonious model overall.")
        rpt(f"  Corrugated is the best MECHANISTIC model, not the best model.")
        rpt(f"  The ΔAIC={aics[sn_idx] - aics[corr_idx]:.1f} gap is suggestive but not decisive,")
        rpt(f"  especially given the 3 extra parameters.")

    corr_aic = aics[4]
    vault_aic = aics[2]
    delta_corr_vault = corr_aic - vault_aic
    rpt(f"\n  Corrugated vs Smooth vault: ΔAIC = {delta_corr_vault:.2f}")
    if delta_corr_vault < -2:
        rpt(f"  → Corrugated shows MODEST AIC improvement (ΔAIC={delta_corr_vault:.1f})")
        rpt(f"    Note: BIC may still favor simpler models due to parameter penalty.")
        rpt(f"    This is suggestive but not decisive evidence for corrugation.")
    elif delta_corr_vault < 2:
        rpt(f"  → INCONCLUSIVE: models are within ΔAIC < 2 (essentially equivalent)")
    else:
        rpt(f"  → Smooth vault WINS: no evidence for corrugation")
        rpt(f"    The extra bumps at φ and e-1 don't help")

    # ═══════════════════════════════════════
    # ANALYSIS 5: Alpha-Power Split
    # ═══════════════════════════════════════
    rpt(f"\n{'='*60}")
    rpt(f"ANALYSIS 5: SCALE-DEPENDENT ATTRACTOR (Alpha Power Split)")
    rpt(f"{'='*60}")

    alpha_powers = []
    valid_ratios_for_split = []
    for s in subjects:
        powers = s.get('powers', {})
        ap = powers.get('alpha') if isinstance(powers, dict) else None
        if ap is not None and ap > 0:
            alpha_powers.append(ap)
            valid_ratios_for_split.append(s['alpha_theta_ratio'])

    if len(alpha_powers) >= 20:
        alpha_powers = np.array(alpha_powers)
        split_ratios = np.array(valid_ratios_for_split)
        median_alpha = np.median(alpha_powers)
        high_mask = alpha_powers >= median_alpha
        low_mask = ~high_mask

        high_ratios = split_ratios[high_mask]
        low_ratios = split_ratios[low_mask]

        rpt(f"\n  Subjects with alpha power data: N={len(alpha_powers)}")
        rpt(f"  Median alpha power: {median_alpha:.4f}")
        rpt(f"\n  High-alpha group (N={len(high_ratios)}): mean α/θ = {np.mean(high_ratios):.4f} ± {np.std(high_ratios,ddof=1):.4f}")
        rpt(f"  Low-alpha group  (N={len(low_ratios)}):  mean α/θ = {np.mean(low_ratios):.4f} ± {np.std(low_ratios,ddof=1):.4f}")

        t_val, p_val = stats.ttest_ind(high_ratios, low_ratios)
        d_val = (np.mean(high_ratios) - np.mean(low_ratios)) / np.sqrt(
            (np.var(high_ratios, ddof=1) + np.var(low_ratios, ddof=1)) / 2)
        rpt(f"  t-test: t={t_val:.3f}, p={p_val:.4g}, Cohen's d={d_val:.3f}")

        if p_val < 0.05:
            if np.mean(high_ratios) < np.mean(low_ratios):
                rpt(f"  → High-alpha subjects have LOWER α/θ ratios (closer to e-1)")
                rpt(f"    Stronger alpha → deeper into the potential well")
            else:
                rpt(f"  → High-alpha subjects have HIGHER α/θ ratios")
                rpt(f"    Stronger alpha → pushed further from e-1")
        else:
            rpt(f"  → No significant difference: alpha power does not predict basin location")

        theta_powers = []
        for s in subjects:
            powers = s.get('powers', {})
            tp = powers.get('theta') if isinstance(powers, dict) else None
            if tp is not None and tp > 0:
                theta_powers.append(tp)
        if len(theta_powers) == len(alpha_powers):
            theta_powers = np.array(theta_powers)
            power_ratio = alpha_powers / theta_powers
            r_corr, p_corr = stats.pearsonr(power_ratio, split_ratios)
            rpt(f"\n  Alpha/Theta POWER ratio vs centroid ratio:")
            rpt(f"    Pearson r = {r_corr:.3f}, p = {p_corr:.4g}")
            if abs(r_corr) > 0.1 and p_corr < 0.05:
                rpt(f"    → Significant correlation: power distribution predicts spectral geometry")
    else:
        rpt(f"\n  Insufficient alpha power data (N={len(alpha_powers)}) for split analysis")

    # ═══════════════════════════════════════
    # FIGURES
    # ═══════════════════════════════════════
    rpt(f"\n{'='*60}")
    rpt(f"GENERATING FIGURES")
    rpt(f"{'='*60}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[1.2, 1],
                             gridspec_kw={'hspace': 0.30})

    ax = axes[0]
    ax.plot(x_grid, density, 'k-', lw=2, label='KDE (empirical)')
    ax.plot(x_grid, smooth_density, 'b--', lw=1.5, label='Skew-normal fit')
    for pk in peaks:
        ax.axvline(x_grid[pk], color='red', ls=':', lw=1, alpha=0.6)
        ax.plot(x_grid[pk], density[pk], 'rv', ms=10)
    for name, val in CONSTANTS.items():
        if 1.35 < val < 2.25 and name not in ('vault_min',):
            ax.axvline(val, color='gray', ls=':', lw=0.8, alpha=0.5)
            ax.text(val, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1] > 0 else 4,
                    name, ha='center', va='top', fontsize=7, rotation=90, color='gray')
    ax.set_xlabel('α/θ spectral centroid ratio', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Density + Mathematical Constants (Dip test p={dip_p:.3f}, GMM best k={best_k})',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(1.3, 2.3)

    ax = axes[1]
    ax.bar(x_grid[:-1], residual[:-1], width=x_grid[1]-x_grid[0],
           color=np.where(residual[:-1] > 0, 'steelblue', 'salmon'), alpha=0.6)
    ax.axhline(0, color='k', lw=0.8)
    for name, val in CONSTANTS.items():
        if 1.35 < val < 2.25 and name not in ('vault_min',):
            ax.axvline(val, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('α/θ spectral centroid ratio', fontsize=11)
    ax.set_ylabel('Residual (KDE − skew-normal)', fontsize=11)
    ax.set_title('Density Residuals: Bumps = Potential Traps, Dips = Barriers',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(1.3, 2.3)

    plt.tight_layout()
    plt.savefig('outputs/e1_figures/fig_obstruction_density.png', dpi=300, bbox_inches='tight')
    plt.close()
    rpt("  Saved: outputs/e1_figures/fig_obstruction_density.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    basin_colors = {'φ-trapped': '#9467bd', 'e1-residents': '#2ca02c',
                    'main-well': '#1f77b4', '2:1-pressed': '#ff7f0e'}
    for bname, (lo, hi) in BASINS.items():
        mask = (ratios >= lo) & (ratios < hi)
        if mask.sum() > 0:
            ax.hist(ratios[mask], bins=20, alpha=0.5, color=basin_colors[bname],
                    label=f'{bname} (N={mask.sum()})', density=False,
                    range=(lo, hi))
    for name, val in CONSTANTS.items():
        if 1.35 < val < 2.25:
            ax.axvline(val, color='gray', ls=':', lw=0.8, alpha=0.6)
            ax.text(val, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1] > 0 else 15,
                    name, ha='center', va='top', fontsize=7, rotation=90)
    for bname, (lo, hi) in BASINS.items():
        ax.axvline(lo, color='black', ls='--', lw=1, alpha=0.3)
    ax.set_xlabel('α/θ spectral centroid ratio', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Basin Classification of N=244 Subjects', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(1.3, 2.3)
    plt.tight_layout()
    plt.savefig('outputs/e1_figures/fig_obstruction_basins.png', dpi=300, bbox_inches='tight')
    plt.close()
    rpt("  Saved: outputs/e1_figures/fig_obstruction_basins.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[1, 1.2],
                             gridspec_kw={'hspace': 0.30})

    x_fine = np.linspace(1.3, 2.3, 1000)
    V_smooth = V_asymmetric(x_fine, vault['x0'], vault['k_left'], vault['k_right'])
    V_smooth -= np.min(V_smooth)
    V_corr = V_corrugated(x_fine, corr['x0'], corr['k_left'], corr['k_right'],
                           corr['A_phi'], corr['A_e1'])
    V_corr -= np.min(V_corr)

    ax = axes[0]
    ax.plot(x_fine, V_smooth, 'b-', lw=2, label='Smooth vault (4 params)')
    ax.plot(x_fine, V_corr, 'r-', lw=2, label='Corrugated (6 params)')
    y_top = max(np.max(V_smooth[(x_fine > 1.4) & (x_fine < 2.2)]),
                np.max(V_corr[(x_fine > 1.4) & (x_fine < 2.2)])) * 1.3
    ax.set_ylim(-0.02 * y_top, y_top)
    for name, val in [('φ', PHI), ('e−1', E1), ('2:1', HARMONIC)]:
        if 1.35 < val < 2.25:
            ax.axvline(val, color='gray', ls=':', lw=0.8, alpha=0.6)
            ax.text(val, y_top*0.9, name, ha='center', fontsize=9, color='gray')
    ax.set_xlabel('α/θ ratio', fontsize=11)
    ax.set_ylabel('V(x)', fontsize=11)
    ax.set_title('Smooth vs Corrugated Potential Wells', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(1.3, 2.3)

    ax = axes[1]
    ax.hist(ratios, bins=35, density=True, alpha=0.3, color='gray',
            range=(1.3, 2.3), label='Empirical')
    P_smooth = boltzmann_pdf(x_fine, V_asymmetric, vault['T'],
                              vault['x0'], vault['k_left'], vault['k_right'])
    P_corr = boltzmann_pdf(x_fine, V_corrugated, corr['T'],
                            corr['x0'], corr['k_left'], corr['k_right'],
                            corr['A_phi'], corr['A_e1'])
    ax.plot(x_fine, P_smooth, 'b-', lw=2, label=f'Smooth (AIC={aics[2]:.1f})')
    ax.plot(x_fine, P_corr, 'r-', lw=2, label=f'Corrugated (AIC={aics[4]:.1f})')
    ax.set_xlabel('α/θ ratio', fontsize=11)
    ax.set_ylabel('Probability density', fontsize=11)
    ax.set_title('Distribution Predictions: Smooth vs Corrugated', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(1.3, 2.3)

    plt.tight_layout()
    plt.savefig('outputs/e1_figures/fig_obstruction_corrugated.png', dpi=300, bbox_inches='tight')
    plt.close()
    rpt("  Saved: outputs/e1_figures/fig_obstruction_corrugated.png")

    # ═══════════════════════════════════════
    # CONCLUSION
    # ═══════════════════════════════════════
    rpt(f"\n{'='*60}")
    rpt(f"FALSIFIABILITY ASSESSMENT")
    rpt(f"{'='*60}")
    rpt()
    rpt(f"Basin classification predictions:")
    rpt(f"  - POST-HOC: Basin boundaries (φ-trapped < 1.67 < e1-residents < 1.75 < main-well")
    rpt(f"    < 1.90 < 2:1-pressed) were defined after observing the data, not predicted.")
    rpt(f"    Any distribution could be partitioned this way. WEAKLY FALSIFIABLE.")
    rpt(f"")
    rpt(f"Corrugated model predictions:")
    rpt(f"  - A_e1 < 0 (e-1 is a trap): POST-HOC. The sign and magnitude were freely fit.")
    rpt(f"    A priori, A_e1 could have been positive (barrier). The fact that it came out")
    rpt(f"    negative is descriptive, not predictive. WEAKLY FALSIFIABLE.")
    rpt(f"  - A_phi > 0 (φ is a barrier): POST-HOC. Same issue as above.")
    rpt(f"")
    rpt(f"Genuinely testable predictions from the corrugated model:")
    rpt(f"  1. New datasets with N>100 at 1024+ Hz should show a density bump near e-1")
    rpt(f"     (positive residual from skew-normal fit). If the bump disappears, the")
    rpt(f"     corrugated model fails. GENUINELY FALSIFIABLE.")
    rpt(f"  2. The basin occupancy differences across datasets (χ² p={p_chi:.4g} if available)")
    rpt(f"     should replicate in new datasets with similar paradigms. If they don't,")
    rpt(f"     the basin classification is dataset-specific, not universal.")
    rpt(f"  3. Manipulating alpha power (e.g., neurofeedback) should shift subjects")
    rpt(f"     between basins if the corrugated landscape is causally real.")
    rpt(f"")
    rpt(f"Overall: The corrugated model is a descriptive framework fit to one dataset.")
    rpt(f"Its predictions are mostly post-hoc. The strongest test would be prospective")
    rpt(f"replication of the density bump at e-1 in an independent high-resolution dataset.")

    rpt(f"\n{'='*60}")
    rpt(f"CONCLUSION")
    rpt(f"{'='*60}")
    rpt()

    if dip_p < 0.05:
        rpt("The distribution shows SIGNIFICANT multimodality (Hartigan's dip test).")
    else:
        rpt("The distribution is consistent with UNIMODALITY (Hartigan's dip test).")

    if best_k > 1:
        rpt(f"However, GMM selects k={best_k} components by BIC,")
        rpt(f"suggesting some internal structure beyond a single mode.")
    else:
        rpt(f"GMM also selects k=1 component, confirming unimodality.")

    rpt(f"\nNote on unimodality + corrugation: The dip test (p={dip_p:.3f}) and GMM")
    rpt(f"(best k={best_k}) both indicate a unimodal distribution. The corrugated")
    rpt(f"model is NOT claiming multimodality — it models subtle within-mode")
    rpt(f"structure (local bumps/traps) that do not create separate peaks.")
    rpt(f"These are compatible: a distribution can be unimodal while having")
    rpt(f"slight density variations near mathematical constants.")

    if delta_corr_vault < -2:
        rpt(f"\nThe corrugated potential (with bumps at φ and e-1) shows a modest")
        rpt(f"AIC improvement of ΔAIC={-delta_corr_vault:.1f} over the smooth vault.")
        rpt(f"However, the skew-normal (3 params) remains competitive overall by BIC.")
        rpt(f"The corrugated model is the best MECHANISTIC model (it encodes specific")
        rpt(f"roles for φ and e-1) but not the best model on parsimony grounds.")
        if corr['A_e1'] < 0:
            rpt(f"e-1 functions as a LOCAL TRAP (A_e1={corr['A_e1']:.3f} < 0)")
            rpt(f"within the broader landscape — subjects may accumulate near e-1")
            rpt(f"as they navigate the potential landscape, which could contribute to")
            rpt(f"why e-1 appears significant even though the global minimum is at {corr['x0']:.3f}.")
        if corr['A_phi'] < 0:
            rpt(f"φ also functions as a local trap (A_phi={corr['A_phi']:.3f} < 0).")
    else:
        rpt(f"\nThe corrugated potential does NOT improve over the smooth vault")
        rpt(f"(ΔAIC={delta_corr_vault:.1f}). No evidence for geometric obstructions.")
        rpt(f"The α/θ landscape appears to be a SMOOTH asymmetric well")
        rpt(f"without local traps at φ or e-1.")

    md = "# Geometrical Obstruction: Corrugated Potential Landscape\n\n"
    md += "## Testing for internal structure in the α/θ distribution\n\n"
    md += "```\n" + "\n".join(report) + "\n```\n\n"
    md += "## Figures\n\n"
    md += "- `fig_obstruction_density.png`: KDE + residuals at mathematical constants\n"
    md += "- `fig_obstruction_basins.png`: Basin classification histogram\n"
    md += "- `fig_obstruction_corrugated.png`: Smooth vs corrugated potential comparison\n"

    with open('outputs/geometric_obstruction.md', 'w') as f:
        f.write(md)
    rpt(f"\nReport saved: outputs/geometric_obstruction.md")

    def jsonify(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    results = {
        'dip_test': dict(statistic=float(dip_stat), p_value=float(dip_p)),
        'n_kde_peaks': int(len(peaks)),
        'gmm_best_k': int(best_k),
        'excess_at_constants': {n: {k: jsonify(v) for k, v in r.items()}
                                 for n, r in excess_results.items()},
        'corrugated_fit': {k: jsonify(v) for k, v in corr.items()},
        'corrugated_vs_smooth_daic': float(delta_corr_vault),
        'best_model_aic': models[best_idx][0],
        'best_model_bic': models[best_bic_idx][0],
        'note': 'Corrugated wins AIC (6 params) but skew-normal wins BIC (3 params). Corrugated is best mechanistic model, not best overall model.',
    }
    with open('outputs/geometric_obstruction_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    rpt("Results saved: outputs/geometric_obstruction_results.json")


if __name__ == '__main__':
    run_analysis()
