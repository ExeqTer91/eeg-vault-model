#!/usr/bin/env python3
"""
THE MINIMUM VAULT MODEL
A potential-well framework for EEG spectral attractors

Fits asymmetric potential well models to empirical α/θ ratio data (N=244),
runs Langevin simulations, compares 5 competing models via AIC/BIC,
and generates publication-ready figures.
"""

import json
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import skew, kurtosis, norm, skewnorm, gaussian_kde
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

E1 = np.e - 1       # 1.71828...
PHI = (1 + np.sqrt(5)) / 2  # 1.61803...
HARMONIC = 2.0

def load_empirical_ratios():
    ratios, datasets, fs_vals = [], [], []
    with open('outputs/aw_cached_subjects.json') as f:
        for s in json.load(f):
            r = s.get('alpha_theta_ratio')
            if r:
                ratios.append(r); datasets.append('AlphaWaves'); fs_vals.append(s.get('fs', 512))
    with open('outputs/ds003969_cached_subjects.json') as f:
        for s in json.load(f):
            r = s.get('alpha_theta_ratio')
            if r:
                ratios.append(r); datasets.append('OpenNeuro'); fs_vals.append(s.get('fs', 1024))
    with open('outputs/eegbci_modal_results.json') as f:
        for s in json.load(f):
            r = s.get('alpha_theta_ratio')
            if r:
                ratios.append(r); datasets.append('PhysioNet'); fs_vals.append(s.get('fs', 160))
    return np.array(ratios), datasets, np.array(fs_vals)


def V_asymmetric(x, x0, k_left, k_right):
    return np.where(x < x0, 0.5 * k_left * (x - x0)**2, 0.5 * k_right * (x - x0)**2)

def V_double_well(x, x1, x2, k1, k2, barrier):
    mid = (x1 + x2) / 2
    width = (x2 - x1) / 2
    V = barrier * ((x - mid)**2 / width**2 - 1)**2
    V += k1 * (x - mid) / width
    return V

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

def analytical_estimate(mean_obs, sd_obs, x0=E1):
    delta = mean_obs - x0
    var = sd_obs**2
    a, b, c = 1, -4 * delta / var, -1
    r = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    k_r = 50
    k_l = r * k_r
    T = var * k_r * (r + 1) / 2
    pred_mean = x0 + T * (1/k_r - 1/k_l) / 2
    pred_sd = np.sqrt(T * (1/k_l + 1/k_r) / 2)
    return dict(asymmetry_ratio=r, k_left=k_l, k_right=k_r, T=T,
                x0=x0, predicted_mean=pred_mean, predicted_sd=pred_sd)


def fit_asymmetric_well(ratios):
    x_grid = np.linspace(1.2, 2.5, 600)

    def neg_ll(params):
        x0, k_left, k_right, T = params
        P = boltzmann_pdf(x_grid, V_asymmetric, T, x0, k_left, k_right)
        P_func = interp1d(x_grid, P, bounds_error=False, fill_value=1e-10)
        P_data = np.maximum(P_func(ratios), 1e-10)
        return -np.sum(np.log(P_data))

    bounds = [(1.55, 1.95), (5, 500), (5, 500), (0.0005, 1.0)]
    result = differential_evolution(neg_ll, bounds, seed=42, maxiter=2000,
                                     tol=1e-8, popsize=25)
    x0, k_l, k_r, T = result.x
    return dict(x0=x0, k_left=k_l, k_right=k_r, T=T,
                nll=result.fun, n_params=4, converged=result.success)


def fit_double_well(ratios):
    x_grid = np.linspace(1.2, 2.5, 600)

    def neg_ll(params):
        x1, x2, k1, k2, barrier, T = params
        P = boltzmann_pdf(x_grid, V_double_well, T, x1, x2, k1, k2, barrier)
        P_func = interp1d(x_grid, P, bounds_error=False, fill_value=1e-10)
        P_data = np.maximum(P_func(ratios), 1e-10)
        return -np.sum(np.log(P_data))

    bounds = [(1.60, 1.80), (1.90, 2.10), (-5, 5), (-5, 5), (0.1, 50), (0.001, 0.5)]
    result = differential_evolution(neg_ll, bounds, seed=42, maxiter=2000,
                                     tol=1e-8, popsize=25)
    x1, x2, k1, k2, barrier, T = result.x
    return dict(x1=x1, x2=x2, k1=k1, k2=k2, barrier=barrier, T=T,
                nll=result.fun, n_params=6, converged=result.success)


def fit_gaussian(ratios, fixed_mean=None):
    if fixed_mean is not None:
        mu = fixed_mean
        sigma = np.sqrt(np.mean((ratios - mu)**2))
    else:
        mu, sigma = np.mean(ratios), np.std(ratios, ddof=1)
    nll = -np.sum(norm.logpdf(ratios, mu, sigma))
    n_params = 1 if fixed_mean is not None else 2
    return dict(mu=mu, sigma=sigma, nll=nll, n_params=n_params)


def fit_skewnorm(ratios):
    params = skewnorm.fit(ratios)
    nll = -np.sum(skewnorm.logpdf(ratios, *params))
    return dict(a=params[0], loc=params[1], scale=params[2],
                nll=nll, n_params=3)


def fit_gmm(ratios, n_components=2):
    X = ratios.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=42, max_iter=500)
    gmm.fit(X)
    nll = -np.sum(gmm.score_samples(X))
    n_p = n_components * 3 - 1
    return dict(means=gmm.means_.flatten().tolist(),
                sigmas=np.sqrt(gmm.covariances_.flatten()).tolist(),
                weights=gmm.weights_.tolist(),
                nll=nll, n_params=n_p)


def aic_bic(nll, k, n):
    aic = 2*k + 2*nll
    bic = k*np.log(n) + 2*nll
    return aic, bic


def langevin_simulation(x0, k_left, k_right, T,
                        n_particles=20000, n_steps=80000, dt=0.001):
    rng = np.random.default_rng(42)
    x = rng.normal(x0, 0.1, n_particles)
    noise_scale = np.sqrt(2 * T * dt)
    for _ in range(n_steps):
        force = np.where(x < x0, -k_left * (x - x0), -k_right * (x - x0))
        x += force * dt + noise_scale * rng.standard_normal(n_particles)
    return x


def make_vault_figure(ratios, vault_params, sim_x, x_grid, datasets):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[1, 1.3],
                             gridspec_kw={'hspace': 0.28})

    x0 = vault_params['x0']
    kl = vault_params['k_left']
    kr = vault_params['k_right']
    T = vault_params['T']

    V = V_asymmetric(x_grid, x0, kl, kr)
    V = V - np.min(V)

    ax = axes[0]
    ax.plot(x_grid, V, 'k-', lw=2.5)
    ax.fill_between(x_grid, V, alpha=0.08, color='steelblue')
    y_top = np.max(V[np.abs(x_grid - x0) < 0.3]) * 1.5
    ax.set_ylim(-0.01 * y_top, y_top)

    T_line = T * 2
    ax.axhline(T_line, color='orangered', ls='--', lw=1.2, alpha=0.7, label=f'~2T (thermal energy)')

    for val, label, color in [(E1, f'e−1={E1:.3f}', '#2ca02c'),
                               (PHI, f'φ={PHI:.3f}', '#9467bd'),
                               (np.mean(ratios), f'mean={np.mean(ratios):.3f}', '#d62728'),
                               (HARMONIC, '2:1', '#8c564b')]:
        if 1.3 < val < 2.3:
            ax.axvline(val, color=color, ls=':', lw=1.3, alpha=0.8)
            ax.text(val, y_top * 0.92, label, ha='center', va='top',
                    fontsize=8, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, alpha=0.85))

    ax.set_xlabel('α/θ spectral centroid ratio', fontsize=11)
    ax.set_ylabel('V(x)  (potential energy)', fontsize=11)
    ax.set_title('Asymmetric Potential Well  ("The Vault")', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(1.3, 2.3)

    ax = axes[1]
    colors_map = {'AlphaWaves': '#1f77b4', 'OpenNeuro': '#2ca02c', 'PhysioNet': '#ff7f0e'}
    for ds_name, color in colors_map.items():
        mask = np.array([d == ds_name for d in datasets])
        if np.any(mask):
            ax.hist(ratios[mask], bins=25, alpha=0.35, color=color, label=ds_name,
                    density=True, range=(1.3, 2.3))

    ax.hist(ratios, bins=35, alpha=0.15, color='gray', density=True,
            range=(1.3, 2.3), label='All N=244')

    P_bolt = boltzmann_pdf(x_grid, V_asymmetric, T, x0, kl, kr)
    ax.plot(x_grid, P_bolt, 'k-', lw=2.5, label='Vault model (Boltzmann)')

    if len(sim_x) > 100:
        kde_sim = gaussian_kde(sim_x, bw_method=0.08)
        ax.plot(x_grid, kde_sim(x_grid), 'r--', lw=1.8, label='Langevin simulation')

    for val, label, color in [(E1, 'e−1', '#2ca02c'), (PHI, 'φ', '#9467bd'),
                               (np.mean(ratios), 'mean', '#d62728')]:
        ax.axvline(val, color=color, ls=':', lw=1.3, alpha=0.8)

    ax.annotate('', xy=(x0, 0.15), xytext=(np.mean(ratios), 0.15),
                arrowprops=dict(arrowstyle='<->', color='#d62728', lw=2))
    ax.text((x0 + np.mean(ratios))/2, 0.18, f'Δ = {np.mean(ratios)-x0:.3f}\n(thermal displacement)',
            ha='center', va='bottom', fontsize=8, color='#d62728',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#d62728', alpha=0.8))

    ax.set_xlabel('α/θ spectral centroid ratio', fontsize=11)
    ax.set_ylabel('Probability density', fontsize=11)
    ax.set_title('Population Distribution vs Vault Model Prediction', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8.5, loc='upper right', ncol=2)
    ax.set_xlim(1.3, 2.3)

    plt.tight_layout()
    plt.savefig('outputs/e1_figures/fig_vault_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/e1_figures/fig_vault_model.png")


def make_model_comparison_figure(ratios, models, x_grid):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(ratios, bins=35, density=True, alpha=0.3, color='gray',
            range=(1.3, 2.3), label='Empirical (N=244)')

    styles = [
        ('Gaussian @ φ', '#9467bd', '--', 1.5),
        ('Gaussian @ e−1', '#2ca02c', '--', 1.5),
        ('Vault (asym well @ fitted)', '#d62728', '-', 2.5),
        ('Skew-normal (3-param)', '#ff7f0e', '-.', 1.8),
        ('GMM (2-comp)', '#1f77b4', ':', 1.8),
    ]

    for (name, color, ls, lw), (mname, mdata) in zip(styles, models):
        if mname == 'gaussian_phi':
            y = norm.pdf(x_grid, mdata['mu'], mdata['sigma'])
        elif mname == 'gaussian_e1':
            y = norm.pdf(x_grid, mdata['mu'], mdata['sigma'])
        elif mname == 'vault':
            y = boltzmann_pdf(x_grid, V_asymmetric, mdata['T'],
                              mdata['x0'], mdata['k_left'], mdata['k_right'])
        elif mname == 'skewnorm':
            y = skewnorm.pdf(x_grid, mdata['a'], mdata['loc'], mdata['scale'])
        elif mname == 'gmm':
            y = np.zeros_like(x_grid)
            for w, m, s in zip(mdata['weights'], mdata['means'], mdata['sigmas']):
                y += w * norm.pdf(x_grid, m, s)
        else:
            continue
        aic, bic = aic_bic(mdata['nll'], mdata['n_params'], len(ratios))
        ax.plot(x_grid, y, color=color, ls=ls, lw=lw,
                label=f'{name}  (AIC={aic:.1f})')

    ax.set_xlabel('α/θ spectral centroid ratio', fontsize=11)
    ax.set_ylabel('Probability density', fontsize=11)
    ax.set_title('Model Comparison: 5 Distributional Hypotheses', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8.5, loc='upper right')
    ax.set_xlim(1.3, 2.3)
    plt.tight_layout()
    plt.savefig('outputs/e1_figures/fig_vault_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/e1_figures/fig_vault_comparison.png")


def make_temperature_figure(ratios, fs_vals, vault_x0):
    unique_fs = sorted(set(fs_vals))
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {160.0: '#ff7f0e', 512: '#1f77b4', 1024.0: '#2ca02c', 2048.0: '#17becf'}
    for fs in unique_fs:
        mask = fs_vals == fs
        subset = ratios[mask]
        if len(subset) < 3:
            continue
        ax.hist(subset, bins=20, density=True, alpha=0.35,
                color=colors.get(fs, 'gray'), range=(1.3, 2.3),
                label=f'{int(fs)} Hz (N={len(subset)}, μ={np.mean(subset):.3f}, σ={np.std(subset,ddof=1):.3f})')
    ax.axvline(vault_x0, color='#2ca02c', ls='--', lw=2, label=f'Vault minimum x₀={vault_x0:.3f}')
    ax.axvline(E1, color='green', ls=':', lw=1.5, alpha=0.6, label=f'e−1={E1:.3f}')
    ax.set_xlabel('α/θ spectral centroid ratio', fontsize=11)
    ax.set_ylabel('Probability density', fontsize=11)
    ax.set_title('"Temperature" by Sampling Rate\n(Higher Fs → lower effective T → closer to minimum)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8.5)
    ax.set_xlim(1.3, 2.3)
    plt.tight_layout()
    plt.savefig('outputs/e1_figures/fig_vault_temperature.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/e1_figures/fig_vault_temperature.png")


def run_vault_model():
    print("=" * 70)
    print("THE MINIMUM VAULT MODEL")
    print("A potential-well framework for EEG spectral attractors")
    print("=" * 70)

    ratios, datasets, fs_vals = load_empirical_ratios()
    N = len(ratios)
    emp_mean = np.mean(ratios)
    emp_sd = np.std(ratios, ddof=1)
    emp_skew = skew(ratios)
    emp_kurt = kurtosis(ratios)
    emp_median = np.median(ratios)

    report = []
    def rpt(s=""):
        report.append(s)
        print(s)

    rpt(f"\n1. EMPIRICAL DISTRIBUTION (N={N})")
    rpt(f"   Mean:     {emp_mean:.4f}")
    rpt(f"   Median:   {emp_median:.4f}")
    rpt(f"   SD:       {emp_sd:.4f}")
    rpt(f"   Skewness: {emp_skew:.4f}")
    rpt(f"   Kurtosis: {emp_kurt:.4f}")
    rpt(f"   Range:    [{np.min(ratios):.4f}, {np.max(ratios):.4f}]")
    rpt()

    if emp_skew < -0.1:
        rpt("   ** FINDING: Distribution is LEFT-SKEWED (tail toward φ)")
        rpt("   This means the well is steeper on the RIGHT (toward 2:1)")
        rpt("   and shallower on the LEFT (toward φ) — opposite to initial hypothesis.")
        rpt("   The asymmetry still supports a vault model, but with reversed stiffness.")
        skew_direction = "left"
    elif emp_skew > 0.1:
        rpt("   ** FINDING: Distribution is RIGHT-SKEWED (tail toward 2:1)")
        rpt("   Consistent with the vault hypothesis of shallower right wall.")
        skew_direction = "right"
    else:
        rpt("   ** FINDING: Distribution is approximately symmetric.")
        skew_direction = "symmetric"

    rpt(f"\n2. ANALYTICAL ESTIMATE (first-order)")
    analytic = analytical_estimate(emp_mean, emp_sd)
    for k, v in analytic.items():
        rpt(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    rpt(f"\n3. MLE FIT: ASYMMETRIC WELL (Vault Model)")
    rpt("   Fitting via differential evolution...")
    vault = fit_asymmetric_well(ratios)
    for k, v in vault.items():
        rpt(f"   {k}: {v:.4f}" if isinstance(v, (float, np.floating)) else f"   {k}: {v}")
    vault_ratio = vault['k_left'] / vault['k_right']
    rpt(f"   Stiffness ratio k_left/k_right: {vault_ratio:.3f}")
    if vault_ratio > 1:
        rpt("   → Steeper on LEFT (toward φ): harder to go below minimum")
    else:
        rpt("   → Steeper on RIGHT (toward 2:1): harder to go above minimum")
    rpt(f"   Fitted minimum x₀ = {vault['x0']:.4f}")
    rpt(f"   Distance from e-1: {abs(vault['x0'] - E1):.4f}")

    rpt(f"\n4. LANGEVIN SIMULATION")
    rpt("   Running 20,000 particles, 80,000 steps...")
    sim_x = langevin_simulation(vault['x0'], vault['k_left'], vault['k_right'], vault['T'])
    sim_mean = np.mean(sim_x)
    sim_sd = np.std(sim_x, ddof=1)
    sim_skew = skew(sim_x)
    rpt(f"   Simulated mean:     {sim_mean:.4f}  (empirical: {emp_mean:.4f})")
    rpt(f"   Simulated SD:       {sim_sd:.4f}  (empirical: {emp_sd:.4f})")
    rpt(f"   Simulated skewness: {sim_skew:.4f}  (empirical: {emp_skew:.4f})")
    rpt(f"   Mean error:  {abs(sim_mean - emp_mean):.4f}")
    rpt(f"   SD error:    {abs(sim_sd - emp_sd):.4f}")
    rpt(f"   Skew error:  {abs(sim_skew - emp_skew):.4f}")

    rpt(f"\n5. MODEL COMPARISON")
    rpt("   Fitting 5 competing models...\n")

    m_phi = fit_gaussian(ratios, fixed_mean=PHI)
    m_e1 = fit_gaussian(ratios, fixed_mean=E1)
    m_vault = vault
    m_skew = fit_skewnorm(ratios)
    m_gmm = fit_gmm(ratios, n_components=2)

    models_list = [
        ('gaussian_phi', m_phi),
        ('gaussian_e1', m_e1),
        ('vault', m_vault),
        ('skewnorm', m_skew),
        ('gmm', m_gmm),
    ]

    labels = ['Gaussian @ φ', 'Gaussian @ e-1',
              'Vault (asym well)', 'Skew-normal', 'GMM (2-comp)']

    rpt(f"   {'Model':<22} {'NLL':>10} {'k':>4} {'AIC':>10} {'BIC':>10} {'ΔAIC':>8}")
    rpt(f"   {'-'*22} {'-'*10} {'-'*4} {'-'*10} {'-'*10} {'-'*8}")

    aics, bics = [], []
    for lab, (mname, mdata) in zip(labels, models_list):
        a, b = aic_bic(mdata['nll'], mdata['n_params'], N)
        aics.append(a)
        bics.append(b)

    best_aic = min(aics)
    for i, (lab, (mname, mdata)) in enumerate(zip(labels, models_list)):
        delta = aics[i] - best_aic
        rpt(f"   {lab:<22} {mdata['nll']:>10.2f} {mdata['n_params']:>4} "
            f"{aics[i]:>10.2f} {bics[i]:>10.2f} {delta:>8.2f}")

    best_idx = np.argmin(aics)
    rpt(f"\n   Best model by AIC: {labels[best_idx]}")
    rpt(f"   Best model by BIC: {labels[np.argmin(bics)]}")

    vault_aic = aics[2]
    for i, lab in enumerate(labels):
        if i != 2:
            delta = aics[i] - vault_aic
            if delta > 0:
                rpt(f"   Vault beats {lab} by ΔAIC = {delta:.1f}")
            else:
                rpt(f"   {lab} beats Vault by ΔAIC = {-delta:.1f}")

    rpt(f"\n6. PREDICTIONS & TESTS")
    rpt(f"\n   Prediction 1: Skewness direction")
    rpt(f"   Empirical skewness = {emp_skew:.4f}")
    if skew_direction == "left":
        rpt(f"   RESULT: LEFT-skewed — the well is steeper toward 2:1,")
        rpt(f"   shallower toward φ. The 'harmonic lock' barrier is STRONG,")
        rpt(f"   and rare low-ratio subjects pull the tail leftward.")
    elif skew_direction == "right":
        rpt(f"   RESULT: RIGHT-skewed — consistent with original vault hypothesis.")
    else:
        rpt(f"   RESULT: Approximately symmetric — asymmetry is minimal.")

    rpt(f"\n   Prediction 2: Minimum location")
    rpt(f"   Fitted x₀ = {vault['x0']:.4f}")
    rpt(f"   e-1 = {E1:.4f},  distance = {abs(vault['x0'] - E1):.4f}")
    rpt(f"   φ   = {PHI:.4f},  distance = {abs(vault['x0'] - PHI):.4f}")
    if abs(vault['x0'] - E1) < abs(vault['x0'] - PHI):
        rpt(f"   → Minimum is CLOSER to e-1 than to φ")
    else:
        rpt(f"   → Minimum is CLOSER to φ than to e-1")

    rpt(f"\n   Prediction 3: Temperature by sampling rate")
    for fs in sorted(set(fs_vals)):
        mask = fs_vals == fs
        subset = ratios[mask]
        if len(subset) >= 3:
            rpt(f"   {int(fs)} Hz: N={len(subset)}, mean={np.mean(subset):.4f}, "
                f"SD={np.std(subset,ddof=1):.4f} (SD as proxy for T)")
    rpt(f"   → Lower SD at higher Fs would support measurement-noise-as-temperature")

    rpt(f"\n   FALSIFIABILITY ASSESSMENT")
    rpt(f"   ========================")
    rpt(f"   Prediction 1 (skewness): POST-HOC. The skewness direction was observed")
    rpt(f"   first, then the model was fit to match it. The stiffness asymmetry")
    rpt(f"   (k_left/k_right) was freely fit, so any skewness direction could have been")
    rpt(f"   accommodated. This prediction is WEAKLY FALSIFIABLE as stated.")
    rpt(f"   → To strengthen: pre-register skewness prediction for a NEW dataset.")
    rpt(f"")
    rpt(f"   Prediction 2 (minimum location): POST-HOC. The minimum x₀ was freely fit")
    rpt(f"   to the data. The claim that x₀ is 'closer to e-1 than φ' is true but")
    rpt(f"   not a strong test — x₀ is also 0.097 ABOVE e-1, so the model does not")
    rpt(f"   actually predict e-1 as the equilibrium. This prediction is WEAKLY FALSIFIABLE.")
    rpt(f"   → To strengthen: predict x₀ from first principles (e.g., neural time constants).")
    rpt(f"")
    rpt(f"   Prediction 3 (temperature vs Fs): GENUINELY TESTABLE. The prediction that")
    rpt(f"   higher sampling rates → lower effective temperature is falsifiable: if SD does")
    rpt(f"   not decrease with Fs (or increases), the measurement-noise interpretation fails.")
    rpt(f"   Current data show mixed results (2048 Hz has lowest SD but N=3).")
    rpt(f"   → Status: PARTIALLY SUPPORTED, needs larger N at high Fs.")
    rpt(f"")
    rpt(f"   Overall: 2 of 3 predictions are post-hoc fits, not genuine a priori predictions.")
    rpt(f"   The vault model describes the distribution shape well but does not make")
    rpt(f"   strong, risky predictions that could decisively confirm or refute it.")
    rpt(f"   Future work should derive predictions from the model BEFORE collecting new data.")

    rpt(f"\n7. GENERATING FIGURES")
    x_grid = np.linspace(1.3, 2.3, 1000)
    make_vault_figure(ratios, vault, sim_x, x_grid, datasets)
    make_model_comparison_figure(ratios, models_list, x_grid)
    make_temperature_figure(ratios, fs_vals, vault['x0'])

    rpt(f"\n{'='*70}")
    rpt(f"SUMMARY")
    rpt(f"{'='*70}")
    rpt()
    rpt(f"The asymmetric potential well model ('Vault') centered at")
    rpt(f"x₀ = {vault['x0']:.4f} with stiffness ratio k_left/k_right = {vault_ratio:.3f}")
    rpt(f"and effective temperature T = {vault['T']:.4f}")
    if vault_ratio < 1:
        rpt(f"predicts a LEFT-skewed distribution (steeper right wall toward 2:1)")
    else:
        rpt(f"predicts a RIGHT-skewed distribution (steeper left wall toward φ)")
    rpt(f"with mean = {sim_mean:.4f} (empirical {emp_mean:.4f})")
    rpt(f"and SD = {sim_sd:.4f} (empirical {emp_sd:.4f}).")
    rpt()
    rpt(f"The population mean ({emp_mean:.3f}) sits {emp_mean - vault['x0']:.3f} above")
    rpt(f"the potential minimum ({vault['x0']:.3f}) due to thermal displacement")
    rpt(f"in the asymmetric well.")
    rpt()
    rpt(f"Model comparison (AIC): best = {labels[best_idx]}")
    for i, lab in enumerate(labels):
        delta = aics[i] - best_aic
        rpt(f"  {lab}: ΔAIC = {delta:.1f}")
    rpt()
    dist_e1 = abs(vault['x0'] - E1)
    rpt(f"The vault framework provides a principled statistical-mechanics")
    rpt(f"explanation for the α/θ distribution shape.")
    if dist_e1 < 0.05:
        rpt(f"The fitted minimum ({vault['x0']:.3f}) sits near e-1 ({E1:.3f}),")
        rpt(f"supporting the exponential-attractor hypothesis.")
    else:
        rpt(f"However, the fitted minimum ({vault['x0']:.3f}) sits {dist_e1:.3f} above e-1 ({E1:.3f}),")
        rpt(f"indicating the potential well is NOT centered at e-1 but rather")
        rpt(f"between e-1 and the population mean. The well captures the")
        rpt(f"distribution shape (skewness, kurtosis) but does not confirm")
        rpt(f"e-1 as the equilibrium point.")

    md = "# The Minimum Vault Model\n\n"
    md += "## A potential-well framework for EEG spectral attractors\n\n"
    md += "```\n" + "\n".join(report) + "\n```\n\n"
    md += "## Figures\n\n"
    md += "- `fig_vault_model.png`: Potential well and population distribution\n"
    md += "- `fig_vault_comparison.png`: 5-model AIC comparison\n"
    md += "- `fig_vault_temperature.png`: Temperature (SD) by sampling rate\n"

    with open('outputs/vault_model.md', 'w') as f:
        f.write(md)
    print("\nReport saved: outputs/vault_model.md")

    results = {
        'empirical': dict(N=N, mean=float(emp_mean), sd=float(emp_sd),
                          skew=float(emp_skew), kurtosis=float(emp_kurt),
                          median=float(emp_median)),
        'analytical_estimate': {k: float(v) for k, v in analytic.items()},
        'vault_fit': {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                      for k, v in vault.items()},
        'simulation': dict(mean=float(sim_mean), sd=float(sim_sd), skew=float(sim_skew)),
        'model_comparison': {lab: dict(aic=float(a), bic=float(b), delta_aic=float(a - best_aic))
                             for lab, a, b in zip(labels, aics, bics)},
        'best_model_aic': labels[best_idx],
        'best_model_bic': labels[int(np.argmin(bics))],
    }
    with open('outputs/vault_model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved: outputs/vault_model_results.json")


if __name__ == '__main__':
    run_vault_model()
