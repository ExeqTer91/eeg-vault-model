import numpy as np
import pandas as pd
import os
import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.stats import norm
from scipy.optimize import minimize

PHI = 1.6180339887
E_MINUS_1 = np.e - 1
HARMONIC = 2.0
FS = 512
BP_LOW = 1.0
BP_HIGH = 45.0
NOTCH = 50

def preprocess(data, fs):
    data = data - np.mean(data, axis=0, keepdims=True)
    nyq = fs / 2.0
    b, a = butter(4, [BP_LOW / nyq, BP_HIGH / nyq], btype='band')
    for ch in range(data.shape[0]):
        data[ch] = filtfilt(b, a, data[ch])
    b_n, a_n = iirnotch(NOTCH, Q=30, fs=fs)
    for ch in range(data.shape[0]):
        data[ch] = filtfilt(b_n, a_n, data[ch])
    return data

def spectral_centroid(freqs, psd, lo, hi):
    idx = np.logical_and(freqs >= lo, freqs <= hi)
    f_band = freqs[idx]
    p_band = psd[idx]
    total = np.sum(p_band)
    if total == 0:
        return (lo + hi) / 2.0
    return np.sum(f_band * p_band) / total

mat_files = sorted(glob.glob('alpha_s[0-9][0-9].mat')) + sorted(glob.glob('alpha_subj_[0-9][0-9].mat'))
mat_files = mat_files[:35]
n_subjects = len(mat_files)
print(f"Loading {n_subjects} subjects...\n")

ratios = np.zeros(n_subjects)
for i, fpath in enumerate(mat_files):
    mat = loadmat(fpath)
    data = mat['SIGNAL'].astype(np.float64).T
    data = preprocess(data, FS)
    nperseg = min(int(4 * FS), data.shape[1])
    avg_psd = np.zeros(nperseg // 2 + 1)
    for ch in range(data.shape[0]):
        freqs, psd = welch(data[ch], fs=FS, nperseg=nperseg)
        avg_psd += psd
    avg_psd /= data.shape[0]
    theta_c = spectral_centroid(freqs, avg_psd, 4, 8)
    alpha_c = spectral_centroid(freqs, avg_psd, 8, 13)
    ratios[i] = alpha_c / theta_c

print(f"Ratios: mean={np.mean(ratios):.4f}, SD={np.std(ratios):.4f}, range=[{np.min(ratios):.4f}, {np.max(ratios):.4f}]")

def neg_log_lik_single(params, data, fixed_mu=None):
    if fixed_mu is not None:
        mu = fixed_mu
        sigma = max(params[0], 1e-6)
    else:
        mu = params[0]
        sigma = max(params[1], 1e-6)
    return -np.sum(norm.logpdf(data, loc=mu, scale=sigma))

def neg_log_lik_mixture2(params, data):
    w1 = 1 / (1 + np.exp(-params[0]))
    mu1 = params[1]
    s1 = max(np.exp(params[2]), 1e-6)
    mu2 = params[3]
    s2 = max(np.exp(params[4]), 1e-6)
    p1 = w1 * norm.pdf(data, loc=mu1, scale=s1)
    p2 = (1 - w1) * norm.pdf(data, loc=mu2, scale=s2)
    return -np.sum(np.log(p1 + p2 + 1e-300))

def neg_log_lik_mixture3(params, data):
    e1 = np.exp(params[0])
    e2 = np.exp(params[1])
    total = 1 + e1 + e2
    w1 = e1 / total
    w2 = e2 / total
    w3 = 1 / total
    mu1 = params[2]
    s1 = max(np.exp(params[3]), 1e-6)
    mu2 = params[4]
    s2 = max(np.exp(params[5]), 1e-6)
    mu3 = params[6]
    s3 = max(np.exp(params[7]), 1e-6)
    p = w1 * norm.pdf(data, loc=mu1, scale=s1) + w2 * norm.pdf(data, loc=mu2, scale=s2) + w3 * norm.pdf(data, loc=mu3, scale=s3)
    return -np.sum(np.log(p + 1e-300))

n = len(ratios)

print("\n--- Fitting Single-Attractor Models ---")
models = {}

for name, mu_fixed in [('phi', PHI), ('e-1', E_MINUS_1), ('2:1', HARMONIC)]:
    res = minimize(neg_log_lik_single, x0=[0.1], args=(ratios, mu_fixed), method='Nelder-Mead')
    nll = res.fun
    k = 1
    bic = k * np.log(n) + 2 * nll
    aic = 2 * k + 2 * nll
    models[f'Single_{name}'] = {'nll': nll, 'bic': bic, 'aic': aic, 'k': k, 'mu': mu_fixed, 'sigma': res.x[0]}
    print(f"  Normal(mu={mu_fixed:.4f}): sigma={res.x[0]:.4f}, NLL={nll:.2f}, BIC={bic:.2f}, AIC={aic:.2f}")

res_free = minimize(neg_log_lik_single, x0=[np.mean(ratios), 0.1], args=(ratios, None), method='Nelder-Mead')
nll_free = res_free.fun
k_free = 2
bic_free = k_free * np.log(n) + 2 * nll_free
aic_free = 2 * k_free + 2 * nll_free
models['Single_free'] = {'nll': nll_free, 'bic': bic_free, 'aic': aic_free, 'k': k_free, 'mu': res_free.x[0], 'sigma': res_free.x[1]}
print(f"  Normal(free): mu={res_free.x[0]:.4f}, sigma={res_free.x[1]:.4f}, NLL={nll_free:.2f}, BIC={bic_free:.2f}, AIC={aic_free:.2f}")

print("\n--- Fitting 2-Component Mixture Models ---")

for name, mu_init in [('phi+harm', (PHI, HARMONIC)), ('phi+e-1', (PHI, E_MINUS_1))]:
    x0 = [0.0, mu_init[0], np.log(0.05), mu_init[1], np.log(0.05)]
    res2 = minimize(neg_log_lik_mixture2, x0=x0, args=(ratios,), method='Nelder-Mead',
                    options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
    nll2 = res2.fun
    k2 = 5
    bic2 = k2 * np.log(n) + 2 * nll2
    aic2 = 2 * k2 + 2 * nll2
    w1 = 1 / (1 + np.exp(-res2.x[0]))
    mu1_fit = res2.x[1]
    s1_fit = np.exp(res2.x[2])
    mu2_fit = res2.x[3]
    s2_fit = np.exp(res2.x[4])
    models[f'Mix2_{name}'] = {'nll': nll2, 'bic': bic2, 'aic': aic2, 'k': k2,
                              'w1': w1, 'mu1': mu1_fit, 's1': s1_fit, 'mu2': mu2_fit, 's2': s2_fit}
    print(f"  {name}: w1={w1:.3f}, mu1={mu1_fit:.4f}(s={s1_fit:.4f}), mu2={mu2_fit:.4f}(s={s2_fit:.4f}), "
          f"NLL={nll2:.2f}, BIC={bic2:.2f}")

x0_free2 = [0.0, PHI, np.log(0.05), np.mean(ratios), np.log(0.1)]
res_free2 = minimize(neg_log_lik_mixture2, x0=x0_free2, args=(ratios,), method='Nelder-Mead',
                     options={'maxiter': 10000})
nll_f2 = res_free2.fun
k_f2 = 5
bic_f2 = k_f2 * np.log(n) + 2 * nll_f2
aic_f2 = 2 * k_f2 + 2 * nll_f2
w1_f = 1 / (1 + np.exp(-res_free2.x[0]))
models['Mix2_free'] = {'nll': nll_f2, 'bic': bic_f2, 'aic': aic_f2, 'k': k_f2,
                       'w1': w1_f, 'mu1': res_free2.x[1], 's1': np.exp(res_free2.x[2]),
                       'mu2': res_free2.x[3], 's2': np.exp(res_free2.x[4])}
print(f"  free 2-mix: w1={w1_f:.3f}, mu1={res_free2.x[1]:.4f}, mu2={res_free2.x[3]:.4f}, "
      f"NLL={nll_f2:.2f}, BIC={bic_f2:.2f}")

print("\n--- Fitting 3-Component Mixture ---")
x0_3 = [0.0, 0.0, PHI, np.log(0.05), E_MINUS_1, np.log(0.05), HARMONIC, np.log(0.05)]
res3 = minimize(neg_log_lik_mixture3, x0=x0_3, args=(ratios,), method='Nelder-Mead',
                options={'maxiter': 20000})
nll3 = res3.fun
k3 = 8
bic3 = k3 * np.log(n) + 2 * nll3
aic3 = 2 * k3 + 2 * nll3
e1 = np.exp(res3.x[0])
e2 = np.exp(res3.x[1])
total_w = 1 + e1 + e2
w3_1 = e1 / total_w
w3_2 = e2 / total_w
w3_3 = 1 / total_w
models['Mix3_phi_e1_harm'] = {'nll': nll3, 'bic': bic3, 'aic': aic3, 'k': k3,
                              'w1': w3_1, 'mu1': res3.x[2], 's1': np.exp(res3.x[3]),
                              'w2': w3_2, 'mu2': res3.x[4], 's2': np.exp(res3.x[5]),
                              'w3': w3_3, 'mu3': res3.x[6], 's3': np.exp(res3.x[7])}
print(f"  phi+e-1+2:1: w=({w3_1:.3f},{w3_2:.3f},{w3_3:.3f}), "
      f"mu=({res3.x[2]:.4f},{res3.x[4]:.4f},{res3.x[6]:.4f}), "
      f"NLL={nll3:.2f}, BIC={bic3:.2f}")

print("\n" + "="*70)
print("MODEL COMPARISON (sorted by BIC)")
print("="*70)

sorted_models = sorted(models.items(), key=lambda x: x[1]['bic'])
best_bic = sorted_models[0][1]['bic']

print(f"\n{'Model':<25s} {'k':>3s} {'NLL':>8s} {'BIC':>8s} {'dBIC':>8s} {'AIC':>8s}")
print("-" * 60)
for name, m in sorted_models:
    dbic = m['bic'] - best_bic
    print(f"{name:<25s} {m['k']:>3d} {m['nll']:>8.2f} {m['bic']:>8.2f} {dbic:>8.2f} {m['aic']:>8.2f}")

best_name = sorted_models[0][0]
best_model = sorted_models[0][1]
print(f"\nBest model: {best_name} (BIC = {best_model['bic']:.2f})")

approx_bf = {}
for name, m in models.items():
    if name != best_name:
        approx_bf[name] = np.exp(-0.5 * (m['bic'] - best_model['bic']))
print(f"\nApproximate Bayes Factors (vs {best_name}):")
for name, bf in sorted(approx_bf.items(), key=lambda x: -x[1]):
    print(f"  {name}: BF = {bf:.4e}")

os.makedirs('outputs', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

ax = axes[0, 0]
x_range = np.linspace(ratios.min() - 0.1, ratios.max() + 0.1, 200)
ax.hist(ratios, bins=12, density=True, alpha=0.5, color='gray', edgecolor='black', label='Data')
for name, mu, color in [('phi', PHI, '#E74C3C'), ('e-1', E_MINUS_1, '#3498DB'), ('2:1', HARMONIC, '#2ECC71')]:
    m = models[f'Single_{name}']
    pdf = norm.pdf(x_range, loc=m['mu'], scale=m['sigma'])
    ax.plot(x_range, pdf, color=color, linewidth=2, label=f'{name} (BIC={m["bic"]:.1f})')
ax.set_xlabel('Alpha/Theta Ratio', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Single-Attractor Models', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)

ax = axes[0, 1]
model_names = [n for n, _ in sorted_models]
bic_vals = [m['bic'] for _, m in sorted_models]
colors_bic = ['#2ECC71' if i == 0 else '#3498DB' for i in range(len(model_names))]
ax.barh(range(len(model_names)), bic_vals, color=colors_bic, alpha=0.8, edgecolor='black')
ax.set_yticks(range(len(model_names)))
ax.set_yticklabels(model_names, fontsize=9)
ax.set_xlabel('BIC', fontsize=12)
ax.set_title('Model Comparison (BIC)', fontsize=13, fontweight='bold')
ax.invert_yaxis()

ax = axes[1, 0]
if 'Mix2_free' in models:
    m = models['Mix2_free']
    ax.hist(ratios, bins=12, density=True, alpha=0.4, color='gray', edgecolor='black', label='Data')
    pdf1 = m['w1'] * norm.pdf(x_range, loc=m['mu1'], scale=m['s1'])
    pdf2 = (1 - m['w1']) * norm.pdf(x_range, loc=m['mu2'], scale=m['s2'])
    ax.plot(x_range, pdf1 + pdf2, 'k-', linewidth=2, label='Total fit')
    ax.fill_between(x_range, pdf1, alpha=0.3, color='#E74C3C', label=f'Comp 1: mu={m["mu1"]:.3f}, w={m["w1"]:.2f}')
    ax.fill_between(x_range, pdf2, alpha=0.3, color='#3498DB', label=f'Comp 2: mu={m["mu2"]:.3f}, w={1-m["w1"]:.2f}')
    ax.axvline(PHI, color='gold', linewidth=2, linestyle='--', alpha=0.7, label=f'phi={PHI:.3f}')
    ax.set_xlabel('Alpha/Theta Ratio', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Best 2-Component Mixture', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8)

ax = axes[1, 1]
attractors = {'phi': PHI, 'e-1': E_MINUS_1, '2:1': HARMONIC}
distances = {name: np.abs(ratios - val) for name, val in attractors.items()}
closest = np.array([min(distances.items(), key=lambda x: x[1][i])[0] for i in range(n)])
counts = {name: np.sum(closest == name) for name in attractors}
colors_pie = ['#E74C3C', '#3498DB', '#2ECC71']
labels_pie = [f'{name}\n({counts[name]}/{n}, {counts[name]/n*100:.0f}%)' for name in attractors]
ax.pie([counts[name] for name in attractors], labels=labels_pie, colors=colors_pie,
       autopct='', startangle=90, textprops={'fontsize': 11})
ax.set_title('Nearest Attractor Classification', fontsize=13, fontweight='bold')

plt.suptitle(f'Bayesian Model Comparison — N={n} subjects\nAlpha/Theta centroid ratios',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/bayesian_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure saved: outputs/bayesian_model_comparison.png")

results = {
    'n_subjects': n,
    'ratio_mean': float(np.mean(ratios)),
    'ratio_sd': float(np.std(ratios)),
    'best_model': best_name,
    'best_bic': float(best_model['bic']),
    'models': {name: {k: float(v) if isinstance(v, (np.floating, float)) else v
                      for k, v in m.items()} for name, m in models.items()},
    'nearest_attractor_counts': {name: int(counts[name]) for name in attractors},
}
with open('outputs/bayesian_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Results saved: outputs/bayesian_results.json")
