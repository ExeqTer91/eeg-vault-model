import numpy as np
import json
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.stats import norm
from scipy.optimize import minimize

mne.set_log_level('ERROR')

PHI = 1.6180339887
E_MINUS_1 = np.e - 1
HARMONIC = 2.0
FS_ALPHA = 512
BP_LOW = 1.0
BP_HIGH = 45.0
NOTCH_FREQ = 50

os.makedirs('outputs', exist_ok=True)

def preprocess_mat(data, fs):
    data = data - np.mean(data, axis=0, keepdims=True)
    nyq = fs / 2.0
    b, a = butter(4, [BP_LOW / nyq, BP_HIGH / nyq], btype='band')
    for ch in range(data.shape[0]):
        data[ch] = filtfilt(b, a, data[ch])
    b_n, a_n = iirnotch(NOTCH_FREQ, Q=30, fs=fs)
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

def compute_pci(ratio):
    d_phi = abs(ratio - PHI)
    d_harm = abs(ratio - 2.0)
    if d_phi < 1e-10:
        return 10.0
    return np.log(d_harm / d_phi)

print("Loading subjects...")
mat_files = sorted(glob.glob('alpha_s[0-9][0-9].mat'))
subjects = []
for fpath in mat_files:
    mat = loadmat(fpath)
    data = mat['SIGNAL'].astype(np.float64).T
    data = preprocess_mat(data, FS_ALPHA)
    subjects.append({'data': data, 'fs': FS_ALPHA})

mne.set_config('MNE_DATASETS_EEGBCI_PATH', '/home/runner/mne_data')
n_needed = 35 - len(subjects)
loaded = 0
for subj_id in range(1, 110):
    if loaded >= n_needed:
        break
    try:
        fnames = mne.datasets.eegbci.load_data(subj_id, [1], update_path=True)
        raw = mne.io.read_raw_edf(fnames[0], preload=True)
        mne.datasets.eegbci.standardize(raw)
        raw.filter(BP_LOW, BP_HIGH, fir_design='firwin')
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        data = raw.get_data(picks=eeg_picks)
        fs = raw.info['sfreq']
        subjects.append({'data': data, 'fs': fs})
        loaded += 1
    except:
        pass

n = len(subjects)
print(f"Total: {n} subjects\n")

all_ratios = np.zeros(n)
all_pcis = np.zeros(n)
for i, subj in enumerate(subjects):
    data = subj['data']
    fs = subj['fs']
    nperseg = min(int(4 * fs), data.shape[1])
    avg_psd = None
    for ch in range(data.shape[0]):
        freqs, psd = welch(data[ch], fs=fs, nperseg=nperseg)
        if avg_psd is None:
            avg_psd = psd.copy()
        else:
            avg_psd += psd
    avg_psd /= data.shape[0]
    theta_c = spectral_centroid(freqs, avg_psd, 4, 8)
    alpha_c = spectral_centroid(freqs, avg_psd, 8, 13)
    all_ratios[i] = alpha_c / theta_c
    all_pcis[i] = compute_pci(all_ratios[i])

print("="*70)
print("TEST 4: SURROGATE NULL MODEL")
print(f"N = {n}, 100 surrogates")
print("="*70)

N_SURROGATES = 50
np.random.seed(42)

cropped = []
for subj in subjects:
    data = subj['data']
    fs = subj['fs']
    max_samp = min(data.shape[1], int(15 * fs))
    cropped.append((data[:, :max_samp], fs))

observed_phi_error = np.abs(all_ratios - PHI)
observed_mean_phi_error = np.mean(observed_phi_error)
observed_pci_rate = np.mean(all_pcis > 0) * 100

surrogate_mean_errors = np.zeros(N_SURROGATES)
surrogate_pci_rates = np.zeros(N_SURROGATES)
surrogate_mean_ratios = np.zeros(N_SURROGATES)

for s_idx in range(N_SURROGATES):
    if (s_idx + 1) % 25 == 0:
        print(f"  Surrogate {s_idx+1}/{N_SURROGATES}")
    surr_ratios = np.zeros(n)
    surr_pcis = np.zeros(n)
    for subj_i in range(n):
        data_c, fs_c = cropped[subj_i]
        n_ch, n_samp = data_c.shape
        random_phases = np.exp(2j * np.pi * np.random.random(n_samp // 2 + 1))
        random_phases[0] = 1
        if n_samp % 2 == 0:
            random_phases[-1] = np.sign(random_phases[-1].real)
        surrogate = np.zeros_like(data_c)
        for ch in range(n_ch):
            fft_vals = np.fft.rfft(data_c[ch])
            surrogate[ch] = np.fft.irfft(np.abs(fft_vals) * random_phases, n=n_samp)

        nperseg = min(int(4 * fs_c), n_samp)
        avg_psd = None
        for ch in range(n_ch):
            freqs, psd = welch(surrogate[ch], fs=fs_c, nperseg=nperseg)
            if avg_psd is None:
                avg_psd = psd.copy()
            else:
                avg_psd += psd
        avg_psd /= n_ch
        theta_c = spectral_centroid(freqs, avg_psd, 4, 8)
        alpha_c = spectral_centroid(freqs, avg_psd, 8, 13)
        r = alpha_c / theta_c
        surr_ratios[subj_i] = r
        surr_pcis[subj_i] = compute_pci(r)

    surrogate_mean_errors[s_idx] = np.mean(np.abs(surr_ratios - PHI))
    surrogate_pci_rates[s_idx] = np.mean(surr_pcis > 0) * 100
    surrogate_mean_ratios[s_idx] = np.mean(surr_ratios)

p_phi_error = np.mean(surrogate_mean_errors <= observed_mean_phi_error)
z_phi_error = (observed_mean_phi_error - np.mean(surrogate_mean_errors)) / np.std(surrogate_mean_errors)
p_pci_rate = np.mean(surrogate_pci_rates >= observed_pci_rate)

print(f"  Observed mean |ratio - phi|:  {observed_mean_phi_error:.4f}")
print(f"  Surrogate mean |ratio - phi|: {np.mean(surrogate_mean_errors):.4f} (SD={np.std(surrogate_mean_errors):.4f})")
print(f"  Z-score: {z_phi_error:.4f}")
print(f"  p-value (phi proximity): {p_phi_error:.4f}")
print(f"  Observed PCI>0 rate: {observed_pci_rate:.1f}%")
print(f"  Surrogate PCI>0 rate: {np.mean(surrogate_pci_rates):.1f}% (SD={np.std(surrogate_pci_rates):.1f}%)")
print(f"  p-value (PCI rate): {p_pci_rate:.4f}")

if p_phi_error < 0.05:
    print(f"  ** SIGNIFICANT: Real EEG closer to phi than surrogates")
else:
    print(f"  Not significant: ratios not closer to phi than surrogates")

surrogate_results = {
    'n_subjects': n,
    'n_surrogates': N_SURROGATES,
    'observed_mean_phi_error': float(observed_mean_phi_error),
    'surrogate_mean_phi_error': float(np.mean(surrogate_mean_errors)),
    'z_score': float(z_phi_error),
    'p_phi_proximity': float(p_phi_error),
    'observed_pci_rate': float(observed_pci_rate),
    'surrogate_pci_rate': float(np.mean(surrogate_pci_rates)),
    'p_pci_rate': float(p_pci_rate),
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ax = axes[0]
ax.hist(surrogate_mean_errors, bins=25, color='#3498DB', alpha=0.7, edgecolor='black', label='Surrogates')
ax.axvline(observed_mean_phi_error, color='red', linewidth=2.5, linestyle='-', label=f'Observed')
ax.set_xlabel('Mean |Ratio - Phi|')
ax.set_ylabel('Count')
ax.set_title(f'Phi-Proximity Test\nZ={z_phi_error:.2f}, p={p_phi_error:.4f}', fontweight='bold')
ax.legend()
ax = axes[1]
ax.hist(surrogate_pci_rates, bins=25, color='#2ECC71', alpha=0.7, edgecolor='black', label='Surrogates')
ax.axvline(observed_pci_rate, color='red', linewidth=2.5, linestyle='-', label=f'Observed = {observed_pci_rate:.1f}%')
ax.set_xlabel('% Subjects with PCI > 0')
ax.set_ylabel('Count')
ax.set_title(f'PCI>0 Rate Test\np={p_pci_rate:.4f}', fontweight='bold')
ax.legend()
ax = axes[2]
ax.hist(surrogate_mean_ratios, bins=25, color='#F39C12', alpha=0.7, edgecolor='black', label='Surrogate means')
ax.axvline(np.mean(all_ratios), color='red', linewidth=2.5, linestyle='-', label=f'Observed')
ax.axvline(PHI, color='gold', linewidth=2, linestyle='--', label=f'phi')
ax.set_xlabel('Mean Alpha/Theta Ratio')
ax.set_ylabel('Count')
ax.set_title('Mean Ratio Distribution', fontweight='bold')
ax.legend()
plt.suptitle(f'Surrogate Null Model — N={n}, {N_SURROGATES} surrogates', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/surrogate_null_model.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*70)
print("TEST 5: BAYESIAN MODEL COMPARISON")
print(f"N = {n}")
print("="*70)

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
    total_w = 1 + e1 + e2
    w1 = e1 / total_w
    w2 = e2 / total_w
    w3 = 1 / total_w
    mu1 = params[2]; s1 = max(np.exp(params[3]), 1e-6)
    mu2 = params[4]; s2 = max(np.exp(params[5]), 1e-6)
    mu3 = params[6]; s3 = max(np.exp(params[7]), 1e-6)
    p = w1*norm.pdf(data, loc=mu1, scale=s1) + w2*norm.pdf(data, loc=mu2, scale=s2) + w3*norm.pdf(data, loc=mu3, scale=s3)
    return -np.sum(np.log(p + 1e-300))

models = {}
for name, mu_fixed in [('phi', PHI), ('e-1', E_MINUS_1), ('2:1', HARMONIC)]:
    res = minimize(neg_log_lik_single, x0=[0.1], args=(all_ratios, mu_fixed), method='Nelder-Mead')
    nll = res.fun; k_p = 1
    bic = k_p * np.log(n) + 2 * nll
    models[f'Single_{name}'] = {'nll': nll, 'bic': bic, 'aic': 2*k_p + 2*nll, 'k': k_p, 'mu': mu_fixed, 'sigma': res.x[0]}

res_free = minimize(neg_log_lik_single, x0=[np.mean(all_ratios), 0.1], args=(all_ratios, None), method='Nelder-Mead')
nll_free = res_free.fun; k_free = 2
models['Single_free'] = {'nll': nll_free, 'bic': k_free*np.log(n)+2*nll_free, 'aic': 2*k_free+2*nll_free, 'k': k_free,
                         'mu': res_free.x[0], 'sigma': res_free.x[1]}

for name, mu_init in [('phi+harm', (PHI, HARMONIC)), ('phi+e-1', (PHI, E_MINUS_1))]:
    x0 = [0.0, mu_init[0], np.log(0.05), mu_init[1], np.log(0.05)]
    res2 = minimize(neg_log_lik_mixture2, x0=x0, args=(all_ratios,), method='Nelder-Mead', options={'maxiter': 10000})
    nll2 = res2.fun; k2 = 5
    w1 = 1/(1+np.exp(-res2.x[0]))
    models[f'Mix2_{name}'] = {'nll': nll2, 'bic': k2*np.log(n)+2*nll2, 'aic': 2*k2+2*nll2, 'k': k2,
                              'w1': w1, 'mu1': res2.x[1], 's1': np.exp(res2.x[2]), 'mu2': res2.x[3], 's2': np.exp(res2.x[4])}

x0_f2 = [0.0, PHI, np.log(0.05), np.mean(all_ratios), np.log(0.1)]
res_f2 = minimize(neg_log_lik_mixture2, x0=x0_f2, args=(all_ratios,), method='Nelder-Mead', options={'maxiter': 10000})
nll_f2 = res_f2.fun; k_f2 = 5
w1_f = 1/(1+np.exp(-res_f2.x[0]))
models['Mix2_free'] = {'nll': nll_f2, 'bic': k_f2*np.log(n)+2*nll_f2, 'aic': 2*k_f2+2*nll_f2, 'k': k_f2,
                       'w1': w1_f, 'mu1': res_f2.x[1], 's1': np.exp(res_f2.x[2]), 'mu2': res_f2.x[3], 's2': np.exp(res_f2.x[4])}

x0_3 = [0.0, 0.0, PHI, np.log(0.05), E_MINUS_1, np.log(0.05), HARMONIC, np.log(0.05)]
res3 = minimize(neg_log_lik_mixture3, x0=x0_3, args=(all_ratios,), method='Nelder-Mead', options={'maxiter': 20000})
nll3 = res3.fun; k3 = 8
models['Mix3_phi_e1_harm'] = {'nll': nll3, 'bic': k3*np.log(n)+2*nll3, 'aic': 2*k3+2*nll3, 'k': k3}

sorted_models = sorted(models.items(), key=lambda x: x[1]['bic'])
best_bic = sorted_models[0][1]['bic']
best_name = sorted_models[0][0]

print(f"\n{'Model':<25s} {'k':>3s} {'NLL':>8s} {'BIC':>8s} {'dBIC':>8s}")
print("-" * 52)
for mname, m in sorted_models:
    dbic = m['bic'] - best_bic
    print(f"{mname:<25s} {m['k']:>3d} {m['nll']:>8.2f} {m['bic']:>8.2f} {dbic:>8.2f}")
print(f"\nBest model: {best_name} (BIC = {best_bic:.2f})")

bayesian_results = {
    'n_subjects': n,
    'best_model': best_name,
    'best_bic': float(best_bic),
    'models': {mname: {'bic': float(m['bic']), 'nll': float(m['nll']), 'k': m['k']} for mname, m in models.items()},
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
x_range = np.linspace(all_ratios.min()-0.1, all_ratios.max()+0.1, 200)
ax.hist(all_ratios, bins=15, density=True, alpha=0.5, color='gray', edgecolor='black', label='Data')
for mname, mu, color in [('phi', PHI, '#E74C3C'), ('e-1', E_MINUS_1, '#3498DB'), ('2:1', HARMONIC, '#2ECC71')]:
    m = models[f'Single_{mname}']
    pdf = norm.pdf(x_range, loc=m['mu'], scale=m['sigma'])
    ax.plot(x_range, pdf, color=color, linewidth=2, label=f'{mname} (BIC={m["bic"]:.1f})')
ax.set_xlabel('Alpha/Theta Ratio'); ax.set_ylabel('Density')
ax.set_title('Single-Attractor Models', fontweight='bold'); ax.legend()
ax = axes[1]
mn_sorted = [mn for mn, _ in sorted_models]
bic_vals = [m['bic'] for _, m in sorted_models]
colors_bic = ['#2ECC71' if i == 0 else '#3498DB' for i in range(len(mn_sorted))]
ax.barh(range(len(mn_sorted)), bic_vals, color=colors_bic, alpha=0.8, edgecolor='black')
ax.set_yticks(range(len(mn_sorted))); ax.set_yticklabels(mn_sorted, fontsize=9)
ax.set_xlabel('BIC'); ax.set_title('Model Comparison (BIC)', fontweight='bold'); ax.invert_yaxis()
plt.suptitle(f'Bayesian Model Comparison — N={n}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/bayesian_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

combined = {
    'surrogate': surrogate_results,
    'bayesian': bayesian_results,
}
with open('outputs/surrogate_results.json', 'w') as f:
    json.dump(surrogate_results, f, indent=2)
with open('outputs/bayesian_results.json', 'w') as f:
    json.dump(bayesian_results, f, indent=2)

print("\n" + "="*70)
print("TESTS 4-5 COMPLETE")
print("="*70)
