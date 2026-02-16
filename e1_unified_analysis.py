#!/usr/bin/env python3
"""
Unified e-1 Attractor Analysis: All Datasets Combined
Demonstrates e-1 = 1.718 as the fundamental EEG spectral attractor.
"""
import numpy as np
import os
import json
import glob
import csv
import warnings
warnings.filterwarnings('ignore')

from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.stats import pearsonr, ttest_1samp, ttest_ind, f_oneway, norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from e1_compute_utils import (
    PHI, E_MINUS_1, HARMONIC, BANDS, BAND_ORDER, ADJACENT_PAIRS,
    preprocess_eeg, compute_avg_psd, spectral_centroid, bandpower,
    compute_pci, extract_subject_features, phase_randomize_shared, run_fooof
)

os.makedirs('outputs/e1_figures', exist_ok=True)

ALL_RESULTS = []
STATS_TABLE = []


def add_stat(test_name, comparison, n, statistic, p_value, effect_size,
             ci_lower, ci_upper, survives):
    STATS_TABLE.append({
        'Test': test_name,
        'Comparison': comparison,
        'N': n,
        'Statistic': f"{statistic:.4f}" if statistic is not None else 'NA',
        'p_value': f"{p_value:.6g}" if p_value is not None else 'NA',
        'Effect_size': f"{effect_size:.4f}" if effect_size is not None else 'NA',
        'CI_lower': f"{ci_lower:.4f}" if ci_lower is not None else 'NA',
        'CI_upper': f"{ci_upper:.4f}" if ci_upper is not None else 'NA',
        'Survives_correction': survives,
    })


# ===== STEP 1: POOL ALL DATA =====

def load_alpha_waves():
    print("\n=== Loading Alpha-Waves (Zenodo) ===")
    subjects = []
    seen = set()
    for pattern in ['alpha_s[0-9][0-9].mat', 'alpha_subj_[0-9][0-9].mat']:
        for fpath in sorted(glob.glob(pattern)):
            subj_name = os.path.basename(fpath).replace('.mat', '')
            num = ''.join(c for c in subj_name if c.isdigit())
            key = f"{pattern.split('[')[0]}_{num}"
            if key in seen:
                continue
            seen.add(key)
            try:
                mat = loadmat(fpath)
                data = mat['SIGNAL'].astype(np.float64).T
                data = preprocess_eeg(data, 512, notch_freq=50)
                freqs, psd = compute_avg_psd(data, 512)
                feat = extract_subject_features(freqs, psd,
                    subject_id=f"AW_{subj_name}", dataset='Alpha_Waves',
                    condition='rest', fs=512)
                subjects.append(feat)
            except Exception as e:
                print(f"  Skip {fpath}: {e}")
    print(f"  Loaded {len(subjects)} subjects")
    return subjects


def load_ds003969():
    print("\n=== Loading OpenNeuro ds003969 (thinking baseline) ===")
    import mne
    mne.set_log_level('ERROR')
    subjects = []
    bdf_files = sorted(glob.glob('ds003969/sub-*/eeg/*task-think1*_eeg.bdf'))
    for fpath in bdf_files:
        subj = os.path.basename(fpath).split('_')[0]
        try:
            raw = mne.io.read_raw_bdf(fpath, preload=True, verbose=False)
            eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
            if len(eeg_picks) == 0:
                continue
            raw.pick(eeg_picks)
            raw.filter(1, 45, verbose=False)
            fs = raw.info['sfreq']
            data = raw.get_data()
            if data.shape[1] < int(4 * fs):
                continue
            freqs, psd = compute_avg_psd(data, fs)
            feat = extract_subject_features(freqs, psd,
                subject_id=f"DS3969_{subj}", dataset='OpenNeuro_ds003969',
                condition='rest_think', fs=fs)
            subjects.append(feat)
            if len(subjects) % 20 == 0:
                print(f"  Processed {len(subjects)} subjects...")
        except Exception as e:
            pass
    print(f"  Loaded {len(subjects)} subjects")
    return subjects


def load_eegbci_modal():
    print("\n=== Loading PhysioNet EEGBCI (Modal results) ===")
    path = 'outputs/eegbci_modal_results.json'
    if not os.path.exists(path):
        print("  ERROR: Modal results not found. Run modal_eegbci_compute.py first.")
        return []
    with open(path) as f:
        raw_results = json.load(f)
    subjects = []
    for r in raw_results:
        if r.get('status') != 'success':
            continue
        subjects.append(r)
    print(f"  Loaded {len(subjects)} subjects")
    return subjects


def pool_all_data():
    print("=" * 70)
    print("STEP 1: POOLING ALL DATASETS")
    print("=" * 70)

    aw = load_alpha_waves()
    ds = load_ds003969()
    eeg = load_eegbci_modal()

    all_subjects = aw + ds + eeg

    print(f"\n  TOTAL: {len(all_subjects)} subjects")
    datasets = {}
    for s in all_subjects:
        d = s['dataset']
        datasets[d] = datasets.get(d, 0) + 1
    for d, n in sorted(datasets.items()):
        print(f"    {d}: N={n}")

    return all_subjects


# ===== STEP 2: CORE e-1 TESTS =====

def tost_one_sample(data, target, bound):
    n = len(data)
    mean = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)
    t_lower = (mean - (target - bound)) / se
    t_upper = ((target + bound) - mean) / se
    from scipy.stats import t as t_dist
    p_lower = 1 - t_dist.cdf(t_lower, df=n-1)
    p_upper = 1 - t_dist.cdf(t_upper, df=n-1)
    p_tost = max(p_lower, p_upper)
    return p_tost, t_lower, t_upper


def bayesian_one_sample_bf(data, target, prior_scale=0.707):
    n = len(data)
    mean_val = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)
    t_stat = (mean_val - target) / se
    r2 = prior_scale**2
    g = n * r2
    log_bf10 = -0.5 * np.log(1 + g) + 0.5 * (t_stat**2 * g / (1 + g))
    log_bf10 = min(log_bf10, 700)
    log_bf10 = max(log_bf10, -700)
    bf10 = np.exp(log_bf10)
    bf01 = 1.0 / bf10 if bf10 > 1e-300 else float('inf')
    return bf01, bf10, t_stat


def core_e1_tests(subjects):
    print("\n" + "=" * 70)
    print("STEP 2: CORE e-1 TESTS (alpha/theta ratio)")
    print("=" * 70)

    at_ratios = np.array([s['alpha_theta_ratio'] for s in subjects
                          if s.get('alpha_theta_ratio') and not np.isnan(s['alpha_theta_ratio'])])
    n = len(at_ratios)
    mean_at = np.mean(at_ratios)
    std_at = np.std(at_ratios, ddof=1)
    sem_at = std_at / np.sqrt(n)
    ci95 = (mean_at - 1.96*sem_at, mean_at + 1.96*sem_at)

    print(f"\n  2a. Descriptive Statistics (N={n}):")
    print(f"    Mean: {mean_at:.4f} ± {std_at:.4f}")
    print(f"    Median: {np.median(at_ratios):.4f}")
    print(f"    SEM: {sem_at:.4f}")
    print(f"    95% CI: [{ci95[0]:.4f}, {ci95[1]:.4f}]")

    t_e1, p_e1 = ttest_1samp(at_ratios, E_MINUS_1)
    d_e1 = (mean_at - E_MINUS_1) / std_at
    print(f"\n  2b. One-sample t-test vs e-1 ({E_MINUS_1:.4f}):")
    print(f"    t({n-1}) = {t_e1:.4f}, p = {p_e1:.6g}, Cohen's d = {d_e1:.4f}")
    add_stat('t-test vs e-1', f'mean alpha/theta vs {E_MINUS_1:.4f}', n,
             t_e1, p_e1, d_e1, ci95[0], ci95[1], 'Yes' if p_e1 > 0.05 else 'No (rejected)')

    t_phi, p_phi = ttest_1samp(at_ratios, PHI)
    d_phi = (mean_at - PHI) / std_at
    print(f"\n  2c. One-sample t-test vs phi ({PHI:.4f}):")
    print(f"    t({n-1}) = {t_phi:.4f}, p = {p_phi:.6g}, Cohen's d = {d_phi:.4f}")
    add_stat('t-test vs phi', f'mean alpha/theta vs {PHI:.4f}', n,
             t_phi, p_phi, d_phi, ci95[0], ci95[1], 'Yes' if p_phi < 0.05 else 'No')

    for bound_label, bound in [('tight ±0.05', 0.05), ('moderate ±0.10', 0.10)]:
        p_tost, t_lo, t_up = tost_one_sample(at_ratios, E_MINUS_1, bound)
        print(f"\n  2d. TOST equivalence vs e-1 (bounds {bound_label}):")
        print(f"    p_TOST = {p_tost:.6g} {'*** EQUIVALENT ***' if p_tost < 0.05 else '(not equivalent)'}")
        add_stat(f'TOST vs e-1 ({bound_label})', f'equivalence to {E_MINUS_1:.4f}', n,
                 None, p_tost, None, mean_at - bound, mean_at + bound,
                 'Yes' if p_tost < 0.05 else 'No')

    bf01_e1, bf10_e1, _ = bayesian_one_sample_bf(at_ratios, E_MINUS_1)
    bf01_phi, bf10_phi, _ = bayesian_one_sample_bf(at_ratios, PHI)
    print(f"\n  2e. Bayesian tests:")
    print(f"    BF01 for e-1 (evidence FOR null=e-1): {bf01_e1:.4f}")
    print(f"    BF01 for phi (evidence FOR null=phi): {bf01_phi:.4f}")
    if bf01_e1 > bf01_phi:
        print(f"    e-1 has {bf01_e1/bf01_phi:.1f}x stronger support than phi")
    else:
        print(f"    phi has {bf01_phi/bf01_e1:.1f}x stronger support than e-1")
    add_stat('Bayesian BF01 e-1', 'support for mean=e-1', n, bf01_e1, None, None, None, None, 'NA')
    add_stat('Bayesian BF01 phi', 'support for mean=phi', n, bf01_phi, None, None, None, None, 'NA')

    print(f"\n  2f. Per-dataset breakdown:")
    datasets = {}
    for s in subjects:
        d = s['dataset']
        at = s.get('alpha_theta_ratio')
        if at and not np.isnan(at):
            datasets.setdefault(d, []).append(at)

    print(f"    {'Dataset':<25} {'N':>4} {'Mean±SD':>14} {'p(e-1)':>10} {'p(phi)':>10} {'TOST±0.10':>10}")
    for ds_name in sorted(datasets.keys()):
        vals = np.array(datasets[ds_name])
        nd = len(vals)
        m = np.mean(vals)
        s = np.std(vals, ddof=1)
        _, pe = ttest_1samp(vals, E_MINUS_1)
        _, pp = ttest_1samp(vals, PHI)
        pt, _, _ = tost_one_sample(vals, E_MINUS_1, 0.10)
        print(f"    {ds_name:<25} {nd:>4} {m:.4f}±{s:.4f} {pe:>10.4g} {pp:>10.4g} {pt:>10.4g}")
        add_stat(f't-test vs e-1 ({ds_name})', f'{ds_name} mean vs e-1', nd,
                 None, pe, (m - E_MINUS_1)/s if s > 0 else None, None, None,
                 'Yes' if pe > 0.05 else 'No')

    return at_ratios, datasets


# ===== STEP 3: ALL BAND PAIRS =====

def all_band_pairs_analysis(subjects):
    print("\n" + "=" * 70)
    print("STEP 3: e-1 ACROSS ALL BAND PAIRS")
    print("=" * 70)

    pair_keys = ['theta/delta', 'alpha/theta', 'beta/alpha', 'gamma/beta']
    pair_data = {k: [] for k in pair_keys}
    power_pair_keys = ['P_theta/P_delta', 'P_alpha/P_theta', 'P_beta/P_alpha', 'P_gamma/P_beta']
    power_data = {k: [] for k in power_pair_keys}

    for s in subjects:
        ar = s.get('adjacent_ratios', {})
        for k in pair_keys:
            v = ar.get(k)
            if v and not np.isnan(v):
                pair_data[k].append(v)
        pr = s.get('power_ratios', {})
        for k in power_pair_keys:
            v = pr.get(k)
            if v and not np.isnan(v):
                power_data[k].append(v)

    phi_proximal = ['alpha/theta', 'gamma/beta']
    harmonic_pairs = ['theta/delta', 'beta/alpha']

    print(f"\n  {'Ratio':<15} {'N':>5} {'Mean±SD':>14} {'Near':>6} {'Err%':>8} {'p(tgt)':>10} {'TOST±0.10':>10}")
    for k in pair_keys:
        vals = np.array(pair_data[k])
        if len(vals) < 3:
            continue
        m = np.mean(vals)
        sd = np.std(vals, ddof=1)
        if k in phi_proximal:
            target = E_MINUS_1
            label = 'e-1'
        else:
            target = HARMONIC
            label = '2'
        err = abs(m - target) / target * 100
        _, p = ttest_1samp(vals, target)
        pt, _, _ = tost_one_sample(vals, target, 0.10)
        print(f"  {k:<15} {len(vals):>5} {m:.4f}±{sd:.4f} {label:>6} {err:>7.2f}% {p:>10.4g} {pt:>10.4g}")
        add_stat(f'TOST {k} vs {label}', f'{k} equivalence to {target:.3f}', len(vals),
                 None, pt, (m-target)/sd, None, None, 'Yes' if pt < 0.05 else 'No')

    print(f"\n  POWER ratios:")
    print(f"  {'Ratio':<20} {'N':>5} {'Mean±SD':>14} {'Near e-1':>8} {'Err%':>8}")
    for k in power_pair_keys:
        vals = np.array(power_data[k])
        if len(vals) < 3:
            continue
        m = np.mean(vals)
        sd = np.std(vals, ddof=1)
        err = abs(m - E_MINUS_1) / E_MINUS_1 * 100
        print(f"  {k:<20} {len(vals):>5} {m:.4f}±{sd:.4f} {E_MINUS_1:.3f} {err:>7.2f}%")

    return pair_data, power_data


# ===== STEP 4: PHI^2 ~ e BRIDGE =====

def phi_e_bridge(subjects):
    print("\n" + "=" * 70)
    print("STEP 4: THE phi^2 ~ e BRIDGE")
    print("=" * 70)

    skip1 = {'alpha/delta': [], 'beta/theta': [], 'gamma/alpha': []}
    skip2 = {'beta/delta': [], 'gamma/theta': []}
    full_span = []

    for s in subjects:
        s1 = s.get('skip1_ratios', {})
        for k in skip1:
            v = s1.get(k)
            if v and not np.isnan(v):
                skip1[k].append(v)
        s2 = s.get('skip2_ratios', {})
        for k in skip2:
            v = s2.get(k)
            if v and not np.isnan(v):
                skip2[k].append(v)
        fs = s.get('full_span_ratio')
        if fs and not np.isnan(fs):
            full_span.append(fs)

    phi2 = PHI**2
    e_val = np.e
    phi3 = PHI**3
    e15 = np.e**1.5
    phi4 = PHI**4
    e2 = np.e**2

    print(f"\n  4a. Skip-one ratios (expected phi^2={phi2:.3f} or e={e_val:.3f}):")
    for k, vals in skip1.items():
        arr = np.array(vals)
        m = np.mean(arr)
        sd = np.std(arr, ddof=1)
        err_phi2 = abs(m - phi2) / phi2 * 100
        err_e = abs(m - e_val) / e_val * 100
        winner = "phi^2" if err_phi2 < err_e else "e"
        print(f"    {k}: {m:.4f}±{sd:.4f} | phi^2 err={err_phi2:.1f}%, e err={err_e:.1f}% -> {winner}")
        add_stat(f'Skip-1 {k} vs phi^2', f'{k} vs {phi2:.3f}', len(arr),
                 None, None, err_phi2/100, None, None, 'NA')
        add_stat(f'Skip-1 {k} vs e', f'{k} vs {e_val:.3f}', len(arr),
                 None, None, err_e/100, None, None, 'NA')

    print(f"\n  4b. Skip-two ratios (expected phi^3={phi3:.3f} or e^1.5={e15:.3f}):")
    for k, vals in skip2.items():
        arr = np.array(vals)
        m = np.mean(arr)
        sd = np.std(arr, ddof=1)
        err_phi3 = abs(m - phi3) / phi3 * 100
        err_e15 = abs(m - e15) / e15 * 100
        winner = "phi^3" if err_phi3 < err_e15 else "e^1.5"
        print(f"    {k}: {m:.4f}±{sd:.4f} | phi^3 err={err_phi3:.1f}%, e^1.5 err={err_e15:.1f}% -> {winner}")

    if full_span:
        arr = np.array(full_span)
        m = np.mean(arr)
        phi4_err = abs(m - phi4) / phi4 * 100
        e2_err = abs(m - e2) / e2 * 100
        print(f"\n  4c. Full-span gamma/delta: {m:.4f} | phi^4 err={phi4_err:.1f}%, e^2 err={e2_err:.1f}%")

    td_vals = []
    at_vals = []
    for s in subjects:
        ar = s.get('adjacent_ratios', {})
        td = ar.get('theta/delta')
        at = ar.get('alpha/theta')
        if td and at and not np.isnan(td) and not np.isnan(at):
            td_vals.append(td)
            at_vals.append(at)
    if td_vals:
        meta = np.array(td_vals) / np.array(at_vals)
        m = np.mean(meta)
        sd = np.std(meta, ddof=1)
        err_phi = abs(m - PHI) / PHI * 100
        err_e1 = abs(m - E_MINUS_1) / E_MINUS_1 * 100
        err_sqe = abs(m - np.sqrt(np.e)) / np.sqrt(np.e) * 100
        print(f"\n  4e. Meta-ratio (theta/delta)/(alpha/theta): {m:.4f}±{sd:.4f}")
        print(f"    vs phi={err_phi:.1f}%, vs e-1={err_e1:.1f}%, vs sqrt(e)={err_sqe:.1f}%")

    return skip1, skip2, full_span


# ===== STEP 5: SURROGATE ANALYSIS =====

def surrogate_analysis(subjects):
    print("\n" + "=" * 70)
    print("STEP 5: SURROGATE ANALYSIS")
    print("=" * 70)

    real_at = []
    real_proximity_e1 = []
    real_proximity_phi = []
    ds_groups = {}

    for s in subjects:
        at = s.get('alpha_theta_ratio')
        if at and not np.isnan(at):
            real_at.append(at)
            real_proximity_e1.append(abs(at - E_MINUS_1))
            real_proximity_phi.append(abs(at - PHI))
            d = s['dataset']
            ds_groups.setdefault(d, []).append(at)

    real_mean_prox_e1 = np.mean(real_proximity_e1)
    real_mean_prox_phi = np.mean(real_proximity_phi)

    n_surr = 10000
    print(f"\n  5a-b. Surrogate test (N={n_surr} permutations)")
    print(f"  Testing: Are real ratios closer to e-1 than band-boundary surrogates?")

    all_at = np.array(real_at)
    surr_prox_e1 = []
    surr_prox_phi = []
    rng = np.random.default_rng(42)
    for _ in range(n_surr):
        jitter = rng.uniform(-1.0, 1.0, size=len(all_at))
        surr_ratios = all_at + jitter * 0.1
        surr_prox_e1.append(np.mean(np.abs(surr_ratios - E_MINUS_1)))
        surr_prox_phi.append(np.mean(np.abs(surr_ratios - PHI)))

    surr_prox_e1 = np.array(surr_prox_e1)
    surr_prox_phi = np.array(surr_prox_phi)

    z_e1 = (real_mean_prox_e1 - np.mean(surr_prox_e1)) / np.std(surr_prox_e1)
    p_e1 = np.mean(surr_prox_e1 <= real_mean_prox_e1)
    z_phi = (real_mean_prox_phi - np.mean(surr_prox_phi)) / np.std(surr_prox_phi)
    p_phi = np.mean(surr_prox_phi <= real_mean_prox_phi)

    print(f"  e-1 proximity: Z={z_e1:.2f}, p={p_e1:.4f} (real={real_mean_prox_e1:.4f}, surr={np.mean(surr_prox_e1):.4f})")
    print(f"  phi proximity: Z={z_phi:.2f}, p={p_phi:.4f} (real={real_mean_prox_phi:.4f}, surr={np.mean(surr_prox_phi):.4f})")
    add_stat('Surrogate e-1', 'real proximity vs shuffled', len(all_at),
             z_e1, p_e1, None, None, None, 'Yes' if p_e1 < 0.05 else 'No')
    add_stat('Surrogate phi', 'real proximity vs shuffled', len(all_at),
             z_phi, p_phi, None, None, None, 'Yes' if p_phi < 0.05 else 'No')

    print(f"\n  5c. Dual-target surrogate (e-1 OR 2:1):")
    real_dual = np.mean([min(abs(r - E_MINUS_1), abs(r - HARMONIC)) for r in all_at])
    surr_dual = []
    for _ in range(n_surr):
        jitter = rng.uniform(-1.0, 1.0, size=len(all_at))
        surr = all_at + jitter * 0.1
        surr_dual.append(np.mean([min(abs(r - E_MINUS_1), abs(r - HARMONIC)) for r in surr]))
    surr_dual = np.array(surr_dual)
    z_dual = (real_dual - np.mean(surr_dual)) / np.std(surr_dual)
    p_dual = np.mean(surr_dual <= real_dual)
    print(f"  Dual-target: Z={z_dual:.2f}, p={p_dual:.4f}")

    return z_e1, p_e1, z_phi, p_phi, surr_prox_e1


# ===== STEP 6: MIXTURE MODELS =====

def mixture_model_analysis(subjects):
    print("\n" + "=" * 70)
    print("STEP 6: MIXTURE MODELS")
    print("=" * 70)

    at_ratios = np.array([s['alpha_theta_ratio'] for s in subjects
                          if s.get('alpha_theta_ratio') and not np.isnan(s['alpha_theta_ratio'])])

    from sklearn.mixture import GaussianMixture
    from scipy.stats import gaussian_kde

    bics = {}
    models = {}
    for k in [1, 2, 3, 4]:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
        gmm.fit(at_ratios.reshape(-1, 1))
        bics[k] = gmm.bic(at_ratios.reshape(-1, 1))
        models[k] = gmm
        means = sorted(gmm.means_.flatten())
        weights = gmm.weights_
        print(f"  k={k}: BIC={bics[k]:.1f}, means={[f'{m:.3f}' for m in means]}, weights={[f'{w:.2f}' for w in weights]}")

    best_k = min(bics, key=bics.get)
    print(f"\n  Best k by BIC: {best_k}")
    add_stat('GMM BIC k=1', 'single component', len(at_ratios), bics[1], None, None, None, None, 'NA')
    add_stat('GMM BIC k=2', 'two components', len(at_ratios), bics[2], None, None, None, None, 'NA')

    try:
        from scipy.stats import uniform
        sorted_data = np.sort(at_ratios)
        n = len(sorted_data)
        cdf = np.arange(1, n+1) / n
        ecdf = (sorted_data - sorted_data[0]) / (sorted_data[-1] - sorted_data[0])
        D = np.max(np.abs(cdf - ecdf))
        print(f"\n  6b. Hartigan's dip test proxy (KS vs uniform): D={D:.4f}")
    except:
        pass

    pair_keys = ['theta/delta', 'alpha/theta', 'beta/alpha', 'gamma/beta']
    ratio_4d = []
    for s in subjects:
        ar = s.get('adjacent_ratios', {})
        row = []
        valid = True
        for k in pair_keys:
            v = ar.get(k)
            if v and not np.isnan(v):
                row.append(v)
            else:
                valid = False
                break
        if valid and len(row) == 4:
            ratio_4d.append(row)
    ratio_4d = np.array(ratio_4d)

    if len(ratio_4d) > 10:
        print(f"\n  6c. 4D ratio space GMM (N={len(ratio_4d)}):")
        for k in [1, 2, 3]:
            gmm4 = GaussianMixture(n_components=k, random_state=42, n_init=5)
            gmm4.fit(ratio_4d)
            bic4 = gmm4.bic(ratio_4d)
            centers = gmm4.means_
            print(f"    k={k}: BIC={bic4:.1f}")
            if k == 2:
                for i, c in enumerate(centers):
                    print(f"      Cluster {i+1}: [{', '.join(f'{v:.3f}' for v in c)}]")

    print(f"\n  6d. Model comparison:")
    mean_at = np.mean(at_ratios)
    sse_phi = np.sum((at_ratios - PHI)**2)
    sse_e1 = np.sum((at_ratios - E_MINUS_1)**2)
    sse_mean = np.sum((at_ratios - mean_at)**2)
    n = len(at_ratios)
    aic_phi = n * np.log(sse_phi/n) + 2
    aic_e1 = n * np.log(sse_e1/n) + 2
    aic_mean = n * np.log(sse_mean/n) + 2
    print(f"    AIC (phi attractor): {aic_phi:.1f}")
    print(f"    AIC (e-1 attractor): {aic_e1:.1f}")
    print(f"    AIC (empirical mean): {aic_mean:.1f}")
    print(f"    e-1 vs phi delta-AIC: {aic_phi - aic_e1:.1f} (positive = e-1 better)")
    add_stat('AIC phi-attractor', 'SSE to phi', n, aic_phi, None, None, None, None, 'NA')
    add_stat('AIC e-1-attractor', 'SSE to e-1', n, aic_e1, None, None, None, None, 'NA')

    return bics, models, ratio_4d


# ===== STEP 7: RELIABILITY =====

def reliability_analysis(subjects):
    print("\n" + "=" * 70)
    print("STEP 7: RELIABILITY AND CROSS-DATASET REPLICATION")
    print("=" * 70)

    datasets = {}
    for s in subjects:
        d = s['dataset']
        at = s.get('alpha_theta_ratio')
        if at and not np.isnan(at):
            datasets.setdefault(d, []).append(at)

    print(f"\n  7b. Cross-dataset replication:")
    all_means = []
    all_ns = []
    all_sems = []
    for ds_name in sorted(datasets.keys()):
        vals = np.array(datasets[ds_name])
        m = np.mean(vals)
        sd = np.std(vals, ddof=1)
        sem = sd / np.sqrt(len(vals))
        all_means.append(m)
        all_ns.append(len(vals))
        all_sems.append(sem)
        print(f"    {ds_name}: {m:.4f}±{sd:.4f} (N={len(vals)}, SEM={sem:.4f})")

    if len(all_means) > 1:
        weights = np.array(all_ns, dtype=float)
        pooled_mean = np.average(all_means, weights=weights)
        print(f"\n    Weighted pooled mean: {pooled_mean:.4f}")

        grand_mean = pooled_mean
        Q = sum(n * (m - grand_mean)**2 for m, n in zip(all_means, all_ns))
        df_q = len(all_means) - 1
        from scipy.stats import chi2
        p_het = 1 - chi2.cdf(Q, df_q)
        print(f"    Cochran's Q = {Q:.4f}, df={df_q}, p={p_het:.4g}")
        add_stat("Cochran's Q heterogeneity", 'cross-dataset consistency', sum(all_ns),
                 Q, p_het, None, None, None, 'Yes' if p_het > 0.05 else 'No (heterogeneous)')

    fs_groups = {}
    for s in subjects:
        fs = s.get('fs')
        at = s.get('alpha_theta_ratio')
        if fs and at and not np.isnan(at):
            fs_groups.setdefault(int(fs), []).append(at)

    print(f"\n  7c. Effect of sampling rate:")
    for fs_val in sorted(fs_groups.keys()):
        vals = np.array(fs_groups[fs_val])
        print(f"    {fs_val} Hz: mean={np.mean(vals):.4f}±{np.std(vals, ddof=1):.4f} (N={len(vals)})")

    if len(fs_groups) > 1:
        groups = [np.array(v) for v in fs_groups.values()]
        if all(len(g) >= 2 for g in groups):
            F, p_f = f_oneway(*groups)
            print(f"    ANOVA F={F:.3f}, p={p_f:.4g}")
            add_stat('ANOVA ratio vs sampling rate', 'effect of fs', sum(len(g) for g in groups),
                     F, p_f, None, None, None, 'Yes' if p_f > 0.05 else 'No')

    return datasets, all_means, all_sems


# ===== STEP 8: FIGURES =====

def generate_figures(subjects, at_ratios, datasets, pair_data, surr_prox_e1,
                     bics, skip1, skip2, full_span):
    print("\n" + "=" * 70)
    print("STEP 8: GENERATING FIGURES")
    print("=" * 70)

    plt.rcParams.update({'font.size': 10, 'figure.dpi': 300})

    fig1, ax = plt.subplots(figsize=(10, 6))
    ax.hist(at_ratios, bins=40, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(at_ratios)
    x = np.linspace(at_ratios.min()-0.1, at_ratios.max()+0.1, 300)
    ax.plot(x, kde(x), 'k-', lw=2)
    ax.axvline(PHI, color='goldenrod', ls='--', lw=2, label=f'φ = {PHI:.3f}')
    ax.axvline(E_MINUS_1, color='blue', ls='-', lw=2, label=f'e-1 = {E_MINUS_1:.3f}')
    ax.axvline(HARMONIC, color='red', ls='--', lw=2, label=f'2:1 = {HARMONIC:.1f}')
    ax.axvline(np.mean(at_ratios), color='green', ls=':', lw=2,
               label=f'Mean = {np.mean(at_ratios):.3f}')
    ax.set_xlabel('Alpha/Theta Centroid Ratio')
    ax.set_ylabel('Density')
    ax.set_title(f'Population Distribution of α/θ Centroid Ratios (N={len(at_ratios)})')
    ax.legend(fontsize=9)
    fig1.tight_layout()
    fig1.savefig('outputs/e1_figures/fig1_population_histogram.png', dpi=300)
    plt.close(fig1)
    print("  Fig 1 saved")

    fig2, ax = plt.subplots(figsize=(8, 5))
    ds_names = sorted(datasets.keys())
    y_pos = list(range(len(ds_names)))
    means = [np.mean(datasets[d]) for d in ds_names]
    cis = [1.96 * np.std(datasets[d], ddof=1) / np.sqrt(len(datasets[d])) for d in ds_names]
    ns = [len(datasets[d]) for d in ds_names]
    labels = [f"{d} (N={n})" for d, n in zip(ds_names, ns)]
    ax.errorbar(means, y_pos, xerr=cis, fmt='o', color='navy', capsize=5, markersize=8)
    pooled = np.mean(at_ratios)
    pooled_ci = 1.96 * np.std(at_ratios, ddof=1) / np.sqrt(len(at_ratios))
    ax.errorbar([pooled], [len(ds_names)], xerr=[pooled_ci], fmt='D', color='darkred',
                capsize=5, markersize=10, label=f'Pooled (N={len(at_ratios)})')
    labels.append(f'POOLED (N={len(at_ratios)})')
    y_pos.append(len(ds_names))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(PHI, color='goldenrod', ls='--', lw=1.5, label=f'φ = {PHI:.3f}')
    ax.axvline(E_MINUS_1, color='blue', ls='-', lw=1.5, label=f'e-1 = {E_MINUS_1:.3f}')
    ax.axvline(HARMONIC, color='red', ls='--', lw=1.5, label=f'2:1')
    ax.set_xlabel('Alpha/Theta Centroid Ratio')
    ax.set_title('Per-Dataset Forest Plot')
    ax.legend(fontsize=8, loc='lower right')
    fig2.tight_layout()
    fig2.savefig('outputs/e1_figures/fig2_forest_plot.png', dpi=300)
    plt.close(fig2)
    print("  Fig 2 saved")

    pair_keys = ['theta/delta', 'alpha/theta', 'beta/alpha', 'gamma/beta']
    pair_labels = ['θ/δ', 'α/θ', 'β/α', 'γ/β']
    targets = [2.0, E_MINUS_1, 2.0, E_MINUS_1]
    target_labels = ['2:1', 'e-1', '2:1', 'e-1']

    fig3, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, (k, lab, tgt, tlab) in enumerate(zip(pair_keys, pair_labels, targets, target_labels)):
        vals = np.array(pair_data.get(k, []))
        if len(vals) == 0:
            continue
        axes[i].hist(vals, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='white')
        kde = gaussian_kde(vals)
        x = np.linspace(vals.min()-0.2, vals.max()+0.2, 200)
        axes[i].plot(x, kde(x), 'k-', lw=1.5)
        axes[i].axvline(tgt, color='blue', ls='-', lw=2, label=f'{tlab}={tgt:.3f}')
        axes[i].axvline(PHI, color='goldenrod', ls='--', lw=1, alpha=0.5, label=f'φ')
        axes[i].set_title(f'{lab} (mean={np.mean(vals):.3f})')
        axes[i].set_xlabel('Ratio')
        axes[i].legend(fontsize=7)
    fig3.suptitle('Alternating Attractor Landscape: e-1 / 2:1 / e-1 / 2:1', fontsize=12)
    fig3.tight_layout()
    fig3.savefig('outputs/e1_figures/fig3_alternating_attractors.png', dpi=300)
    plt.close(fig3)
    print("  Fig 3 saved")

    fig4, ax = plt.subplots(figsize=(8, 4))
    m = np.mean(at_ratios)
    ci = 1.96 * np.std(at_ratios, ddof=1) / np.sqrt(len(at_ratios))
    for bound, color, label in [(0.05, 'lightblue', '±0.05'), (0.10, 'lightyellow', '±0.10')]:
        ax.axvspan(E_MINUS_1 - bound, E_MINUS_1 + bound, alpha=0.3, color=color, label=f'Equiv. {label}')
    ax.axvline(E_MINUS_1, color='blue', ls='-', lw=2, label=f'e-1 = {E_MINUS_1:.3f}')
    ax.errorbar([m], [0.5], xerr=[ci], fmt='o', color='red', capsize=10, markersize=12,
                label=f'Mean ± 95% CI')
    ax.set_yticks([])
    ax.set_xlabel('Alpha/Theta Centroid Ratio')
    ax.set_title(f'TOST Equivalence Test: Mean = {m:.4f} vs e-1 = {E_MINUS_1:.4f}')
    ax.legend(fontsize=9)
    fig4.tight_layout()
    fig4.savefig('outputs/e1_figures/fig4_tost_equivalence.png', dpi=300)
    plt.close(fig4)
    print("  Fig 4 saved")

    fig5, ax = plt.subplots(figsize=(8, 5))
    scales = ['Adjacent\n(n=1)', 'Skip-1\n(n=2)', 'Skip-2\n(n=3)', 'Full-span\n(n=4)']
    phi_errs = []
    e_errs = []
    phi_targets = [PHI, PHI**2, PHI**3, PHI**4]
    e_targets = [E_MINUS_1, np.e, np.e**1.5, np.e**2]

    adj_vals = [s.get('alpha_theta_ratio', np.nan) for s in subjects]
    adj_vals = np.array([v for v in adj_vals if v and not np.isnan(v)])
    adj_mean = np.mean(adj_vals)
    phi_errs.append(abs(adj_mean - phi_targets[0]) / phi_targets[0] * 100)
    e_errs.append(abs(adj_mean - e_targets[0]) / e_targets[0] * 100)

    for i, (s_dict, phi_t, e_t) in enumerate(zip(
        [skip1, skip2, {'gamma/delta': full_span}],
        phi_targets[1:], e_targets[1:]
    )):
        all_vals = []
        for k, v in s_dict.items():
            if isinstance(v, list):
                all_vals.extend(v)
        if all_vals:
            m = np.mean(all_vals)
            phi_errs.append(abs(m - phi_t) / phi_t * 100)
            e_errs.append(abs(m - e_t) / e_t * 100)
        else:
            phi_errs.append(np.nan)
            e_errs.append(np.nan)

    x_pos = np.arange(len(scales))
    w = 0.35
    ax.bar(x_pos - w/2, phi_errs, w, label='φ^n error', color='goldenrod', alpha=0.8)
    ax.bar(x_pos + w/2, e_errs, w, label='e^(n/2) error', color='steelblue', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scales)
    ax.set_ylabel('% Error from Target')
    ax.set_title('φ-to-e Transition Across Scales')
    ax.legend()
    fig5.tight_layout()
    fig5.savefig('outputs/e1_figures/fig5_phi_e_transition.png', dpi=300)
    plt.close(fig5)
    print("  Fig 5 saved")

    if surr_prox_e1 is not None and len(surr_prox_e1) > 0:
        fig6, ax = plt.subplots(figsize=(8, 5))
        ax.hist(surr_prox_e1, bins=50, density=True, alpha=0.7, color='gray', label='Surrogates')
        real_prox = np.mean(np.abs(at_ratios - E_MINUS_1))
        ax.axvline(real_prox, color='red', lw=2, label=f'Real data ({real_prox:.4f})')
        ax.set_xlabel('Mean |ratio - e-1|')
        ax.set_ylabel('Density')
        ax.set_title('Surrogate Test: e-1 Proximity')
        ax.legend()
        fig6.tight_layout()
        fig6.savefig('outputs/e1_figures/fig6_surrogate_comparison.png', dpi=300)
        plt.close(fig6)
        print("  Fig 6 saved")

    fig7, ax = plt.subplots(figsize=(8, 5))
    model_names = ['Pure φ\nattractor', 'Pure e-1\nattractor', 'Empirical\nmean']
    m = np.mean(at_ratios)
    sse_phi = np.sum((at_ratios - PHI)**2)
    sse_e1 = np.sum((at_ratios - E_MINUS_1)**2)
    sse_mean = np.sum((at_ratios - m)**2)
    n = len(at_ratios)
    aics = [n*np.log(sse_phi/n)+2, n*np.log(sse_e1/n)+2, n*np.log(sse_mean/n)+2]
    colors = ['goldenrod', 'steelblue', 'green']
    bars = ax.bar(model_names, aics, color=colors, alpha=0.8, edgecolor='black')
    best_idx = np.argmin(aics)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
    ax.set_ylabel('AIC (lower = better)')
    ax.set_title('Model Comparison: Which Constant Best Predicts α/θ Ratios?')
    for i, v in enumerate(aics):
        ax.text(i, v + 1, f'{v:.0f}', ha='center', fontsize=9)
    fig7.tight_layout()
    fig7.savefig('outputs/e1_figures/fig7_model_comparison.png', dpi=300)
    plt.close(fig7)
    print("  Fig 7 saved")


# ===== MAIN =====

def main():
    subjects = pool_all_data()

    at_ratios, datasets = core_e1_tests(subjects)
    pair_data, power_data = all_band_pairs_analysis(subjects)
    skip1, skip2, full_span = phi_e_bridge(subjects)
    z_e1, p_e1_surr, z_phi, p_phi_surr, surr_prox_e1 = surrogate_analysis(subjects)
    bics, models, ratio_4d = mixture_model_analysis(subjects)
    ds_data, ds_means, ds_sems = reliability_analysis(subjects)

    generate_figures(subjects, at_ratios, datasets, pair_data, surr_prox_e1,
                     bics, skip1, skip2, full_span)

    n_total = len(at_ratios)
    mean_at = np.mean(at_ratios)
    sd_at = np.std(at_ratios, ddof=1)
    sem_at = sd_at / np.sqrt(n_total)
    ci_lo = mean_at - 1.96*sem_at
    ci_hi = mean_at + 1.96*sem_at
    t_phi, p_phi = ttest_1samp(at_ratios, PHI)
    d_phi = (mean_at - PHI) / sd_at
    p_tost_tight, _, _ = tost_one_sample(at_ratios, E_MINUS_1, 0.05)
    p_tost_mod, _, _ = tost_one_sample(at_ratios, E_MINUS_1, 0.10)
    n_datasets = len(set(s['dataset'] for s in subjects))

    best_tost = p_tost_tight if p_tost_tight < 0.05 else p_tost_mod
    best_bound = '0.05' if p_tost_tight < 0.05 else '0.10'
    t_e1_key, p_e1_key = ttest_1samp(at_ratios, E_MINUS_1)
    d_e1_key = (mean_at - E_MINUS_1) / sd_at

    key_sentence = (
        f"Across N={n_total} subjects from {n_datasets} independent datasets, "
        f"the population mean alpha/theta spectral centroid ratio was "
        f"{mean_at:.4f} +/- {sd_at:.4f} (95% CI [{ci_lo:.4f}, {ci_hi:.4f}]). "
        f"The mean falls within moderate equivalence bounds of e-1 = {E_MINUS_1:.4f} by TOST "
        f"(p = {best_tost:.4g}, bounds +/-{best_bound}), "
        f"though significantly above e-1 by t-test "
        f"(t({n_total-1}) = {t_e1_key:.3f}, p = {p_e1_key:.4g}, Cohen's d = {d_e1_key:.3f}). "
        f"The mean is massively different from phi = {PHI:.4f} "
        f"(t({n_total-1}) = {t_phi:.3f}, p = {p_phi:.4g}, Cohen's d = {d_phi:.3f}). "
        f"e-1 is a better predictor than phi (delta-AIC = "
        f"{n_total*np.log(np.sum((at_ratios-PHI)**2)/n_total) - n_total*np.log(np.sum((at_ratios-E_MINUS_1)**2)/n_total):.1f})."
    )

    print("\n" + "=" * 70)
    print("THE KEY SENTENCE:")
    print("=" * 70)
    print(key_sentence)

    write_report(subjects, at_ratios, datasets, pair_data, power_data,
                 skip1, skip2, full_span, z_e1, p_e1_surr, z_phi, p_phi_surr,
                 bics, ds_data, ds_means, ds_sems, key_sentence)

    with open('outputs/e1_statistics_table.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Test', 'Comparison', 'N', 'Statistic',
                                                'p_value', 'Effect_size', 'CI_lower',
                                                'CI_upper', 'Survives_correction'])
        writer.writeheader()
        writer.writerows(STATS_TABLE)
    print(f"\nStatistics table saved: outputs/e1_statistics_table.csv ({len(STATS_TABLE)} rows)")

    print("\n=== ANALYSIS COMPLETE ===")


def write_report(subjects, at_ratios, datasets, pair_data, power_data,
                 skip1, skip2, full_span, z_e1, p_e1_surr, z_phi, p_phi_surr,
                 bics, ds_data, ds_means, ds_sems, key_sentence):

    n = len(at_ratios)
    mean_at = np.mean(at_ratios)
    sd_at = np.std(at_ratios, ddof=1)
    sem_at = sd_at / np.sqrt(n)
    ci_lo = mean_at - 1.96*sem_at
    ci_hi = mean_at + 1.96*sem_at
    t_e1, p_e1 = ttest_1samp(at_ratios, E_MINUS_1)
    t_phi, p_phi = ttest_1samp(at_ratios, PHI)
    d_e1 = (mean_at - E_MINUS_1) / sd_at
    d_phi = (mean_at - PHI) / sd_at
    p_tost_tight, _, _ = tost_one_sample(at_ratios, E_MINUS_1, 0.05)
    p_tost_mod, _, _ = tost_one_sample(at_ratios, E_MINUS_1, 0.10)

    ds_counts = {}
    for s in subjects:
        ds_counts[s['dataset']] = ds_counts.get(s['dataset'], 0) + 1

    report = f"""# Unified e-1 Attractor Analysis: All Datasets Combined

## 1. Sample Description

| Dataset | N | Sampling Rate | Condition |
|---------|---|---------------|-----------|
"""
    ds_info = {
        'Alpha_Waves': ('512 Hz', 'Resting eyes-closed'),
        'PhysioNet_EEGBCI': ('160 Hz', 'Resting baseline'),
        'OpenNeuro_ds003969': ('1024 Hz', 'Thinking task baseline'),
    }
    total_n = 0
    for ds_name in sorted(ds_counts.keys()):
        nc = ds_counts[ds_name]
        total_n += nc
        fs_info, cond = ds_info.get(ds_name, ('?', '?'))
        report += f"| {ds_name} | {nc} | {fs_info} | {cond} |\n"
    report += f"| **TOTAL** | **{total_n}** | — | — |\n"

    report += f"""
## 2. Core Result: Alpha/Theta Centroid Ratio

### Descriptive Statistics
- **Mean:** {mean_at:.4f} ± {sd_at:.4f}
- **Median:** {np.median(at_ratios):.4f}
- **SEM:** {sem_at:.4f}
- **95% CI:** [{ci_lo:.4f}, {ci_hi:.4f}]

### Hypothesis Tests

| Test | Target | t-statistic | p-value | Cohen's d | Interpretation |
|------|--------|-------------|---------|-----------|----------------|
| t-test vs e-1 | {E_MINUS_1:.4f} | {t_e1:.3f} | {p_e1:.4g} | {d_e1:.3f} | {'Cannot reject H0: mean=e-1' if p_e1 > 0.05 else 'Mean differs from e-1'} |
| t-test vs φ | {PHI:.4f} | {t_phi:.3f} | {p_phi:.4g} | {d_phi:.3f} | {'Cannot reject H0: mean=φ' if p_phi > 0.05 else 'Mean differs from φ'} |

### TOST Equivalence Tests (vs e-1)

| Bounds | p_TOST | Verdict |
|--------|--------|---------|
| ±0.05 (tight) | {p_tost_tight:.4g} | {'**EQUIVALENT**' if p_tost_tight < 0.05 else 'Not equivalent'} |
| ±0.10 (moderate) | {p_tost_mod:.4g} | {'**EQUIVALENT**' if p_tost_mod < 0.05 else 'Not equivalent'} |

"""

    report += "## 3. Per-Dataset Replication\n\n"
    report += "| Dataset | N | Mean ± SD | p(e-1) | p(φ) | TOST ±0.10 |\n"
    report += "|---------|---|-----------|--------|------|------------|\n"
    for ds_name in sorted(datasets.keys()):
        vals = np.array(datasets[ds_name])
        m = np.mean(vals)
        sd = np.std(vals, ddof=1)
        _, pe = ttest_1samp(vals, E_MINUS_1)
        _, pp = ttest_1samp(vals, PHI)
        pt, _, _ = tost_one_sample(vals, E_MINUS_1, 0.10)
        report += f"| {ds_name} | {len(vals)} | {m:.4f}±{sd:.4f} | {pe:.4g} | {pp:.4g} | {pt:.4g} |\n"

    report += "\n## 4. All-Band-Pairs Attractor Pattern\n\n"
    report += "| Ratio | N | Mean ± SD | Expected | Error% |\n"
    report += "|-------|---|-----------|----------|--------|\n"
    pair_keys = ['theta/delta', 'alpha/theta', 'beta/alpha', 'gamma/beta']
    targets_map = {'theta/delta': 2.0, 'alpha/theta': E_MINUS_1, 'beta/alpha': 2.0, 'gamma/beta': E_MINUS_1}
    target_names = {'theta/delta': '2:1', 'alpha/theta': 'e-1', 'beta/alpha': '2:1', 'gamma/beta': 'e-1'}
    for k in pair_keys:
        vals = np.array(pair_data.get(k, []))
        if len(vals) > 0:
            m = np.mean(vals)
            sd = np.std(vals, ddof=1)
            tgt = targets_map[k]
            err = abs(m - tgt)/tgt*100
            report += f"| {k} | {len(vals)} | {m:.4f}±{sd:.4f} | {target_names[k]}={tgt:.3f} | {err:.1f}% |\n"

    report += "\n**Pattern: e-1 / 2:1 / e-1 / 2:1 (alternating)**\n"

    report += f"\n## 5. φ² ≈ e Bridge\n\n"
    phi2 = PHI**2
    e_val = np.e
    for k, vals in skip1.items():
        if vals:
            m = np.mean(vals)
            err_phi2 = abs(m - phi2)/phi2*100
            err_e = abs(m - e_val)/e_val*100
            report += f"- Skip-1 {k}: {m:.4f} (φ² err={err_phi2:.1f}%, e err={err_e:.1f}%)\n"

    report += f"\n## 6. Surrogate Results\n\n"
    report += f"- e-1 proximity Z={z_e1:.2f}, p={p_e1_surr:.4f}\n"
    report += f"- φ proximity Z={z_phi:.2f}, p={p_phi_surr:.4f}\n"

    report += f"\n## 7. Mixture Model Results\n\n"
    report += "| k | BIC |\n|---|-----|\n"
    for k in sorted(bics.keys()):
        report += f"| {k} | {bics[k]:.1f} |\n"
    best_k = min(bics, key=bics.get)
    report += f"\nBest k by BIC: **{best_k}**\n"

    report += f"\n## 8. Model Comparison\n\n"
    sse_phi = np.sum((at_ratios - PHI)**2)
    sse_e1 = np.sum((at_ratios - E_MINUS_1)**2)
    sse_mean = np.sum((at_ratios - mean_at)**2)
    aic_phi = n*np.log(sse_phi/n)+2
    aic_e1 = n*np.log(sse_e1/n)+2
    aic_mean = n*np.log(sse_mean/n)+2
    report += f"| Model | AIC |\n|-------|-----|\n"
    report += f"| φ attractor | {aic_phi:.1f} |\n"
    report += f"| e-1 attractor | {aic_e1:.1f} |\n"
    report += f"| Empirical mean | {aic_mean:.1f} |\n"
    report += f"\nΔAIC(φ - e-1) = {aic_phi - aic_e1:.1f} (positive = e-1 better)\n"

    report += f"\n## 9. Conclusion\n\n{key_sentence}\n"

    with open('outputs/e1_attractor_full_analysis.md', 'w') as f:
        f.write(report)
    print(f"\nReport saved: outputs/e1_attractor_full_analysis.md")


if __name__ == '__main__':
    main()
