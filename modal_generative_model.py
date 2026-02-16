#!/usr/bin/env python3
"""
Generative Model: Exponential Oscillator Network → e-1 Spectral Spacing
Runs on Modal for fast parallel processing.

4 experiments:
  1. Default exponential Wilson-Cowan → what comes out?
  2. Parameter sweep over tau_scale
  3. Exponential vs Linear vs Random vs Uniform vs Inverse (critical control)
  4. Kuramoto oscillators with exponential natural frequencies
"""
import modal
import json
import os

app = modal.App("eeg-generative-model")

sim_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "scipy")
)

N_POPS = 5


@app.function(image=sim_image, timeout=300, retries=1)
def run_wilson_cowan_sim(seed: int, tau_base: float, tau_scale: float,
                         coupling_decay: float, coupling_strength: float,
                         noise_level: float, T: float = 30.0,
                         dt: float = 0.001, fs_out: int = 256) -> dict:
    import numpy as np
    from scipy.signal import welch
    from scipy.integrate import solve_ivp

    np.random.seed(seed)
    N = 5

    taus_e = np.array([tau_base * np.exp(k * tau_scale) for k in range(N)])
    taus_i = taus_e * 1.5

    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                W[i, j] = coupling_strength * np.exp(-coupling_decay * abs(i - j))

    a_ee, a_ei, a_ie, a_ii = 10.0, 12.0, 8.0, 3.0
    P = np.linspace(2.0, 6.0, N)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def dynamics(t, y):
        E = y[:N]
        I = y[N:]
        coupling_E = W @ E
        dE = (-E + sigmoid(a_ee * E - a_ei * I + P + coupling_E)) / taus_e
        dI = (-I + sigmoid(a_ie * E - a_ii * I)) / taus_i
        noise = noise_level * np.random.randn(2 * N) / np.sqrt(taus_e.mean())
        return np.concatenate([dE, dI]) + noise

    y0 = np.random.rand(2 * N) * 0.5
    t_eval = np.arange(0, T, dt)

    try:
        sol = solve_ivp(dynamics, (0, T), y0, t_eval=t_eval, method='RK45', max_step=dt)
        eeg = np.sum(sol.y[:N], axis=0)
        ds_factor = max(1, int(1.0 / (dt * fs_out)))
        eeg_ds = eeg[::ds_factor]

        freqs, psd = welch(eeg_ds, fs=fs_out, nperseg=min(4 * fs_out, len(eeg_ds)))

        bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
                 'beta': (13, 30), 'gamma': (30, 45)}
        centroids = {}
        for name, (fmin, fmax) in bands.items():
            mask = (freqs >= fmin) & (freqs <= fmax)
            if mask.sum() > 0 and psd[mask].sum() > 0:
                centroids[name] = float(np.average(freqs[mask], weights=psd[mask]))
            else:
                centroids[name] = None

        band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        ratios = {}
        for i in range(len(band_names) - 1):
            key = f"{band_names[i + 1]}/{band_names[i]}"
            c1 = centroids.get(band_names[i + 1])
            c0 = centroids.get(band_names[i])
            if c1 and c0 and c0 > 0:
                ratios[key] = float(c1 / c0)
            else:
                ratios[key] = None

        return {
            'status': 'success', 'seed': seed,
            'tau_scale': tau_scale,
            'centroids': centroids, 'ratios': ratios,
            'alpha_theta': ratios.get('alpha/theta'),
            'taus_e': taus_e.tolist(),
        }
    except Exception as e:
        return {'status': 'error', 'seed': seed, 'error': str(e)}


@app.function(image=sim_image, timeout=300, retries=1)
def run_custom_tau_sim(seed: int, taus_e_list: list, condition_name: str,
                       coupling_decay: float = 0.3, coupling_strength: float = 0.5,
                       noise_level: float = 0.02, T: float = 20.0,
                       dt: float = 0.001, fs_out: int = 250) -> dict:
    import numpy as np
    from scipy.signal import welch
    from scipy.integrate import solve_ivp

    np.random.seed(seed)
    N = 5
    taus_e = np.array(taus_e_list)
    taus_i = taus_e * 1.5

    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                W[i, j] = coupling_strength * np.exp(-coupling_decay * abs(i - j))

    a_ee, a_ei, a_ie, a_ii = 10.0, 12.0, 8.0, 3.0
    P = np.linspace(2.0, 6.0, N)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def dynamics(t, y):
        E = y[:N]
        I = y[N:]
        coupling_E = W @ E
        dE = (-E + sigmoid(a_ee * E - a_ei * I + P + coupling_E)) / taus_e
        dI = (-I + sigmoid(a_ie * E - a_ii * I)) / taus_i
        noise = noise_level * np.random.randn(2 * N) / np.sqrt(taus_e.mean())
        return np.concatenate([dE, dI]) + noise

    y0 = np.random.rand(2 * N) * 0.5
    t_eval = np.arange(0, T, dt)

    try:
        sol = solve_ivp(dynamics, (0, T), y0, t_eval=t_eval, method='RK45', max_step=dt)
        eeg = np.sum(sol.y[:N], axis=0)
        ds_factor = max(1, int(1.0 / (dt * fs_out)))
        eeg_ds = eeg[::ds_factor]

        freqs, psd = welch(eeg_ds, fs=fs_out, nperseg=min(4 * fs_out, len(eeg_ds)))

        bands = {'theta': (4, 8), 'alpha': (8, 13)}
        centroids = {}
        for name, (fmin, fmax) in bands.items():
            mask = (freqs >= fmin) & (freqs <= fmax)
            if mask.sum() > 0 and psd[mask].sum() > 0:
                centroids[name] = float(np.average(freqs[mask], weights=psd[mask]))

        at = None
        if 'alpha' in centroids and 'theta' in centroids and centroids['theta'] > 0:
            at = centroids['alpha'] / centroids['theta']
            if not (1.0 < at < 3.0):
                at = None

        return {
            'status': 'success', 'seed': seed,
            'condition': condition_name,
            'alpha_theta': at,
            'centroids': centroids,
            'taus_e': taus_e_list,
        }
    except Exception as e:
        return {'status': 'error', 'seed': seed, 'condition': condition_name, 'error': str(e)}


@app.function(image=sim_image, timeout=300, retries=1)
def run_kuramoto_sim(seed: int, n_oscillators: int = 100, T: float = 60.0,
                     dt: float = 0.001, K: float = 2.0,
                     f_base: float = 2.0, f_scale: float = 0.6,
                     freq_spacing: str = 'exponential') -> dict:
    import numpy as np
    from scipy.signal import welch

    np.random.seed(seed)
    n_per_band = n_oscillators // 5

    if freq_spacing == 'exponential':
        band_centers = np.array([f_base * np.exp(k * f_scale) for k in range(5)])
    elif freq_spacing == 'linear':
        band_centers = np.linspace(f_base, f_base * np.exp(4 * f_scale), 5)
    elif freq_spacing == 'uniform':
        band_centers = np.ones(5) * np.mean([f_base, f_base * np.exp(4 * f_scale)])
    else:
        band_centers = np.sort(np.random.uniform(f_base, f_base * np.exp(4 * f_scale), 5))

    omega = np.zeros(n_oscillators)
    for i in range(5):
        start = i * n_per_band
        end = start + n_per_band
        omega[start:end] = band_centers[i] + np.random.randn(n_per_band) * band_centers[i] * 0.1
    omega *= 2 * np.pi

    n_steps = int(T / dt)
    theta = np.random.uniform(0, 2 * np.pi, n_oscillators)
    fs_record = 256
    record_every = max(1, int(1.0 / (dt * fs_record)))
    eeg = []

    for step in range(n_steps):
        mean_field = np.mean(np.exp(1j * theta))
        r = np.abs(mean_field)
        psi = np.angle(mean_field)
        dtheta = omega + K * r * np.sin(psi - theta) + 0.1 * np.random.randn(n_oscillators)
        theta += dtheta * dt
        if step % record_every == 0:
            eeg.append(float(np.sum(np.cos(theta))))

    eeg = np.array(eeg)
    freqs, psd = welch(eeg, fs=fs_record, nperseg=min(4 * fs_record, len(eeg)))

    bands_def = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
                 'beta': (13, 30), 'gamma': (30, 45)}
    centroids = {}
    for name, (fmin, fmax) in bands_def.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        if mask.sum() > 0 and psd[mask].sum() > 0:
            centroids[name] = float(np.average(freqs[mask], weights=psd[mask]))

    at = None
    if 'alpha' in centroids and 'theta' in centroids and centroids['theta'] > 0:
        at = centroids['alpha'] / centroids['theta']
        if not (1.0 < at < 3.0):
            at = None

    return {
        'status': 'success', 'seed': seed,
        'freq_spacing': freq_spacing,
        'band_centers': band_centers.tolist(),
        'centroids': centroids,
        'alpha_theta': at,
    }


@app.local_entrypoint()
def main():
    import numpy as np

    os.makedirs('outputs', exist_ok=True)
    all_results = {}

    print("=" * 60)
    print("EXPERIMENT 1: Default exponential oscillator network (N=50)")
    print("=" * 60)

    default_params = {
        'tau_base': 0.08, 'tau_scale': 0.5,
        'coupling_decay': 0.3, 'coupling_strength': 0.5,
        'noise_level': 0.02,
    }

    exp1_futures = []
    for seed in range(50):
        exp1_futures.append(
            run_wilson_cowan_sim.spawn(
                seed=seed, T=30.0, **default_params
            )
        )

    exp1_results = [f.get() for f in exp1_futures]
    exp1_at = [r['alpha_theta'] for r in exp1_results
               if r['status'] == 'success' and r.get('alpha_theta') is not None]

    if exp1_at:
        m = np.mean(exp1_at)
        print(f"  α/θ = {m:.4f} ± {np.std(exp1_at):.4f} (N={len(exp1_at)})")
        print(f"  Δe-1 = {abs(m - 1.7183):.4f}, Δφ = {abs(m - 1.618):.4f}")

    all_results['exp1'] = {
        'params': default_params,
        'ratios': exp1_at,
        'full_results': [r for r in exp1_results if r['status'] == 'success'],
    }

    print(f"\n{'='*60}")
    print("EXPERIMENT 2: Parameter sweep over tau_scale (15 values × 20 sims)")
    print("=" * 60)

    tau_scales = list(np.linspace(0.1, 1.5, 15))
    exp2_futures = []
    for ts in tau_scales:
        for seed in range(20):
            exp2_futures.append(
                run_wilson_cowan_sim.spawn(
                    seed=seed + 1000, tau_base=0.08, tau_scale=ts,
                    coupling_decay=0.3, coupling_strength=0.5,
                    noise_level=0.02, T=20.0
                )
            )

    exp2_results = [f.get() for f in exp2_futures]

    sweep_data = {}
    for r in exp2_results:
        if r['status'] == 'success' and r.get('alpha_theta') is not None:
            ts = round(r['tau_scale'], 4)
            sweep_data.setdefault(ts, []).append(r['alpha_theta'])

    exp2_summary = []
    for ts in sorted(sweep_data.keys()):
        vals = sweep_data[ts]
        if len(vals) >= 5:
            m = np.mean(vals)
            entry = {
                'tau_scale': ts, 'mean': m, 'sd': float(np.std(vals)),
                'dist_e1': abs(m - 1.7183), 'dist_phi': abs(m - 1.618),
                'n': len(vals),
            }
            exp2_summary.append(entry)
            print(f"  τ_scale={ts:.2f}: α/θ={m:.4f}±{np.std(vals):.4f} "
                  f"(Δe-1={abs(m-1.7183):.4f}, Δφ={abs(m-1.618):.4f})")

    if exp2_summary:
        best = min(exp2_summary, key=lambda x: x['dist_e1'])
        print(f"\n  Closest to e-1: τ_scale={best['tau_scale']:.2f} → α/θ={best['mean']:.4f}")

    all_results['exp2'] = {'sweep': exp2_summary}

    print(f"\n{'='*60}")
    print("EXPERIMENT 3: CRITICAL CONTROL — Exponential vs Others (5 conditions × 30 sims)")
    print("=" * 60)

    np.random.seed(42)
    conditions = {
        'Exponential': [0.08 * np.exp(k * 0.5) for k in range(5)],
        'Linear': list(np.linspace(0.08, 0.08 * np.exp(4 * 0.5), 5)),
        'Random': sorted(np.random.uniform(0.05, 1.0, 5).tolist()),
        'Uniform': [0.3] * 5,
        'Inverse_exp': [0.08 * np.exp((4 - k) * 0.5) for k in range(5)],
    }

    exp3_futures = []
    for cond_name, taus in conditions.items():
        for seed in range(30):
            exp3_futures.append(
                run_custom_tau_sim.spawn(
                    seed=seed + 2000, taus_e_list=taus,
                    condition_name=cond_name, T=20.0
                )
            )

    exp3_results = [f.get() for f in exp3_futures]

    exp3_summary = {}
    for r in exp3_results:
        if r['status'] == 'success' and r.get('alpha_theta') is not None:
            cond = r['condition']
            exp3_summary.setdefault(cond, []).append(r['alpha_theta'])

    for cond_name in ['Exponential', 'Linear', 'Random', 'Uniform', 'Inverse_exp']:
        vals = exp3_summary.get(cond_name, [])
        taus = conditions[cond_name]
        if len(vals) >= 5:
            m = np.mean(vals)
            print(f"  {cond_name:15s} (τ={[f'{t:.3f}' for t in taus]}): "
                  f"α/θ={m:.4f}±{np.std(vals):.4f} | "
                  f"Δe-1={abs(m-1.7183):.4f} | Δφ={abs(m-1.618):.4f} | N={len(vals)}")
        else:
            print(f"  {cond_name:15s}: insufficient valid sims ({len(vals)})")

    all_results['exp3'] = {
        cond: {'ratios': vals, 'taus': conditions[cond]}
        for cond, vals in exp3_summary.items()
    }

    print(f"\n{'='*60}")
    print("EXPERIMENT 4: Kuramoto oscillators (3 freq spacings × 50 sims)")
    print("=" * 60)

    exp4_futures = []
    for spacing in ['exponential', 'linear', 'random']:
        for seed in range(50):
            exp4_futures.append(
                run_kuramoto_sim.spawn(
                    seed=seed + 3000, freq_spacing=spacing,
                    f_base=2.0, f_scale=0.6, T=60.0
                )
            )

    exp4_results = [f.get() for f in exp4_futures]

    exp4_summary = {}
    for r in exp4_results:
        if r['status'] == 'success' and r.get('alpha_theta') is not None:
            sp = r['freq_spacing']
            exp4_summary.setdefault(sp, []).append(r['alpha_theta'])

    for sp in ['exponential', 'linear', 'random']:
        vals = exp4_summary.get(sp, [])
        if len(vals) >= 5:
            m = np.mean(vals)
            print(f"  Kuramoto ({sp:12s}): α/θ={m:.4f}±{np.std(vals):.4f} | "
                  f"Δe-1={abs(m-1.7183):.4f} | Δφ={abs(m-1.618):.4f} | N={len(vals)}")
        else:
            print(f"  Kuramoto ({sp:12s}): insufficient ({len(vals)})")

    all_results['exp4'] = {
        sp: {'ratios': vals}
        for sp, vals in exp4_summary.items()
    }

    out_path = 'outputs/generative_model_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"Total simulations: {len(exp1_results) + len(exp2_results) + len(exp3_results) + len(exp4_results)}")
