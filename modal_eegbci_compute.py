import modal
import json
import os

app = modal.App("eeg-e1-analysis")

eeg_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "numpy",
        "scipy",
        "mne",
        "fooof",
        "matplotlib",
    )
)


@app.function(image=eeg_image, timeout=600, retries=1)
def process_eegbci_subject(subject_id: int) -> dict:
    import numpy as np
    from scipy.signal import welch, butter, filtfilt
    import warnings
    warnings.filterwarnings('ignore')
    import mne
    mne.set_log_level('ERROR')
    from mne.datasets import eegbci
    from mne.io import read_raw_edf

    PHI = 1.6180339887

    def spectral_centroid(freqs, psd, lo, hi):
        idx = (freqs >= lo) & (freqs <= hi)
        f_band = freqs[idx]
        p_band = psd[idx]
        total = np.sum(p_band)
        if total == 0:
            return (lo + hi) / 2.0
        return float(np.sum(f_band * p_band) / total)

    def bandpower(freqs, psd, lo, hi):
        idx = (freqs >= lo) & (freqs <= hi)
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        return float(np.sum(psd[idx]) * df)

    def compute_pci(ratio):
        d_phi = abs(ratio - PHI)
        d_harm = abs(ratio - 2.0)
        if d_phi < 1e-10:
            return 10.0
        return float(np.log(d_harm / d_phi))

    def run_fooof_remote(freqs, psd):
        try:
            from fooof import FOOOF
            mask = (freqs >= 1) & (freqs <= 50)
            f = freqs[mask]
            p = psd[mask]
            if len(f) < 10:
                return {'aperiodic_offset': None, 'aperiodic_slope': None, 'n_peaks': 0, 'peaks': []}
            fm = FOOOF(peak_width_limits=[1, 12], max_n_peaks=8, min_peak_height=0.05, verbose=False)
            fm.fit(f, p)
            ap = fm.aperiodic_params_
            peaks = fm.peak_params_.tolist() if len(fm.peak_params_) > 0 else []
            return {
                'aperiodic_offset': float(ap[0]),
                'aperiodic_slope': float(ap[-1]),
                'n_peaks': len(peaks),
                'peaks': peaks,
            }
        except Exception:
            return {'aperiodic_offset': None, 'aperiodic_slope': None, 'n_peaks': 0, 'peaks': []}

    BANDS = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45),
    }
    ADJACENT_PAIRS = [('theta', 'delta'), ('alpha', 'theta'), ('beta', 'alpha'), ('gamma', 'beta')]
    SKIP1_PAIRS = [('alpha', 'delta'), ('beta', 'theta'), ('gamma', 'alpha')]
    SKIP2_PAIRS = [('beta', 'delta'), ('gamma', 'theta')]

    try:
        fnames = eegbci.load_data(subject_id, [1], update_path=False, verbose=False)
        raw = read_raw_edf(fnames[0], preload=True, verbose=False)
        raw.filter(1, 45, verbose=False)
        fs = raw.info['sfreq']
        data = raw.get_data()

        nperseg = min(int(4 * fs), data.shape[1])
        avg_psd = None
        n_good = 0
        for ch in range(data.shape[0]):
            freqs, psd = welch(data[ch], fs=fs, nperseg=nperseg)
            if avg_psd is None:
                avg_psd = psd.copy()
            else:
                avg_psd += psd
            n_good += 1
        avg_psd /= n_good

        centroids = {}
        powers = {}
        for band, (lo, hi) in BANDS.items():
            centroids[band] = spectral_centroid(freqs, avg_psd, lo, hi)
            powers[band] = bandpower(freqs, avg_psd, lo, hi)

        ratios = {}
        for num_b, den_b in ADJACENT_PAIRS:
            key = f"{num_b}/{den_b}"
            ratios[key] = centroids[num_b] / centroids[den_b] if centroids[den_b] > 0 else None

        skip1 = {}
        for num_b, den_b in SKIP1_PAIRS:
            key = f"{num_b}/{den_b}"
            skip1[key] = centroids[num_b] / centroids[den_b] if centroids[den_b] > 0 else None

        skip2 = {}
        for num_b, den_b in SKIP2_PAIRS:
            key = f"{num_b}/{den_b}"
            skip2[key] = centroids[num_b] / centroids[den_b] if centroids[den_b] > 0 else None

        full_span = centroids['gamma'] / centroids['delta'] if centroids['delta'] > 0 else None

        alpha_theta = ratios.get('alpha/theta')
        pci = compute_pci(alpha_theta) if alpha_theta else None

        power_ratios = {}
        for num_b, den_b in ADJACENT_PAIRS:
            key = f"P_{num_b}/P_{den_b}"
            power_ratios[key] = powers[num_b] / powers[den_b] if powers[den_b] > 0 else None

        fooof_params = run_fooof_remote(freqs, avg_psd)

        return {
            'subject_id': f"EEGBCI_S{subject_id:03d}",
            'dataset': 'PhysioNet_EEGBCI',
            'condition': 'rest',
            'fs': float(fs),
            'n_channels': int(data.shape[0]),
            'centroids': centroids,
            'powers': powers,
            'adjacent_ratios': ratios,
            'skip1_ratios': skip1,
            'skip2_ratios': skip2,
            'full_span_ratio': full_span,
            'alpha_theta_ratio': alpha_theta,
            'pci': pci,
            'power_ratios': power_ratios,
            'fooof': fooof_params,
            'status': 'success',
        }
    except Exception as e:
        return {
            'subject_id': f"EEGBCI_S{subject_id:03d}",
            'dataset': 'PhysioNet_EEGBCI',
            'status': 'failed',
            'error': str(e),
        }


@app.local_entrypoint()
def main():
    subject_ids = list(range(1, 110))
    print(f"Processing {len(subject_ids)} EEGBCI subjects on Modal...")

    results = []
    for result in process_eegbci_subject.map(subject_ids, order_outputs=False):
        sid = result.get('subject_id', '?')
        status = result.get('status', 'unknown')
        if status == 'success':
            at = result.get('alpha_theta_ratio', 0)
            print(f"  {sid}: alpha/theta = {at:.4f}")
        else:
            print(f"  {sid}: FAILED - {result.get('error', 'unknown')}")
        results.append(result)

    successes = [r for r in results if r.get('status') == 'success']
    failures = [r for r in results if r.get('status') != 'success']
    print(f"\nDone: {len(successes)} succeeded, {len(failures)} failed")

    os.makedirs('outputs', exist_ok=True)
    outpath = 'outputs/eegbci_modal_results.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {outpath}")

    if successes:
        at_ratios = [r['alpha_theta_ratio'] for r in successes if r.get('alpha_theta_ratio')]
        import numpy as np
        arr = np.array(at_ratios)
        print(f"\nAlpha/Theta summary (N={len(arr)}):")
        print(f"  Mean: {np.mean(arr):.4f} ± {np.std(arr):.4f}")
        print(f"  Median: {np.median(arr):.4f}")
        print(f"  e-1 = {np.e - 1:.4f}")
        print(f"  phi = {(1 + np.sqrt(5))/2:.4f}")
