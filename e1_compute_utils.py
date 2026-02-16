import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch

PHI = 1.6180339887
E_MINUS_1 = float(np.e - 1)
HARMONIC = 2.0

BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),
}
BAND_ORDER = ['delta', 'theta', 'alpha', 'beta', 'gamma']
ADJACENT_PAIRS = [('theta', 'delta'), ('alpha', 'theta'), ('beta', 'alpha'), ('gamma', 'beta')]


def preprocess_eeg(data, fs, bp_low=1.0, bp_high=45.0, notch_freq=None):
    data = data - np.mean(data, axis=-1, keepdims=True)
    nyq = fs / 2.0
    if bp_high >= nyq:
        bp_high = nyq - 1
    b, a = butter(4, [bp_low / nyq, bp_high / nyq], btype='band')
    for ch in range(data.shape[0]):
        data[ch] = filtfilt(b, a, data[ch])
    if notch_freq and notch_freq < nyq:
        b_n, a_n = iirnotch(notch_freq, Q=30, fs=fs)
        for ch in range(data.shape[0]):
            data[ch] = filtfilt(b_n, a_n, data[ch])
    return data


def compute_avg_psd(data, fs, nperseg=None):
    if nperseg is None:
        nperseg = min(int(4 * fs), data.shape[1])
    avg_psd = None
    for ch in range(data.shape[0]):
        freqs, psd = welch(data[ch], fs=fs, nperseg=nperseg)
        if avg_psd is None:
            avg_psd = psd.copy()
        else:
            avg_psd += psd
    avg_psd /= data.shape[0]
    return freqs, avg_psd


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


def extract_subject_features(freqs, psd, subject_id, dataset, condition='rest', fs=None):
    centroids = {}
    powers = {}
    for band, (lo, hi) in BANDS.items():
        centroids[band] = spectral_centroid(freqs, psd, lo, hi)
        powers[band] = bandpower(freqs, psd, lo, hi)

    ratios = {}
    for num_band, den_band in ADJACENT_PAIRS:
        key = f"{num_band}/{den_band}"
        if centroids[den_band] > 0:
            ratios[key] = centroids[num_band] / centroids[den_band]
        else:
            ratios[key] = float('nan')

    skip1_ratios = {}
    skip1_pairs = [('alpha', 'delta'), ('beta', 'theta'), ('gamma', 'alpha')]
    for num_band, den_band in skip1_pairs:
        key = f"{num_band}/{den_band}"
        if centroids[den_band] > 0:
            skip1_ratios[key] = centroids[num_band] / centroids[den_band]
        else:
            skip1_ratios[key] = float('nan')

    skip2_ratios = {}
    skip2_pairs = [('beta', 'delta'), ('gamma', 'theta')]
    for num_band, den_band in skip2_pairs:
        key = f"{num_band}/{den_band}"
        if centroids[den_band] > 0:
            skip2_ratios[key] = centroids[num_band] / centroids[den_band]
        else:
            skip2_ratios[key] = float('nan')

    full_span = centroids['gamma'] / centroids['delta'] if centroids['delta'] > 0 else float('nan')

    alpha_theta = ratios.get('alpha/theta', float('nan'))
    pci = compute_pci(alpha_theta)

    power_ratios = {}
    for num_band, den_band in ADJACENT_PAIRS:
        key = f"P_{num_band}/P_{den_band}"
        if powers[den_band] > 0:
            power_ratios[key] = powers[num_band] / powers[den_band]
        else:
            power_ratios[key] = float('nan')

    fooof_params = run_fooof(freqs, psd)

    return {
        'subject_id': subject_id,
        'dataset': dataset,
        'condition': condition,
        'fs': fs,
        'centroids': centroids,
        'powers': powers,
        'adjacent_ratios': ratios,
        'skip1_ratios': skip1_ratios,
        'skip2_ratios': skip2_ratios,
        'full_span_ratio': full_span,
        'alpha_theta_ratio': alpha_theta,
        'pci': pci,
        'power_ratios': power_ratios,
        'fooof': fooof_params,
    }


def run_fooof(freqs, psd):
    try:
        import warnings
        warnings.filterwarnings('ignore')
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


def phase_randomize_shared(data):
    n_ch, n_samp = data.shape
    surrogate = np.zeros_like(data)
    random_phases = np.exp(2j * np.pi * np.random.random(n_samp // 2 + 1))
    random_phases[0] = 1
    if n_samp % 2 == 0:
        random_phases[-1] = np.sign(random_phases[-1].real)
    for ch in range(n_ch):
        fft_vals = np.fft.rfft(data[ch])
        amplitudes = np.abs(fft_vals)
        surrogate[ch] = np.fft.irfft(amplitudes * random_phases, n=n_samp)
    return surrogate
