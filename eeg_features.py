import numpy as np
from scipy import signal
from scipy.signal import hilbert, coherence as scipy_coherence
from itertools import combinations


BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (13, 30),
    'gamma': (30, 45),
}


def compute_bandpower(data, fs, bands=None, nperseg=None):
    if bands is None:
        bands = BANDS
    if nperseg is None:
        nperseg = min(int(4 * fs), data.shape[-1])

    n_channels = data.shape[0]
    bp = {name: np.zeros(n_channels) for name in bands}

    for ch in range(n_channels):
        freqs, psd = signal.welch(data[ch], fs=fs, nperseg=nperseg)
        for name, (lo, hi) in bands.items():
            idx = np.logical_and(freqs >= lo, freqs <= hi)
            bp[name][ch] = np.trapezoid(psd[idx], freqs[idx])

    return bp


def compute_psd(data, fs, nperseg=None):
    if nperseg is None:
        nperseg = min(int(4 * fs), data.shape[-1])
    freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg)
    return freqs, psd


def compute_gfp(data):
    return np.std(data, axis=0)


def bandpass_filter(data, fs, lo, hi, order=4):
    nyq = fs / 2.0
    lo_n = max(lo / nyq, 1e-5)
    hi_n = min(hi / nyq, 0.9999)
    b, a = signal.butter(order, [lo_n, hi_n], btype='band')
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered[ch] = signal.filtfilt(b, a, data[ch])
    return filtered


def compute_plv(data, fs, band=(8, 12)):
    filtered = bandpass_filter(data, fs, band[0], band[1])
    n_channels = filtered.shape[0]
    analytic = np.zeros_like(filtered, dtype=complex)
    for ch in range(n_channels):
        analytic[ch] = hilbert(filtered[ch])
    phases = np.angle(analytic)

    plv_matrix = np.ones((n_channels, n_channels))
    for i, j in combinations(range(n_channels), 2):
        phase_diff = phases[i] - phases[j]
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        plv_matrix[i, j] = plv
        plv_matrix[j, i] = plv

    return plv_matrix


def compute_coherence_matrix(data, fs, band=(8, 12), nperseg=None):
    if nperseg is None:
        nperseg = min(int(4 * fs), data.shape[-1])
    n_channels = data.shape[0]
    coh_matrix = np.ones((n_channels, n_channels))

    for i, j in combinations(range(n_channels), 2):
        freqs, cxy = scipy_coherence(data[i], data[j], fs=fs, nperseg=nperseg)
        idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        mean_coh = np.mean(cxy[idx]) if np.any(idx) else 0.0
        coh_matrix[i, j] = mean_coh
        coh_matrix[j, i] = mean_coh

    return coh_matrix


def compute_kuramoto(data, fs, band=(8, 12)):
    filtered = bandpass_filter(data, fs, band[0], band[1])
    n_channels = filtered.shape[0]
    analytic = np.zeros_like(filtered, dtype=complex)
    for ch in range(n_channels):
        analytic[ch] = hilbert(filtered[ch])
    phases = np.angle(analytic)

    r_t = np.abs(np.mean(np.exp(1j * phases), axis=0))
    mean_r = np.mean(r_t)

    return r_t, mean_r


def compute_mean_plv(plv_matrix):
    n = plv_matrix.shape[0]
    if n < 2:
        return 0.0
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return np.mean(plv_matrix[mask])
