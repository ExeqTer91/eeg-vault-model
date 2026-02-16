import numpy as np
import os
import glob
import json
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.stats import pearsonr, ttest_1samp

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

def load_all_subjects():
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
                data = preprocess_mat(data, FS_ALPHA)
                subjects.append({
                    'name': subj_name,
                    'data': data,
                    'fs': FS_ALPHA,
                    'source': 'alpha_waves',
                    'file': fpath
                })
            except Exception as e:
                print(f"  Skip {fpath}: {e}")
    print(f"Loaded {len(subjects)} subjects from Alpha-Waves dataset (all 512 Hz)")
    return subjects

def compute_subject_psd(subj):
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
    return freqs, avg_psd

def compute_all_band_centroids(freqs, psd):
    delta_c = spectral_centroid(freqs, psd, 1, 4)
    theta_c = spectral_centroid(freqs, psd, 4, 8)
    alpha_c = spectral_centroid(freqs, psd, 8, 13)
    beta_c = spectral_centroid(freqs, psd, 13, 30)
    gamma_c = spectral_centroid(freqs, psd, 30, 45)
    return {'delta': delta_c, 'theta': theta_c, 'alpha': alpha_c, 'beta': beta_c, 'gamma': gamma_c}

def load_and_compute():
    subjects = load_all_subjects()
    n = len(subjects)

    all_freqs = []
    all_psds = []
    all_centroids = []
    all_ratios = np.zeros(n)
    all_pcis = np.zeros(n)

    for i, subj in enumerate(subjects):
        freqs, avg_psd = compute_subject_psd(subj)
        all_freqs.append(freqs)
        all_psds.append(avg_psd)
        centroids = compute_all_band_centroids(freqs, avg_psd)
        all_centroids.append(centroids)
        ratio = centroids['alpha'] / centroids['theta']
        all_ratios[i] = ratio
        all_pcis[i] = compute_pci(ratio)

    return subjects, all_freqs, all_psds, all_centroids, all_ratios, all_pcis

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
