import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import io
import tempfile

from eeg_features import (
    compute_bandpower, compute_psd, compute_gfp,
    compute_plv, compute_coherence_matrix,
    compute_kuramoto, compute_mean_plv, BANDS, bandpass_filter
)

st.set_page_config(page_title="EEG Analysis Tool", page_icon="🧠", layout="wide")
st.title("EEG Analysis Tool")
st.markdown("Upload EEG data and compute bandpower, connectivity, and synchrony measures.")

SUPPORTED_FORMATS = {
    '.edf': 'EDF',
    '.fif': 'FIF',
    '.vhdr': 'BrainVision',
    '.set': 'EEGLAB',
    '.csv': 'CSV',
    '.mat': 'MATLAB',
}


def load_eeg_mne(filepath, fmt):
    import mne
    mne.set_log_level('ERROR')

    if fmt == 'EDF':
        raw = mne.io.read_raw_edf(filepath, preload=True)
    elif fmt == 'FIF':
        raw = mne.io.read_raw_fif(filepath, preload=True)
    elif fmt == 'BrainVision':
        raw = mne.io.read_raw_brainvision(filepath, preload=True)
    elif fmt == 'EEGLAB':
        raw = mne.io.read_raw_eeglab(filepath, preload=True)
    else:
        raise ValueError(f"Unsupported MNE format: {fmt}")

    return raw


def load_csv(filepath, fs):
    df = pd.read_csv(filepath)
    if 'time' in [c.lower() for c in df.columns]:
        time_col = [c for c in df.columns if c.lower() == 'time'][0]
        df = df.drop(columns=[time_col])
    ch_names = list(df.columns)
    data = df.values.T.astype(np.float64)
    return data, ch_names, fs


def load_mat(filepath):
    from scipy.io import loadmat
    mat = loadmat(filepath)

    data_key = None
    fs_val = None
    ch_names = None

    skip_keys = {'__header__', '__version__', '__globals__'}
    candidates = {k: v for k, v in mat.items() if k not in skip_keys}

    for k, v in candidates.items():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            if data_key is None or v.size > mat[data_key].size:
                data_key = k

    for k, v in candidates.items():
        kl = k.lower()
        if kl in ('fs', 'srate', 'sfreq', 'sampling_rate', 'sr'):
            fs_val = float(np.squeeze(v))

    if data_key is None:
        raise ValueError("No 2D data array found in .mat file.")

    data = mat[data_key].astype(np.float64)
    if data.shape[0] > data.shape[1]:
        data = data.T

    n_ch = data.shape[0]
    if ch_names is None:
        ch_names = [f"Ch{i+1}" for i in range(n_ch)]

    return data, ch_names, fs_val


def preprocess_mne(raw, notch_freq, bp_low, bp_high):
    import mne

    try:
        raw.set_montage('standard_1020', on_missing='warn')
        montage_set = True
    except Exception:
        montage_set = False
        st.warning("Could not set standard_1020 montage. Topomaps will be unavailable.")

    raw.filter(bp_low, bp_high, fir_design='firwin')

    if notch_freq:
        raw.notch_filter(notch_freq)

    raw.set_eeg_reference('average', projection=False)

    return raw, montage_set


def preprocess_array(data, fs, notch_freq, bp_low, bp_high):
    from scipy.signal import butter, filtfilt, iirnotch

    avg_ref = data - np.mean(data, axis=0, keepdims=True)
    data = avg_ref

    nyq = fs / 2.0
    lo_n = max(bp_low / nyq, 1e-5)
    hi_n = min(bp_high / nyq, 0.9999)
    b, a = butter(4, [lo_n, hi_n], btype='band')
    for ch in range(data.shape[0]):
        data[ch] = filtfilt(b, a, data[ch])

    if notch_freq:
        for nf in (notch_freq if isinstance(notch_freq, list) else [notch_freq]):
            b_n, a_n = iirnotch(nf, Q=30, fs=fs)
            for ch in range(data.shape[0]):
                data[ch] = filtfilt(b_n, a_n, data[ch])

    return data


with st.sidebar:
    st.header("Settings")

    notch_option = st.selectbox("Notch filter", ["50 Hz", "60 Hz", "Both (50 & 60)", "None"])
    bp_low = st.number_input("Bandpass low (Hz)", value=1.0, min_value=0.1, max_value=10.0, step=0.5)
    bp_high = st.number_input("Bandpass high (Hz)", value=45.0, min_value=20.0, max_value=100.0, step=5.0)
    alpha_low = st.number_input("Alpha band low (Hz)", value=8.0, min_value=4.0, max_value=15.0, step=0.5)
    alpha_high = st.number_input("Alpha band high (Hz)", value=12.0, min_value=8.0, max_value=20.0, step=0.5)

    csv_fs = st.number_input("Sampling rate (for CSV/MAT without metadata)", value=256.0, min_value=1.0, max_value=10000.0, step=1.0)

    if notch_option == "50 Hz":
        notch_freq = [50]
    elif notch_option == "60 Hz":
        notch_freq = [60]
    elif notch_option == "Both (50 & 60)":
        notch_freq = [50, 60]
    else:
        notch_freq = None


uploaded_file = st.file_uploader(
    "Upload EEG file",
    type=['edf', 'fif', 'vhdr', 'set', 'csv', 'mat'],
    help="Supported: EDF, FIF, BrainVision (.vhdr), EEGLAB (.set), CSV, MATLAB (.mat)"
)

if uploaded_file is not None:
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    fmt = SUPPORTED_FORMATS.get(ext)

    if fmt is None:
        st.error(f"Unsupported file format: {ext}")
        st.stop()

    st.info(f"Detected format: **{fmt}** | File: {uploaded_file.name}")

    with st.spinner("Loading and preprocessing data..."):
        try:
            if fmt == 'CSV':
                with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                data, ch_names, fs = load_csv(tmp_path, csv_fs)
                os.unlink(tmp_path)
                data = preprocess_array(data, fs, notch_freq, bp_low, bp_high)
                montage_set = False

            elif fmt == 'MATLAB':
                with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                data, ch_names, mat_fs = load_mat(tmp_path)
                os.unlink(tmp_path)
                fs = mat_fs if mat_fs else csv_fs
                data = preprocess_array(data, fs, notch_freq, bp_low, bp_high)
                montage_set = False

            else:
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                raw = load_eeg_mne(tmp_path, fmt)
                os.unlink(tmp_path)
                raw, montage_set = preprocess_mne(raw, notch_freq, bp_low, bp_high)
                data = raw.get_data()
                ch_names = raw.ch_names
                fs = raw.info['sfreq']

            n_channels, n_samples = data.shape
            duration = n_samples / fs
            st.success(f"Loaded: {n_channels} channels, {n_samples} samples, {fs:.1f} Hz, {duration:.1f}s duration")

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()

    alpha_band = (alpha_low, alpha_high)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Bandpower & PSD", "Connectivity (PLV)", "Coherence", "Synchrony (Kuramoto)", "Summary & Export"
    ])

    with tab1:
        st.subheader("Power Spectral Density")

        with st.spinner("Computing PSD..."):
            freqs, psd = compute_psd(data, fs)
            avg_psd = np.mean(psd, axis=0)

        fig_psd, ax_psd = plt.subplots(figsize=(10, 4))
        ax_psd.semilogy(freqs, avg_psd, color='steelblue', linewidth=1.5)
        for name, (lo, hi) in BANDS.items():
            ax_psd.axvspan(lo, hi, alpha=0.1, label=name)
        ax_psd.set_xlabel('Frequency (Hz)')
        ax_psd.set_ylabel('Power Spectral Density (V^2/Hz)')
        ax_psd.set_title('Average PSD across channels')
        ax_psd.legend(loc='upper right', fontsize=8)
        ax_psd.set_xlim(0, bp_high + 5)
        plt.tight_layout()
        st.pyplot(fig_psd)
        plt.close(fig_psd)

        st.subheader("Bandpower per Channel")
        with st.spinner("Computing bandpower..."):
            bp = compute_bandpower(data, fs)

        bp_df = pd.DataFrame(bp, index=ch_names)
        st.dataframe(bp_df.style.format("{:.6f}"), use_container_width=True)

        fig_bp, ax_bp = plt.subplots(figsize=(12, max(4, n_channels * 0.3)))
        bp_array = np.array([bp[b] for b in BANDS.keys()])
        im = ax_bp.imshow(bp_array, aspect='auto', cmap='YlOrRd')
        ax_bp.set_yticks(range(len(BANDS)))
        ax_bp.set_yticklabels(list(BANDS.keys()))
        if n_channels <= 30:
            ax_bp.set_xticks(range(n_channels))
            ax_bp.set_xticklabels(ch_names, rotation=90, fontsize=7)
        ax_bp.set_xlabel('Channel')
        ax_bp.set_title('Bandpower Heatmap')
        plt.colorbar(im, ax=ax_bp, label='Power')
        plt.tight_layout()
        st.pyplot(fig_bp)
        plt.close(fig_bp)

        if montage_set:
            st.subheader("Alpha Bandpower Topomap")
            try:
                import mne
                info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
                info.set_montage('standard_1020', on_missing='warn')
                fig_topo, ax_topo = plt.subplots(figsize=(5, 5))
                mne.viz.plot_topomap(bp['alpha'], info, axes=ax_topo, show=False)
                ax_topo.set_title('Alpha Bandpower Topomap')
                st.pyplot(fig_topo)
                plt.close(fig_topo)
            except Exception as e:
                st.warning(f"Could not generate topomap: {e}")

        st.subheader("Global Field Power (GFP)")
        gfp = compute_gfp(data)
        time_axis = np.arange(len(gfp)) / fs
        fig_gfp, ax_gfp = plt.subplots(figsize=(10, 3))
        ax_gfp.plot(time_axis, gfp, color='darkgreen', linewidth=0.5)
        ax_gfp.set_xlabel('Time (s)')
        ax_gfp.set_ylabel('GFP (std across channels)')
        ax_gfp.set_title('Global Field Power')
        plt.tight_layout()
        st.pyplot(fig_gfp)
        plt.close(fig_gfp)

    with tab2:
        st.subheader(f"Phase-Locking Value (PLV) - Alpha ({alpha_low}-{alpha_high} Hz)")

        if n_channels > 64:
            st.warning(f"PLV computation for {n_channels} channels may be slow. Consider selecting fewer channels.")

        with st.spinner("Computing PLV matrix..."):
            plv_matrix = compute_plv(data, fs, band=alpha_band)
            mean_plv = compute_mean_plv(plv_matrix)

        st.metric("Mean PLV (all pairs)", f"{mean_plv:.4f}")

        fig_plv, ax_plv = plt.subplots(figsize=(8, 7))
        im_plv = ax_plv.imshow(plv_matrix, cmap='hot', vmin=0, vmax=1, aspect='equal')
        if n_channels <= 30:
            ax_plv.set_xticks(range(n_channels))
            ax_plv.set_xticklabels(ch_names, rotation=90, fontsize=7)
            ax_plv.set_yticks(range(n_channels))
            ax_plv.set_yticklabels(ch_names, fontsize=7)
        ax_plv.set_title(f'PLV Matrix (Alpha {alpha_low}-{alpha_high} Hz)')
        plt.colorbar(im_plv, ax=ax_plv, label='PLV')
        plt.tight_layout()
        st.pyplot(fig_plv)
        plt.close(fig_plv)

    with tab3:
        st.subheader(f"Magnitude-Squared Coherence - Alpha ({alpha_low}-{alpha_high} Hz)")

        with st.spinner("Computing coherence matrix..."):
            coh_matrix = compute_coherence_matrix(data, fs, band=alpha_band)

        fig_coh, ax_coh = plt.subplots(figsize=(8, 7))
        im_coh = ax_coh.imshow(coh_matrix, cmap='viridis', vmin=0, vmax=1, aspect='equal')
        if n_channels <= 30:
            ax_coh.set_xticks(range(n_channels))
            ax_coh.set_xticklabels(ch_names, rotation=90, fontsize=7)
            ax_coh.set_yticks(range(n_channels))
            ax_coh.set_yticklabels(ch_names, fontsize=7)
        ax_coh.set_title(f'Coherence Matrix (Alpha {alpha_low}-{alpha_high} Hz)')
        plt.colorbar(im_coh, ax=ax_coh, label='Coherence')
        plt.tight_layout()
        st.pyplot(fig_coh)
        plt.close(fig_coh)

    with tab4:
        st.subheader(f"Kuramoto Order Parameter - Alpha ({alpha_low}-{alpha_high} Hz)")

        with st.spinner("Computing Kuramoto order parameter..."):
            r_t, mean_r = compute_kuramoto(data, fs, band=alpha_band)

        col1, col2 = st.columns(2)
        col1.metric("Mean Kuramoto r", f"{mean_r:.4f}")
        col2.metric("Std Kuramoto r", f"{np.std(r_t):.4f}")

        time_r = np.arange(len(r_t)) / fs
        fig_r, ax_r = plt.subplots(figsize=(10, 3))
        ax_r.plot(time_r, r_t, color='purple', linewidth=0.5, alpha=0.7)
        ax_r.axhline(mean_r, color='red', linestyle='--', label=f'Mean = {mean_r:.3f}')
        ax_r.set_xlabel('Time (s)')
        ax_r.set_ylabel('r(t)')
        ax_r.set_title('Kuramoto Order Parameter Over Time')
        ax_r.set_ylim(0, 1)
        ax_r.legend()
        plt.tight_layout()
        st.pyplot(fig_r)
        plt.close(fig_r)

        st.markdown("""
        **Interpretation:**
        - r = 0: no synchrony (random phases)
        - r = 1: perfect synchrony (all channels in phase)
        """)

    with tab5:
        st.subheader("Analysis Summary")

        results = {
            'file': uploaded_file.name,
            'format': fmt,
            'n_channels': int(n_channels),
            'n_samples': int(n_samples),
            'sampling_rate': float(fs),
            'duration_seconds': float(duration),
            'preprocessing': {
                'bandpass': [float(bp_low), float(bp_high)],
                'notch': notch_freq,
                'reference': 'average',
            },
            'alpha_band': [float(alpha_low), float(alpha_high)],
            'mean_plv': float(mean_plv),
            'mean_kuramoto_r': float(mean_r),
            'bandpower_means': {name: float(np.mean(vals)) for name, vals in bp.items()},
        }

        st.json(results)

        os.makedirs('outputs', exist_ok=True)
        with open('outputs/results.json', 'w') as f:
            json.dump(results, f, indent=2)
        bp_df.to_csv('outputs/bandpower.csv')
        st.success("Results saved to `outputs/results.json` and `outputs/bandpower.csv`")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download results.json",
                data=json.dumps(results, indent=2),
                file_name="results.json",
                mime="application/json"
            )
        with col2:
            csv_data = bp_df.to_csv()
            st.download_button(
                "Download bandpower.csv",
                data=csv_data,
                file_name="bandpower.csv",
                mime="text/csv"
            )

        st.markdown("---")
        st.subheader("Console-Style Summary")
        summary_lines = [
            f"File: {uploaded_file.name} ({fmt})",
            f"Channels: {n_channels} | Samples: {n_samples} | Fs: {fs:.1f} Hz | Duration: {duration:.1f}s",
            f"Preprocessing: BP {bp_low}-{bp_high} Hz, Notch {notch_freq}, Avg ref",
            f"Alpha band: {alpha_low}-{alpha_high} Hz",
            "",
            "--- Bandpower (mean across channels) ---",
        ]
        for name, vals in bp.items():
            summary_lines.append(f"  {name:>6s}: {np.mean(vals):.6e}")
        summary_lines.extend([
            "",
            "--- Connectivity ---",
            f"  Mean PLV (alpha): {mean_plv:.4f}",
            f"  Mean Kuramoto r:  {mean_r:.4f}",
        ])
        st.code('\n'.join(summary_lines))

else:
    st.markdown("---")
    st.markdown("### Getting Started")
    st.markdown("""
    1. **Upload an EEG file** using the uploader above
    2. **Adjust settings** in the sidebar (filter parameters, alpha band range)
    3. **View results** across the analysis tabs

    **Supported formats:**
    - **EDF** (.edf) - European Data Format
    - **FIF** (.fif) - MNE-Python native format
    - **BrainVision** (.vhdr) - Brain Products format
    - **EEGLAB** (.set) - EEGLAB format
    - **CSV** (.csv) - Plain CSV with channel columns
    - **MATLAB** (.mat) - MATLAB data files

    **For CSV files:** Provide one column per channel. Optionally include a 'time' column.
    Set the sampling rate in the sidebar.

    **For MATLAB files:** The largest 2D array will be used as EEG data.
    Variables named 'fs', 'srate', or 'sfreq' will be used as sampling rate.
    """)

    existing_mat = [f for f in os.listdir('.') if f.endswith('.mat') and 'alpha' in f.lower()]
    if existing_mat:
        st.markdown("---")
        st.markdown(f"**Tip:** You have {len(existing_mat)} .mat files in the workspace (e.g., `{existing_mat[0]}`). Upload one to analyze it!")
