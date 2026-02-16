# EEG Vault Model

**Minimum Vault Model and Geometric Obstruction Analysis for EEG Spectral Attractors**

Companion code for: *Ursachi (2026). The φ-Organized Brain: Scale-Free Dynamics, Bridge-State Control, and a Seven-State Neural Architecture.* Frontiers in Computational Neuroscience. Manuscript ID: 1781338.

## Overview

This repository provides the full analysis pipeline for investigating whether human EEG alpha/theta frequency ratios organize near mathematical constants (e-1 ≈ 1.718, φ ≈ 1.618, √e, etc.). It implements:

- **Cross-dataset validation** (N=244: EEGBCI N=109, Alpha Waves N=37, OpenNeuro ds003969 N=98)
- **Vault model** and **geometric obstruction** analysis
- **Kuramoto generative modeling** with exponential frequency spacing
- **FOOOF aperiodic correction** (specparam)
- **22 publication-ready figures** (300 DPI)
- **4 peer-review response analyses** (Jobs 9–12)

## Key Findings

| Analysis | Result | Interpretation |
|----------|--------|----------------|
| Raw α/θ ratio (N=244) | 1.778 ± 0.15 | Between φ and e-1, nearest e-1 |
| FOOOF periodic-only (N=42) | 1.667 | Shifts toward φ after 1/f removal |
| Expanded constants AIC (Job 9) | Empirical > all constants | No single constant is "best" |
| Noble number control (Job 10) | [1;3,1̄] beats φ | φ not uniquely optimal for Kuramoto |
| FOOOF correction (Job 11) | 61.5% attrition | Selection bias confirmed (p=0.0004) |
| gedBounds (Job 12) | r=0.21, p=0.20 | Landscape not robust to individualization |

## Repository Structure

### Core Analysis Scripts

| Script | Manuscript Section | Description |
|--------|-------------------|-------------|
| `e1_unified_analysis.py` | Methods §2, Results §3 | Cross-dataset N=244 unified analysis |
| `vault_model.py` | Methods §2.4 | Minimum vault model implementation |
| `geometric_obstruction.py` | Methods §2.5 | Geometric obstruction analysis |
| `generate_publication_figures.py` | All figures | Generates figures 1–18 |
| `offset_investigation.py` | Results §3.2 | Sampling rate offset (η²=6.9%) |
| `split_half_validation.py` | Methods §2.6 | Anti-circularity validation |

### Generative Modeling

| Script | Manuscript Section | Description |
|--------|-------------------|-------------|
| `modal_generative_model.py` | Methods §2.7 | Kuramoto model with exponential frequencies |
| `run_job3_kuramoto.py` | Fig 13 | Kuramoto parameter sweep |
| `run_kuramoto_circle.py` | Fig 14 | Kuramoto circle architecture |
| `run_job5_biphasic.py` | Fig 15 | Biphasic transition analysis |
| `run_diagonal_spin.py` | Fig 18 | Symmetry-breaking Kuramoto-Sakaguchi |
| `run_job7_self_similar.py` | Fig 17 | Self-similar cascade (φ-ladder) |

### Peer-Review Response Analyses (Jobs 9–12)

| Script | Figure | Description |
|--------|--------|-------------|
| `run_job9_constants.py` | Fig 19 | Expanded constant set (φ, e-1, √e, √π, etc.) with AIC |
| `run_job10_noble.py` | Fig 20 | Noble number control for Kuramoto collapse |
| `run_job11_fooof.py` | Fig 21 | FOOOF aperiodic correction + selection bias |
| `run_job12_gedbounds.py` | Fig 22 | Individualized band boundaries (gedBounds) |

### Statistical Validation

| Script | Description |
|--------|-------------|
| `test_surrogates.py` | Surrogate comparison (Fig 10) |
| `test_bayesian_models.py` | Bayesian model comparison |
| `run_job2_downsampling.py` | Downsampling robustness (Fig 12) |
| `peer_review_statistics.py` | Complete statistical methods |
| `comprehensive_validation_tests.py` | Full validation suite |

### Data Processing

| Script | Description |
|--------|-------------|
| `cache_all_subjects.py` | Cache EEGBCI N=109 results |
| `cache_ds003969.py` | Cache OpenNeuro ds003969 N=98 results |
| `eeg_features.py` | Helper: bandpower, PLV, coherence, GFP, Kuramoto |
| `e1_compute_utils.py` | FOOOF/specparam utilities |
| `modal_eegbci_compute.py` | Modal cloud compute for EEGBCI |

### Output Files

| File | Description |
|------|-------------|
| `outputs/Ursachi_2026_Figures_v3.7.zip` | All 22 publication figures (300 DPI PNG) |
| `outputs/publication_figures/fig*.png` | Individual figure files |
| `outputs/eegbci_modal_results.json` | EEGBCI N=109 cached results with FOOOF |
| `outputs/aw_cached_subjects.json` | Alpha Waves N=37 cached results |
| `outputs/ds003969_cached_subjects.json` | OpenNeuro ds003969 N=98 cached results |
| `outputs/expanded_constants_results.json` | Job 9 results |
| `outputs/noble_number_results.json` | Job 10 results |
| `outputs/fooof_correction_results.json` | Job 11 results |
| `outputs/gedbounds_results.json` | Job 12 results |
| `eeg-processing/results/physionet_*.csv` | PhysioNet processing results |

## Installation

```bash
git clone https://github.com/ExeqTer91/eeg-vault-model.git
cd eeg-vault-model
pip install -r requirements.txt
```

## Dependencies

### Core (required for main analyses and figures)

| Package | Version | Purpose |
|---------|---------|---------|
| `mne` | ≥1.6 | EEG data loading, preprocessing, EEGBCI dataset |
| `scipy` | ≥1.11 | Signal processing (Welch PSD, filters), statistics |
| `numpy` | ≥1.24 | Numerical computation |
| `matplotlib` | ≥3.8 | Publication figure generation (300 DPI) |
| `fooof` (specparam) | ≥1.0 | Aperiodic spectral decomposition (Job 11) |
| `pandas` | ≥2.1 | Data manipulation and CSV export |
| `seaborn` | ≥0.13 | Statistical visualization |
| `scikit-learn` | ≥1.3 | Clustering, PCA, classification |
| `statsmodels` | ≥0.14 | Statistical tests and regression |
| `pingouin` | any | Bayesian and effect size statistics |
| `networkx` | any | Graph analysis (coherence networks) |
| `Pillow` | ≥10.0 | Image handling for figure composites |
| `streamlit` | ≥1.30 | Interactive dashboard (`app.py`) |
| `requests` | ≥2.31 | Dataset downloads (Zenodo) |

### Optional (specific analysis scripts)

| Package | Used by | Purpose |
|---------|---------|---------|
| `diptest` | `deep_dive_block*.py` | Hartigan's dip test for bimodality |
| `hmmlearn` | `hierarchical_state_analysis.py` | Hidden Markov model state analysis |
| `torch` + `torchvision` | `progressive_vit_experiment.py` | Vision transformer experiment |
| `runpod` | `create_runpod_pods.py` | RunPod cloud compute orchestration |

## Reproducing Results

### Quick Start (uses cached data)

```bash
# Cross-dataset analysis (uses cached JSON files in outputs/)
python e1_unified_analysis.py

# Generate all 22 publication figures
python generate_publication_figures.py

# Peer-review response jobs (use cached EEGBCI data)
python run_job9_constants.py   # Expanded constants (Fig 19)
python run_job10_noble.py      # Noble numbers (Fig 20)
python run_job11_fooof.py      # FOOOF correction (Fig 21)
python run_job12_gedbounds.py  # gedBounds (Fig 22)
```

### Full Pipeline (downloads and processes raw data)

```bash
# Step 1: Cache EEGBCI data (downloads ~2GB via MNE)
python cache_all_subjects.py

# Step 2: Cache OpenNeuro ds003969
python cache_ds003969.py

# Step 3: Run unified analysis
python e1_unified_analysis.py

# Step 4: Generate figures
python generate_publication_figures.py
```

### Data Sources

| Dataset | N | Access |
|---------|---|--------|
| PhysioNet EEGBCI | 109 | Automatic via `mne.datasets.eegbci` |
| Alpha Waves | 37 | Zenodo (downloaded via `requests`) |
| OpenNeuro ds003969 | 98 | BIDS format in `ds003969/` folder |

## Figure–Script Mapping

| Figure | Script | Key Result |
|--------|--------|------------|
| Fig 1 | `generate_publication_figures.py` | Population corrugated landscape |
| Fig 2 | `generate_publication_figures.py` | Corrugated landscape detail |
| Fig 3 | `generate_publication_figures.py` | Basin classification |
| Fig 4 | `generate_publication_figures.py` | Temperature/sampling rate effects |
| Fig 5 | `generate_publication_figures.py` | Generative model (Kuramoto) |
| Fig 6 | `generate_publication_figures.py` | Alternating attractors |
| Fig 7 | `generate_publication_figures.py` | AIC model comparison |
| Fig 8 | `generate_publication_figures.py` | TOST equivalence test |
| Fig 9 | `generate_publication_figures.py` | Cross-domain analysis |
| Fig 10 | `test_surrogates.py` | Surrogate comparison |
| Fig 11 | `generate_publication_figures.py` | Permutation test |
| Fig 12 | `run_job2_downsampling.py` | Downsampling robustness |
| Fig 13 | `run_job3_kuramoto.py` | Kuramoto parameter sweep |
| Fig 14 | `run_kuramoto_circle.py` | Kuramoto circle architecture |
| Fig 15 | `run_job5_biphasic.py` | Biphasic transition |
| Fig 16 | `generate_publication_figures.py` | Trapezoidal test |
| Fig 17 | `run_job7_self_similar.py` | Self-similar cascade |
| Fig 18 | `run_diagonal_spin.py` | Diagonal spin (Kuramoto-Sakaguchi) |
| Fig 19 | `run_job9_constants.py` | Expanded constants comparison |
| Fig 20 | `run_job10_noble.py` | Noble number control |
| Fig 21 | `run_job11_fooof.py` | FOOOF aperiodic correction |
| Fig 22 | `run_job12_gedbounds.py` | gedBounds individualized boundaries |

## Statistical Methods

- **AIC Gaussian comparison**: k=1 (fixed-μ) vs k=2 (empirical mean)
- **TOST equivalence testing**: Bounds ±0.10 around e-1
- **Paired bootstrap CI**: 10,000 resamples for Kuramoto collapse
- **Mann-Whitney U**: Selection bias check (Job 11)
- **FOOOF peak extraction**: With attrition tracking and bias reporting
- **Split-half validation**: Odd/even epochs to rule out circularity

## Honest Negative Findings

This repository includes several findings that challenge the original hypotheses:

1. **Job 9**: Empirical mean (1.778) beats all mathematical constants by AIC — no single constant is needed
2. **Job 10**: φ is NOT uniquely best for Kuramoto temporal collapse — other noble numbers beat it
3. **Job 11**: FOOOF correction causes 61.5% attrition with confirmed selection bias (Cohen d = -0.54)
4. **Job 12**: The ratio landscape does NOT robustly survive individualized band boundaries (r = 0.21)

These are reported transparently as they strengthen the scientific credibility of the work.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@article{ursachi2026phi,
  title={The $\varphi$-Organized Brain: Scale-Free Dynamics, Bridge-State Control, and a Seven-State Neural Architecture},
  author={Ursachi, ...},
  journal={Frontiers in Computational Neuroscience},
  year={2026},
  note={Manuscript ID: 1781338}
}
```
