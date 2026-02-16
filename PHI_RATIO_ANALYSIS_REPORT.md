# PHI-RATIO ANALYSIS ON EEG FREQUENCY BANDS

## Summary

**Hypothesis tested:** Adjacent EEG frequency band ratios converge to the golden ratio φ ≈ 1.618

**Dataset:** 42 EEG recordings (19 channels each, ~90,000 samples per recording)

**Result:** ❌ No frequency band ratios statistically converge to φ within 95% confidence intervals

---

## Key Findings

### Frequency Band Peak Frequencies (Mean ± SD)

| Band | Range (Hz) | Peak Freq (Hz) | Center of Gravity (Hz) |
|------|------------|----------------|------------------------|
| Delta | 0.5-4 | 0.70 ± 0.62 | 1.45 ± 0.27 |
| Theta | 4-8 | 4.82 ± 1.00 | 5.76 ± 0.26 |
| Alpha | 8-13 | 12.18 ± 1.16 | 11.16 ± 0.59 |
| Beta | 13-30 | 15.64 ± 3.97 | 20.44 ± 0.71 |
| Gamma | 30-45 | 35.85 ± 2.88 | 36.72 ± 0.32 |

---

## Phi-Ratio Analysis Results

### Peak Frequency Method

| Ratio | Mean | Distance from φ | % Error | p-value | φ in 95% CI |
|-------|------|-----------------|---------|---------|-------------|
| Theta/Delta | 9.01 | 7.40 | 457.1% | < 0.0001 | NO |
| Alpha/Theta | 2.61 | 0.99 | 61.1% | < 0.0001 | NO |
| **Beta/Alpha** | **1.29** | **0.32** | **20.1%** | < 0.0001 | NO |
| Gamma/Beta | 2.39 | 0.78 | 48.0% | < 0.0001 | NO |

### Center-of-Gravity Method

| Ratio | Mean | Distance from φ | % Error | p-value | φ in 95% CI |
|-------|------|-----------------|---------|---------|-------------|
| Theta/Delta | 4.12 | 2.50 | 154.7% | < 0.0001 | NO |
| Alpha/Theta | 1.94 | 0.32 | 20.1% | < 0.0001 | NO |
| Beta/Alpha | 1.84 | 0.22 | 13.6% | < 0.0001 | NO |
| **Gamma/Beta** | **1.80** | **0.18** | **11.1%** | < 0.0001 | NO |

---

## Interpretation

### Closest to φ (1.618):

1. **Gamma/Beta (CoG method):** 1.80 - only 11.1% error
2. **Beta/Alpha (CoG method):** 1.84 - 13.6% error
3. **Beta/Alpha (Peak method):** 1.29 - 20.1% error (below φ)

### Observed Patterns:

- **Gamma/Beta** and **Beta/Alpha** ratios cluster around 1.8-2.0
- This is closer to **harmonic ratios** (2:1) than φ
- **Alpha/Theta** ratio ≈ 2.0 (exactly harmonic)
- **Theta/Delta** highly variable, not following any constant ratio

### Why No Convergence to φ?

1. **Resting state data:** These are likely rest/baseline recordings
   - Hypothesis predicts φ-ratio in "optimal desynchronization" (Pletzer 2010)
   - May need task/arousal conditions for φ emergence

2. **Individual alpha frequency (IAF) variation:**
   - Each person has different alpha peak
   - Averaging across subjects blurs individual φ-patterns

3. **Frequency band definitions are arbitrary:**
   - Standard bands (delta, theta, etc.) may not reflect natural φ-based divisions
   - Neural oscillations may organize by individual IAF, not fixed bands

---

## Recommendations for Further Analysis

### 1. Arousal Condition Comparison
- Compare HIGH vs LOW arousal trials (if labels available)
- Prediction: φ-ratios stronger in low arousal/rest

### 2. Individual Subject Analysis
- Calculate φ-ratios per subject
- Look for subjects who DO show φ-convergence

### 3. Alternative Ratio Calculations
- Use individual alpha frequency (IAF) as anchor
- Calculate ratios relative to IAF: IAF/θ, β/IAF

### 4. Time-Frequency Analysis
- Look at φ-ratios during specific cognitive states
- Event-related changes in band ratios

---

## Files Generated

- `detailed_phi_analysis_results.png` - Visualization of all results
- `detailed_phi_results.csv` - Raw numerical results
- `phi_ratio_results.csv` - Summary statistics

---

## Conclusion

The current analysis on 42 EEG recordings does **NOT support** the hypothesis that adjacent frequency band ratios converge to φ ≈ 1.618. 

The closest observed ratios are:
- Gamma/Beta = 1.80 (11% from φ)
- Beta/Alpha = 1.84 (14% from φ)

These are closer to **harmonic ratios (2:1)** than to the golden ratio.

However, this does not definitively reject the φ-hypothesis because:
1. Data may be rest-only (not optimal for φ detection)
2. Averaging across subjects may mask individual patterns
3. Standard frequency bands may not align with natural φ-divisions

**Next step:** Analyze arousal-labeled data (DREAMER/DEAP) to test if high-arousal shifts ratios toward integers and low-arousal toward φ.
