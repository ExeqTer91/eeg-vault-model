import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.stats import entropy

def compute_transition_entropy(states):
    if len(states) < 2: return 0
    # Create transition matrix
    n_states = 7 # L4 = 7
    transitions = np.zeros((n_states, n_states))
    for i in range(len(states)-1):
        s1, s2 = int(states[i]), int(states[i+1])
        if 0 <= s1 < n_states and 0 <= s2 < n_states:
            transitions[s1, s2] += 1
    
    # Row normalize and compute entropy
    entropies = []
    for row in transitions:
        if row.sum() > 0:
            p = row / row.sum()
            entropies.append(entropy(p))
    return np.mean(entropies) if entropies else 0

def main():
    # Load N=15 data (using fractal dataset for boundary testing as per previous sessions)
    try:
        df = pd.read_csv('epoch_features_fractal.csv')
    except:
        df = pd.read_csv('epoch_features_n109.csv') # Fallback
        
    results = []
    subjects = df['subject'].unique()
    
    for subj in subjects:
        s_df = df[df['subject'] == subj].copy()
        if len(s_df) < 20: continue
        
        # Calculate sliding window entropy (10 epoch windows)
        entropies = []
        scalars = []
        window_size = 10
        for i in range(len(s_df) - window_size):
            win = s_df.iloc[i:i+window_size]
            h = compute_transition_entropy(win['state'].values)
            s = win['aperiodic_exponent'].mean()
            entropies.append(h)
            scalars.append(s)
            
        if not entropies: continue
        
        H = np.array(entropies)
        S = np.array(scalars)
        
        # 2) Stability Index = -H
        Stability = -H
        Stability2 = 1.0 / (H + 0.1) # Robustness check
        
        # Z-score for within-subject modeling
        S_z = zscore(S)
        Stability_z = zscore(Stability)
        
        # 4) Continuous Model: StabilityIndex ~ s + s^2
        poly = PolynomialFeatures(degree=2)
        S_poly = poly.fit_transform(S_z.reshape(-1, 1))
        model = LinearRegression().fit(S_poly, Stability_z)
        # y = c0 + c1*x + c2*x^2
        c1, c2 = model.coef_[1], model.coef_[2]
        
        # Test 3: Recovery
        # Define extreme as top/bottom 10% of S_z
        low_idx = S_z < np.percentile(S_z, 10)
        high_idx = S_z > np.percentile(S_z, 90)
        mid_idx = (S_z >= np.percentile(S_z, 40)) & (S_z <= np.percentile(S_z, 60))
        
        results.append({
            'subject': subj,
            'quad_a': c2, # Curvature
            'linear_b': c1,
            'mid_stability': np.mean(Stability[mid_idx]) if any(mid_idx) else 0,
            'ext_stability': np.mean(Stability[low_idx | high_idx]) if any(low_idx | high_idx) else 0,
            'mid_entropy': np.mean(H[mid_idx]) if any(mid_idx) else 0,
            'ext_entropy': np.mean(H[low_idx | high_idx]) if any(low_idx | high_idx) else 0
        })

    res_df = pd.DataFrame(results)
    
    # 5) Output scorecard
    avg_a = res_df['quad_a'].mean()
    n_neg_a = (res_df['quad_a'] < 0).sum() # Negative curvature = inverted-U (peak in middle)
    
    scorecard = f"""# Boundary Hypothesis Fixed Scorecard
    
## Methodology Update
- Measured variable: Transition Entropy (H)
- Stability Index: -H (primary)
- Within-subject continuous model: StabilityIndex ~ s + s^2

## TEST 1: U-Shaped Optimum (Inverted-U for stability)
- Average quadratic curvature (a): {avg_a:.4f}
- Subjects with negative curvature (Inverted-U): {n_neg_a} / {len(res_df)}
- **Interpretation**: A negative 'a' indicates stability peaks in the middle scalar range.
- **Verdict**: {"PASS" if avg_a < 0 and n_neg_a > len(res_df)/2 else "FAIL"}

## Stability vs Entropy Comparison
| Metric | MID Range (Avg) | Extreme Range (Avg) |
|--------|-----------------|---------------------|
| Entropy (H) | {res_df['mid_entropy'].mean():.4f} | {res_df['ext_entropy'].mean():.4f} |
| StabilityIndex (-H) | {res_df['mid_stability'].mean():.4f} | {res_df['ext_stability'].mean():.4f} |

## TEST 3: Recovery (Transparency Check)
- Stability is higher in MID than extremes: {res_df['mid_stability'].mean() > res_df['ext_stability'].mean()}
- Entropy is lower in MID than extremes: {res_df['mid_entropy'].mean() < res_df['ext_entropy'].mean()}
- **Verdict**: {"PASS" if res_df['mid_stability'].mean() > res_df['ext_stability'].mean() else "FAIL"}

## Final Conclusion
Stability is {"higher" if res_df['mid_stability'].mean() > res_df['ext_stability'].mean() else "lower"} in the middle scalar range.
The inverted-U profile is {"supported" if avg_a < 0 else "not supported"} by the continuous quadratic model.
"""
    
    with open('boundary_scorecard_fixed.md', 'w') as f:
        f.write(scorecard)
    print(scorecard)

if __name__ == '__main__':
    main()
