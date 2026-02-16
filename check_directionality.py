import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore, linregress
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os

def main():
    df = pd.read_csv('epoch_features_n109.csv')
    
    # 1) Metric Definitions
    # s(t) = aperiodic exponent
    # stability(t) = 1.0 / (abs(r - mean_r) + 0.1)
    
    results = []
    
    for subj in df['subject'].unique():
        s_df = df[df['subject'] == subj].copy()
        if len(s_df) < 10: continue
        
        # Calculate local stability: proximity to subject's own mean ratio
        mean_r = s_df['r'].mean()
        s_df['stability'] = 1.0 / (np.abs(s_df['r'] - mean_r) + 0.1)
        
        # Features
        X = s_df['aperiodic_exponent'].values
        y = s_df['stability'].values
        
        # Z-score
        X_z = zscore(X)
        y_z = zscore(y)
        
        # Linear Fit
        slope, intercept, r_val, p_val, std_err = linregress(X_z, y_z)
        
        # Quadratic Fit
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X_z.reshape(-1, 1))
        model = LinearRegression().fit(X_poly, y_z)
        # y = c0 + c1*x + c2*x^2
        c1, c2 = model.coef_[1], model.coef_[2]
        
        results.append({
            'subject': subj,
            'linear_slope': slope,
            'quad_c2': c2,
            'p_val': p_val,
            'r_val': r_val
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv('stability_directionality_results.csv', index=False)
    
    # Group analysis
    mean_slope = res_df['linear_slope'].mean()
    std_slope = res_df['linear_slope'].std()
    n = len(res_df)
    ci = 1.96 * (std_slope / np.sqrt(n))
    
    print(f"Group Linear Slope: {mean_slope:.4f} +/- {ci:.4f}")
    
    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(res_df['subject'], res_df['linear_slope'], alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Stability vs Aperiodic Exponent (Linear Slopes per Subject)')
    plt.xlabel('Subject Index')
    plt.ylabel('Z-scored Slope (Stability ~ Exponent)')
    plt.savefig('stability_slopes.png')
    
    # Summary text
    relation = "proportional" if mean_slope > 0 else "inversely proportional"
    with open('stability_directionality_summary.txt', 'w') as f:
        f.write(f"Stability is {relation} to the scalar (aperiodic exponent).\n")
        f.write(f"Group Mean Slope: {mean_slope:.4f} (95% CI: {mean_slope-ci:.4f} to {mean_slope+ci:.4f})\n")
        f.write(f"N subjects: {n}\n")
        f.write(f"Positive slopes: {(res_df['linear_slope'] > 0).sum()} / {n}\n")

if __name__ == '__main__':
    main()
