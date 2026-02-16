import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

RES = 'results_metatron'
FIG = 'figures_metatron'
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

CONSTANTS = {
    'φ': (1 + np.sqrt(5)) / 2,
    'π': np.pi,
    '√2': np.sqrt(2),
    '5/3': 5/3,
    '8/5': 1.6,
    '7/4': 1.75,
    'π/2': np.pi/2
}

HARMONICS = [0.5, 0.666, 0.75, 0.8, 1.0, 1.25, 1.333, 1.5, 1.618, 2.0]

def generate_metatron_geometry():
    """
    Constructs a 2D projection of Metatron's Cube (13 points).
    Points are arranged in a hexagonal pattern (Flower of Life seed).
    """
    points = []
    # Center
    points.append(np.array([0, 0]))
    # Inner ring (6 points)
    for i in range(6):
        angle = i * np.pi / 3
        points.append(np.array([np.cos(angle), np.sin(angle)]))
    # Outer ring (6 points)
    for i in range(6):
        angle = i * np.pi / 3
        points.append(np.array([2 * np.cos(angle), 2 * np.sin(angle)]))
    
    points = np.array(points)
    
    # Calculate all unique distances (ratios)
    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = np.linalg.norm(points[i] - points[j])
            distances.append(d)
            
    unique_distances = np.unique(np.round(distances, 6))
    ratios = []
    for i in range(len(unique_distances)):
        for j in range(len(unique_distances)):
            if i != j:
                ratios.append(unique_distances[i] / unique_distances[j])
                
    return np.array(ratios)

def analyze_metatron(ratios, perturbations=100):
    print("Analyzing Metatron's Cube...")
    
    # 1. Constant Proximity
    dist_counts = {k: 0 for k in CONSTANTS}
    for r in ratios:
        dists = {k: abs(r - v) for k, v in CONSTANTS.items()}
        closest = min(dists, key=dists.get)
        if dists[closest] < 0.05:
            dist_counts[closest] += 1
            
    # 2. Stability Shelves
    shelves = []
    for r in ratios:
        noise = r + np.random.normal(0, 0.02, perturbations)
        hits = sum(1 for n in noise if any(abs(n - v) < 0.01 for v in CONSTANTS.values()))
        shelves.append(hits / perturbations)
        
    return {
        'name': 'Metatron',
        'counts': dist_counts,
        'shelf_strength': np.mean(shelves),
        'harmonic_hit': np.mean([any(abs(r - h) < 0.02 for h in HARMONICS) for r in ratios])
    }

def main():
    metatron_ratios = generate_metatron_geometry()
    results = analyze_metatron(metatron_ratios)
    
    # Load previous results for comparison
    if os.path.exists('results_geometry/cross_domain_stability.csv'):
        prev_df = pd.read_csv('results_geometry/cross_domain_stability.csv')
        # Add metatron results
        new_row = {
            'name': 'Metatron',
            'shelf_strength': results['shelf_strength'],
            'harmonic_hit': results['harmonic_hit']
        }
        full_df = pd.concat([prev_df, pd.DataFrame([new_row])], ignore_index=True)
        full_df.to_csv('results_geometry/final_comparison_table.csv', index=False)
        
        # Plotting
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        colors = ['teal', 'orange', 'brown', 'purple', 'red']
        
        full_df.plot(kind='bar', x='name', y='shelf_strength', ax=ax[0], color=colors[:len(full_df)])
        ax[0].set_title('Stability Shelf Strength (Null Control: Metatron)')
        ax[0].set_ylabel('Persistence Probability')
        
        full_df.plot(kind='bar', x='name', y='harmonic_hit', ax=ax[1], color=colors[:len(full_df)])
        ax[1].set_title('Harmonic Clustering (Null Control: Metatron)')
        ax[1].set_ylabel('Ratio Hit Rate')
        
        plt.tight_layout()
        plt.savefig(f"{FIG}/metatron_comparison.tiff", dpi=300)
        
        # Report update
        with open('final_report.md', 'a') as f:
            f.write("\n\n## TEST 6: METATRON'S CUBE (ULTIMATE NULL CONTROL)\n")
            f.write(f"- Metatron Shelf Strength: {results['shelf_strength']:.4f}\n")
            f.write(f"- Metatron Harmonic Hit Rate: {results['harmonic_hit']:.4f}\n")
            f.write("\n### INTERPRETATION:\n")
            if results['shelf_strength'] > 0.4: # Arbitrary high baseline
                 f.write("- Metatron produces strong shelves; this confirms shelves are a generic geometric baseline.\n")
            else:
                 f.write("- Metatron produces weak shelves; if EEG shelves are stronger, they represent non-trivial dynamics.\n")
            
            f.write("\n### FINAL COMPARISON TABLE\n")
            f.write(full_df.to_markdown(index=False))

if __name__ == "__main__":
    main()
