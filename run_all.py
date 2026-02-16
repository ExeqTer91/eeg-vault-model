#!/usr/bin/env python3
"""
THE φ-ORGANIZED BRAIN: Complete Analysis Pipeline
==================================================
Run all analyses from scratch in one shot.

Usage:
    # Full pipeline (downloads EEG data + all analyses):
    python run_all.py

    # Skip EEG download, use cached data only (fast):
    python run_all.py --cached-only

    # Run a single stage:
    python run_all.py --stage unified
    python run_all.py --stage offset
    python run_all.py --stage generative
    python run_all.py --stage vault
    python run_all.py --stage obstruction

Setup (first time):
    pip install -r setup_requirements.txt
"""

import subprocess
import sys
import os
import time
import argparse


def check_dependencies():
    """Check all required packages are installed, install missing ones."""
    required = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'sklearn': 'scikit-learn',
        'mne': 'mne',
        'diptest': 'diptest',
        'streamlit': 'streamlit',
    }

    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"\nInstalling missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Installation complete.\n")
    else:
        print("All dependencies satisfied.\n")


def run_stage(name, script, description, required_inputs=None, skip_if_exists=None):
    """Run a single analysis stage."""
    print(f"\n{'='*70}")
    print(f"STAGE: {name}")
    print(f"  {description}")
    print(f"{'='*70}")

    if required_inputs:
        for f in required_inputs:
            if not os.path.exists(f):
                print(f"  ERROR: Required input missing: {f}")
                print(f"  Run earlier stages first or use --cached-only with cached data.")
                return False

    if skip_if_exists:
        all_exist = all(os.path.exists(f) for f in skip_if_exists)
        if all_exist:
            print(f"  Outputs already exist. Regenerating...")

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=False,
            timeout=600
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"\n  COMPLETED in {elapsed:.1f}s")
            return True
        else:
            print(f"\n  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
            return False

    except subprocess.TimeoutExpired:
        print(f"\n  TIMEOUT after 600s")
        return False
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run the complete EEG spectral attractor analysis pipeline.')
    parser.add_argument('--cached-only', action='store_true',
                        help='Skip EEG data download; use cached JSON data only')
    parser.add_argument('--stage', type=str, default=None,
                        choices=['unified', 'offset', 'generative', 'vault', 'obstruction'],
                        help='Run only a specific stage')
    parser.add_argument('--skip-setup', action='store_true',
                        help='Skip dependency check')
    args = parser.parse_args()

    print("="*70)
    print("THE φ-ORGANIZED BRAIN: Complete Analysis Pipeline")
    print("="*70)

    if not args.skip_setup:
        print("\n[1/6] Checking dependencies...")
        check_dependencies()

    os.makedirs('outputs/e1_figures', exist_ok=True)

    cached_data = [
        'outputs/aw_cached_subjects.json',
        'outputs/ds003969_cached_subjects.json',
        'outputs/eegbci_modal_results.json',
    ]

    stages = {
        'unified': {
            'script': 'e1_unified_analysis.py',
            'desc': 'Unified e-1 analysis across all datasets (downloads EEG data)',
            'inputs': [],
            'outputs': cached_data + ['outputs/e1_attractor_full_analysis.md'],
            'needs_download': True,
        },
        'offset': {
            'script': 'offset_investigation.py',
            'desc': 'Investigate the 0.06 offset (biological vs technical)',
            'inputs': cached_data,
            'outputs': ['outputs/data_source_investigation.md'],
            'needs_download': False,
        },
        'generative': {
            'script': 'generative_model_report.py',
            'desc': 'Generate figures/report from generative model results',
            'inputs': ['outputs/generative_model_results.json'],
            'outputs': ['outputs/generative_model.md'],
            'needs_download': False,
        },
        'vault': {
            'script': 'vault_model.py',
            'desc': 'Minimum Vault Model (potential well framework)',
            'inputs': cached_data,
            'outputs': ['outputs/vault_model.md', 'outputs/vault_model_results.json'],
            'needs_download': False,
        },
        'obstruction': {
            'script': 'geometric_obstruction.py',
            'desc': 'Geometric obstruction (corrugated potential landscape)',
            'inputs': cached_data,
            'outputs': ['outputs/geometric_obstruction.md',
                        'outputs/geometric_obstruction_results.json'],
            'needs_download': False,
        },
    }

    if args.stage:
        run_stages = [args.stage]
    else:
        run_stages = ['unified', 'offset', 'generative', 'vault', 'obstruction']

    if args.cached_only:
        run_stages = [s for s in run_stages if not stages[s]['needs_download']]
        has_cached = all(os.path.exists(f) for f in cached_data)
        if not has_cached:
            print("\nWARNING: --cached-only specified but cached data files are missing:")
            for f in cached_data:
                status = "OK" if os.path.exists(f) else "MISSING"
                print(f"  [{status}] {f}")
            print("\nRun without --cached-only first to generate cached data,")
            print("or ensure the cached JSON files are in the outputs/ directory.")
            if not any(os.path.exists(f) for f in cached_data):
                sys.exit(1)

    results = {}
    total_start = time.time()

    for i, stage_name in enumerate(run_stages, 1):
        stage = stages[stage_name]
        print(f"\n[{i}/{len(run_stages)}] ", end="")
        success = run_stage(
            stage_name.upper(),
            stage['script'],
            stage['desc'],
            required_inputs=stage['inputs'] if stage['inputs'] else None,
            skip_if_exists=stage['outputs'],
        )
        results[stage_name] = success

        if not success and stage_name == 'unified' and not args.cached_only:
            print("\nUnified analysis failed. Checking if cached data exists...")
            if all(os.path.exists(f) for f in cached_data):
                print("Cached data found! Continuing with remaining stages.")
            else:
                print("No cached data. Cannot continue. Fix unified analysis first.")
                break

    total_elapsed = time.time() - total_start

    print(f"\n\n{'='*70}")
    print(f"PIPELINE SUMMARY  (total time: {total_elapsed:.1f}s)")
    print(f"{'='*70}")
    for stage_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {stage_name}")

    print(f"\nOutput files:")
    for f in sorted(set(
        f for s in stages.values() for f in s['outputs']
        if os.path.exists(f)
    )):
        size = os.path.getsize(f)
        print(f"  {f} ({size:,} bytes)")

    import glob as g
    figs = sorted(g.glob('outputs/e1_figures/*.png'))
    print(f"\nFigures: {len(figs)} PNG files in outputs/e1_figures/")
    for f in figs:
        print(f"  {f}")

    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)
    if n_pass == n_total:
        print(f"\nAll {n_total} stages completed successfully.")
    else:
        print(f"\n{n_pass}/{n_total} stages completed. Check failures above.")


if __name__ == '__main__':
    main()
