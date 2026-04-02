"""Evaluate synthetic data quality: fidelity and utility.

Fidelity metrics (how well the synthetic data preserves the real distribution):
  - **1-way TVD**: Total Variation Distance between each feature's marginal
    distribution in real vs synthetic data.  Mean ± std across all features.
  - **2-way TVD (col, target)**: TVD of the joint (feature, target) marginal.
    Mean ± std across all feature columns.
  Lower TVD = better fidelity (0 = identical distributions, 1 = completely different).

Utility metric (how useful the synthetic data is as a training set):
  - Train a Logistic Regression classifier on the synthetic data.
  - Evaluate on the held-out *real* test set.
  - Baseline: same LR trained on 80 % of the real data, tested on the remaining 20 %.
  Higher accuracy / F1 = better utility.

Usage
-----
    # Evaluate all CSVs in a directory against a real dataset:
    python3 eval_synth_quality.py \\
        --synth-dir data/synth_data/ \\
        --data data/tcga_combined_full_100f.csv \\
        --target-col Subtype \\
        --n-bins 4

    # Single synthetic file:
    python3 eval_synth_quality.py \\
        --synth data/synth_data/tcga_eps1.00_run1_train300_bins4.csv \\
        --data data/tcga_combined_full_100f.csv

    # Multiple specific files:
    python3 eval_synth_quality.py \\
        --synth data/synth_data/tcga_eps1.00_run1*.csv data/synth_data/tcga_eps7*.csv \\
        --data data/tcga_combined_full_100f.csv

Output
------
    Per-file metrics are printed to stdout and saved to a CSV.
    Columns: synth_file, mean_tvd_1way, std_tvd_1way, mean_tvd_2way, std_tvd_2way,
             lr_accuracy, lr_f1_macro, baseline_accuracy, baseline_f1_macro
"""

import sys
import os
import glob
import argparse
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append("reprosyn-main/src/reprosyn/methods/mbi/")

import mbi_patch  # must precede any mbi import

os.makedirs("data/experiment_artifacts", exist_ok=True)
os.makedirs("intermediate/experiment_artifacts/focalpoints", exist_ok=True)

from util import C, Config, get_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# ---------------------------------------------------------------------------
# Fidelity helpers
# ---------------------------------------------------------------------------

def _to_prob(series):
    """Return a value-count dict normalised to probabilities."""
    vc = series.value_counts(normalize=True)
    return vc.to_dict()


def tvd(p: dict, q: dict) -> float:
    """Total Variation Distance between two discrete distributions.

    TVD = 0.5 * sum_x |P(x) - Q(x)|   ∈ [0, 1]
    """
    all_vals = set(p.keys()) | set(q.keys())
    return 0.5 * sum(abs(p.get(v, 0.0) - q.get(v, 0.0)) for v in all_vals)


def marginal_tvd_1way(real: pd.DataFrame, synth: pd.DataFrame,
                      feature_cols: list) -> np.ndarray:
    """Per-feature 1-way marginal TVD between real and synthetic."""
    return np.array([
        tvd(_to_prob(real[col]), _to_prob(synth[col]))
        for col in feature_cols
    ])


def marginal_tvd_2way(real: pd.DataFrame, synth: pd.DataFrame,
                      feature_cols: list, target_col: str) -> np.ndarray:
    """Per-feature 2-way (col, target) marginal TVD between real and synthetic."""
    tvds = []
    for col in feature_cols:
        real_joint  = real[[col, target_col]].apply(tuple, axis=1).value_counts(normalize=True).to_dict()
        synth_joint = synth[[col, target_col]].apply(tuple, axis=1).value_counts(normalize=True).to_dict()
        tvds.append(tvd(real_joint, synth_joint))
    return np.array(tvds)


# ---------------------------------------------------------------------------
# Utility helper
# ---------------------------------------------------------------------------

def train_lr(X_train, y_train, X_test, y_test):
    """Fit LogisticRegression on (X_train, y_train), evaluate on (X_test, y_test).

    Returns (accuracy, f1_macro).
    """
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="auto",
                             C=1.0, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = float((y_pred == y_test).mean())
    f1  = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    return acc, f1


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_file(synth_path: str,
                  real_df: pd.DataFrame,
                  feature_cols: list,
                  target_col: str,
                  test_frac: float = 0.2,
                  random_state: int = 42):
    """Evaluate one synthetic CSV against the real dataset.

    Parameters
    ----------
    synth_path   : path to synthetic CSV (ordinal int columns matching real_df)
    real_df      : preprocessed real dataset (ordinal ints, no HHID)
    feature_cols : list of feature column names
    target_col   : name of the classification target column
    test_frac    : fraction of real_df held out as the utility test set

    Returns
    -------
    dict with fidelity and utility metrics
    """
    synth = pd.read_csv(synth_path)

    # Ensure only columns present in real_df are used.
    shared_feat = [c for c in feature_cols if c in synth.columns]
    if target_col not in synth.columns:
        raise ValueError(f"Target column '{target_col}' not found in {synth_path}. "
                         f"Columns: {list(synth.columns[:5])}...")

    # ---- Fidelity ----
    tvd_1 = marginal_tvd_1way(real_df, synth, shared_feat)
    tvd_2 = marginal_tvd_2way(real_df, synth, shared_feat, target_col)

    # ---- Utility ----
    # Split real data → 20 % test (held-out from both LR experiments)
    X_real = real_df[shared_feat].values
    y_real = real_df[target_col].values

    X_train_real, X_test, y_train_real, y_test = train_test_split(
        X_real, y_real, test_size=test_frac, stratify=y_real,
        random_state=random_state
    )

    # Train-on-synth, test-on-real
    X_synth = synth[shared_feat].values
    y_synth = synth[target_col].values
    acc_synth, f1_synth = train_lr(X_synth, y_synth, X_test, y_test)

    # Baseline: train on 80 % real, test on the same 20 % real
    acc_base, f1_base = train_lr(X_train_real, y_train_real, X_test, y_test)

    return {
        "synth_file":        os.path.basename(synth_path),
        "n_synth_rows":      len(synth),
        "n_real_rows":       len(real_df),
        "n_features":        len(shared_feat),
        "mean_tvd_1way":     float(np.mean(tvd_1)),
        "std_tvd_1way":      float(np.std(tvd_1)),
        "mean_tvd_2way":     float(np.mean(tvd_2)),
        "std_tvd_2way":      float(np.std(tvd_2)),
        "lr_accuracy":       acc_synth,
        "lr_f1_macro":       f1_synth,
        "baseline_accuracy": acc_base,
        "baseline_f1_macro": f1_base,
    }


def print_results(rows: list):
    """Pretty-print a table of evaluation results."""
    if not rows:
        print("No results to display.")
        return

    # Header
    print(f"\n{'='*100}")
    print(
        f"{'File':<45}"
        f"{'TVD-1w mean':>11} {'std':>6}"
        f"  {'TVD-2w mean':>11} {'std':>6}"
        f"  {'LR acc':>7} {'F1':>7}"
        f"  {'Base acc':>8} {'F1':>7}"
    )
    print("-" * 100)

    for r in rows:
        fname = r["synth_file"]
        if len(fname) > 44:
            fname = "..." + fname[-41:]
        print(
            f"{fname:<45}"
            f"{r['mean_tvd_1way']:>11.4f} {r['std_tvd_1way']:>6.4f}"
            f"  {r['mean_tvd_2way']:>11.4f} {r['std_tvd_2way']:>6.4f}"
            f"  {r['lr_accuracy']:>7.4f} {r['lr_f1_macro']:>7.4f}"
            f"  {r['baseline_accuracy']:>8.4f} {r['baseline_f1_macro']:>7.4f}"
        )

    print("=" * 100)

    if len(rows) > 1:
        df = pd.DataFrame(rows)
        print(f"\nAggregate over {len(rows)} files:")
        for col in ["mean_tvd_1way", "mean_tvd_2way", "lr_accuracy", "lr_f1_macro"]:
            print(f"  {col:<22}: {df[col].mean():.4f} ± {df[col].std():.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate synthetic data quality (fidelity + utility) "
            "against the original dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Synth input: exactly one of --synth or --synth-dir
    synth_group = parser.add_mutually_exclusive_group(required=True)
    synth_group.add_argument(
        "--synth", nargs="+", metavar="CSV",
        help="One or more synthetic CSV file paths (glob patterns accepted).",
    )
    synth_group.add_argument(
        "--synth-dir", metavar="DIR",
        help="Directory; all *.csv files inside are evaluated.",
    )

    parser.add_argument(
        "--data", required=True,
        help="Path to the real dataset CSV (same file used with eval_epsilon_sweep.py).",
    )
    parser.add_argument(
        "--target-col", default=None,
        help="Target/label column name (default: Subtype, or cancer_subtype for COMBINED).",
    )
    parser.add_argument(
        "--n-bins", type=int, default=4,
        help="Equal-depth discretisation bins used when the sweep was run.",
    )
    parser.add_argument(
        "--name", default=None,
        help="Artifact namespace passed to encode_data (matches the sweep --name flag).",
    )
    parser.add_argument(
        "--test-frac", type=float, default=0.2,
        help="Fraction of real data held out as the utility test set.",
    )
    parser.add_argument(
        "--output", default=None,
        help="CSV path for results (default: synth_quality_<timestamp>.csv).",
    )
    args = parser.parse_args()

    # ---- Resolve synthetic file list ----
    if args.synth_dir:
        synth_files = sorted(glob.glob(os.path.join(args.synth_dir, "*.csv")))
        if not synth_files:
            print(f"No CSV files found in {args.synth_dir}")
            sys.exit(1)
    else:
        # Expand glob patterns in the explicit list
        synth_files = []
        for pat in args.synth:
            matches = sorted(glob.glob(pat))
            if matches:
                synth_files.extend(matches)
            elif os.path.isfile(pat):
                synth_files.append(pat)
            else:
                print(f"WARNING: no match for pattern '{pat}'")
        if not synth_files:
            print("No synthetic files found.")
            sys.exit(1)

    print(f"Found {len(synth_files)} synthetic file(s) to evaluate.")

    # ---- Load and preprocess real data (same pipeline as sweep) ----
    C.n_bins = args.n_bins

    cfg = Config(
        data_name="tcga",
        train_size=300,
        train_sizes={300: 100},
        set_MI=False,
        overlapping_aux=True,
        check_arbitrary_fps=False,
        pgm_target_variable=args.target_col or "Subtype",
    )
    cfg.csv_path = os.path.abspath(args.data)
    if args.name:
        cfg.artifact_name = args.name
    if args.target_col:
        cfg.pgm_target_variable = args.target_col

    _, aux, columns, meta, _ = get_data(cfg)
    target_col  = cfg.pgm_target_variable
    feature_cols = [c for c in columns if c != target_col]

    # aux has HHID; use only model columns.
    real_df = aux[columns].copy()

    print(f"Real dataset: {real_df.shape[0]} rows × {len(feature_cols)} features "
          f"+ target='{target_col}'  ({real_df[target_col].nunique()} classes)")

    # ---- Evaluate each file ----
    rows = []
    for i, spath in enumerate(synth_files, 1):
        print(f"\n[{i}/{len(synth_files)}] {os.path.basename(spath)}")
        t0 = time.time()
        try:
            result = evaluate_file(
                spath, real_df, feature_cols, target_col,
                test_frac=args.test_frac,
            )
            rows.append(result)
            print(
                f"  TVD-1way={result['mean_tvd_1way']:.4f}±{result['std_tvd_1way']:.4f}  "
                f"TVD-2way={result['mean_tvd_2way']:.4f}±{result['std_tvd_2way']:.4f}  "
                f"LR acc={result['lr_accuracy']:.4f}  F1={result['lr_f1_macro']:.4f}  "
                f"(baseline acc={result['baseline_accuracy']:.4f} F1={result['baseline_f1_macro']:.4f})  "
                f"[{time.time()-t0:.1f}s]"
            )
        except Exception as e:
            print(f"  ERROR: {e}")

    # ---- Print summary ----
    print_results(rows)

    # ---- Save results ----
    output_path = args.output or f"synth_quality_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
