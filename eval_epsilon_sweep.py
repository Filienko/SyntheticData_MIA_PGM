"""Evaluate MAMA-MIA attack on Private-PGM across epsilon values (TCGA dataset).

For each epsilon this script:
  1. Determines the fixed PGM focal points (deterministic – no shadow modelling).
  2. Runs N attack trials: trains Private-PGM on a sampled training set, then
     runs the MAMA-MIA marginal likelihood-ratio attack on held-out targets.
  3. Reports per-run and summary AUC / Membership-Advantage (MA) statistics.
  4. Saves a CSV results table.

Usage
-----
    python3 eval_epsilon_sweep.py [options]

    # defaults: epsilons=[1,2,5,7,10], 3 runs, train_size=300
    python3 eval_epsilon_sweep.py

    # custom sweep
    python3 eval_epsilon_sweep.py --epsilons 1 2 5 10 --n-runs 5 --train-size 200

    # use all columns from a different TCGA CSV (e.g. full 978-gene file)
    python3 eval_epsilon_sweep.py --data path/to/tcga_full.csv --target-col Subtype

Prerequisites
-------------
    data/tcga_combined_full_100f.csv   (or supply --data path)

Output
------
    results_epsilon_sweep_<timestamp>.csv  – one row per epsilon with
        mean_AUC, std_AUC, mean_MA, std_MA, all individual run values.
"""

import sys
import os
import argparse
import time
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap paths (same as run_tcga_pgm.py)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append('reprosyn-main/src/reprosyn/methods/mbi/')

import mbi_patch  # must precede any mbi import

os.makedirs("data/experiment_artifacts", exist_ok=True)
os.makedirs("intermediate/experiment_artifacts/focalpoints", exist_ok=True)
os.makedirs("intermediate/experiment_artifacts/satml25-rebuttal/mamamia_results",
            exist_ok=True)

from util import C, Config, get_data, sample_experimental_data
from determine_focal_points import determine_privatepgm_marginals
from conduct_attacks import attack_privatepgm


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(epsilons, n_runs, train_size, n_targets, n_bins,
              data_path=None, target_col=None):
    """Run MAMA-MIA attack across epsilons.

    Parameters
    ----------
    epsilons   : list[float]  DP epsilon values to test
    n_runs     : int          attack trials per epsilon
    train_size : int          records used to train PGM per trial
    n_targets  : int          target records to classify per trial (members + non-members).
                              Half are members, half non-members.
                              AUC variance ∝ 1/sqrt(n_targets) – use ≥100 for stable estimates.
                              Must satisfy: train_size + n_targets ≤ total dataset rows.
    n_bins     : int          equal-depth discretization bins for continuous features
    data_path  : str|None     override CSV path (default: data/tcga_combined_full_100f.csv)
    target_col : str|None     PGM pivot column; None → infer from dataset loader

    Returns
    -------
    pd.DataFrame  summary table (one row per epsilon)
    pd.DataFrame  per-run detail table
    """
    C.n_bins = n_bins

    cfg = Config(
        data_name="tcga",
        train_size=train_size,
        train_sizes={train_size: n_targets},
        set_MI=False,
        overlapping_aux=True,
        check_arbitrary_fps=False,
        pgm_target_variable=target_col or "Subtype",
        epsilons=epsilons,
    )

    # Allow overriding the data path for full-gene datasets.
    if data_path:
        import encode_data as _ed
        _orig = _ed.DATA_DIR
        _ed.DATA_DIR = os.path.dirname(data_path) + "/"
        # Temporarily monkey-patch the CSV filename expected by tcga_data().
        import encode_data
        _orig_fn = None
        # Rebuild the expected filename: tcga_data() reads DATA_DIR + "tcga_combined_full_100f.csv"
        # If a different file is supplied, symlink or point directly via environment.
        # Simplest: just set the path and reload.
        # (For now, if data_path differs from default, the user must name it correctly
        #  or use the default path; see note in --help.)

    _, aux, columns, meta, _ = get_data(cfg)
    print(f"\nLoaded dataset: {aux.shape[0]} rows × {len(columns)} columns")
    print(f"Subtype classes: {aux['Subtype'].nunique()}")
    print(f"Epsilons: {epsilons}  |  runs/ε: {n_runs}  |  train_size: {train_size}\n")

    summary_rows = []
    detail_rows  = []

    for eps in epsilons:
        print(f"{'='*60}")
        print(f"ε = {eps}")

        # Focal points are deterministic for PGM – compute once per epsilon.
        fp_file = f"FP_tcga_pgm_e{eps:.2f}_n{train_size}"
        fps = determine_privatepgm_marginals(
            cfg, aux, columns, cfg.categorical_columns, meta,
            eps, train_size, filename=fp_file,
        )
        print(f"  Focal points: {len(fps)} cliques "
              f"({sum(1 for f in fps if len(f)==1)} 1-way, "
              f"{sum(1 for f in fps if len(f)==2)} 2-way)")

        run_aucs, run_mas = [], []

        for run in range(n_runs):
            t0 = time.process_time()

            target_ids, targets, membership, train, kde_seed = \
                sample_experimental_data(cfg, aux, columns)

            result = attack_privatepgm(
                cfg, meta, aux, columns, train, eps,
                targets, target_ids, membership,
                kde_sample_seed=kde_seed,
                fps=fps,
            )

            elapsed = time.process_time() - t0

            # result tuple positions (from conduct_attacks.attack_privatepgm):
            # (kde_ma, kde_auc, kde_time, mm_ma, mm_auc,
            #  mm_ma_w, mm_auc_w, mm_time, mm_arbitrary_ma, distance,
            #  kde_roc, mm_roc)
            ma_w  = result[5]
            auc_w = result[6]

            label = (f"  Run {run+1}/{n_runs}: "
                     f"MA={ma_w:.4f}  AUC={auc_w:.4f}  ({elapsed:.1f}s CPU)")
            print(label)

            if ma_w  is not None: run_mas.append(ma_w)
            if auc_w is not None: run_aucs.append(auc_w)

            detail_rows.append({
                'epsilon':    eps,
                'run':        run + 1,
                'AUC':        auc_w,
                'MA':         ma_w,
                'cpu_s':      round(elapsed, 2),
                'train_size': train_size,
                'n_targets':  n_targets,
                'n_bins':     n_bins,
            })

        mean_auc = np.mean(run_aucs) if run_aucs else float('nan')
        std_auc  = np.std(run_aucs)  if run_aucs else float('nan')
        mean_ma  = np.mean(run_mas)  if run_mas  else float('nan')
        std_ma   = np.std(run_mas)   if run_mas  else float('nan')

        print(f"\n  ε={eps:>5}  AUC={mean_auc:.4f}±{std_auc:.4f}"
              f"  MA={mean_ma:.4f}±{std_ma:.4f}")

        summary_rows.append({
            'epsilon':    eps,
            'mean_AUC':   round(mean_auc, 4),
            'std_AUC':    round(std_auc, 4),
            'mean_MA':    round(mean_ma, 4),
            'std_MA':     round(std_ma, 4),
            'n_runs':     len(run_aucs),
            'train_size': train_size,
            'n_targets':  n_targets,
            'n_bins':     n_bins,
            'AUC_runs':   run_aucs,
            'MA_runs':    run_mas,
        })

    summary_df = pd.DataFrame(summary_rows)
    detail_df  = pd.DataFrame(detail_rows)
    return summary_df, detail_df


# ---------------------------------------------------------------------------
# Pretty-print summary table
# ---------------------------------------------------------------------------

def print_summary(df):
    n_tgt = int(df['n_targets'].iloc[0]) if 'n_targets' in df.columns else '?'
    n_tr  = int(df['train_size'].iloc[0])
    print(f"\n{'='*68}")
    print(f"SUMMARY  (MAMA-MIA on Private-PGM × TCGA  |  "
          f"train={n_tr}  targets={n_tgt})")
    print(f"{'='*68}")
    print(f"{'ε':>6}  {'AUC mean':>9}  {'AUC std':>8}  {'MA mean':>8}  {'MA std':>7}  runs  per-run AUC")
    print("-" * 68)
    for _, row in df.iterrows():
        runs_str = "  ".join(f"{v:.3f}" for v in row.get('AUC_runs', []))
        print(f"{row['epsilon']:>6}  "
              f"{row['mean_AUC']:>9.4f}  "
              f"{row['std_AUC']:>8.4f}  "
              f"{row['mean_MA']:>8.4f}  "
              f"{row['std_MA']:>7.4f}  "
              f"{int(row['n_runs']):>4}  [{runs_str}]")
    print("=" * 68)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sweep DP epsilon for MAMA-MIA attack on Private-PGM (TCGA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epsilons", nargs="+", type=float,
        default=[1.0, 2.0, 5.0, 7.0, 10.0],
        help="List of DP epsilon values to evaluate",
    )
    parser.add_argument(
        "--n-runs", type=int, default=3,
        help="Attack trials per epsilon",
    )
    parser.add_argument(
        "--train-size", type=int, default=300,
        help="Records used to train PGM per trial",
    )
    parser.add_argument(
        "--n-targets", type=int, default=100,
        help=(
            "Target records classified per trial (half members, half non-members). "
            "AUC std ∝ 1/sqrt(n_targets): 30 targets → std≈0.12 (noisy); "
            "100 → std≈0.06; 200 → std≈0.04. "
            "Rule of thumb for TCGA (1089 rows): train_size + n_targets ≤ 900."
        ),
    )
    parser.add_argument(
        "--n-bins", type=int, default=10,
        help="Equal-depth discretization bins for continuous features",
    )
    parser.add_argument(
        "--data", default=None,
        help="Path to TCGA CSV (default: data/tcga_combined_full_100f.csv). "
             "File must be named tcga_combined_full_100f.csv in its parent directory, "
             "or placed at that default path.",
    )
    parser.add_argument(
        "--target-col", default=None,
        help="PGM target/pivot column for 2-way marginals (default: Subtype)",
    )
    parser.add_argument(
        "--output", default=None,
        help="CSV path for results (default: results_epsilon_sweep_<timestamp>.csv)",
    )
    args = parser.parse_args()

    output_path = args.output or (
        f"results_epsilon_sweep_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )

    total_needed = args.train_size + args.n_targets
    if total_needed > 1000:
        print(f"WARNING: train_size ({args.train_size}) + n_targets ({args.n_targets}) "
              f"= {total_needed}. TCGA has 1089 rows; leaving only "
              f"{1089 - total_needed} for aux. Consider reducing one of these.")

    summary_df, detail_df = run_sweep(
        epsilons=sorted(args.epsilons),
        n_runs=args.n_runs,
        train_size=args.train_size,
        n_targets=args.n_targets,
        n_bins=args.n_bins,
        data_path=args.data,
        target_col=args.target_col,
    )

    print_summary(summary_df)

    # Save results – drop the list columns (AUC_runs, MA_runs) for the CSV.
    detail_df.to_csv(output_path, index=False)
    summary_csv = output_path.replace(".csv", "_summary.csv")
    summary_df.drop(columns=["AUC_runs", "MA_runs"]).to_csv(summary_csv, index=False)

    print(f"\nPer-run results  → {output_path}")
    print(f"Summary table    → {summary_csv}")


if __name__ == "__main__":
    main()
