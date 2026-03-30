"""MAMA-MIA attack integrated with Health-Privacy-Challenge competition format.

Adapts our Private-PGM MAMA-MIA attack to the competition's black-box setting.

Competition data facts (from running_baseline_example.zip)
----------------------------------------------------------
- Synthetic data:  CSV, 978 ENSG gene columns, ~871 rows, no index column.
- Membership test: TSV (transposed), same 978 genes, TCGA sample IDs as index.
- Ground truth:    CSV with 1089 rows, index + `membership_label` (0.0 / 1.0).
- Reference:       Same gene format as synthetic data (optional).
- Preprocessing:   StandardScaler is applied before density estimation in
                   the competition baselines.

Attack strategy for DP-PGM
---------------------------
Private-PGM uses a fixed marginal structure (no exponential-mechanism
randomness), so we know the cliques from the algorithm spec alone:
  - All 1-way singletons:  {gene} for each gene column.
  - All 2-way pairs:       {gene, target_col} for each gene, if target_col set.

We treat the competition's pre-generated synthetic data as the member
distribution proxy (P_synth) and the reference data as the non-member
distribution (P_ref), then score each target record by the MAMA-MIA
likelihood-ratio sum over all focal-point cliques.

Output format (required by competition)
----------------------------------------
Single-column CSV with header `membership_label`.
Values are continuous membership scores (higher = more likely member).
Files: synthetic_data_1_predictions.csv … synthetic_data_4_predictions.csv

Usage – standalone (Mode A, explicit paths)
-------------------------------------------
    python3 competition_mia.py \\
        --synthetic  path/synthetic_data_split_1.csv \\
        --reference  path/reference_data.csv \\
        --targets    path/membership_test.tsv \\
        --output     path/synthetic_data_1_predictions.csv \\
        [--gt        path/synthetic_data_1_gt.csv] \\
        [--epsilon   10.0] \\
        [--n-bins    10] \\
        [--target-col ""]         # "" = 1-way only; default for gene data

Usage – competition directory layout (Mode B)
---------------------------------------------
    python3 competition_mia.py \\
        --competition-dir ~/Health-Privacy-Challenge \\
        --dataset TCGA-BRCA \\
        --generator dp_pgm \\
        --experiment-name epsilon_10.0

Usage – our own TCGA data (reproduce run_tcga_pgm.py in black-box style)
-------------------------------------------------------------------------
    python3 competition_mia.py \\
        --synthetic  data/tcga_synthetic_eps10_n300.csv \\
        --reference  data/tcga_combined_full_100f.csv \\
        --targets    data/tcga_combined_full_100f.csv \\
        --output     results/tcga_predictions.csv \\
        --target-col Subtype --n-bins 10 \\
        --gt         data/tcga_gt.csv
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup (mirrors run_tcga_pgm.py bootstrap)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append('reprosyn-main/src/reprosyn/methods/mbi/')

import mbi_patch  # must be imported before any mbi usage

from encode_data import (
    fit_continuous_features_equaldepth,
    discretize_continuous_features_equaldepth,
)
from util import C

os.makedirs("data/experiment_artifacts", exist_ok=True)


# ===========================================================================
# Data loading helpers
# ===========================================================================

def load_csv_or_tsv(path, **kwargs):
    """Auto-detect separator and load a flat data file."""
    sep = '\t' if path.endswith('.tsv') else ','
    return pd.read_csv(path, sep=sep, index_col=0, **kwargs)


def load_synthetic(path):
    """Load Blue Team synthetic data.  No index column."""
    return pd.read_csv(path)


def load_membership_test(path):
    """Load membership test file.

    Competition format: TSV with sample IDs as index, genes as columns.
    Falls back to standard CSV if .tsv extension is absent.
    """
    sep = '\t' if path.endswith('.tsv') else ','
    df = pd.read_csv(path, sep=sep, index_col=0)
    # If the file is transposed (genes as rows, samples as columns), un-transpose.
    # Heuristic: if first column name starts with 'TCGA' it's already correct;
    # if first row index looks like an ENSG id, we need to transpose.
    if df.index[0].startswith('ENSG') and not df.columns[0].startswith('ENSG'):
        df = df.T
    return df


def load_gt(path, label_col='membership_label'):
    """Load ground-truth membership labels. Returns np.ndarray of 0/1."""
    df = pd.read_csv(path, index_col=0)
    return df[label_col].values.astype(int)


# ===========================================================================
# Encoding helpers
# ===========================================================================

def _is_already_encoded(df, feature_cols, n_bins):
    """Heuristic: True if columns hold non-negative integers < 2*n_bins."""
    sub = df[feature_cols].select_dtypes(include=[np.number])
    if sub.empty:
        return False
    vals = sub.values
    return (
        np.allclose(vals, vals.astype(int), equal_nan=True)
        and (vals >= 0).all()
        and (vals < n_bins * 2).all()
    )


def encode_dataframes(synth, ref, targets, feature_cols, n_bins,
                      standardize=True, name='competition'):
    """Discretize all DataFrames into equal-depth bins.

    Pipeline:
      1. (Optional) StandardScaler – fit on synth+ref, transform all three.
      2. Equal-depth binning – fit on synth+ref, discretize all three.

    If data already appears integer-encoded, steps are skipped.
    """
    if _is_already_encoded(synth, feature_cols, n_bins):
        print("  Data appears already discretized – skipping encoding.")
        return synth.copy(), ref.copy(), targets.copy()

    # Step 1: StandardScaler (matches competition baseline preprocessing).
    if standardize:
        scaler = StandardScaler()
        combined_vals = pd.concat([synth[feature_cols], ref[feature_cols]],
                                  ignore_index=True)
        scaler.fit(combined_vals)

        def _scale(df):
            out = df.copy()
            out[feature_cols] = scaler.transform(df[feature_cols])
            return out

        synth   = _scale(synth)
        ref     = _scale(ref)
        targets = _scale(targets)

    # Step 2: Equal-depth binning (same as our tcga_data() pipeline).
    print(f"  Discretizing {len(feature_cols)} features into {n_bins} bins …")
    C.n_bins = n_bins
    combined = pd.concat([synth[feature_cols], ref[feature_cols]], ignore_index=True)
    fit_continuous_features_equaldepth(combined, name)

    def _enc(df):
        out = df.copy()
        out[feature_cols] = discretize_continuous_features_equaldepth(
            df[feature_cols], name
        )
        return out

    return _enc(synth), _enc(ref), _enc(targets)


# ===========================================================================
# Focal points
# ===========================================================================

def build_pgm_focal_points(columns, target_col=None):
    """Fixed marginal structure of Private-PGM (deterministic).

    Returns
    -------
    dict  {clique_tuple: weight}   all weights = 1.0
    """
    fps = {(col,): 1.0 for col in columns}
    if target_col and target_col in columns:
        for col in columns:
            if col != target_col:
                fps[(col, target_col)] = 1.0
    n1 = sum(1 for c in fps if len(c) == 1)
    n2 = sum(1 for c in fps if len(c) == 2)
    print(f"  PGM focal points: {len(fps)} cliques ({n1} 1-way, {n2} 2-way)")
    return fps


# ===========================================================================
# MAMA-MIA scoring
# ===========================================================================

def mama_mia_score(synth_enc, ref_enc, targets_enc, focal_points):
    """Compute MAMA-MIA likelihood-ratio scores for all target rows.

    Mirrors ``custom_mst_attack`` in conduct_attacks.py.
    For each clique C and target x:
        score(x) += weight * P_synth(x[C]) / P_ref(x[C])

    Returns
    -------
    np.ndarray  shape (n_targets,)
    """
    n         = len(targets_enc)
    A         = np.zeros(n)
    num_used  = np.zeros(n)
    default_v = 1e-10

    for clique, weight in focal_points.items():
        cols = [c for c in clique
                if c in synth_enc.columns
                and c in ref_enc.columns
                and c in targets_enc.columns]
        if not cols:
            continue

        D_synth = synth_enc[cols].value_counts(normalize=True)
        D_ref   = ref_enc[cols].value_counts(normalize=True)

        for i, val in enumerate(targets_enc[cols].values):
            key   = tuple(val)
            p_s   = D_synth.get(key, default=default_v)
            p_r   = D_ref.get(key,   default=default_v)
            A[i] += weight * (p_s / p_r)
            num_used[i] += weight

    return A / np.maximum(num_used, 1.0)


def scores_to_probs(scores):
    """Min-max scale raw LR scores to [0, 1]."""
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.full_like(scores, 0.5, dtype=float)
    return (scores - lo) / (hi - lo)


def _tpr_at_fpr(labels, scores, fpr_target=0.1):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(fpr, fpr_target)
    return float(tpr[min(idx, len(tpr) - 1)])


# ===========================================================================
# Top-level attack runner
# ===========================================================================

def run_attack(
    synthetic_path,
    reference_path,
    targets_path,
    output_path,
    epsilon=10.0,
    n_bins=10,
    target_col=None,
    gt_path=None,
    label_col='membership_label',
    standardize=True,
):
    """Full MAMA-MIA attack pipeline for one competition data split.

    Parameters
    ----------
    synthetic_path : str   Blue Team's DP-PGM synthetic CSV
    reference_path : str   Reference (non-training) data CSV
    targets_path   : str   Membership test TSV/CSV (samples × genes)
    output_path    : str   Where to write prediction CSV
    epsilon        : float DP epsilon used by Blue Team (informational)
    n_bins         : int   Discretization bins (match Blue Team's PGM setup)
    target_col     : str|None  PGM pivot column for 2-way marginals (None = 1-way only)
    gt_path        : str|None  Ground-truth CSV for evaluation
    label_col      : str   Membership label column name
    standardize    : bool  Apply StandardScaler before binning (competition default)

    Returns
    -------
    scores : np.ndarray   raw MAMA-MIA LR scores (one per target)
    """
    print(f"\n=== MAMA-MIA Competition Attack (ε={epsilon}) ===")

    # --- Load ---
    synth   = load_synthetic(synthetic_path)
    ref     = load_csv_or_tsv(reference_path) if reference_path else None
    targets = load_membership_test(targets_path)

    print(f"  Synth  : {synth.shape}   ({synthetic_path})")
    if ref is not None:
        print(f"  Ref    : {ref.shape}   ({reference_path})")
    print(f"  Targets: {targets.shape}  ({targets_path})")

    # --- Ground truth (optional) ---
    membership = None
    if gt_path and os.path.exists(gt_path):
        membership = load_gt(gt_path, label_col)
    elif label_col in targets.columns:
        membership = targets[label_col].values.astype(int)
        targets = targets.drop(columns=[label_col])

    # Drop label from synth/ref if present.
    synth = synth.drop(columns=[label_col], errors='ignore')
    if ref is not None:
        ref = ref.drop(columns=[label_col], errors='ignore')

    # --- Align columns ---
    if ref is not None:
        common_cols = [c for c in synth.columns
                       if c in ref.columns and c in targets.columns]
    else:
        common_cols = [c for c in synth.columns if c in targets.columns]
        # Without a reference, fall back to using synth itself as ref (weak).
        print("  WARNING: no reference data – using synthetic data as ref (weak attack).")
        ref = synth.copy()

    synth   = synth[common_cols]
    ref     = ref[common_cols]
    targets = targets[common_cols]

    feature_cols = [c for c in common_cols if c != target_col]
    print(f"  Columns: {len(common_cols)} features"
          + (f" + target_col={target_col!r}" if target_col else " (1-way only)"))

    # --- Encode ---
    synth_enc, ref_enc, targets_enc = encode_dataframes(
        synth, ref, targets, feature_cols, n_bins, standardize=standardize
    )

    # --- Focal points + scoring ---
    fps    = build_pgm_focal_points(common_cols, target_col)
    scores = mama_mia_score(synth_enc, ref_enc, targets_enc, fps)

    # --- Save predictions in competition format ---
    # Single column `membership_label` with raw scores.
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pd.DataFrame({'membership_label': scores}).to_csv(output_path, index=False)
    print(f"\n  Predictions → {output_path}")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    # --- Evaluate ---
    if membership is not None:
        try:
            probs = scores_to_probs(scores)
            auc   = roc_auc_score(membership, probs)
            ma    = 2 * auc - 1
            tpr01 = _tpr_at_fpr(membership, probs, 0.1)
            print(f"\n  AUC-ROC        : {auc:.4f}")
            print(f"  Membership Adv : {ma:.4f}")
            print(f"  TPR@FPR=0.1    : {tpr01:.4f}")
        except Exception as exc:
            print(f"  Evaluation error: {exc}")

    return scores


# ===========================================================================
# Competition directory helper
# ===========================================================================

def build_paths_from_competition_dir(
    competition_dir, dataset, generator, experiment_name, split_idx=1, output_dir=None
):
    """Resolve file paths from the Health-Privacy-Challenge directory layout.

    Expected layout:
        {competition_dir}/
        ├── data_splits/{dataset}/
        │   ├── synthetic_data_split_{split_idx}.csv   ← Blue Team output
        │   ├── synthetic_data_{split_idx}_gt.csv      ← ground truth
        │   ├── membership_test_{split_idx}.tsv        ← records to classify
        │   └── reference_data.csv
        └── results/mia/{dataset}/mama_mia/{generator}/{experiment_name}/
    """
    splits_dir = os.path.join(competition_dir, 'data_splits', dataset)
    out_dir = output_dir or os.path.join(
        competition_dir, 'results', 'mia', dataset,
        'mama_mia', generator, experiment_name,
        f'synthetic_data_{split_idx}',
    )
    os.makedirs(out_dir, exist_ok=True)

    def _first_existing(candidates):
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[0]

    synth_path = _first_existing([
        os.path.join(splits_dir, f'synthetic_data_split_{split_idx}.csv'),
        os.path.join(splits_dir, f'synthetic_{generator}_{experiment_name}_{split_idx}.csv'),
    ])
    ref_path = _first_existing([
        os.path.join(splits_dir, 'reference_data.csv'),
        os.path.join(splits_dir, 'reference.csv'),
    ])
    target_path = _first_existing([
        os.path.join(splits_dir, f'membership_test_{split_idx}.tsv'),
        os.path.join(splits_dir, f'target_data_{split_idx}.csv'),
    ])
    gt_candidates = [
        os.path.join(splits_dir, f'synthetic_data_{split_idx}_gt.csv'),
        os.path.join(splits_dir, f'gt_{split_idx}.csv'),
    ]
    gt_path = next((p for p in gt_candidates if os.path.exists(p)), None)

    return {
        'synthetic': synth_path,
        'reference': ref_path,
        'targets':   target_path,
        'output':    os.path.join(out_dir, f'synthetic_data_{split_idx}_predictions.csv'),
        'gt':        gt_path,
    }


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "MAMA-MIA black-box attack on DP-PGM Blue Team "
            "(Health-Privacy-Challenge competition)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[1] if "Usage" in __doc__ else "",
    )

    grp_a = parser.add_argument_group('Mode A – explicit file paths')
    grp_a.add_argument('--synthetic',  help='Blue Team synthetic CSV')
    grp_a.add_argument('--reference',  help='Reference (non-training) CSV (optional)')
    grp_a.add_argument('--targets',    help='Membership test TSV/CSV to classify')
    grp_a.add_argument('--output',     help='Output predictions CSV')
    grp_a.add_argument('--gt',         default=None,
                       help='Ground-truth labels CSV (optional, for evaluation)')

    grp_b = parser.add_argument_group('Mode B – competition directory layout')
    grp_b.add_argument('--competition-dir', metavar='DIR',
                       help='Root of Health-Privacy-Challenge clone')
    grp_b.add_argument('--dataset',         default='TCGA-BRCA')
    grp_b.add_argument('--generator',       default='dp_pgm')
    grp_b.add_argument('--experiment-name', default='epsilon_10.0')
    grp_b.add_argument('--split',           type=int, default=1,
                       help='Data split index 1–4 (default 1)')
    grp_b.add_argument('--all-splits',      action='store_true',
                       help='Run attack on all 4 splits')
    grp_b.add_argument('--output-dir',      default=None)

    grp_p = parser.add_argument_group('Attack parameters')
    grp_p.add_argument('--epsilon',      type=float, default=10.0)
    grp_p.add_argument('--n-bins',       type=int,   default=10,
                       help='Discretization bins (default 10)')
    grp_p.add_argument('--target-col',   default='',
                       help='PGM pivot column for 2-way marginals. '
                            'Empty string = 1-way only (default for competition gene data). '
                            'Use "Subtype" for our local TCGA data.')
    grp_p.add_argument('--label-col',    default='membership_label')
    grp_p.add_argument('--no-standardize', action='store_true',
                       help='Skip StandardScaler (use when data is already scaled)')

    args = parser.parse_args()
    target_col = args.target_col if args.target_col else None
    standardize = not args.no_standardize

    if args.competition_dir:
        splits = list(range(1, 5)) if args.all_splits else [args.split]
        for s in splits:
            paths = build_paths_from_competition_dir(
                competition_dir=os.path.expanduser(args.competition_dir),
                dataset=args.dataset,
                generator=args.generator,
                experiment_name=args.experiment_name,
                split_idx=s,
                output_dir=args.output_dir,
            )
            print(f"\n{'='*60}")
            print(f"Split {s}/4")
            run_attack(
                synthetic_path=paths['synthetic'],
                reference_path=paths['reference'],
                targets_path=paths['targets'],
                output_path=paths['output'],
                gt_path=paths['gt'],
                epsilon=args.epsilon,
                n_bins=args.n_bins,
                target_col=target_col,
                label_col=args.label_col,
                standardize=standardize,
            )

    elif args.synthetic and args.targets and args.output:
        run_attack(
            synthetic_path=args.synthetic,
            reference_path=args.reference,
            targets_path=args.targets,
            output_path=args.output,
            gt_path=args.gt,
            epsilon=args.epsilon,
            n_bins=args.n_bins,
            target_col=target_col,
            label_col=args.label_col,
            standardize=standardize,
        )
    else:
        parser.error(
            "Provide either --competition-dir (Mode B) or "
            "--synthetic / --targets / --output (Mode A)."
        )


if __name__ == '__main__':
    main()
