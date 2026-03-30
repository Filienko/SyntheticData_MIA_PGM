"""MAMA-MIA attack integrated with Health-Privacy-Challenge competition format.

Adapts our Private-PGM MAMA-MIA attack to the competition's black-box setting:
  - Uses the competition's pre-generated DP-PGM synthetic data as the member
    distribution proxy (instead of training PGM ourselves).
  - Uses the competition's reference data as the non-member distribution.
  - Computes MAMA-MIA marginal likelihood-ratio scores for each target record.
  - Outputs membership probability predictions in competition CSV format.

Why this works for DP-PGM
--------------------------
Private-PGM uses a *fixed* marginal structure (deterministic, no exponential-
mechanism randomness):
  - All 1-way singletons: {col} for every feature column
  - All 2-way pairs:      {col, target_variable} for every non-target column

Because we know these cliques exactly from the algorithm spec, shadow modelling
is unnecessary. The competition's synthetic data encodes the learned distribution
over these cliques; we score each target by how much better its marginal values
match the synthetic distribution vs. the reference distribution.

Usage (standalone)
------------------
    python3 competition_mia.py \\
        --synthetic  <path/synthetic_data.csv> \\
        --reference  <path/reference_data.csv> \\
        --targets    <path/targets.csv> \\
        --output     <path/predictions.csv> \\
        [--epsilon   10.0] \\
        [--n-bins    10] \\
        [--target-col Subtype] \\
        [--gt        <path/gt_labels.csv>]

Usage with competition directory layout
----------------------------------------
    python3 competition_mia.py \\
        --competition-dir ~/Health-Privacy-Challenge \\
        --dataset TCGA-BRCA \\
        --generator dp_pgm \\
        --experiment-name epsilon_10.0 \\
        --output-dir ~/Health-Privacy-Challenge/results/mia

Expected competition directory structure
-----------------------------------------
    {competition_dir}/
    ├── data_splits/
    │   └── {dataset}/
    │       ├── synthetic_data_{generator}_{experiment}.csv   <- Blue Team output
    │       ├── target_data_{generator}_{experiment}.csv      <- records to classify
    │       └── reference_data.csv                            <- non-training pool
    └── results/
        └── mia/
            └── {dataset}/
                └── mama_mia/
                    └── {generator}/
                        └── {experiment}/
                            └── predictions.csv

Output CSV format
-----------------
    membership_probability  (float in [0, 1], higher = more likely member)
    score                   (raw unnormalised MAMA-MIA LR score)

Notes
-----
- If the data is already integer-encoded (as PGM synthetic data often is),
  discretization is skipped automatically.
- For datasets other than TCGA-BRCA, set --target-col to the appropriate
  categorical column used by the Blue Team's PGM (or '' to use 1-way only).
- The --epsilon argument is used only for the weight threshold in clique
  selection; all cliques pass through at the default threshold (0.0).
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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
from util import C, dump_artifact

# Ensure required directories exist.
os.makedirs("data/experiment_artifacts", exist_ok=True)


# ===========================================================================
# Core helpers
# ===========================================================================

def build_pgm_focal_points(columns, target_col):
    """Return the fixed marginal structure of Private-PGM.

    Structure (deterministic – no shadow modelling required):
      - 1-way singletons: {col} for every column
      - 2-way pairs:      {col, target_col} for every col != target_col

    Returns
    -------
    dict  {clique_tuple: weight}   all weights = 1.0 (uniform)
    """
    fps = {(col,): 1.0 for col in columns}
    if target_col and target_col in columns:
        for col in columns:
            if col != target_col:
                fps[(col, target_col)] = 1.0
    print(f"  PGM focal points: {len(fps)} cliques "
          f"({sum(1 for c in fps if len(c) == 1)} 1-way, "
          f"{sum(1 for c in fps if len(c) == 2)} 2-way)")
    return fps


def _is_already_encoded(df, feature_cols, n_bins):
    """Heuristic: True if all feature columns are non-negative integers < 2*n_bins."""
    sub = df[feature_cols].select_dtypes(include=[np.number])
    if sub.empty:
        return False
    if not sub.dtypes.apply(lambda d: np.issubdtype(d, np.integer)).all():
        # Allow float columns that are actually integer-valued
        if not np.allclose(sub.values, sub.values.astype(int), equal_nan=True):
            return False
    return (sub.values >= 0).all() and (sub.values < n_bins * 2).all()


def encode_dataframes(synth, ref, targets, feature_cols, n_bins, name='competition'):
    """Fit equal-depth bins on synth+ref, then discretize all three DataFrames.

    If data appears already integer-encoded, returns the inputs unchanged.
    Encoding state is saved as an artifact so downstream calls are consistent.
    """
    if _is_already_encoded(synth, feature_cols, n_bins):
        print("  Data appears already discretized – skipping re-encoding.")
        return synth.copy(), ref.copy(), targets.copy()

    print(f"  Discretizing continuous features into {n_bins} equal-depth bins …")
    C.n_bins = n_bins

    # Fit on synthetic + reference combined so the bin edges are consistent.
    combined = pd.concat([synth[feature_cols], ref[feature_cols]], ignore_index=True)
    fit_continuous_features_equaldepth(combined, name)

    def _enc(df):
        out = df.copy()
        out[feature_cols] = discretize_continuous_features_equaldepth(
            df[feature_cols], name
        )
        return out

    return _enc(synth), _enc(ref), _enc(targets)


def mama_mia_score(synth_enc, ref_enc, targets_enc, focal_points):
    """Compute MAMA-MIA likelihood-ratio scores for all target rows.

    Mirrors ``custom_mst_attack`` in conduct_attacks.py but dependency-free
    (no cfg / target_ids / membership needed).

    For each clique C and each target x:
        score(x) += weight * P_synth(x[C]) / P_ref(x[C])

    where P_synth and P_ref are the empirical marginal distributions of the
    synthetic data and the reference data respectively.

    Parameters
    ----------
    synth_enc, ref_enc, targets_enc : pd.DataFrame  (integer-encoded)
    focal_points : dict  {clique_tuple: float weight}

    Returns
    -------
    np.ndarray  shape (n_targets,)  – raw (unnormalised) LR sum scores
    """
    n = len(targets_enc)
    A         = np.zeros(n)
    num_used  = np.zeros(n)
    default_v = 1e-10

    for clique, weight in focal_points.items():
        cols = [c for c in clique
                if c in synth_enc.columns and c in ref_enc.columns
                and c in targets_enc.columns]
        if not cols:
            continue

        D_synth = synth_enc[cols].value_counts(normalize=True)
        D_ref   = ref_enc[cols].value_counts(normalize=True)

        for i, val in enumerate(targets_enc[cols].values):
            key    = tuple(val)
            p_s    = D_synth.get(key, default=default_v)
            p_r    = D_ref.get(key,   default=default_v)
            A[i]  += weight * (p_s / p_r)
            num_used[i] += weight

    # Normalise by total weight used per target (avoids scale sensitivity).
    normaliser = np.maximum(num_used, 1.0)
    return A / normaliser


def scores_to_probs(scores):
    """Min-max scale raw LR scores to membership probabilities in [0, 1]."""
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.full_like(scores, 0.5, dtype=float)
    return (scores - lo) / (hi - lo)


# ===========================================================================
# Main attack pipeline
# ===========================================================================

def run_attack(
    synthetic_path,
    reference_path,
    targets_path,
    output_path,
    epsilon=10.0,
    n_bins=10,
    target_col='Subtype',
    gt_path=None,
    label_col='membership_label',
):
    """Full MAMA-MIA attack pipeline for one competition data split.

    Parameters
    ----------
    synthetic_path : str   Blue Team's generated synthetic CSV
    reference_path : str   Reference (non-training) data CSV
    targets_path   : str   Target records CSV (mix of members / non-members)
    output_path    : str   Where to write prediction CSV
    epsilon        : float DP epsilon used by Blue Team (informational / logging)
    n_bins         : int   Discretization bins (must match Blue Team's PGM setup)
    target_col     : str   PGM target variable column name (e.g. 'Subtype')
    gt_path        : str   Optional ground-truth labels CSV for evaluation
    label_col      : str   Name of the membership label column

    Returns
    -------
    (scores, probs)  – raw LR scores and [0,1] membership probabilities
    """
    print("\n=== MAMA-MIA Competition Attack ===")
    print(f"  epsilon    : {epsilon}")
    print(f"  n_bins     : {n_bins}")
    print(f"  target_col : {target_col!r}")

    # ------------------------------------------------------------------
    # Load CSVs
    # ------------------------------------------------------------------
    synth   = pd.read_csv(synthetic_path)
    ref     = pd.read_csv(reference_path)
    targets = pd.read_csv(targets_path)

    print(f"  Synth  : {len(synth)} rows, {synth.shape[1]} cols  ({synthetic_path})")
    print(f"  Ref    : {len(ref)} rows  ({reference_path})")
    print(f"  Targets: {len(targets)} rows  ({targets_path})")

    # ------------------------------------------------------------------
    # Extract ground-truth membership labels (for evaluation only)
    # ------------------------------------------------------------------
    membership = None
    if label_col in targets.columns:
        membership = targets[label_col].values.copy()
        targets = targets.drop(columns=[label_col])
    if gt_path and os.path.exists(gt_path):
        gt_df = pd.read_csv(gt_path)
        if label_col in gt_df.columns:
            membership = gt_df[label_col].values

    # Drop label column from synth / ref if accidentally present.
    synth = synth.drop(columns=[label_col], errors='ignore')
    ref   = ref.drop(columns=[label_col], errors='ignore')

    # ------------------------------------------------------------------
    # Align columns across all three DataFrames
    # ------------------------------------------------------------------
    common_cols = [c for c in synth.columns
                   if c in ref.columns and c in targets.columns]
    synth   = synth[common_cols]
    ref     = ref[common_cols]
    targets = targets[common_cols]

    feature_cols = [c for c in common_cols if c != target_col]
    print(f"  Feature cols: {len(feature_cols)} | target_col: {target_col!r}")

    # ------------------------------------------------------------------
    # Encode / discretize
    # ------------------------------------------------------------------
    synth_enc, ref_enc, targets_enc = encode_dataframes(
        synth, ref, targets, feature_cols, n_bins
    )

    # ------------------------------------------------------------------
    # Build PGM focal points & run attack
    # ------------------------------------------------------------------
    fps    = build_pgm_focal_points(common_cols, target_col)
    scores = mama_mia_score(synth_enc, ref_enc, targets_enc, fps)
    probs  = scores_to_probs(scores)

    # ------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out_df = pd.DataFrame({
        'membership_probability': probs,
        'score': scores,
    })
    out_df.to_csv(output_path, index=False)
    print(f"\n  Predictions saved → {output_path}")

    # ------------------------------------------------------------------
    # Evaluate if ground truth is available
    # ------------------------------------------------------------------
    if membership is not None:
        try:
            auc = roc_auc_score(membership, probs)
            ma  = 2 * auc - 1
            tpr_at_fpr01 = _tpr_at_fpr(membership, probs, fpr_target=0.1)
            print(f"\n  --- Evaluation ---")
            print(f"  AUC-ROC         : {auc:.4f}")
            print(f"  Membership Adv. : {ma:.4f}")
            print(f"  TPR @ FPR=0.1   : {tpr_at_fpr01:.4f}")
        except Exception as exc:
            print(f"  Evaluation error: {exc}")

    return scores, probs


def _tpr_at_fpr(labels, scores, fpr_target=0.1):
    """Return TPR at the threshold where FPR ≈ fpr_target."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(fpr, fpr_target)
    if idx >= len(tpr):
        return tpr[-1]
    return float(tpr[idx])


# ===========================================================================
# Competition directory-layout helper
# ===========================================================================

def build_paths_from_competition_dir(
    competition_dir, dataset, generator, experiment_name, output_dir=None
):
    """Resolve file paths from the Health-Privacy-Challenge directory layout.

    Expected layout (based on competition config.yaml):
        {competition_dir}/
        ├── data_splits/{dataset}/
        │   ├── synthetic_data_{generator}_{experiment_name}.csv
        │   ├── target_data_{generator}_{experiment_name}.csv
        │   └── reference_data.csv
        └── results/mia/{dataset}/mama_mia/{generator}/{experiment_name}/

    Returns
    -------
    dict with keys: synthetic, reference, targets, output, gt (may be None)
    """
    splits_dir = os.path.join(competition_dir, 'data_splits', dataset)
    out_dir = output_dir or os.path.join(
        competition_dir, 'results', 'mia', dataset,
        'mama_mia', generator, experiment_name
    )
    os.makedirs(out_dir, exist_ok=True)

    def _find(pattern_candidates):
        """Return first existing path from a list of candidates."""
        for p in pattern_candidates:
            if os.path.exists(p):
                return p
        return pattern_candidates[0]  # return first candidate even if missing

    synth_path = _find([
        os.path.join(splits_dir, f'synthetic_data_{generator}_{experiment_name}.csv'),
        os.path.join(splits_dir, f'synthetic_data_{experiment_name}.csv'),
        os.path.join(splits_dir, f'synthetic_data_split_1.csv'),
    ])
    ref_path = _find([
        os.path.join(splits_dir, 'reference_data.csv'),
        os.path.join(splits_dir, 'reference.csv'),
    ])
    target_path = _find([
        os.path.join(splits_dir, f'target_data_{generator}_{experiment_name}.csv'),
        os.path.join(splits_dir, f'targets_{generator}_{experiment_name}.csv'),
        os.path.join(splits_dir, 'target_data.csv'),
    ])
    gt_path = _find([
        os.path.join(splits_dir, f'synthetic_data_{generator}_{experiment_name}_gt.csv'),
        os.path.join(splits_dir, 'gt_labels.csv'),
    ])
    gt_path = gt_path if os.path.exists(gt_path) else None
    output_path = os.path.join(out_dir, 'predictions.csv')

    return {
        'synthetic': synth_path,
        'reference': ref_path,
        'targets':   target_path,
        'output':    output_path,
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
        epilog="""
Examples
--------
# Direct paths:
  python3 competition_mia.py \\
      --synthetic  data_splits/TCGA-BRCA/synthetic_data_dp_pgm_eps10.csv \\
      --reference  data_splits/TCGA-BRCA/reference_data.csv \\
      --targets    data_splits/TCGA-BRCA/target_data_dp_pgm_eps10.csv \\
      --output     results/mia/TCGA-BRCA/mama_mia/dp_pgm/eps10/predictions.csv

# Competition directory layout:
  python3 competition_mia.py \\
      --competition-dir ~/Health-Privacy-Challenge \\
      --dataset TCGA-BRCA \\
      --generator dp_pgm \\
      --experiment-name epsilon_10.0
""",
    )

    # --- Mode A: explicit paths ---
    grp_paths = parser.add_argument_group('Explicit file paths (Mode A)')
    grp_paths.add_argument('--synthetic',  help='Blue Team synthetic CSV')
    grp_paths.add_argument('--reference',  help='Reference (non-training) CSV')
    grp_paths.add_argument('--targets',    help='Target records CSV to classify')
    grp_paths.add_argument('--output',     help='Output predictions CSV')
    grp_paths.add_argument('--gt',         default=None,
                           help='Ground-truth labels CSV (optional, for evaluation)')

    # --- Mode B: competition directory layout ---
    grp_comp = parser.add_argument_group('Competition directory layout (Mode B)')
    grp_comp.add_argument('--competition-dir', metavar='DIR',
                          help='Root of Health-Privacy-Challenge repo clone')
    grp_comp.add_argument('--dataset',         default='TCGA-BRCA',
                          help='Dataset name, e.g. TCGA-BRCA or TCGA-COMBINED')
    grp_comp.add_argument('--generator',       default='dp_pgm',
                          help='Blue Team generator name, e.g. dp_pgm')
    grp_comp.add_argument('--experiment-name', default='epsilon_10.0',
                          help='Blue Team experiment variant, e.g. epsilon_10.0')
    grp_comp.add_argument('--output-dir',      default=None,
                          help='Override output directory (Mode B only)')

    # --- Attack parameters ---
    grp_atk = parser.add_argument_group('Attack parameters')
    grp_atk.add_argument('--epsilon',    type=float, default=10.0,
                         help='DP epsilon used by Blue Team (default 10.0)')
    grp_atk.add_argument('--n-bins',     type=int,   default=10,
                         help='Discretization bins (default 10, match Blue Team)')
    grp_atk.add_argument('--target-col', default='Subtype',
                         help='PGM target variable column (default: Subtype). '
                              'Use empty string "" for 1-way marginals only.')
    grp_atk.add_argument('--label-col',  default='membership_label',
                         help='Membership label column in target CSV')

    args = parser.parse_args()

    # Resolve paths
    if args.competition_dir:
        paths = build_paths_from_competition_dir(
            competition_dir=os.path.expanduser(args.competition_dir),
            dataset=args.dataset,
            generator=args.generator,
            experiment_name=args.experiment_name,
            output_dir=args.output_dir,
        )
        synthetic_path = paths['synthetic']
        reference_path = paths['reference']
        targets_path   = paths['targets']
        output_path    = paths['output']
        gt_path        = paths['gt']
    elif args.synthetic and args.reference and args.targets and args.output:
        synthetic_path = args.synthetic
        reference_path = args.reference
        targets_path   = args.targets
        output_path    = args.output
        gt_path        = args.gt
    else:
        parser.error(
            "Provide either --competition-dir (Mode B) or "
            "--synthetic / --reference / --targets / --output (Mode A)."
        )

    target_col = args.target_col if args.target_col else None

    run_attack(
        synthetic_path=synthetic_path,
        reference_path=reference_path,
        targets_path=targets_path,
        output_path=output_path,
        epsilon=args.epsilon,
        n_bins=args.n_bins,
        target_col=target_col,
        gt_path=gt_path,
        label_col=args.label_col,
    )


if __name__ == '__main__':
    main()
