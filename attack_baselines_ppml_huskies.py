#!/usr/bin/env python3
"""attack_baselines_ppml_huskies.py — DOMIAS baselines on the PPML-Huskies submission.

Imports run_baselines() (and individual scoring functions) from the competition's
baseline.py without modifying it.  Data loading mirrors attack_ppml_huskies.py.

Baselines run per split
------------------------
  MC              – nearest-neighbour count (Hilprecht median heuristic)
  gan_leaks       – exp(-d_min / scale)  (scaled GAN-Leaks)
  conf_lr         – downstream LR classifier max-class confidence
  conf_rf         – downstream RF classifier max-class confidence
  LOGAN_D1        – reference-calibrated nearest-neighbour (requires ref)
  gan_leaks_cal   – calibrated GAN-Leaks                  (requires ref)
  domias_kde      – KDE density ratio on PCA-150 projection (requires ref)

Usage
-----
    python3 attack_baselines_ppml_huskies.py \\
        --submission-dir /path/to/blueteam_PPML-Huskies_TCGA-BRCA \\
        --competition-home ~/Health-Privacy-Challenge \\
        --output-dir results/baselines_brca

    # Point to the cloned competition repo so baseline.py is importable:
    python3 attack_baselines_ppml_huskies.py \\
        --submission-dir ... \\
        --competition-repo ~/Health-Privacy-Challenge \\
        --output-dir results/baselines_combined

Output
------
    <output-dir>/
        split_<N>_<baseline>_predictions.csv   # competition-format predictions
        baselines_summary.csv                   # AUC / MA per split per baseline
"""

import sys
import os
import argparse
import warnings
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Reuse data-loading helpers from attack_ppml_huskies.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attack_ppml_huskies import (
    load_blue_team_config,
    load_synth_with_labels,
    resolve_path,
)
from attack_submission import (
    load_tsv_with_subtypes,
    load_membership_from_yaml,
    _tpr_at_fpr,
)


def _setup_baseline_import(competition_repo: str):
    """Add competition repo src/ to sys.path so baseline.py is importable."""
    src_dir = os.path.join(os.path.expanduser(competition_repo), 'src')
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(
            f"Expected competition repo src/ at: {src_dir}\n"
            f"Clone with: git clone https://github.com/PMBio/Health-Privacy-Challenge.git"
        )
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Import the three standalone scoring functions we need
    from mia.models.baseline import run_baselines  # noqa: F401
    return run_baselines


# ---------------------------------------------------------------------------
# Per-split attack
# ---------------------------------------------------------------------------

def attack_split_baselines(
    split_idx:        int,
    submission_dir:   str,
    competition_home: str,
    blue_cfg:         dict,
    output_dir:       str,
    run_baselines_fn,
    use_reference:    bool,
) -> pd.DataFrame:
    """Run all DOMIAS baselines on one split.

    Returns a DataFrame with columns [baseline, AUC, MA, TPR@FPR=0.1]
    (empty if membership labels are unavailable).
    """
    print(f"\n{'='*65}")
    print(f"Split {split_idx}  —  DOMIAS baselines")
    print(f"{'='*65}")

    ds_cfg    = blue_cfg['dataset_config']
    dataset   = ds_cfg['name']
    label_col = ds_cfg['subtype_col_name']
    count_rel = ds_cfg['count_file']
    annot_rel = ds_cfg['annot_file']

    test_tsv = resolve_path(competition_home, count_rel)
    sub_csv  = resolve_path(competition_home, annot_rel)

    ref_tsv_candidate = test_tsv.replace('.tsv', '_reference.tsv')
    ref_tsv = ref_tsv_candidate if os.path.exists(ref_tsv_candidate) else None

    synth_path  = os.path.join(submission_dir, f'synthetic_data_split_{split_idx}.csv')
    labels_path = os.path.join(submission_dir, f'synthetic_labels_split_{split_idx}.csv')
    splits_yaml = os.path.join(submission_dir, f'{dataset}_splits.yaml')

    for p in [synth_path, labels_path, test_tsv, sub_csv]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    if not os.path.exists(splits_yaml):
        print(f"  WARNING: splits YAML not found at {splits_yaml} — "
              "AUC evaluation will be skipped.")
        splits_yaml = None

    # ---- Load -----------------------------------------------------------
    print("  Loading data …")
    synth   = load_synth_with_labels(synth_path, labels_path, label_col)
    targets = load_tsv_with_subtypes(test_tsv, sub_csv)

    if ref_tsv and use_reference:
        ref = load_tsv_with_subtypes(ref_tsv, sub_csv)
        ref_has_label = (label_col in ref.columns and
                         not (ref[label_col] == 'Unknown').all())
        if not ref_has_label:
            # Baseline attacks use raw gene arrays only (no label col) —
            # an unlabelled reference TSV is perfectly fine as P_ref.
            print(f"  Note: reference TSV has no '{label_col}' labels "
                  f"(OK — baseline attacks use gene arrays only).")
    else:
        if not ref_tsv:
            print("  No reference TSV → using test TSV as P_ref.")
        else:
            print("  --no-reference: skipping ref-dependent baselines.")
        ref = targets.copy() if use_reference else None

    # ---- Align gene columns --------------------------------------------
    ensg_synth   = [c for c in synth.columns   if c.startswith('ENSG')]
    ensg_ref     = {c for c in targets.columns if c.startswith('ENSG')}
    gene_cols    = [c for c in ensg_synth if c in ensg_ref]

    if not gene_cols:
        raise RuntimeError("No shared ENSG gene columns between synth and targets.")
    print(f"  Shared gene columns : {len(gene_cols)}")
    print(f"  Synth  : {synth.shape}   Targets : {targets.shape}")

    # ---- Build numpy arrays (raw floats — baselines use Euclidean dist) -
    X_G    = synth[gene_cols].values.astype(np.float64)
    X_test = targets[gene_cols].values.astype(np.float64)

    # Encode synth label column to integers for downstream classifier
    le    = LabelEncoder()
    y_G   = le.fit_transform(synth[label_col].fillna('Unknown').values)

    X_ref     = ref[gene_cols].values.astype(np.float64) if ref is not None else None
    X_ref_glc = X_ref  # GAN_leaks_cal uses same reference

    print(f"  X_G : {X_G.shape}   X_test : {X_test.shape}"
          f"   X_ref : {X_ref.shape if X_ref is not None else 'None'}")

    # ---- Membership labels ----------------------------------------------
    membership = None
    if splits_yaml:
        membership = load_membership_from_yaml(splits_yaml, split_idx, targets.index)

    # ---- Run baselines --------------------------------------------------
    print("  Running DOMIAS baselines …")
    scores_dict = run_baselines_fn(X_test, X_G, y_G, X_ref, X_ref_glc, None)

    # ---- Save per-baseline predictions ----------------------------------
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for baseline_name, raw_scores in scores_dict.items():
        raw = np.array(raw_scores, dtype=np.float64)

        # Normalise to [0,1]
        lo, hi = raw.min(), raw.max()
        probs  = (raw - lo) / (hi - lo) if hi > lo else np.full_like(raw, 0.5)

        out_path = os.path.join(
            output_dir, f'split_{split_idx}_{baseline_name}_predictions.csv'
        )
        pd.DataFrame({'membership_label': probs}).to_csv(out_path, index=False)
        print(f"  [{baseline_name}]  score range [{raw.min():.4f}, {raw.max():.4f}]"
              f"  → {os.path.basename(out_path)}")

        if membership is not None:
            auc_  = roc_auc_score(membership, probs)
            ma_   = 2 * auc_ - 1
            tpr_  = _tpr_at_fpr(membership, probs, 0.1)
            n_mem = int(membership.sum())
            print(f"         Members {n_mem}/{len(membership)}"
                  f"  AUC {auc_:.4f}  MA {ma_:.4f}  TPR@0.1 {tpr_:.4f}")
            rows.append({
                'split': split_idx, 'baseline': baseline_name,
                'AUC': auc_, 'MA': ma_, 'TPR@FPR=0.1': tpr_,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DOMIAS baselines on the PPML-Huskies Blue Team submission",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--submission-dir', required=True,
                        help='Blue Team submission directory (contains config.yaml, '
                             'synthetic_data_split_N.csv, synthetic_labels_split_N.csv, '
                             '{DATASET}_splits.yaml)')
    parser.add_argument('--competition-home', default='~/Health-Privacy-Challenge',
                        help='Root of Health-Privacy-Challenge clone (data files).')
    parser.add_argument('--competition-repo', default=None,
                        help='Root of Health-Privacy-Challenge clone (source code / '
                             'baseline.py). Defaults to --competition-home.')
    parser.add_argument('--output-dir', default='results/baselines_ppml_huskies',
                        help='Directory for prediction CSVs and summary.')
    parser.add_argument('--splits', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        help='Which splits to attack.')
    parser.add_argument('--no-reference', action='store_true', default=False,
                        help='Skip reference-dependent baselines '
                             '(LOGAN_D1, gan_leaks_cal, domias_kde). '
                             'Faster; useful when no reference TSV is available.')
    args = parser.parse_args()

    submission_dir   = os.path.expanduser(args.submission_dir)
    competition_home = os.path.expanduser(args.competition_home)
    competition_repo = os.path.expanduser(args.competition_repo or args.competition_home)
    output_dir       = os.path.expanduser(args.output_dir)

    # ---- Import run_baselines from the competition repo ----------------
    print(f"Setting up baseline.py import from: {competition_repo}")
    run_baselines_fn = _setup_baseline_import(competition_repo)
    print("baseline.py imported OK\n")

    # ---- Read Blue Team config -----------------------------------------
    blue_cfg = load_blue_team_config(submission_dir)
    ds_cfg   = blue_cfg['dataset_config']
    dataset  = ds_cfg['name']
    eps      = blue_cfg.get('pgg_pgm_config', {}).get('epsilon', '?')

    print(f"Blue Team: dataset={dataset}  ε={eps}")
    print(f"Splits   : {args.splits}")
    print(f"Output   : {output_dir}")

    # ---- Attack each split ---------------------------------------------
    all_rows = []
    for s in args.splits:
        df = attack_split_baselines(
            split_idx        = s,
            submission_dir   = submission_dir,
            competition_home = competition_home,
            blue_cfg         = blue_cfg,
            output_dir       = output_dir,
            run_baselines_fn = run_baselines_fn,
            use_reference    = not args.no_reference,
        )
        all_rows.append(df)

    # ---- Summary -------------------------------------------------------
    if all_rows:
        full = pd.concat(all_rows, ignore_index=True)
        if not full.empty:
            print(f"\n{'='*72}")
            print(f"Summary  ({dataset}  |  ε={eps})")
            print(f"{'='*72}")

            for baseline, grp in full.groupby('baseline'):
                print(f"\n  [{baseline}]")
                print(f"  {'Split':>6}  {'AUC':>8}  {'MA':>8}  {'TPR@0.1':>9}")
                print('  ' + '-'*36)
                for _, row in grp.iterrows():
                    print(f"  {int(row['split']):>6}  {row['AUC']:>8.4f}"
                          f"  {row['MA']:>8.4f}  {row['TPR@FPR=0.1']:>9.4f}")
                if len(grp) > 1:
                    print('  ' + '-'*36)
                    print(f"  {'mean':>6}  {grp['AUC'].mean():>8.4f}"
                          f"  {grp['MA'].mean():>8.4f}"
                          f"  {grp['TPR@FPR=0.1'].mean():>9.4f}")

            print(f"\n{'='*72}")
            summary_path = os.path.join(output_dir, 'baselines_summary.csv')
            full.to_csv(summary_path, index=False)
            print(f"Summary  → {summary_path}")

    print(f"\nPrediction files in {output_dir}/")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('_predictions.csv'):
            print(f"  {f}")


if __name__ == '__main__':
    main()
