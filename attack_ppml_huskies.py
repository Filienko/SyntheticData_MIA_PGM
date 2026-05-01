#!/usr/bin/env python3
"""attack_ppml_huskies.py — MAMA-MIA attack on the PPML-Huskies Blue Team submission.

Reads the Blue Team's own config.yaml to determine all dataset / model
parameters, then runs the MAMA-MIA likelihood-ratio attack on all 5 splits.

Parameters extracted automatically from Blue Team config
---------------------------------------------------------
  dataset_config.name             → TCGA-BRCA or TCGA-COMBINED
  dataset_config.subtype_col_name → label column ("Subtype" / "cancer_type")
  dataset_config.count_file       → path to gene-expression TSV
  dataset_config.annot_file       → path to sample-annotation CSV
  pgg_pgm_config.epsilon          → 7.0
  pgg_pgm_config.iterations       → 10000

Usage
-----
    # TCGA-COMBINED (all 5 splits):
    python3 attack_ppml_huskies.py \\
        --submission-dir /path/to/blueteam_PPML-Huskies_TCGA-COMBINED \\
        --competition-home ~/Health-Privacy-Challenge \\
        --output-dir results/ppml_huskies_combined

    # TCGA-BRCA:
    python3 attack_ppml_huskies.py \\
        --submission-dir /path/to/blueteam_PPML-Huskies_TCGA-BRCA \\
        --competition-home ~/Health-Privacy-Challenge \\
        --output-dir results/ppml_huskies_brca

    # Override n-bins or add 2-way marginals:
    python3 attack_ppml_huskies.py \\
        --submission-dir ... --competition-home ... \\
        --n-bins 4 --use-target-col

Output
------
    results/<output-dir>/
        synthetic_data_1_predictions.csv   # competition-format, single column
        synthetic_data_2_predictions.csv
        ...
        synthetic_data_5_predictions.csv
        attack_summary.csv                 # AUC / MA per split (if labels available)
"""

import sys
import os
import argparse
import warnings
import yaml
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append('reprosyn-main/src/reprosyn/methods/mbi/')

import mbi_patch  # noqa: F401

# Import the encoding / scoring helpers from attack_submission.py
from attack_submission import (
    load_tsv_with_subtypes,
    load_membership_from_yaml,
    encode_all,
    build_focal_points,
    mama_mia_score,
    compute_metrics,
)


# ---------------------------------------------------------------------------
# Helpers specific to the PPML-Huskies submission format
# ---------------------------------------------------------------------------

def load_blue_team_config(submission_dir: str) -> dict:
    """Load and return the Blue Team's config.yaml."""
    cfg_path = os.path.join(submission_dir, 'config.yaml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"config.yaml not found in submission directory: {submission_dir}\n"
            f"Expected: {cfg_path}"
        )
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def load_synth_with_labels(synth_path: str, labels_path: str, label_col: str) -> pd.DataFrame:
    """Load synthetic gene CSV and attach the label column.

    Handles two cases:
      - labels CSV column is already named `label_col`
      - labels CSV has a single column with a different name → auto-rename
    """
    synth  = pd.read_csv(synth_path)
    labels = pd.read_csv(labels_path)

    # Determine actual column name in the labels CSV
    actual_col = label_col if label_col in labels.columns else labels.columns[0]
    if actual_col != label_col:
        print(f"  Note: label column in CSV is '{actual_col}' (expected '{label_col}')")

    synth[label_col] = labels[actual_col].values
    return synth


def resolve_path(competition_home: str, relative_path: str) -> str:
    """Expand ~ and join competition_home with a relative path from config."""
    home = os.path.expanduser(competition_home)
    return os.path.join(home, relative_path)


# ---------------------------------------------------------------------------
# Per-split attack
# ---------------------------------------------------------------------------

def attack_split(
    split_idx:         int,
    submission_dir:    str,
    competition_home:  str,
    blue_cfg:          dict,
    output_dir:        str,
    n_bins:            int,
    use_target_col:    bool,
) -> tuple:
    """Attack one split of the PPML-Huskies submission.

    Returns (auc, ma) or (None, None) if labels not available.
    """
    print(f"\n{'='*65}")
    print(f"Split {split_idx}")
    print(f"{'='*65}")

    # ---- Extract parameters from Blue Team config -------------------
    ds_cfg      = blue_cfg['dataset_config']
    dataset     = ds_cfg['name']                    # TCGA-BRCA or TCGA-COMBINED
    label_col   = ds_cfg['subtype_col_name']        # "Subtype" or "cancer_type"
    count_rel   = ds_cfg['count_file']              # relative path to TSV
    annot_rel   = ds_cfg['annot_file']              # relative path to subtypes CSV

    target_col  = label_col if use_target_col else None

    # Absolute paths
    test_tsv  = resolve_path(competition_home, count_rel)
    sub_csv   = resolve_path(competition_home, annot_rel)

    # Reference TSV: look for a _reference.tsv sibling of count_file
    ref_tsv_candidate = test_tsv.replace('.tsv', '_reference.tsv')
    ref_tsv = ref_tsv_candidate if os.path.exists(ref_tsv_candidate) else None

    # ---- Submission files -------------------------------------------
    synth_path  = os.path.join(submission_dir, f'synthetic_data_split_{split_idx}.csv')
    labels_path = os.path.join(submission_dir, f'synthetic_labels_split_{split_idx}.csv')
    splits_yaml = os.path.join(submission_dir, f'{dataset}_splits.yaml')

    for p in [synth_path, test_tsv, sub_csv]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    if not os.path.exists(splits_yaml):
        print(f"  WARNING: splits YAML not found at {splits_yaml} — "
              "membership labels will not be available.")
        splits_yaml = None

    # ---- Load -------------------------------------------------------
    print("  Loading data …")
    if os.path.exists(labels_path):
        synth = load_synth_with_labels(synth_path, labels_path, label_col)
    else:
        synth = pd.read_csv(synth_path)
        if label_col not in synth.columns:
            raise FileNotFoundError(
                f"No labels file at {labels_path} and "
                f"no '{label_col}' column in {synth_path}.\n"
                f"  Columns in synth: {list(synth.columns[:10])}"
            )
        print(f"  No separate labels file — using '{label_col}' column from synth CSV")
    targets = load_tsv_with_subtypes(test_tsv, sub_csv)

    # Reference population priority:
    #  1. _reference.tsv sibling of count_file (held-out non-members)
    #  2. test_split_N.csv in submission_dir  (non-members for this split)
    #  3. Full test TSV as last resort         (contains members — degrades LR)
    test_split_csv = os.path.join(submission_dir, f'test_split_{split_idx}.csv')

    if ref_tsv:
        ref = load_tsv_with_subtypes(ref_tsv, sub_csv)
        ref_has_label = (label_col in ref.columns and
                         not (ref[label_col] == 'Unknown').all())
        if target_col and not ref_has_label:
            print(f"  Reference TSV has no '{label_col}' labels and 2-way marginals "
                  f"were requested → using test TSV as P_aux.")
            ref = targets.copy()
        elif not ref_has_label:
            print(f"  Note: reference TSV has no '{label_col}' labels "
                  f"(OK — using it for 1-way gene marginals only).")
    elif os.path.exists(test_split_csv):
        # Non-member split CSV (samples × genes, comma-separated, no subtype join needed)
        ref_raw = pd.read_csv(test_split_csv)
        # Attach subtype labels if possible (needed for 2-way marginals)
        if label_col not in ref_raw.columns:
            ref_raw[label_col] = 'Unknown'
        ref = ref_raw
        print(f"  No _reference.tsv → using non-member split: "
              f"{os.path.basename(test_split_csv)}  ({ref.shape[0]} samples)")
        if target_col and (ref[label_col] == 'Unknown').all():
            print(f"  Warning: test_split CSV has no '{label_col}' labels — "
                  f"2-way marginals will be degraded. Consider --use-target-col=False.")
    else:
        print(f"  No reference TSV or test_split CSV found → using full test TSV as P_aux.")
        print(f"  WARNING: test TSV contains training members; LR scores may collapse.")
        ref = targets.copy()

    print(f"  Synth  : {synth.shape}")
    print(f"  Ref    : {ref.shape}")
    print(f"  Targets: {targets.shape}")

    # ---- Membership labels ------------------------------------------
    membership = None
    if splits_yaml:
        membership = load_membership_from_yaml(splits_yaml, split_idx, targets.index)

    # ---- Align gene columns -----------------------------------------
    ensg_synth   = [c for c in synth.columns   if c.startswith('ENSG')]
    ensg_ref     = {c for c in ref.columns     if c.startswith('ENSG')}
    ensg_targets = {c for c in targets.columns if c.startswith('ENSG')}
    gene_cols    = [c for c in ensg_synth if c in ensg_ref and c in ensg_targets]

    if not gene_cols:
        raise RuntimeError(
            f"No shared ENSG gene columns found across synth / ref / targets.\n"
            f"  Synth columns (first 5): {list(synth.columns[:5])}\n"
            f"  Targets columns (first 5): {list(targets.columns[:5])}"
        )
    print(f"  Shared gene columns: {len(gene_cols)}")

    # ---- Build working DataFrames (genes + optional target_col) -----
    use_cols  = gene_cols + ([label_col] if target_col else [])
    synth_w   = synth[[c for c in use_cols if c in synth.columns]].copy()
    ref_w     = ref[[c for c in use_cols if c in ref.columns]].copy()
    targets_w = targets[[c for c in use_cols if c in targets.columns]].copy()

    # ---- Encode -----------------------------------------------------
    synth_enc, ref_enc, targets_enc = encode_all(
        synth_w, ref_w, targets_w, gene_cols, n_bins,
        name=f'{dataset.lower().replace("-", "_")}_split{split_idx}'
    )

    # ---- Focal points + scores -------------------------------------
    fps    = build_focal_points(gene_cols, target_col if target_col else None)
    scores = mama_mia_score(synth_enc, ref_enc, targets_enc, fps,
                            membership=membership)

    # ---- Save predictions ------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'synthetic_data_{split_idx}_predictions.csv')
    pd.DataFrame({'membership_label': scores}).to_csv(out_path, index=False)
    print(f"\n  Predictions → {out_path}")
    print(f"  Score range : [{scores.min():.4f}, {scores.max():.4f}]")

    # ---- Evaluate --------------------------------------------------
    if membership is not None:
        lo, hi = scores.min(), scores.max()
        probs  = (scores - lo) / (hi - lo) if hi > lo else np.full_like(scores, 0.5)
        m      = compute_metrics(probs, membership)
        n_mem  = int(membership.sum())
        print(f"  Members: {n_mem} / Non-members: {len(membership) - n_mem}")
        print(f"  AUC-ROC        : {m['AUC']:.4f}")
        print(f"  Membership Adv : {m['MA']:.4f}")
        print(f"  TPR@FPR=0.1    : {m['TPR@FPR=0.1']:.4f}")
        print(f"  PR_AUC         : {m['PR_AUC']:.4f}")
        print(f"  Precision@5pct : {m['Precision@5pct']:.4f}")
        return m

    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MAMA-MIA attack on the PPML-Huskies Blue Team submission",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--submission-dir', required=True,
        help=(
            'Path to the Blue Team submission directory, e.g. '
            '~/blueteam_PPML-Huskies_TCGA-COMBINED  '
            '(must contain config.yaml, synthetic_data_split_N.csv, '
            'synthetic_labels_split_N.csv, {DATASET}_splits.yaml)'
        ),
    )
    parser.add_argument(
        '--competition-home', default='~/Health-Privacy-Challenge',
        help=(
            'Root of the Health-Privacy-Challenge repository clone. '
            'Data files are resolved as competition_home / dataset_config.count_file '
            '(and annot_file) from the Blue Team config.yaml.'
        ),
    )
    parser.add_argument(
        '--output-dir', default='results/ppml_huskies_attack',
        help='Directory for prediction CSVs and summary.',
    )
    parser.add_argument(
        '--splits', nargs='+', type=int, default=[1, 2, 3, 4, 5],
        help='Which splits to attack (default: all 5).',
    )
    parser.add_argument(
        '--n-bins', type=int, default=4,
        help=(
            'Discretization bins. Blue Team uses 4 (Q25/Q50/Q75). '
            'Changing this will mismatch synth vs real encoding.'
        ),
    )
    parser.add_argument(
        '--use-target-col', action='store_true', default=False,
        help=(
            'Add 2-way (gene, label) marginals to the focal point set. '
            'Disabled by default because ~1000 genes × K subtypes creates '
            'very sparse 2-way marginals. '
            'Enable only when using a small gene subset (< 200 features).'
        ),
    )
    args = parser.parse_args()

    submission_dir   = os.path.expanduser(args.submission_dir)
    competition_home = os.path.expanduser(args.competition_home)
    output_dir       = os.path.expanduser(args.output_dir)

    # ---- Read Blue Team config --------------------------------------
    print(f"Reading Blue Team config: {os.path.join(submission_dir, 'config.yaml')}")
    blue_cfg = load_blue_team_config(submission_dir)

    ds_cfg    = blue_cfg['dataset_config']
    dataset   = ds_cfg['name']
    label_col = ds_cfg['subtype_col_name']
    eps       = blue_cfg.get('pgg_pgm_config', {}).get('epsilon', 10.0)
    iters     = blue_cfg.get('pgg_pgm_config', {}).get('iterations', 10000)

    print(f"\nBlue Team parameters:")
    print(f"  Dataset   : {dataset}")
    print(f"  Label col : {label_col}")
    print(f"  Epsilon   : {eps}")
    print(f"  Iterations: {iters}")
    print(f"  n_bins    : {args.n_bins}  (Blue Team hardcodes 4 bins)")
    print(f"  2-way marginals: {'yes, with ' + label_col if args.use_target_col else 'no (1-way only)'}")

    # ---- Attack each split ------------------------------------------
    rows = []
    for s in args.splits:
        m = attack_split(
            split_idx        = s,
            submission_dir   = submission_dir,
            competition_home = competition_home,
            blue_cfg         = blue_cfg,
            output_dir       = output_dir,
            n_bins           = args.n_bins,
            use_target_col   = args.use_target_col,
        )
        if m is not None:
            rows.append({'split': s, **m})

    # ---- Print summary ----------------------------------------------
    if rows:
        df   = pd.DataFrame(rows)
        mean = df.drop(columns='split').mean().to_dict()
        mean['split'] = 'mean'
        df_out = pd.concat([df, pd.DataFrame([mean])], ignore_index=True)

        hdr_metrics = ['AUC','MA','acc_best','f1_best',
                       'TPR@FPR=0.01','TPR@FPR=0.1','PR_AUC','Precision@5pct']
        print(f"\n{'='*80}")
        print(f"Summary  ({dataset}  |  ε={eps}  |  bins={args.n_bins})")
        print(f"{'='*80}")
        header = f"  {'Split':>6}" + "".join(f"  {m:>14}" for m in hdr_metrics)
        print(header)
        print('  ' + '-' * (len(header) - 2))
        for _, row in df.iterrows():
            vals = "".join(f"  {row[m]:>14.4f}" for m in hdr_metrics)
            print(f"  {int(row['split']):>6}{vals}")
        if len(df) > 1:
            print('  ' + '-' * (len(header) - 2))
            vals = "".join(f"  {mean[m]:>14.4f}" for m in hdr_metrics)
            print(f"  {'mean':>6}{vals}")
        print('=' * 80)

        summary_path = os.path.join(output_dir, 'attack_summary.csv')
        df_out.to_csv(summary_path, index=False)
        print(f"\nSummary → {summary_path}")

    print(f"\nPrediction files in {output_dir}/")
    for s in args.splits:
        p = os.path.join(output_dir, f'synthetic_data_{s}_predictions.csv')
        if os.path.exists(p):
            print(f"  {os.path.basename(p)}")


if __name__ == '__main__':
    main()
