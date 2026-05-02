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
    python3 attack_ppml_huskies.py \
        --submission-dir /path/to/blueteam_PPML-Huskies_TCGA-COMBINED \
        --competition-home ~/Health-Privacy-Challenge \
        --output-dir results/ppml_huskies_combined

    # TCGA-BRCA:
    python3 attack_ppml_huskies.py \
        --submission-dir /path/to/blueteam_PPML-Huskies_TCGA-BRCA \
        --competition-home ~/Health-Privacy-Challenge \
        --output-dir results/ppml_huskies_brca

    # Override n-bins or add 2-way marginals:
    python3 attack_ppml_huskies.py \
        --submission-dir ... --competition-home ... \
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
# Shared helper
# ---------------------------------------------------------------------------

def _try_join_subtype_early(df: pd.DataFrame, sub_csv: str,
                             label_col: str) -> pd.DataFrame:
    """Join label_col from sub_csv onto df using its index as sample IDs."""
    if label_col in df.columns and not (df[label_col] == 'Unknown').all():
        return df
    sub = pd.read_csv(sub_csv, index_col=0)
    if 'samplesID' in sub.columns:
        sub = sub.set_index('samplesID')
    lbl_src = label_col if label_col in sub.columns else sub.columns[0]
    joined = df.join(sub[[lbl_src]].rename(columns={lbl_src: label_col}),
                     how='left')
    n_joined = int(joined[label_col].notna().sum())
    joined[label_col] = joined[label_col].fillna('Unknown')
    if n_joined > 0:
        print(f"    Subtype join: {n_joined}/{len(df)} samples matched")
    else:
        print(f"    Subtype join: no sample IDs matched sub_csv — "
              f"'{label_col}' left as 'Unknown'")
    return joined


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
    ref_mode:          str   = 'auto',
    ref_csv:           str   = None,
    ref_tsv:           str   = None,
    test_tsv_override: str   = None,
    sub_csv_override:  str   = None,
    splits_yaml_override: str = None,
    decontaminate:     bool  = False,
    member_frac:       float = None,
) -> tuple:
    """Attack one split of the PPML-Huskies submission.

    Parameters
    ----------
    ref_mode      : 'auto' | 'full'
    ref_csv       : override P_ref with this CSV (auto Subtype join attempted).
    decontaminate : algebraically remove the member contribution from P_ref:
                    P_nonmem = (P_pool − α·P_synth) / (1−α).
                    Intended for use with --ref-csv (whole population pool).
                    α is computed as N_synth/N_ref unless --member-frac is set.
    member_frac   : explicit α (0 < α < 1).  If None, auto-computed.

    Returns metric dict, or None if labels are unavailable.
    """
    print(f"\n{'='*65}")
    print(f"Split {split_idx}")
    print(f"{'='*65}")

    # ---- Extract parameters from Blue Team config -------------------
    ds_cfg      = blue_cfg['dataset_config']
    dataset     = ds_cfg['name']                     # TCGA-BRCA or TCGA-COMBINED
    label_col   = ds_cfg['subtype_col_name']         # "Subtype" or "cancer_type"
    count_rel   = ds_cfg['count_file']               # relative path to TSV
    annot_rel   = ds_cfg['annot_file']               # relative path to subtypes CSV

    target_col  = label_col if use_target_col else None

    # Absolute paths (CLI overrides take priority over config.yaml)
    test_tsv  = test_tsv_override if test_tsv_override else resolve_path(competition_home, count_rel)
    sub_csv   = sub_csv_override  if sub_csv_override  else resolve_path(competition_home, annot_rel)

    # Reference TSV: look for a _reference.tsv sibling of test_tsv
    # Only use auto-detected path if no explicit --ref-tsv was passed
    ref_tsv_candidate = test_tsv.replace('.tsv', '_reference.tsv')
    ref_tsv_auto = ref_tsv_candidate if os.path.exists(ref_tsv_candidate) else None
    if ref_tsv is None:
        ref_tsv = ref_tsv_auto

    # ---- Submission files -------------------------------------------
    synth_path  = os.path.join(submission_dir, f'synthetic_data_split_{split_idx}.csv')
    labels_path = os.path.join(submission_dir, f'synthetic_labels_split_{split_idx}.csv')
    splits_yaml = (splits_yaml_override
                   if splits_yaml_override
                   else os.path.join(submission_dir, f'{dataset}_splits.yaml'))

    for p in [synth_path, test_tsv, sub_csv]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    if not os.path.exists(splits_yaml):
        print(f"  WARNING: splits YAML not found at {splits_yaml} — "
              "membership labels will not be available.")
        splits_yaml = None

    # ---- Print resolved paths for verification ----------------------
    print(f"  [paths] test_tsv  : {test_tsv}")
    print(f"  [paths] sub_csv   : {sub_csv}")
    print(f"  [paths] synth_csv : {synth_path}")
    if os.path.exists(labels_path):
        print(f"  [paths] labels_csv: {labels_path}")
    else:
        print(f"  [paths] labels_csv: (none — expecting label col in synth CSV)")
    ref_tsv_exists = ref_tsv is not None and os.path.exists(ref_tsv)
    print(f"  [paths] ref_tsv   : {ref_tsv}  (exists={ref_tsv_exists})")

    # ---- Load -------------------------------------------------------
    print("  Loading data …")
    targets = load_tsv_with_subtypes(test_tsv, sub_csv)

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

    # ---- Reference population selection --------------------------------
    # Priority (highest to lowest):
    #  --ref-csv <path>   → load that CSV directly as P_ref
    #  --ref-mode full    → full test TSV  (members+non-members, has subtype labels)
    #  _reference.tsv     → dedicated held-out TSV (competition standard)
    #  test_split_N.csv   → per-split non-member CSV (auto Subtype join attempted)
    #  full test TSV      → last resort (warns about member contamination)
    test_split_csv = os.path.join(submission_dir, f'test_split_{split_idx}.csv')

    def _try_join_subtype(df: pd.DataFrame) -> pd.DataFrame:
        return _try_join_subtype_early(df, sub_csv, label_col)

    if ref_csv:
        ref = pd.read_csv(ref_csv, index_col=0)
        ref = _try_join_subtype(ref)
        print(f"  --ref-csv → P_ref = {os.path.basename(ref_csv)}  "
              f"(n={ref.shape[0]}, '{label_col}' known: "
              f"{(ref[label_col] != 'Unknown').sum()})")
        if target_col and (ref[label_col] == 'Unknown').all():
            print(f"  Warning: ref_csv has no '{label_col}' labels — "
                  f"2-way marginals degraded.")
    elif ref_tsv:
        ref = load_tsv_with_subtypes(ref_tsv, sub_csv)
        print(f"  --ref-tsv → P_ref = {os.path.basename(ref_tsv)}  "
              f"(n={ref.shape[0]}, '{label_col}' known: "
              f"{(ref[label_col] != 'Unknown').sum() if label_col in ref.columns else 0})")
    elif ref_mode == 'full':
        ref = targets.copy()
        print(f"  ref_mode=full → P_ref = full test TSV  "
              f"(n={ref.shape[0]}, has '{label_col}' labels).")
        print(f"  NOTE: full test TSV contains training members (~80%); "
              f"1-way LR will collapse. Only 2-way (gene,subtype) terms give signal.")
    elif ref_tsv:
        ref = load_tsv_with_subtypes(ref_tsv, sub_csv)
        ref_has_label = (label_col in ref.columns and
                         not (ref[label_col] == 'Unknown').all())
        if target_col and not ref_has_label:
            print(f"  Reference TSV has no '{label_col}' labels and 2-way marginals "
                  f"were requested → falling back to full test TSV as P_aux.")
            ref = targets.copy()
        elif not ref_has_label:
            print(f"  Note: reference TSV has no '{label_col}' labels "
                  f"(OK — 1-way gene marginals only).")
    elif os.path.exists(test_split_csv):
        ref_raw = pd.read_csv(test_split_csv, index_col=0)
        ref_raw = _try_join_subtype(ref_raw)
        ref = ref_raw
        has_lbl = (ref[label_col] != 'Unknown').sum()
        print(f"  No _reference.tsv → non-member split: "
              f"{os.path.basename(test_split_csv)}  "
              f"(n={ref.shape[0]}, '{label_col}' known: {has_lbl})")
        if target_col and has_lbl == 0:
            print(f"  Warning: no subtype labels in P_ref — "
                  f"2-way marginals degraded. Try --ref-csv data/tcga_brca_full.csv.")
    else:
        print(f"  No reference source found → full test TSV as P_aux (last resort).")
        print(f"  WARNING: ~80% members in P_ref; use --ref-csv for a cleaner reference.")
        ref = targets.copy()

    # ---- Spot-checks on loaded data ---------------------------------
    first_gene = next((c for c in synth.columns if c.startswith('ENSG')), None)
    synth_lbl_vc = synth[label_col].value_counts().to_dict() if label_col in synth.columns else {}
    ref_lbl_vc   = ref[label_col].value_counts().to_dict()   if label_col in ref.columns   else {}

    print(f"  Synth  : {synth.shape}  |  {label_col} dist: {synth_lbl_vc}")
    if first_gene:
        uniq = sorted(synth[first_gene].dropna().unique())
        print(f"    first gene ({first_gene}): {len(uniq)} unique vals → {uniq[:8]}")
    print(f"  Ref    : {ref.shape}  |  {label_col} dist: {ref_lbl_vc}")
    if first_gene and first_gene in ref.columns:
        ref_uniq = sorted(ref[first_gene].dropna().unique())
        print(f"    first gene in ref: {len(ref_uniq)} unique vals, "
              f"range [{ref[first_gene].min():.3f}, {ref[first_gene].max():.3f}]")
    print(f"  Targets: {targets.shape}  |  index sample: {list(targets.index[:3])}")

    # ---- Membership labels ------------------------------------------
    membership = None
    if splits_yaml:
        membership = load_membership_from_yaml(splits_yaml, split_idx, targets.index)
        n_mem  = int(membership.sum())
        n_non  = len(membership) - n_mem
        print(f"  Membership labels: {n_mem} members / {n_non} non-members "
              f"(out of {len(membership)} candidates)  "
              f"[from {os.path.basename(splits_yaml)}]")

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
                            membership=membership,
                            decontaminate=decontaminate,
                            alpha=member_frac)

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
            'Doubles focal point count (978 → 1956 for BRCA). '
            'Requires subtype labels in P_ref; use --ref-mode full when '
            'test_split CSVs carry no subtype annotations.'
        ),
    )
    parser.add_argument(
        '--ref-mode', choices=['auto', 'full'], default='auto',
        help=(
            "'auto': use _reference.tsv > test_split_N.csv > full test TSV. "
            "'full': always use the full test TSV (members+non-members) as P_ref."
        ),
    )
    parser.add_argument(
        '--ref-csv', default=None,
        help=(
            'Path to a CSV file to use directly as P_ref (overrides --ref-mode). '
            'Must have ENSG* gene columns and sample IDs as the index. '
            'Subtype labels are auto-joined from sub_csv if missing. '
            'Example: --ref-csv data/tcga_brca_full.csv'
        ),
    )
    parser.add_argument(
        '--ref-tsv', default=None,
        help=(
            'Path to a TSV file to use as P_ref. '
            'Loaded via load_tsv_with_subtypes (handles genes×samples transpose). '
            'Use this for the competition _reference.tsv when auto-detection misses it. '
            'Example: --ref-tsv data/processed/TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes_reference.tsv'
        ),
    )
    parser.add_argument(
        '--test-tsv', default=None,
        help=(
            'Override the gene-expression TSV path from config.yaml. '
            'Also triggers auto-detection of a _reference.tsv sibling at this location. '
            'Example: --test-tsv data/processed/TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes.tsv'
        ),
    )
    parser.add_argument(
        '--sub-csv', default=None,
        help=(
            'Override the sample-annotation CSV path from config.yaml. '
            'Example: --sub-csv data/meta/TCGA-COMBINED_primary_tumor_subtypes.csv'
        ),
    )
    parser.add_argument(
        '--splits-yaml', default=None,
        help=(
            'Path to the splits YAML with train_index/test_index per split. '
            'Overrides the default lookup in the submission dir. '
            'Example: --splits-yaml /path/to/split_indices/TCGA-COMBINED_splits.yaml'
        ),
    )
    parser.add_argument(
        '--decontaminate', action='store_true', default=False,
        help=(
            'Estimate the non-member distribution by subtracting the member '
            'contribution from P_ref:  P_nonmem = (P_pool − α·P_synth)/(1−α). '
            'Intended for use with --ref-csv <whole-population-pool.csv> where '
            'the pool contains both members and non-members. '
            'α is auto-computed as N_synth/N_ref; override with --member-frac.'
        ),
    )
    parser.add_argument(
        '--member-frac', type=float, default=None,
        help=(
            'Explicit fraction of members in P_ref (α, between 0 and 1). '
            'Auto-computed as N_synth/N_ref when not set. '
            'Only used with --decontaminate.'
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
    
    ref_csv              = os.path.expanduser(args.ref_csv)     if args.ref_csv     else None
    ref_tsv_override     = os.path.expanduser(args.ref_tsv)     if args.ref_tsv     else None
    test_tsv_override    = os.path.expanduser(args.test_tsv)    if args.test_tsv    else None
    sub_csv_override     = os.path.expanduser(args.sub_csv)     if args.sub_csv     else None
    splits_yaml_override = os.path.expanduser(args.splits_yaml) if args.splits_yaml else None
    print(f"  ref_mode  : {args.ref_mode}"
          + (f"  (overridden by --ref-csv {os.path.basename(ref_csv)})" if ref_csv else "")
          + (f"  (overridden by --ref-tsv {os.path.basename(ref_tsv_override)})" if ref_tsv_override else ""))
    
    if ref_csv:
        print(f"  ref_csv   : {ref_csv}")
    if args.decontaminate:
        a_str = f"{args.member_frac:.3f}" if args.member_frac else "N_synth/N_ref (auto)"
        print(f"  decontaminate: ON  α={a_str}")
        if not ref_csv:
            print(f"  NOTE: --decontaminate is most useful with --ref-csv <pool.csv> "
                  f"where the pool contains members + non-members.")

    # ---- Attack each split ------------------------------------------
    rows = []
    for s in args.splits:
        m = attack_split(
            split_idx            = s,
            submission_dir       = submission_dir,
            competition_home     = competition_home,
            blue_cfg             = blue_cfg,
            output_dir           = output_dir,
            n_bins               = args.n_bins,
            use_target_col       = args.use_target_col,
            ref_mode             = args.ref_mode,
            ref_csv              = ref_csv,
            ref_tsv              = ref_tsv_override,
            test_tsv_override    = test_tsv_override,
            sub_csv_override     = sub_csv_override,
            splits_yaml_override = splits_yaml_override,
            decontaminate        = args.decontaminate,
            member_frac          = args.member_frac,
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
        marginals  = f"2-way({label_col})" if args.use_target_col else "1-way"
        ref_desc   = os.path.basename(ref_csv)   if ref_csv   else args.ref_mode
        decon_desc = f"+decontam(α={args.member_frac or 'auto'})" if args.decontaminate else ""
        
        # Hardcoded synth description since training data was removed
        print(f"Summary  ({dataset}  |  ε={eps}  |  bins={args.n_bins}  |  "
              f"{marginals}  |  P_synth=synth  |  P_ref={ref_desc}{decon_desc})")
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
