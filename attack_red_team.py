#!/usr/bin/env python3
"""attack_red_team.py — MAMA-MIA attack on RED-team submission folders.

Expected folder layout:
    RED_TCGA-BRCA_v2/
        synthetic_data_1.csv
        synthetic_labels_1.csv
        TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv
        TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes_reference.tsv  (optional)

    RED_TCGA-COMBINED_v2/
        synthetic_data_2.csv
        synthetic_labels_2.csv
        TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes.tsv
        TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes_reference.tsv

Usage
-----
    python3 attack_red_team.py --submission-dir RED_TCGA-BRCA_v2     --split 2 --output brca_2_predictions.csv
    python3 attack_red_team.py --submission-dir RED_TCGA-COMBINED_v2 --split 2 --output combined_2_predictions.csv
"""

import sys
import os
import argparse
import warnings
import glob

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append('reprosyn-main/src/reprosyn/methods/mbi/')

import mbi_patch  # noqa: F401
from attack_submission import encode_all, build_focal_points, mama_mia_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_tsv(folder: str, suffix: str = '') -> str | None:
    """Return the first .tsv file in folder whose name contains suffix."""
    for f in sorted(glob.glob(os.path.join(folder, '*.tsv'))):
        if suffix in os.path.basename(f):
            return f
    return None


def load_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', index_col=0)
    if df.shape[1] > df.shape[0]:          # genes×samples → transpose
        df = df.T
    return df


def detect_label_col(folder: str) -> str:
    name = os.path.basename(os.path.abspath(folder)).upper()
    if 'BRCA' in name:
        return 'Subtype'
    return 'cancer_type'


# ---------------------------------------------------------------------------
# Main attack
# ---------------------------------------------------------------------------

def attack(submission_dir: str, split: int, output_csv: str,
           n_bins: int = 4, use_target_col: bool = False) -> None:

    folder = os.path.abspath(submission_dir)
    label_col = detect_label_col(folder)

    # ---- Locate data files -----------------------------------------------
    synth_path  = os.path.join(folder, f'synthetic_data_{split}.csv')
    labels_path = os.path.join(folder, f'synthetic_labels_{split}.csv')

    # Find the main test TSV (not the reference one)
    test_tsv = find_tsv(folder, suffix='lmgenes.tsv')
    if test_tsv is None:
        # fallback: any tsv that is NOT the reference
        all_tsvs = [f for f in glob.glob(os.path.join(folder, '*.tsv'))
                    if 'reference' not in f]
        test_tsv = all_tsvs[0] if all_tsvs else None
    if test_tsv is None:
        raise FileNotFoundError(f"No test TSV found in {folder}")

    # Reference TSV: prefer dedicated _reference.tsv, fall back to test TSV
    ref_tsv = find_tsv(folder, suffix='_reference.tsv')

    for p in [synth_path, labels_path, test_tsv]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file missing: {p}")

    print(f"\n{'='*65}")
    print(f"Dataset folder : {folder}")
    print(f"Split          : {split}")
    print(f"Label column   : {label_col}")
    print(f"Synth CSV      : {os.path.basename(synth_path)}")
    print(f"Labels CSV     : {os.path.basename(labels_path)}")
    print(f"Test TSV       : {os.path.basename(test_tsv)}")
    print(f"Reference TSV  : {os.path.basename(ref_tsv) if ref_tsv else '(none — using test TSV)'}")
    print(f"{'='*65}")

    # ---- Load ------------------------------------------------------------
    print("Loading data …")
    targets = load_tsv(test_tsv)
    print(f"  Targets: {targets.shape}  index sample: {list(targets.index[:3])}")

    if ref_tsv:
        ref = load_tsv(ref_tsv)
        print(f"  Ref    : {ref.shape}  (dedicated reference TSV)")
    else:
        ref = targets.copy()
        print(f"  Ref    : using full test TSV as P_ref (no reference TSV found)")

    synth  = pd.read_csv(synth_path)
    labels = pd.read_csv(labels_path)
    lbl_col_actual = label_col if label_col in labels.columns else labels.columns[0]
    synth[label_col] = labels[lbl_col_actual].values
    print(f"  Synth  : {synth.shape}  {label_col} dist: {synth[label_col].value_counts().to_dict()}")

    first_gene = next((c for c in synth.columns if c.startswith('ENSG')), None)
    if first_gene:
        uniq = sorted(synth[first_gene].dropna().unique())
        print(f"  First gene ({first_gene}): {len(uniq)} unique vals → {uniq[:6]}")

    # ---- Align gene columns ----------------------------------------------
    ensg_synth   = [c for c in synth.columns   if c.startswith('ENSG')]
    ensg_ref     = {c for c in ref.columns     if c.startswith('ENSG')}
    ensg_targets = {c for c in targets.columns if c.startswith('ENSG')}
    gene_cols    = [c for c in ensg_synth if c in ensg_ref and c in ensg_targets]
    print(f"  Shared gene columns: {len(gene_cols)}")

    # ---- Build working DataFrames ----------------------------------------
    target_col = label_col if use_target_col else None
    use_cols   = gene_cols + ([label_col] if target_col else [])

    synth_w   = synth[[c for c in use_cols if c in synth.columns]].copy()
    ref_w     = ref[[c for c in use_cols if c in ref.columns]].copy()
    targets_w = targets[[c for c in use_cols if c in targets.columns]].copy()

    # Add placeholder label col to ref/targets if needed for 2-way but missing
    if target_col:
        for df, name in [(ref_w, 'ref'), (targets_w, 'targets')]:
            if label_col not in df.columns:
                df[label_col] = 'Unknown'

    # ---- Encode + score --------------------------------------------------
    synth_enc, ref_enc, targets_enc = encode_all(
        synth_w, ref_w, targets_w, gene_cols, n_bins,
        name=f'red_split{split}'
    )

    fps    = build_focal_points(gene_cols, target_col)
    scores = mama_mia_score(synth_enc, ref_enc, targets_enc, fps)

    print(f"\n  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Targets    : {len(scores)} samples")

    # ---- Save ------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    pd.DataFrame({'membership_label': scores},
                 index=targets.index).to_csv(output_csv, index=False)
    print(f"  Saved → {output_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MAMA-MIA attack on RED-team submission folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--submission-dir', required=True,
                        help='Path to RED_TCGA-BRCA_v2/ or RED_TCGA-COMBINED_v2/')
    parser.add_argument('--split', type=int, required=True,
                        help='Which split number to attack (e.g. 2)')
    parser.add_argument('--output', required=True,
                        help='Output CSV path (single column: membership_label)')
    parser.add_argument('--n-bins', type=int, default=4,
                        help='Discretization bins')
    parser.add_argument('--use-target-col', action='store_true', default=False,
                        help='Add 2-way (gene, label) marginals')
    args = parser.parse_args()

    attack(
        submission_dir = os.path.expanduser(args.submission_dir),
        split          = args.split,
        output_csv     = os.path.expanduser(args.output),
        n_bins         = args.n_bins,
        use_target_col = args.use_target_col,
    )


if __name__ == '__main__':
    main()
