"""Prepare a flat CSV (samples × genes + Subtype) from a TCGA TSV + subtypes CSV.

This is the preprocessing step required before running eval_epsilon_sweep.py
on any TCGA dataset (BRCA, COMBINED, etc.).

Input files
-----------
  genes_tsv   : TSV with genes as rows and sample IDs as columns
                e.g. TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv
  subtypes_csv: CSV with columns [samplesID, cancer_type, ...]
                e.g. TCGA-BRCA_primary_tumor_subtypes.csv

Output
------
  A flat CSV: one row per sample, one column per gene, plus a 'Subtype' column.
  Suitable for use with eval_epsilon_sweep.py --data <output_csv>.

Usage
-----
  python3 prepare_dataset.py \\
      --genes    ~/privacy/Health-Privacy-Challenge/data/RED_TCGA-BRCA/TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv \\
      --subtypes ~/privacy/Health-Privacy-Challenge/data/RED_TCGA-BRCA/TCGA-BRCA_primary_tumor_subtypes.csv \\
      --output   data/tcga_brca_full.csv

  python3 prepare_dataset.py \\
      --genes    ~/privacy/Health-Privacy-Challenge/data/RED_TCGA-COMBINED/TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes.tsv \\
      --subtypes ~/privacy/Health-Privacy-Challenge/data/RED_TCGA-COMBINED/TCGA-COMBINED_primary_tumor_subtypes.csv \\
      --output   data/tcga_combined_full.csv
"""

import os
import argparse
import pandas as pd


def prepare(genes_tsv, subtypes_csv, output_csv, subtype_col="cancer_type"):
    print(f"Reading genes TSV: {genes_tsv}")
    df = pd.read_csv(genes_tsv, sep='\t', index_col=0).T  # → samples × genes
    df.index.name = 'samplesID'
    print(f"  Shape after transpose: {df.shape}")

    print(f"Reading subtypes CSV: {subtypes_csv}")
    sub = pd.read_csv(subtypes_csv, index_col=0)
    if 'samplesID' in sub.columns:
        sub = sub.set_index('samplesID')
    if subtype_col not in sub.columns:
        raise ValueError(
            f"Column '{subtype_col}' not found in subtypes CSV. "
            f"Available: {list(sub.columns)}"
        )
    sub = sub[[subtype_col]]

    df = df.join(sub, how='left')
    n_missing = df[subtype_col].isna().sum()
    if n_missing:
        print(f"  WARNING: {n_missing}/{len(df)} samples have no subtype — dropping them.")
        df = df.dropna(subset=[subtype_col])

    # Rename subtype column to 'Subtype' for compatibility with tcga_data().
    df = df.rename(columns={subtype_col: 'Subtype'})

    n_subtypes = df['Subtype'].nunique()
    print(f"  Samples: {len(df)}  |  Genes: {df.shape[1] - 1}  |  Subtypes: {n_subtypes}")
    print(f"  Subtype distribution:\n{df['Subtype'].value_counts().to_string()}")

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved → {output_csv}")
    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Build flat samples×genes+Subtype CSV from TCGA TSV + subtypes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--genes',    required=True,
                        help='Genes×samples TSV (genes as rows, sample IDs as columns)')
    parser.add_argument('--subtypes', required=True,
                        help='Subtypes CSV with samplesID and cancer_type columns')
    parser.add_argument('--output',   required=True,
                        help='Output CSV path')
    parser.add_argument('--subtype-col', default='cancer_type',
                        help='Column name in subtypes CSV to use as Subtype')
    args = parser.parse_args()

    prepare(
        genes_tsv=os.path.expanduser(args.genes),
        subtypes_csv=os.path.expanduser(args.subtypes),
        output_csv=os.path.expanduser(args.output),
        subtype_col=args.subtype_col,
    )


if __name__ == '__main__':
    main()
