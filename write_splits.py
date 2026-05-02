#!/usr/bin/env python3
"""write_splits.py — Build TCGA-BRCA_splits.yaml / TCGA-COMBINED_splits.yaml
from the real train-split CSVs and the full test TSV.

The YAML format expected by attack_ppml_huskies.py / load_membership_from_yaml:

    splits:
      split_1:
        train_index: [TCGA-A1-0001-01, ...]   # members
        test_index:  [TCGA-A1-0099-01, ...]   # non-members
      split_2:
        ...

Usage
-----
    # All 5 splits, TCGA-COMBINED:
    python3 write_splits.py \\
        --dataset      TCGA-COMBINED \\
        --train-csvs   path/to/train_split_1.csv path/to/train_split_2.csv ... \\
        --test-tsv     ~/Health-Privacy-Challenge/data/processed/TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes.tsv \\
        --output       path/to/submission_dir/TCGA-COMBINED_splits.yaml

    # Single split, auto-glob from submission dir:
    python3 write_splits.py \\
        --dataset      TCGA-COMBINED \\
        --submission-dir path/to/blueteam_PPML-Huskies_TCGA-COMBINED \\
        --test-tsv     ~/Health-Privacy-Challenge/data/processed/TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes.tsv

The sample IDs are taken from the CSV index (first column) and matched against
the TSV index.  If a TSV is genes×samples it is transposed automatically.
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd
import yaml


def load_tsv_index(path: str) -> list:
    """Return the list of sample IDs from a TSV file (auto-transpose if genes×samples)."""
    df = pd.read_csv(path, sep='\t', index_col=0, nrows=2)  # peek at shape
    if str(df.index[0]).startswith('ENSG'):
        # genes×samples — column names are sample IDs
        full = pd.read_csv(path, sep='\t', index_col=0)
        return list(full.columns)
    else:
        # samples×genes — index values are sample IDs; read only the index
        full = pd.read_csv(path, sep='\t', index_col=0, usecols=[0])
        return list(full.index)


def load_train_ids(csv_path: str) -> list:
    """Return sample IDs from the first column of a train CSV."""
    df = pd.read_csv(csv_path, index_col=0, usecols=[0])
    return list(df.index)


def main():
    p = argparse.ArgumentParser(
        description='Build splits YAML for attack_ppml_huskies.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--dataset', required=True, choices=['TCGA-BRCA', 'TCGA-COMBINED'],
                   help='Dataset name (used as prefix in the output filename)')
    p.add_argument('--test-tsv', required=True,
                   help='Path to the full gene-expression TSV (all candidate samples)')
    p.add_argument('--train-csvs', nargs='+', default=None,
                   help='Explicit list of train_split_N.csv files (ordered: split 1, 2, …)')
    p.add_argument('--submission-dir', default=None,
                   help='Auto-find train_split_*.csv files in this directory '
                        '(used when --train-csvs is not given)')
    p.add_argument('--output', default=None,
                   help='Output YAML path.  Default: <submission-dir>/<dataset>_splits.yaml '
                        'or current dir if --submission-dir is not set.')
    args = p.parse_args()

    # ---- Collect train CSVs ---------------------------------------------------
    if args.train_csvs:
        train_files = [os.path.expanduser(f) for f in args.train_csvs]
    elif args.submission_dir:
        pattern = os.path.join(os.path.expanduser(args.submission_dir),
                               'train_split_*.csv')
        train_files = sorted(glob.glob(pattern))
        if not train_files:
            sys.exit(f"No train_split_*.csv files found in {args.submission_dir}")
    else:
        sys.exit("Provide either --train-csvs or --submission-dir")

    print(f"Found {len(train_files)} train CSV(s):")
    for f in train_files:
        print(f"  {f}")

    # ---- Load full candidate pool from test TSV ---------------------------------
    test_tsv = os.path.expanduser(args.test_tsv)
    print(f"\nLoading test TSV: {test_tsv}")
    all_ids = load_tsv_index(test_tsv)
    all_ids_set = set(all_ids)
    print(f"  Candidates: {len(all_ids)}")

    # ---- Build splits dict ------------------------------------------------------
    splits = {}
    for i, csv_path in enumerate(train_files, start=1):
        # Infer split number from filename (train_split_3.csv → 3), fallback to i
        m = re.search(r'split_(\d+)', os.path.basename(csv_path))
        split_num = int(m.group(1)) if m else i

        train_ids = load_train_ids(csv_path)
        train_set = set(train_ids)

        # Members must be present in the candidate pool
        members     = [sid for sid in train_ids if sid in all_ids_set]
        non_members = [sid for sid in all_ids  if sid not in train_set]

        missing = len(train_ids) - len(members)
        if missing:
            print(f"  WARNING split_{split_num}: {missing} train IDs not found "
                  f"in test TSV (they won't appear as members in YAML)")

        splits[f'split_{split_num}'] = {
            'train_index': members,
            'test_index':  non_members,
        }
        print(f"  split_{split_num}: {len(members)} members, "
              f"{len(non_members)} non-members  (source: {os.path.basename(csv_path)})")

    # ---- Determine output path --------------------------------------------------
    if args.output:
        out_path = os.path.expanduser(args.output)
    elif args.submission_dir:
        out_path = os.path.join(os.path.expanduser(args.submission_dir),
                                f'{args.dataset}_splits.yaml')
    else:
        out_path = f'{args.dataset}_splits.yaml'

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump({'splits': splits}, f, default_flow_style=False, sort_keys=False)

    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
