#!/usr/bin/env python3
"""run_privatepgm.py — Train Private-PGM on any gene-expression CSV and
generate a synthetic dataset.

Usage
-----
    # Single combined CSV (label column = Subtype):
    python3 run_privatepgm.py \\
        --input  /path/to/real_train.csv \\
        --output /path/to/syn_train_privatepgm_eps10.0_iters10000.csv \\
        --epsilon 10.0 --num-iters 10000

    # TCGA-COMBINED (label column = cancer_type):
    python3 run_privatepgm.py \\
        --input  /path/to/miav_sweep_01_comb/real_train.csv \\
        --output /path/to/miav_sweep_01_comb/syn_train_privatepgm_eps10.0_iters10000.csv \\
        --epsilon 10.0 --num-iters 10000 --target-col cancer_type

    # Split-dir layout (X_train / y_train / column_names separate files):
    python3 run_privatepgm.py \\
        --split-dir PPML-H_data_splits/TCGA-COMBINED/real \\
        --split 1 \\
        --output submission/internal_fixed/blueteam_PPML-Huskies_TCGA-COMBINED/synthetic_data_split_1.csv \\
        --epsilon 10.0 --num-iters 10000 --target-col cancer_type

Pipeline
--------
1. Load CSV (or merge X_train + y_train + column_names)  →  detect / validate target column
2. Ordinal-encode target column (sorted labels → 0..K-1)
3. Quantile-discretize all gene columns into n_bins equal-depth bins (0..n_bins-1)
4. Build reprosyn metadata  →  run PRIVATEPGM
5. Decode back to original labels, save CSV
"""

import sys
import os
import argparse
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup — works when run from SyntheticData_MIA_PGM/ directory OR
# when the repo root is passed via --repo-dir.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.append(os.path.join(_HERE, 'reprosyn-main/src/reprosyn/methods/mbi/'))
sys.path.append(os.path.join(_HERE, 'reprosyn-main/src/'))

import mbi_patch  # noqa: F401 — patches mbi imports
import privatepgm as pgm_module


# ---------------------------------------------------------------------------
# Split-dir loader (X_train / y_train / column_names layout)
# ---------------------------------------------------------------------------

def load_split(split_dir: str, split: int, target_col: str) -> pd.DataFrame:
    """Merge X_train_real_split_N.csv + y_train_real_split_N.csv + column_names.csv.

    Expected files in split_dir:
      column_names.csv            — one ENSG ID per row (header = 'column_names')
      X_train_real_split_N.csv    — float matrix, NO header row (rows = samples)
      y_train_real_split_N.csv    — label column, with or without header

    Returns a combined DataFrame with ENSG columns + target_col.
    """
    col_names_path = os.path.join(split_dir, 'column_names.csv')
    x_path         = os.path.join(split_dir, f'X_train_real_split_{split}.csv')
    y_path         = os.path.join(split_dir, f'y_train_real_split_{split}.csv')

    for p in [col_names_path, x_path, y_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    # Column names: one ENSG ID per row, header line is 'column_names'
    col_df    = pd.read_csv(col_names_path)
    gene_cols = col_df.iloc[:, 0].tolist()

    # X matrix: try with header first; fall back to headerless
    x_peek = pd.read_csv(x_path, nrows=1)
    if str(x_peek.columns[0]).startswith('ENSG'):
        X = pd.read_csv(x_path)
        X = X[[c for c in gene_cols if c in X.columns]]  # reorder to column_names order
    else:
        X = pd.read_csv(x_path, header=None)
        if X.shape[1] != len(gene_cols):
            raise ValueError(
                f"X_train has {X.shape[1]} columns but column_names.csv has "
                f"{len(gene_cols)} entries."
            )
        X.columns = gene_cols

    # y labels: single column, may or may not have a header
    y_peek = pd.read_csv(y_path, nrows=1)
    first_val = str(y_peek.iloc[0, 0])
    # If the first value looks like a cancer-type label (not a number), file has NO header
    try:
        float(first_val)
        # numeric first value → treat as headerless
        y = pd.read_csv(y_path, header=None, names=[target_col])
    except ValueError:
        # non-numeric first value → file has a proper header row
        y = pd.read_csv(y_path)
        y.columns = [target_col]

    if len(X) != len(y):
        raise ValueError(f"X ({len(X)} rows) and y ({len(y)} rows) have different lengths.")

    df = X.copy()
    df[target_col] = y[target_col].values
    print(f"  Split-dir load: {df.shape}  "
          f"genes={len(gene_cols)}  "
          f"{target_col} dist: {df[target_col].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# Discretization (same scheme as tcga_data() in encode_data.py)
# ---------------------------------------------------------------------------

def quantile_discretize(df: pd.DataFrame, cols: list, n_bins: int):
    """Discretize continuous columns into n_bins equal-depth integer bins.

    Returns:
      disc_df      : copy of df with each col replaced by integers 0..n_bins-1
      bin_means    : dict {col: array of length n_bins} — mean original value
                     per bin, used to map synthetic integers back to floats
    """
    out = df.copy()
    bin_means = {}
    for col in cols:
        vals = df[col].values
        boundaries = np.quantile(vals, np.linspace(0, 1, n_bins + 1)[1:-1])
        bin_idx = np.digitize(vals, boundaries).astype(int)   # 0..n_bins-1
        out[col] = bin_idx
        # Mean of original values falling in each bin
        bin_means[col] = np.array([
            vals[bin_idx == b].mean() if (bin_idx == b).any() else np.nan
            for b in range(n_bins)
        ])
    return out, bin_means


def dediscretize(synth: pd.DataFrame, gene_cols: list, bin_means: dict) -> pd.DataFrame:
    """Map synthetic integer bin indices back to the mean value of each bin."""
    out = synth.copy()
    for col in gene_cols:
        means = bin_means[col]
        out[col] = synth[col].map(lambda b, m=means: float(m[int(b)]))
    return out


def build_metadata(gene_cols: list, target_col: str,
                   n_bins: int, n_subtypes: int) -> list:
    """Build reprosyn-style metadata list."""
    meta = [
        {"name": col, "type": "finite/ordered",
         "representation": list(range(n_bins))}
        for col in gene_cols
    ] + [
        {"name": target_col, "type": "finite/ordered",
         "representation": list(range(n_subtypes))}
    ]
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(output_csv:  str,
        epsilon:     float,
        delta:       float,
        num_iters:   int,
        n_bins:      int,
        target_col:  str,
        synth_size:  int | None,
        input_csv:   str | None = None,
        df:          pd.DataFrame | None = None):
    """Train Private-PGM and save synthetic CSV.

    Supply exactly one of `input_csv` (path to combined CSV) or `df`
    (pre-loaded DataFrame, e.g. from load_split()).
    """
    print(f"\n{'='*65}")
    print(f"Private-PGM  |  ε={epsilon}  δ={delta}  iters={num_iters}  bins={n_bins}")
    print(f"  Input : {input_csv or '(pre-loaded DataFrame)'}")
    print(f"  Output: {output_csv}")
    print(f"{'='*65}")

    # ---- Load -----------------------------------------------------------
    if df is None:
        if input_csv is None:
            raise ValueError("Provide either input_csv or df.")
        df = pd.read_csv(input_csv)
        print(f"  Loaded: {df.shape}  columns (first 5): {list(df.columns[:5])}")

    # ---- Detect target column ------------------------------------------
    if target_col not in df.columns:
        # Auto-detect: look for known label columns
        candidates = [c for c in ['Subtype', 'cancer_type', 'subtype', 'label']
                      if c in df.columns]
        if not candidates:
            raise ValueError(
                f"Target column '{target_col}' not found.  "
                f"Non-ENSG columns: {[c for c in df.columns if not c.startswith('ENSG')]}"
            )
        target_col = candidates[0]
        print(f"  Auto-detected target column: '{target_col}'")

    # ---- Separate gene columns -----------------------------------------
    gene_cols = [c for c in df.columns
                 if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
    if not gene_cols:
        raise RuntimeError("No numeric gene columns found.")
    print(f"  Gene columns : {len(gene_cols)}")

    # ---- Encode target -------------------------------------------------
    subtypes = sorted(df[target_col].unique().tolist())
    subtype_map = {s: i for i, s in enumerate(subtypes)}
    inv_map     = {i: s for s, i in subtype_map.items()}
    df[target_col] = df[target_col].map(subtype_map)
    print(f"  Target col   : '{target_col}'  classes={subtypes}")

    # ---- Discretize gene columns  --------------------------------------
    print(f"  Discretizing {len(gene_cols)} gene columns into {n_bins} bins …")
    df_disc, bin_means = quantile_discretize(df, gene_cols, n_bins)
    df_disc[target_col] = df[target_col].values  # target already integer

    columns = gene_cols + [target_col]
    meta    = build_metadata(gene_cols, target_col, n_bins, len(subtypes))

    n_rows   = synth_size if synth_size else len(df_disc)
    print(f"  Training rows: {len(df_disc)}   Synthetic rows: {n_rows}")

    # ---- Run Private-PGM -----------------------------------------------
    print(f"\n  Training Private-PGM …")
    domain_config = {m['name']: len(m['representation']) for m in meta}
    pgm_gen = pgm_module.Private_PGM(
        target_variable = target_col,
        enable_privacy  = True,
        target_epsilon  = epsilon,
        target_delta    = delta,
    )
    pgm_gen.train(df_disc[columns], domain_config, num_iters=num_iters)

    synth_df = pgm_gen.generate(n_rows)

    # ---- Map gene bins back to original float space --------------------
    print("  Mapping gene bins → original float values (bin means) …")
    synth_df = dediscretize(synth_df, gene_cols, bin_means)

    # ---- Decode target column back to original labels ------------------
    synth_df[target_col] = synth_df[target_col].map(inv_map)

    # ---- Save ----------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    synth_df.to_csv(output_csv, index=False)
    print(f"\n  Saved {synth_df.shape} → {output_csv}")

    # Quick sanity check
    gene_unique = synth_df[gene_cols[0]].nunique()
    label_vc    = synth_df[target_col].value_counts().to_dict()
    print(f"  First gene unique values : {gene_unique}  (expect ~{n_bins}, one float per bin)")
    print(f"  Label distribution       : {label_vc}")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Config + splits YAML helpers
# ---------------------------------------------------------------------------

_DATASET_DEFAULTS = {
    'TCGA-BRCA': {
        'subtype_col_name': 'Subtype',
        'count_file': 'data/processed/TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv',
        'annot_file': 'data/meta/TCGA-BRCA_primary_tumor_subtypes.csv',
    },
    'TCGA-COMBINED': {
        'subtype_col_name': 'cancer_type',
        'count_file': 'data/processed/TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes.tsv',
        'annot_file': 'data/meta/TCGA-COMBINED_primary_tumor_subtypes.csv',
    },
}

def _infer_dataset(target_col: str) -> str:
    return 'TCGA-BRCA' if target_col == 'Subtype' else 'TCGA-COMBINED'


def write_config(submission_dir: str, target_col: str, epsilon: float,
                 num_iters: int) -> str:
    dataset = _infer_dataset(target_col)
    d = _DATASET_DEFAULTS[dataset]
    cfg = {
        'dataset_config': {
            'name': dataset,
            'subtype_col_name': d['subtype_col_name'],
            'count_file': d['count_file'],
            'annot_file': d['annot_file'],
        },
        'pgg_pgm_config': {
            'epsilon': epsilon,
            'iterations': num_iters,
        },
    }
    out = os.path.join(submission_dir, 'config.yaml')
    with open(out, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  Wrote config        → {out}")
    return out


def write_splits(submission_dir: str, split: int, train_ids: list,
                 test_tsv: str | None, target_col: str) -> str | None:
    dataset = _infer_dataset(target_col)

    # Non-members: all candidates in test TSV that are not in train set
    if test_tsv and os.path.exists(test_tsv):
        peek = pd.read_csv(test_tsv, sep='\t', index_col=0, nrows=2)
        if str(peek.index[0]).startswith('ENSG'):
            full = pd.read_csv(test_tsv, sep='\t', index_col=0)
            all_ids = list(full.columns)
        else:
            full = pd.read_csv(test_tsv, sep='\t', index_col=0, usecols=[0])
            all_ids = list(full.index)
        train_set  = set(train_ids)
        members    = [sid for sid in train_ids if sid in set(all_ids)]
        non_members = [sid for sid in all_ids if sid not in train_set]
        print(f"  Splits ({split}): {len(members)} members, "
              f"{len(non_members)} non-members  (from test TSV)")
    else:
        # No TSV available — write train_index only; attack will treat rest as members
        members     = train_ids
        non_members = []
        if test_tsv:
            print(f"  WARNING: test TSV not found at {test_tsv} — "
                  f"writing train_index only (no test_index)")
        else:
            print(f"  NOTE: no --test-tsv given — writing train_index only")

    splits_data = {
        'splits': {
            f'split_{split}': {
                'train_index': members,
                **(({'test_index': non_members}) if non_members else {}),
            }
        }
    }

    out = os.path.join(submission_dir, f'{dataset}_splits.yaml')
    # Merge with existing YAML if present (other splits already written)
    if os.path.exists(out):
        with open(out) as f:
            existing = yaml.safe_load(f) or {}
        existing.setdefault('splits', {}).update(splits_data['splits'])
        splits_data = existing

    with open(out, 'w') as f:
        yaml.dump(splits_data, f, default_flow_style=False, sort_keys=False)
    print(f"  Wrote splits YAML   → {out}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run Private-PGM on a gene-expression CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Input: either a single combined CSV or a split-dir layout ----
    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument('--input',
                           help='Path to combined real_train.csv '
                                '(gene columns + label column)')
    input_grp.add_argument('--split-dir',
                           help='Directory containing X_train_real_split_N.csv, '
                                'y_train_real_split_N.csv, and column_names.csv')

    parser.add_argument('--split',      type=int, default=None,
                        help='Split number to use with --split-dir (e.g. 1)')
    parser.add_argument('--output',     required=True,
                        help='Path for the output synthetic CSV')
    parser.add_argument('--epsilon',    type=float, default=10.0,
                        help='DP privacy budget ε')
    parser.add_argument('--delta',      type=float, default=1e-5,
                        help='DP delta parameter')
    parser.add_argument('--num-iters',  type=int,   default=10000,
                        help='FactoredInference mirror-descent iterations')
    parser.add_argument('--n-bins',     type=int,   default=4,
                        help='Equal-depth quantile bins for gene discretization')
    parser.add_argument('--target-col', default='Subtype',
                        help='Label column name (Subtype for BRCA, '
                             'cancer_type for COMBINED). '
                             'Auto-detected if not found.')
    parser.add_argument('--synth-size', type=int, default=None,
                        help='Number of synthetic rows to generate '
                             '(default: same as training set size)')
    parser.add_argument('--test-tsv', default=None,
                        help='Path to the full gene-expression TSV (all candidate samples). '
                             'Used to derive test_index (non-members) in the splits YAML. '
                             'If omitted, splits YAML will contain train_index only.')
    args = parser.parse_args()

    output_csv = os.path.expanduser(args.output)
    submission_dir = os.path.dirname(os.path.abspath(output_csv))

    if args.split_dir:
        if args.split is None:
            parser.error('--split is required when using --split-dir')
        df = load_split(
            split_dir  = os.path.expanduser(args.split_dir),
            split      = args.split,
            target_col = args.target_col,
        )
        run(
            df         = df,
            output_csv = output_csv,
            epsilon    = args.epsilon,
            delta      = args.delta,
            num_iters  = args.num_iters,
            n_bins     = args.n_bins,
            target_col = args.target_col,
            synth_size = args.synth_size,
        )
        write_config(submission_dir, args.target_col, args.epsilon, args.num_iters)
        # Splits YAML needs real sample IDs — only possible if test TSV is given
        # (X_train in split-dir format has no row sample IDs, so we derive members
        #  as the intersection of test-TSV IDs that match by position, which is
        #  only meaningful when the TSV row order matches the training set).
        # Most reliable: skip splits YAML here and use write_splits.py separately.
        if args.test_tsv:
            print("  NOTE: split-dir X_train has no sample IDs — "
                  "splits YAML requires a separate run of write_splits.py "
                  "with --train-csvs pointing to a CSV that has TCGA IDs as its index.")
    else:
        df_loaded = pd.read_csv(os.path.expanduser(args.input))
        # Extract sample IDs if the first column is non-numeric (TCGA-xxx IDs)
        first_col = df_loaded.columns[0]
        if not first_col.startswith('ENSG') and not pd.api.types.is_numeric_dtype(df_loaded[first_col]):
            train_ids = list(df_loaded[first_col])
        else:
            train_ids = None
        run(
            df         = df_loaded,
            output_csv = output_csv,
            epsilon    = args.epsilon,
            delta      = args.delta,
            num_iters  = args.num_iters,
            n_bins     = args.n_bins,
            target_col = args.target_col,
            synth_size = args.synth_size,
        )
        write_config(submission_dir, args.target_col, args.epsilon, args.num_iters)
        if args.split is not None:
            if train_ids is None:
                print("  NOTE: no string sample IDs found in input CSV — skipping splits YAML")
            else:
                write_splits(submission_dir, args.split, train_ids,
                             os.path.expanduser(args.test_tsv) if args.test_tsv else None,
                             args.target_col)


if __name__ == '__main__':
    main()
