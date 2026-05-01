#!/usr/bin/env python3
"""run_privatepgm.py — Train Private-PGM on any gene-expression CSV and
generate a synthetic dataset.

Usage
-----
    # TCGA-BRCA (label column = Subtype):
    python3 run_privatepgm.py \\
        --input  /path/to/real_train.csv \\
        --output /path/to/syn_train_privatepgm_eps10.0_iters10000.csv \\
        --epsilon 10.0 --num-iters 10000

    # TCGA-COMBINED (label column = cancer_type):
    python3 run_privatepgm.py \\
        --input  /path/to/miav_sweep_01_comb/real_train.csv \\
        --output /path/to/miav_sweep_01_comb/syn_train_privatepgm_eps10.0_iters10000.csv \\
        --epsilon 10.0 --num-iters 10000 --target-col cancer_type

Pipeline
--------
1. Load CSV  →  detect / validate target column
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

def run(input_csv:   str,
        output_csv:  str,
        epsilon:     float,
        delta:       float,
        num_iters:   int,
        n_bins:      int,
        target_col:  str,
        synth_size:  int | None):

    print(f"\n{'='*65}")
    print(f"Private-PGM  |  ε={epsilon}  δ={delta}  iters={num_iters}  bins={n_bins}")
    print(f"  Input : {input_csv}")
    print(f"  Output: {output_csv}")
    print(f"{'='*65}")

    # ---- Load -----------------------------------------------------------
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


def main():
    parser = argparse.ArgumentParser(
        description="Run Private-PGM on a gene-expression CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input',      required=True,
                        help='Path to real_train.csv')
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
    args = parser.parse_args()

    run(
        input_csv   = os.path.expanduser(args.input),
        output_csv  = os.path.expanduser(args.output),
        epsilon     = args.epsilon,
        delta       = args.delta,
        num_iters   = args.num_iters,
        n_bins      = args.n_bins,
        target_col  = args.target_col,
        synth_size  = args.synth_size,
    )


if __name__ == '__main__':
    main()
