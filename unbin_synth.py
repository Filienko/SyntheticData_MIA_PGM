#!/usr/bin/env python3
"""unbin_synth.py — Map already-generated integer bin indices back to floats.

Fits per-bin means from real_train.csv, then replaces each integer bin
index in the synthetic CSV with the mean float value of that bin.

Usage
-----
    python3 unbin_synth.py \
        --real  /path/to/real_train.csv \
        --synth /path/to/syn_train_privatepgm_eps10.0_iters1000.csv \
        --target-col cancer_type

    # Writes result to the same file by default; use --output to redirect:
    python3 unbin_synth.py \
        --real  real_train.csv \
        --synth syn_train.csv \
        --output syn_train_continuous.csv
"""

import argparse
import os
import numpy as np
import pandas as pd


def compute_bin_means(real: pd.DataFrame, gene_cols: list, n_bins: int) -> dict:
    """Fit quantile boundaries on real data, return mean per bin per column."""
    bin_means = {}
    for col in gene_cols:
        vals = real[col].values
        boundaries = np.quantile(vals, np.linspace(0, 1, n_bins + 1)[1:-1])
        bin_idx = np.digitize(vals, boundaries).astype(int)
        bin_means[col] = np.array([
            vals[bin_idx == b].mean() if (bin_idx == b).any() else np.nan
            for b in range(n_bins)
        ])
    return bin_means


def unbin(synth: pd.DataFrame, gene_cols: list, bin_means: dict) -> pd.DataFrame:
    out = synth.copy()
    for col in gene_cols:
        means = bin_means[col]
        out[col] = synth[col].map(lambda b, m=means: float(m[int(b)]))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Convert integer bin indices in a synthetic CSV to float bin means",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--real',       required=True,
                        help='real_train.csv used to compute bin means')
    parser.add_argument('--synth',      required=True,
                        help='Synthetic CSV with integer bin indices')
    parser.add_argument('--output',     default=None,
                        help='Output path (default: overwrite --synth)')
    parser.add_argument('--target-col', default='Subtype',
                        help='Label column to skip (Subtype or cancer_type)')
    parser.add_argument('--n-bins',     type=int, default=4,
                        help='Number of bins used during generation')
    args = parser.parse_args()

    real  = pd.read_csv(os.path.expanduser(args.real))
    synth = pd.read_csv(os.path.expanduser(args.synth))

    # Auto-detect target column if not present
    target_col = args.target_col
    if target_col not in real.columns:
        for candidate in ['Subtype', 'cancer_type', 'subtype', 'label']:
            if candidate in real.columns:
                target_col = candidate
                print(f"Auto-detected target column: '{target_col}'")
                break

    gene_cols = [c for c in synth.columns
                 if c != target_col and pd.api.types.is_numeric_dtype(synth[c])]
    print(f"Gene columns : {len(gene_cols)}")
    print(f"Synth shape  : {synth.shape}")
    print(f"Unique values in first gene before: {synth[gene_cols[0]].nunique()}")

    bin_means = compute_bin_means(real, gene_cols, args.n_bins)
    result    = unbin(synth, gene_cols, bin_means)

    print(f"Unique values in first gene after : {result[gene_cols[0]].nunique()}")

    out_path = os.path.expanduser(args.output or args.synth)
    result.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


if __name__ == '__main__':
    main()
