"""MAMA-MIA attack on a Blue Team submission directory.

Handles the PPML-Huskies / PRO-GENE-GEN TCGA-COMBINED layout:

  submission_dir/
    synthetic_data_split_{1..5}.csv     # genes as columns, no index
    synthetic_labels_split_{1..5}.csv   # cancer_type column (one row per synth sample)
    TCGA-COMBINED_splits.yaml           # train/test sample IDs per split

  tcga_dir/
    TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes.tsv         # genes×samples
    TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes_reference.tsv
    TCGA-COMBINED_primary_tumor_subtypes.csv                        # samplesID + cancer_type

Attack pipeline (per split):
  P_synth  ← synthetic_data_split_N.csv  + synthetic_labels_split_N.csv
  P_aux    ← reference.tsv               + subtypes.csv  (joined on sample ID)
  targets  ← test.tsv                    + subtypes.csv  (joined on sample ID)
  labels   ← TCGA-COMBINED_splits.yaml  (train set → members)

Usage
-----
  python3 attack_submission.py \\
      --submission-dir  submission/blueteam_PPML-Huskies_TCGA-COMBINED \\
      --tcga-dir        ~/privacy/Health-Privacy-Challenge/data/RED_TCGA-COMBINED \\
      --splits          1 2 3 4 5 \\
      --target-col      cancer_type \\
      --n-bins          10 \\
      --output-dir      results/ppml_huskies_attack
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append('reprosyn-main/src/reprosyn/methods/mbi/')

import mbi_patch  # noqa: F401 – must precede any mbi import

from encode_data import (
    fit_continuous_features_equaldepth,
    discretize_continuous_features_equaldepth,
)
from util import C


# ===========================================================================
# Data loading
# ===========================================================================

def load_tsv_with_subtypes(tsv_path, subtypes_path):
    """Load a genes×samples TSV, transpose to samples×genes, join cancer_type.

    Parameters
    ----------
    tsv_path      : path to genes-as-rows, sample-IDs-as-columns TSV
    subtypes_path : path to CSV with columns [samplesID, cancer_type, ...]

    Returns
    -------
    DataFrame  index = sample IDs, columns = gene IDs + cancer_type
    """
    df = pd.read_csv(tsv_path, sep='\t', index_col=0).T  # → samples × genes
    df.index.name = 'samplesID'

    sub = pd.read_csv(subtypes_path, index_col=0)
    # normalise: keep only samplesID and cancer_type
    if 'samplesID' in sub.columns:
        sub = sub.set_index('samplesID')
    sub = sub[['cancer_type']]

    df = df.join(sub, how='left')
    n_missing = df['cancer_type'].isna().sum()
    if n_missing:
        print(f"  WARNING: {n_missing}/{len(df)} rows have no cancer_type after join "
              f"({tsv_path})")
        df['cancer_type'] = df['cancer_type'].fillna('Unknown')
    return df


def load_synthetic_with_labels(synth_path, labels_path):
    """Load synthetic data CSV and attach the cancer_type label column.

    Parameters
    ----------
    synth_path  : CSV with gene columns (no index)
    labels_path : CSV with a single 'cancer_type' column (one row per synth sample)

    Returns
    -------
    DataFrame  columns = gene IDs + cancer_type
    """
    synth = pd.read_csv(synth_path)
    labels = pd.read_csv(labels_path)

    if 'cancer_type' not in labels.columns:
        raise ValueError(
            f"Expected 'cancer_type' column in {labels_path}; "
            f"found: {list(labels.columns)}"
        )
    if len(labels) != len(synth):
        raise ValueError(
            f"Length mismatch: synthetic ({len(synth)}) vs labels ({len(labels)})"
        )
    synth['cancer_type'] = labels['cancer_type'].values
    return synth


def load_membership_from_yaml(splits_yaml_path, split_idx, test_sample_ids):
    """Derive binary membership labels from a splits YAML file.

    Handles the PPML-Huskies layout:

        splits:
          split_1:
            test_index:  [TCGA-..., ...]   # held-out (non-members)
            train_index: [TCGA-..., ...]   # training set (members)  ← optional

    If only test_index is present, members = all candidates NOT in test_index.

    Parameters
    ----------
    splits_yaml_path : path to YAML
    split_idx        : int, 1-based split number
    test_sample_ids  : list/Index of sample IDs in the test TSV (all candidates)

    Returns
    -------
    np.ndarray  shape (len(test_sample_ids),)  dtype int  0=non-member 1=member
    """
    with open(splits_yaml_path) as f:
        root = yaml.safe_load(f)

    # Navigate top-level 'splits:' wrapper if present.
    splits = root.get('splits', root)

    key = f'split_{split_idx}'
    if key not in splits:
        raise KeyError(
            f"Cannot find '{key}' in {splits_yaml_path}. "
            f"Available keys: {list(splits.keys())}"
        )

    split_data = splits[key]

    if 'train_index' in split_data:
        # Explicit train list → members
        member_set = set(split_data['train_index'])
        labels = np.array(
            [1 if sid in member_set else 0 for sid in test_sample_ids],
            dtype=int,
        )
    elif 'test_index' in split_data:
        # Only test list given → non-members are in test_index, everyone else is a member
        non_member_set = set(split_data['test_index'])
        labels = np.array(
            [0 if sid in non_member_set else 1 for sid in test_sample_ids],
            dtype=int,
        )
    else:
        raise ValueError(
            f"Cannot parse split '{key}': expected 'train_index' or 'test_index'. "
            f"Got: {list(split_data.keys())}"
        )

    n_members = labels.sum()
    print(f"  Membership labels: {n_members} members / "
          f"{len(labels) - n_members} non-members "
          f"(out of {len(labels)} candidates)")
    return labels


# ===========================================================================
# Encoding
# ===========================================================================

def encode_all(synth, ref, targets, gene_cols, n_bins, name='attack'):
    """StandardScale then equal-depth bin all three DataFrames consistently."""
    scaler = StandardScaler()
    combined_vals = pd.concat(
        [synth[gene_cols], ref[gene_cols]], ignore_index=True
    )
    scaler.fit(combined_vals)

    def _scale(df):
        out = df.copy()
        out[gene_cols] = scaler.transform(df[gene_cols].values)
        return out

    synth   = _scale(synth)
    ref     = _scale(ref)
    targets = _scale(targets)

    print(f"  Discretizing {len(gene_cols)} gene features into {n_bins} bins …")
    C.n_bins = n_bins
    combined = pd.concat(
        [synth[gene_cols], ref[gene_cols]], ignore_index=True
    )
    fit_continuous_features_equaldepth(combined, name)

    def _enc(df):
        out = df.copy()
        out[gene_cols] = discretize_continuous_features_equaldepth(
            df[gene_cols], name
        )
        return out

    return _enc(synth), _enc(ref), _enc(targets)


# ===========================================================================
# Focal points + scoring
# ===========================================================================

def build_focal_points(gene_cols, target_col=None):
    fps = {(col,): 1.0 for col in gene_cols}
    if target_col:
        for col in gene_cols:
            fps[(col, target_col)] = 1.0
    n1 = sum(1 for c in fps if len(c) == 1)
    n2 = sum(1 for c in fps if len(c) == 2)
    print(f"  Focal points: {len(fps)} ({n1} 1-way, {n2} 2-way with '{target_col}')")
    return fps


def mama_mia_score(synth_enc, ref_enc, targets_enc, focal_points):
    """Likelihood-ratio score for each target row."""
    n = len(targets_enc)
    A = np.zeros(n)
    W = np.zeros(n)
    eps = 1e-10

    for clique, weight in focal_points.items():
        cols = [c for c in clique
                if c in synth_enc.columns
                and c in ref_enc.columns
                and c in targets_enc.columns]
        if not cols:
            continue

        D_synth = synth_enc[cols].value_counts(normalize=True)
        D_ref   = ref_enc[cols].value_counts(normalize=True)

        vals = targets_enc[cols].values
        for i, row in enumerate(vals):
            key = tuple(row)
            p_s = D_synth.get(key, default=eps)
            p_r = D_ref.get(key,   default=eps)
            A[i] += weight * (p_s / p_r)
            W[i] += weight

    return A / np.maximum(W, 1.0)


def _tpr_at_fpr(labels, scores, fpr_target=0.1):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(fpr, fpr_target)
    return float(tpr[min(idx, len(tpr) - 1)])


# ===========================================================================
# Per-split attack
# ===========================================================================

def attack_split(
    split_idx,
    submission_dir,
    tcga_dir,
    output_dir,
    target_col,
    n_bins,
):
    print(f"\n{'='*65}")
    print(f"Split {split_idx}")
    print(f"{'='*65}")

    # --- File paths ---
    synth_path  = os.path.join(submission_dir, f'synthetic_data_split_{split_idx}.csv')
    labels_path = os.path.join(submission_dir, f'synthetic_labels_split_{split_idx}.csv')
    splits_yaml = os.path.join(submission_dir, 'TCGA-COMBINED_splits.yaml')

    # TCGA files
    test_tsv  = os.path.join(tcga_dir, 'TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes.tsv')
    ref_tsv   = os.path.join(tcga_dir, 'TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes_reference.tsv')
    sub_csv   = os.path.join(tcga_dir, 'TCGA-COMBINED_primary_tumor_subtypes.csv')

    for p in [synth_path, labels_path, test_tsv, ref_tsv, sub_csv]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    # --- Load ---
    print("  Loading data …")
    synth   = load_synthetic_with_labels(synth_path, labels_path)
    ref     = load_tsv_with_subtypes(ref_tsv, sub_csv)
    targets = load_tsv_with_subtypes(test_tsv, sub_csv)

    print(f"  Synth  : {synth.shape}")
    print(f"  Ref    : {ref.shape}")
    print(f"  Targets: {targets.shape}")

    # --- Membership labels from splits YAML ---
    membership = None
    if os.path.exists(splits_yaml):
        membership = load_membership_from_yaml(
            splits_yaml, split_idx, targets.index
        )
    else:
        print(f"  WARNING: splits YAML not found at {splits_yaml} — "
              "AUC evaluation will be skipped.")

    # --- Align gene columns ---
    # gene_cols = ENSG columns present in all three DataFrames
    ensg_synth   = [c for c in synth.columns   if c.startswith('ENSG')]
    ensg_ref     = set(c for c in ref.columns     if c.startswith('ENSG'))
    ensg_targets = set(c for c in targets.columns if c.startswith('ENSG'))
    gene_cols = [c for c in ensg_synth if c in ensg_ref and c in ensg_targets]
    print(f"  Gene columns: {len(gene_cols)} shared ENSG features")

    # Build working DataFrames (genes + target_col)
    use_cols = gene_cols + ([target_col] if target_col else [])
    synth_w   = synth[use_cols].copy()
    ref_w     = ref[use_cols].copy()
    targets_w = targets[use_cols].copy()

    # --- Encode ---
    synth_enc, ref_enc, targets_enc = encode_all(
        synth_w, ref_w, targets_w, gene_cols, n_bins,
        name=f'split{split_idx}'
    )

    # --- Focal points + scoring ---
    fps    = build_focal_points(gene_cols, target_col if target_col else None)
    scores = mama_mia_score(synth_enc, ref_enc, targets_enc, fps)

    # --- Output ---
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'synthetic_data_split_{split_idx}_predictions.csv')

    out_df = pd.DataFrame(
        {'membership_label': scores},
        index=targets.index,
    )
    out_df.to_csv(out_path)
    print(f"\n  Predictions → {out_path}")
    print(f"  Score range : [{scores.min():.4f}, {scores.max():.4f}]")

    # --- Evaluate ---
    if membership is not None:
        lo, hi = scores.min(), scores.max()
        probs = (scores - lo) / (hi - lo) if hi > lo else np.full_like(scores, 0.5)
        auc  = roc_auc_score(membership, probs)
        ma   = 2 * auc - 1
        tpr  = _tpr_at_fpr(membership, probs, 0.1)
        print(f"  AUC-ROC        : {auc:.4f}")
        print(f"  Membership Adv : {ma:.4f}")
        print(f"  TPR@FPR=0.1    : {tpr:.4f}")
        return auc, ma
    return None, None


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MAMA-MIA attack on a PPML-Huskies-style Blue Team submission",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--submission-dir', required=True,
                        help='Directory containing synthetic_data_split_N.csv, '
                             'synthetic_labels_split_N.csv, TCGA-COMBINED_splits.yaml')
    parser.add_argument('--tcga-dir', required=True,
                        help='Directory with the TCGA TSV files and subtypes CSV')
    parser.add_argument('--output-dir', default='results/attack',
                        help='Where to write prediction CSVs')
    parser.add_argument('--splits', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help='Which splits to attack (default: all 5)')
    parser.add_argument('--target-col', default='cancer_type',
                        help='Column for 2-way marginals. '
                             'Set to "" to use 1-way only.')
    parser.add_argument('--n-bins', type=int, default=10,
                        help='Equal-depth bins for discretization')

    args = parser.parse_args()
    target_col = args.target_col if args.target_col else None

    results = []
    for s in args.splits:
        auc, ma = attack_split(
            split_idx=s,
            submission_dir=os.path.expanduser(args.submission_dir),
            tcga_dir=os.path.expanduser(args.tcga_dir),
            output_dir=os.path.expanduser(args.output_dir),
            target_col=target_col,
            n_bins=args.n_bins,
        )
        results.append((s, auc, ma))

    if any(auc is not None for _, auc, _ in results):
        print(f"\n{'='*65}")
        print("Summary")
        print(f"{'='*65}")
        print(f"{'Split':>6}  {'AUC':>8}  {'Memb. Adv':>10}")
        aucs = []
        for s, auc, ma in results:
            if auc is not None:
                print(f"{s:>6}  {auc:>8.4f}  {ma:>10.4f}")
                aucs.append(auc)
        if len(aucs) > 1:
            print(f"{'mean':>6}  {np.mean(aucs):>8.4f}  {2*np.mean(aucs)-1:>10.4f}")


if __name__ == '__main__':
    main()
