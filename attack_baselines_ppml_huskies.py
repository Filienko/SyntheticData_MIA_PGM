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
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, auc, accuracy_score, f1_score,
)
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
)


# ---------------------------------------------------------------------------
# Metric computation  (matches BaseMIAModel._compute_metrics)
# ---------------------------------------------------------------------------

METRIC_COLS = [
    'split', 'baseline',
    'AUC', 'MA',
    'acc_median', 'acc_best',
    'AP', 'PR_AUC',
    'f1_median', 'f1_best',
    'TPR@FPR=0.01', 'TPR@FPR=0.1',
    'Precision@5pct',
]


def _compute_precision_top_percent(y_true, scores, top_percent=5):
    n  = len(scores)
    k  = max(1, int(np.ceil(n * top_percent / 100)))
    top_idx    = np.argsort(scores)[-k:][::-1]
    top_members = y_true[top_idx].sum()
    return top_members / k


def _compute_metrics(y_scores: np.ndarray, y_true: np.ndarray) -> dict:
    """Full metric suite matching BaseMIAModel._compute_metrics."""
    y_pred_median = (y_scores > np.median(y_scores)).astype(int)

    thresholds = np.sort(np.unique(y_scores))
    if len(thresholds) >= 2:
        f1s = [f1_score(y_true, y_scores > t, zero_division=0) for t in thresholds]
        best_t   = thresholds[np.argmax(f1s)]
        y_pred_best = (y_scores > best_t).astype(int)
    else:
        y_pred_best = y_pred_median

    auc_sc = roc_auc_score(y_true, y_scores)
    ap     = average_precision_score(y_true, y_scores)
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(rec, prec)

    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    tpr_at_001  = float(tpr[(fpr >= 0.01).argmax()])
    tpr_at_01   = float(tpr[(fpr >= 0.1).argmax()])

    prec5 = _compute_precision_top_percent(y_true, y_scores, top_percent=5)

    return {
        'AUC':           auc_sc,
        'MA':            2 * auc_sc - 1,
        'acc_median':    accuracy_score(y_true, y_pred_median),
        'acc_best':      accuracy_score(y_true, y_pred_best),
        'AP':            ap,
        'PR_AUC':        pr_auc,
        'f1_median':     f1_score(y_true, y_pred_median, zero_division=0),
        'f1_best':       f1_score(y_true, y_pred_best,   zero_division=0),
        'TPR@FPR=0.01':  tpr_at_001,
        'TPR@FPR=0.1':   tpr_at_01,
        'Precision@5pct': prec5,
    }


def _setup_baseline_import(competition_repo: str):
    """Set up sys.path so mia/baselines.py (local) is importable.

    The local mia/baselines.py is imported for run_baselines().
    The competition repo src/ is added afterwards so that its transitive
    dependencies (mia.utils, mia.models.base, domias) resolve correctly.
    """
    # Local repo root first → mia.baselines resolves to ./mia/baselines.py
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Competition repo for transitive deps (mia.utils, mia.models.base, domias)
    src_dir = os.path.join(os.path.expanduser(competition_repo), 'src')
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(
            f"Expected competition repo src/ at: {src_dir}\n"
            f"Clone with: git clone https://github.com/PMBio/Health-Privacy-Challenge.git"
        )
    if src_dir not in sys.path:
        sys.path.append(src_dir)

    from mia.utils.baseline import run_baselines  # noqa: F401  — local mia/utils/baseline.py
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

    for p in [synth_path, test_tsv, sub_csv]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    if not os.path.exists(splits_yaml):
        print(f"  WARNING: splits YAML not found at {splits_yaml} — "
              "AUC evaluation will be skipped.")
        splits_yaml = None

    # ---- Load -----------------------------------------------------------
    print("  Loading data …")
    synth_raw = pd.read_csv(synth_path)

    if os.path.exists(labels_path):
        # Standard format: separate labels file
        labels_df  = pd.read_csv(labels_path)
        actual_col = label_col if label_col in labels_df.columns else labels_df.columns[0]
        synth_raw[label_col] = labels_df[actual_col].values
        print(f"  Labels loaded from {os.path.basename(labels_path)}")
    elif label_col in synth_raw.columns:
        # Internal format: label column already present in synth CSV
        print(f"  No separate labels file — using '{label_col}' column from synth CSV")
    else:
        raise FileNotFoundError(
            f"No labels file at {labels_path} and "
            f"no '{label_col}' column in {synth_path}.\n"
            f"  Columns in synth: {list(synth_raw.columns[:10])}"
        )
    synth   = synth_raw
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

    # ---- Impute NaNs (column means from X_G) ---------------------------
    # baseline.py's LR/RF classifiers reject NaN; impute before passing.
    col_means = np.nanmean(X_G, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)

    def _impute(X):
        if not np.isnan(X).any():
            return X
        out = X.copy()
        nan_mask = np.isnan(out)
        out[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        n_filled = nan_mask.sum()
        print(f"  Imputed {n_filled} NaN values with column means")
        return out

    X_G       = _impute(X_G)
    X_test    = _impute(X_test)
    if X_ref is not None:
        X_ref     = _impute(X_ref)
        X_ref_glc = X_ref

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

        if membership is not None:
            m = _compute_metrics(probs, membership)
            n_mem = int(membership.sum())
            print(f"  [{baseline_name}]  Members {n_mem}/{len(membership)}"
                  f"  AUC {m['AUC']:.4f}  MA {m['MA']:.4f}"
                  f"  TPR@0.1 {m['TPR@FPR=0.1']:.4f}"
                  f"  Prec@5% {m['Precision@5pct']:.4f}"
                  f"  → {os.path.basename(out_path)}")
            rows.append({'split': split_idx, 'baseline': baseline_name, **m})
        else:
            print(f"  [{baseline_name}]  score range [{raw.min():.4f}, {raw.max():.4f}]"
                  f"  → {os.path.basename(out_path)}")

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
            metric_cols = [c for c in full.columns if c not in ('split', 'baseline')]

            # Build mean rows (split='mean') per baseline
            mean_rows = []
            for baseline, grp in full.groupby('baseline'):
                row = {'split': 'mean', 'baseline': baseline}
                row.update(grp[metric_cols].mean().to_dict())
                mean_rows.append(row)
            mean_df = pd.DataFrame(mean_rows)
            full_with_mean = pd.concat([full, mean_df], ignore_index=True)

            print(f"\n{'='*80}")
            print(f"Summary  ({dataset}  |  ε={eps})")
            print(f"{'='*80}")
            hdr_metrics = ['AUC','MA','acc_best','f1_best',
                           'TPR@FPR=0.01','TPR@FPR=0.1','Precision@5pct']
            for baseline, grp in full.groupby('baseline'):
                print(f"\n  [{baseline}]")
                header = f"  {'Split':>6}" + "".join(f"  {m:>12}" for m in hdr_metrics)
                print(header)
                print('  ' + '-' * (len(header) - 2))
                for _, row in grp.iterrows():
                    vals = "".join(f"  {row[m]:>12.4f}" for m in hdr_metrics)
                    print(f"  {int(row['split']):>6}{vals}")
                if len(grp) > 1:
                    print('  ' + '-' * (len(header) - 2))
                    mrow = mean_df[mean_df['baseline'] == baseline].iloc[0]
                    vals = "".join(f"  {mrow[m]:>12.4f}" for m in hdr_metrics)
                    print(f"  {'mean':>6}{vals}")

            print(f"\n{'='*80}")
            summary_path = os.path.join(output_dir, 'baselines_summary.csv')
            full_with_mean.to_csv(summary_path, index=False)
            print(f"Summary  → {summary_path}")

    print(f"\nPrediction files in {output_dir}/")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('_predictions.csv'):
            print(f"  {f}")


if __name__ == '__main__':
    main()
