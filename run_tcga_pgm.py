"""Self-contained MAMA-MIA experiment: TCGA dataset + Private-PGM synthesizer.

Usage
-----
    python3 run_tcga_pgm.py [epsilon]  [train_size]  [n_runs]

    epsilon    : DP epsilon  (default 1.0)
    train_size : records used to train PGM  (default 500)
    n_runs     : how many attack trials  (default 10)

Prerequisites
-------------
    1. Place  data/tcga_combined_full_100f.csv  in the project root's data/ dir.
    2. pip install mbi  (already installed; version providing FactoredInference)
    3. The reprosyn submodule must exist (reprosyn-main/).

What it does
------------
    Because Private-PGM measures a *fixed* set of marginals:
      • all 1-way singletons
      • all 2-way (feature_col, Subtype) pairs
    shadow-modelling is skipped – the focal-point set is constructed directly
    from the domain.  The attack then runs `custom_mst_attack` (shared with
    the MST/PrivBayes attack paths) on those fixed cliques.

Output
------
    Per-run Membership-Advantage (MA) and AUC printed to stdout, plus a
    summary across all runs.
"""

import sys
import os
import time
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parse CLI args
# ---------------------------------------------------------------------------
epsilon    = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
train_size = int(sys.argv[2])   if len(sys.argv) > 2 else 500
n_runs     = int(sys.argv[3])   if len(sys.argv) > 3 else 10

# ---------------------------------------------------------------------------
# Bootstrap paths (mirrors mamamia_experiments.py)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
sys.path.append('reprosyn-main/src/reprosyn/methods/mbi/')

# Mirror the directory structure expected by mamamia_experiments.py.
# DATA_DIR = "data/"  → artifacts go into data/experiment_artifacts/
# DIR     = "intermediate" → focal-points go into intermediate/experiment_artifacts/focalpoints/
os.makedirs("data/experiment_artifacts", exist_ok=True)
os.makedirs("intermediate/experiment_artifacts/focalpoints", exist_ok=True)
os.makedirs("intermediate/experiment_artifacts/satml25-rebuttal/mamamia_results", exist_ok=True)

from util import *                          # Config, get_data, C, dump_artifact …
from determine_focal_points import determine_privatepgm_marginals
from conduct_attacks import attack_privatepgm

# ---------------------------------------------------------------------------
# Config  –  TCGA is small (1089 rows) so use a small train_size.
# train_sizes maps train_size → num_targets; set manually for TCGA.
# ---------------------------------------------------------------------------
C.n_bins = 10                               # bins for continuous features

cfg = Config(
    data_name="tcga",
    train_size=train_size,
    # With 1089 rows we need many rows left over for aux/targets.
    train_sizes={train_size: max(6, train_size // 10)},
    set_MI=False,
    overlapping_aux=True,
    check_arbitrary_fps=False,
    pgm_target_variable="Subtype",
    epsilons=[epsilon],
)

# ---------------------------------------------------------------------------
# Load + pre-process data
# ---------------------------------------------------------------------------
print("Loading TCGA data …")
_, aux, columns, meta, _ = get_data(cfg)
print(f"  aux shape : {aux.shape}")
print(f"  columns   : {len(columns)} ({columns[:3]} … {columns[-3:]})")
print(f"  Subtype   : {aux['Subtype'].nunique()} classes, "
      f"counts {dict(aux['Subtype'].value_counts().sort_index())}")

# ---------------------------------------------------------------------------
# Focal points  –  for Private-PGM these are deterministic.
# ---------------------------------------------------------------------------
fp_file = f"intermediate/experiment_artifacts/focalpoints/FP_tcga_pgm_e{epsilon:.2f}"
fps = determine_privatepgm_marginals(
    cfg, aux, columns, cfg.categorical_columns, meta,
    epsilon, train_size, filename=fp_file
)
print(f"\nFocal points: {len(fps)} cliques "
      f"({sum(1 for f in fps if len(f)==1)} 1-way, "
      f"{sum(1 for f in fps if len(f)==2)} 2-way)")

# ---------------------------------------------------------------------------
# Attack loop
# ---------------------------------------------------------------------------
all_ma, all_auc = [], []

for run in range(n_runs):
    print(f"\n── Run {run+1}/{n_runs}  (ε={epsilon}, n={train_size}) ──")

    # Sample train / targets from aux.
    target_ids, targets, membership, train, kde_seed = \
        sample_experimental_data(cfg, aux, columns)

    result = attack_privatepgm(
        cfg, meta, aux, columns, train, epsilon,
        targets, target_ids, membership,
        kde_sample_seed=kde_seed,
        fps=fps,
    )

    # result tuple: (kde_ma, kde_auc, kde_time, mm_ma, mm_auc,
    #                ma_w, auc_w, time, arbitrary_ma, distance,
    #                kde_roc, mm_roc)
    ma_w  = result[5]
    auc_w = result[6]

    print(f"  MA  = {ma_w:.4f}" if ma_w  is not None else "  MA  = N/A")
    print(f"  AUC = {auc_w:.4f}" if auc_w is not None else "  AUC = N/A")

    if ma_w  is not None: all_ma.append(ma_w)
    if auc_w is not None: all_auc.append(auc_w)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print(f"TCGA + Private-PGM  |  ε={epsilon}  |  n_train={train_size}  |  runs={n_runs}")
if all_ma:
    print(f"  Mean MA  = {np.mean(all_ma):.4f}  (std {np.std(all_ma):.4f})")
if all_auc:
    print(f"  Mean AUC = {np.mean(all_auc):.4f}  (std {np.std(all_auc):.4f})")
print("="*60)
