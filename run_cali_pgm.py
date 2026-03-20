"""MAMA-MIA experiment: California Housing (sklearn) + Private-PGM.

Mirrors the structure of attack_experiment_D from mamamia_experiments.py,
but hardcoded for cali + pgm so it runs without any CLI scaffolding.

Usage
-----
    python3 run_cali_pgm.py [epsilon]  [train_size]  [n_runs]

    epsilon    : DP epsilon  (default 1.0)
    train_size : records to train PGM on  (default 1000, matches expD.n)
    n_runs     : attack trials  (default 30, matches C.n_runs)

Dataset
-------
    California Housing from sklearn (20 640 rows × 9 features).
    Columns are renamed 0–8 ("8" = MedHouseVal, used as pgm target).
    All features are binned into C.n_bins=20 equal-depth bins.
    Households are synthetic blocks of 5 (household_min_size=5),
    but set_MI=False so targets are individual records.

Private-PGM marginal structure (fixed, not stochastic)
-------------------------------------------------------
    1-way : one per column  →  9 cliques
    2-way : (col_i, "8") for i in 0..7  →  8 cliques
    Total : 17 cliques  (no shadow-modelling uncertainty)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append('reprosyn-main/src/reprosyn/methods/mbi/')

import mbi_patch  # must be before any mbi import; patches FactoredInference + JIT

# Create required directory structure before importing (dump_artifact needs it).
os.makedirs("data/experiment_artifacts", exist_ok=True)
os.makedirs("intermediate/experiment_artifacts/focalpoints", exist_ok=True)
os.makedirs("intermediate/experiment_artifacts/satml25-rebuttal/mamamia_results", exist_ok=True)

from tqdm import tqdm
from util import *                              # Config, get_data, C, sample_experimental_data, dump_artifact …
from determine_focal_points import determine_privatepgm_marginals
from conduct_attacks import attack_privatepgm

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
epsilon    = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
train_size = int(sys.argv[2])   if len(sys.argv) > 2 else expD.n   # 1000
n_runs     = int(sys.argv[3])   if len(sys.argv) > 3 else C.n_runs  # 30

C.n_bins = 20          # standard cali setting (20 equal-depth bins per feature)
overlap  = True        # attacker's aux overlaps with the pool (standard setup)
set_MI   = False       # individual-record membership inference

# ---------------------------------------------------------------------------
# Config + data
# ---------------------------------------------------------------------------
cfg = Config(
    data_name="cali",
    train_size=train_size,
    overlapping_aux=overlap,
    set_MI=set_MI,
    check_arbitrary_fps=False,
    pgm_target_variable="8",          # column "8" = MedHouseVal
)

print("Loading California Housing data …")
_, full_aux, columns, meta, _ = get_data(cfg)
print(f"  aux shape : {full_aux.shape}")
print(f"  columns   : {columns}")   # ['0','1',…,'8']

# ---------------------------------------------------------------------------
# Focal points (deterministic for Private-PGM)
# ---------------------------------------------------------------------------
fp_file = f"FP_cali_pgm_e{epsilon:.2f}_n{train_size}"
fps = determine_privatepgm_marginals(
    cfg, full_aux, columns, cfg.categorical_columns, meta,
    epsilon, train_size, filename=fp_file,
)
print(f"\nFocal points: {len(fps)} cliques "
      f"({sum(1 for f in fps if len(f)==1)} 1-way, "
      f"{sum(1 for f in fps if len(f)==2)} 2-way)")

# ---------------------------------------------------------------------------
# Results store (same keys as attack_experiment_D)
# ---------------------------------------------------------------------------
results_file = (
    f"intermediate/experiment_artifacts/"
    f"results_pgm_e{epsilon:.2f}_n{train_size}_cali_o{overlap}_set{set_MI}"
)
results = load_artifact(results_file) or {
    "KDE_MA": [], "KDE_AUC": [], "KDE_time": [],
    "MM_MA": [], "MM_AUC": [],
    "MM_MA_weighted": [], "MM_AUC_weighted": [],
    "MM_time": [], "MM_arbitrary_MA": [], "distance": [],
    "KDE_ROC": [], "MM_ROC": [],
}
already_done = len(results["MM_AUC_weighted"])
remaining    = n_runs - already_done
print(f"\nRuns already completed: {already_done}  |  runs to go: {remaining}")

# ---------------------------------------------------------------------------
# Attack loop
# ---------------------------------------------------------------------------
for run in tqdm(range(remaining), desc=f"pgm cali ε={epsilon}"):
    target_ids, targets, membership, train, kde_seed = \
        sample_experimental_data(cfg, full_aux, columns)
    aux = full_aux if overlap else full_aux[~full_aux.index.isin(train.index)]

    (kde_ma, kde_auc, kde_time,
     mm_ma, mm_auc,
     mm_ma_w, mm_auc_w,
     mm_time, mm_arbitrary_ma,
     distance, kde_roc, mm_roc) = attack_privatepgm(
        cfg, meta, aux, columns, train, epsilon,
        targets, target_ids, membership, kde_seed, fps,
    )

    if kde_ma          is not None: results["KDE_MA"].append(kde_ma)
    if kde_auc         is not None: results["KDE_AUC"].append(kde_auc)
    if kde_time        is not None: results["KDE_time"].append(kde_time)
    if mm_ma           is not None: results["MM_MA"].append(mm_ma)
    if mm_auc          is not None: results["MM_AUC"].append(mm_auc)
    if mm_ma_w         is not None: results["MM_MA_weighted"].append(mm_ma_w)
    if mm_auc_w        is not None: results["MM_AUC_weighted"].append(mm_auc_w)
    if mm_time         is not None: results["MM_time"].append(mm_time)
    if distance        is not None: results["distance"].append(distance)
    if mm_arbitrary_ma is not None: results["MM_arbitrary_MA"].append(mm_arbitrary_ma)
    if kde_roc         is not None: results["KDE_ROC"].append(kde_roc)
    if mm_roc          is not None: results["MM_ROC"].append(mm_roc)

    dump_artifact(results, results_file)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"California Housing + Private-PGM  |  ε={epsilon}  |  n_train={train_size}  |  runs={n_runs}")
for key in ("MM_MA_weighted", "MM_AUC_weighted", "MM_MA", "MM_AUC"):
    vals = results[key]
    if vals:
        print(f"  {key:20s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  (n={len(vals)})")
print("=" * 60)
print(f"Results saved to: {results_file}")
