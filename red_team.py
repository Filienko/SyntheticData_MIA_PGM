"""red_team.py – MAMA-MIA attack class for Health-Privacy-Challenge submission.

This file is the submission entry point.  Place it (together with
competition_mia.py and any supporting files) inside the submission zip:

    redteam_{teamname}_TCGA-BRCA.zip
    ├── red_team.py                ← this file
    ├── competition_mia.py         ← our attack engine
    ├── encode_data.py             ← discretization helpers
    ├── util.py                    ← C namespace and dump/load helpers
    ├── mbi_patch.py               ← MBI pandas-3.x compatibility patch
    ├── config.yaml                ← competition config (updated below)
    ├── environment.yaml           ← conda environment
    ├── synthetic_data_1_predictions.csv   ← pre-generated predictions
    ├── synthetic_data_2_predictions.csv
    ├── synthetic_data_3_predictions.csv
    └── synthetic_data_4_predictions.csv

How the competition framework uses this file
--------------------------------------------
The competition calls run_mia() from its red_team.py entry point, which:
  1. Reads config.yaml from the current directory.
  2. Instantiates the class mapped to `attack_model` in `mia_classes`.
  3. Calls `mia_model.run_attack()` → Dict[str, np.ndarray].
  4. Calls `mia_model.save_predictions(predictions)`.

Relevant BaseMIAModel constructor signature (from competition source):
    BaseMIAModel(
        config,                  # dict from config.yaml
        synthetic_file,          # path to Blue Team synthetic CSV
        membership_test_file,    # path to membership test TSV/CSV
        membership_lbl_file,     # path to ground-truth labels (may be None)
        mia_experiment_name,     # e.g. "synthetic_data_1"
        reference_file=None,     # path to reference data (may be None)
    )
"""

import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Attempt to import BaseMIAModel from the competition framework.
# Falls back to a minimal stub when running outside the competition repo.
# ---------------------------------------------------------------------------
try:
    from src.mia.models.base import BaseMIAModel  # competition package
except ImportError:
    # Stub so this file is importable during standalone development.
    from abc import ABC, abstractmethod

    class BaseMIAModel(ABC):           # noqa: F811  (redefined locally)
        def __init__(self, config, synthetic_file, membership_test_file,
                     membership_lbl_file, mia_experiment_name, reference_file=None):
            self.config              = config
            self.synthetic_file      = synthetic_file
            self.membership_test_file = membership_test_file
            self.membership_lbl_file  = membership_lbl_file
            self.mia_experiment_name  = mia_experiment_name
            self.reference_file       = reference_file

        @abstractmethod
        def run_attack(self):
            ...

        def save_predictions(self, scores_dict):
            out_dir = self.config.get('mia_files', 'results/mia')
            os.makedirs(out_dir, exist_ok=True)
            for method, scores in scores_dict.items():
                fname = os.path.join(out_dir, f'{self.mia_experiment_name}_{method}_predictions.csv')
                import pandas as pd
                pd.DataFrame({'membership_label': scores}).to_csv(fname, index=False)
                print(f"  Saved → {fname}")

        def evaluate_attack(self, scores_dict, labels, file_name):
            from sklearn.metrics import roc_auc_score
            for method, scores in scores_dict.items():
                try:
                    auc = roc_auc_score(labels, scores)
                    print(f"  [{method}] AUC = {auc:.4f}  MA = {2*auc-1:.4f}")
                except Exception as e:
                    print(f"  [{method}] evaluation error: {e}")


# ---------------------------------------------------------------------------
# Import our attack engine
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from competition_mia import (
    load_synthetic,
    load_membership_test,
    load_gt,
    build_pgm_focal_points,
    encode_dataframes,
    mama_mia_score,
)


# ===========================================================================
# Our MAMA-MIA attack class
# ===========================================================================

class MamaMIAPGMModel(BaseMIAModel):
    """MAMA-MIA membership inference attack against the DP-PGM Blue Team.

    Attack overview
    ---------------
    Private-PGM uses a fixed marginal structure (all 1-way + all 2-way with
    a target variable). Because these cliques are deterministic, we can
    reconstruct the exact focal-point set from the domain without shadow
    modelling. We then score each target record by the MAMA-MIA likelihood-
    ratio sum over the pre-computed focal-point cliques, using:
      - P_synth  ← empirical marginal distribution of Blue Team synthetic data
      - P_ref    ← empirical marginal distribution of the reference dataset

    Config keys read from competition_config section in config.yaml
    --------------------------------------------------------------
    mama_mia_config:
      epsilon:    float  – DP epsilon used by Blue Team (informational only)
      n_bins:     int    – discretization bins (default 10)
      target_col: str    – PGM pivot column for 2-way marginals
                           (empty string or omit for 1-way only)
    """

    def run_attack(self):
        """Execute the MAMA-MIA attack.

        Returns
        -------
        dict  {"mama_mia": np.ndarray}
            Membership scores, one per test sample (higher = more likely member).
        """
        # --- Read attack parameters from config ---
        mm_cfg     = self.config.get('mama_mia_config', {})
        n_bins     = int(mm_cfg.get('n_bins', 10))
        target_col = mm_cfg.get('target_col', '') or None   # '' → None (1-way only)
        label_col  = (self.config.get('dataset_config', {})
                      .get('membership_label_col', 'membership_label'))

        print(f"\n[MamaMIAPGMModel] n_bins={n_bins}, target_col={target_col!r}")

        # --- Load data ---
        synth   = load_synthetic(self.synthetic_file)
        targets = load_membership_test(self.membership_test_file)

        if self.reference_file and os.path.exists(self.reference_file):
            import pandas as pd
            ref = pd.read_csv(self.reference_file, index_col=0)
        else:
            print("  WARNING: no reference file – using synthetic as ref (weak attack)")
            ref = synth.copy()

        # Drop label columns if present.
        synth   = synth.drop(columns=[label_col], errors='ignore')
        ref     = ref.drop(columns=[label_col], errors='ignore')
        targets = targets.drop(columns=[label_col], errors='ignore')

        # --- Align columns ---
        common_cols = [c for c in synth.columns
                       if c in ref.columns and c in targets.columns]
        synth   = synth[common_cols]
        ref     = ref[common_cols]
        targets = targets[common_cols]
        feature_cols = [c for c in common_cols if c != target_col]

        print(f"  Aligned on {len(common_cols)} columns, {len(targets)} target records")

        # --- Encode ---
        synth_enc, ref_enc, targets_enc = encode_dataframes(
            synth, ref, targets, feature_cols, n_bins, standardize=True
        )

        # --- Build focal points & score ---
        fps    = build_pgm_focal_points(common_cols, target_col)
        scores = mama_mia_score(synth_enc, ref_enc, targets_enc, fps)

        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        return {'mama_mia': scores}


# ===========================================================================
# mia_classes registry (required by the competition framework)
# ===========================================================================
mia_classes = {
    'mama_mia_pgm': MamaMIAPGMModel,
}
