"""red_team.py – MAMA-MIA submission for Health-Privacy-Challenge.

Drop this file (plus competition_mia.py, encode_data.py, util.py, mbi_patch.py,
mbi/, reprosyn-main/, and config.yaml) into the submission zip.

Competition evaluators run it as:
    python red_team.py run-mia <synthetic_file> <mmb_test_file> <experiment_name> \\
        [--mmb_labels_file <gt_csv>] [--reference_file <ref_tsv>]

The competition framework (src/mia/red_team.py) also calls it dynamically via
mia_classes; our standalone click commands replicate that flow so this file
works both ways.

Data loading follows MIADataLoader conventions exactly
(src/mia/utils/prepare_data.py):
  - synthetic_file : CSV, no index column, numeric gene columns.
  - mmb_test_file  : TSV, genes as rows / samples as columns → transposed.
  - reference_file : TSV, same layout as mmb_test_file.
  - mmb_labels_file: CSV with index col, column membership_label_col.

run_attack() returns (Dict[str, np.ndarray], np.ndarray | None) as expected
by src/mia/red_team.py:
    predictions, y_test = mia_model.run_attack()
"""

import os
import sys
import click
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Import BaseMIAModel – works inside the competition repo or standalone.
# ---------------------------------------------------------------------------
try:
    from src.mia.models.base import BaseMIAModel
except ImportError:
    # Stub for standalone use / development outside the competition repo.
    from abc import ABC, abstractmethod

    class BaseMIAModel(ABC):                    # noqa: F811
        def __init__(self, config, synthetic_file, membership_test_file,
                     membership_lbl_file, mia_experiment_name,
                     reference_file=None):
            self.config               = config
            self.home_dir             = config["dir_list"]["home"]
            self.generator_model      = config["generator_config"]["name"]
            self.experiment_name      = config["generator_config"]["experiment_name"]
            self.attack_model         = config["attack_model"]
            self.dataset_config       = config["dataset_config"]
            self.dataset_name         = self.dataset_config["name"]
            self.membership_label_col = self.dataset_config["membership_label_col"]
            self.synthetic_file       = synthetic_file
            self.reference_file       = reference_file
            self.membership_test_file = membership_test_file
            self.membership_lbl_file  = membership_lbl_file
            self.results_save_dir     = os.path.join(
                os.path.expanduser(self.home_dir),
                config["dir_list"]["mia_files"],
                self.dataset_name,
                self.attack_model,
                self.generator_model,
                self.experiment_name,
                mia_experiment_name,
            )
            os.makedirs(self.results_save_dir, exist_ok=True)
            config_key = f"{self.attack_model}_config"
            if config_key in config:
                self.mia_config = config[config_key]
            else:
                raise ValueError(
                    f"config.yaml must contain '{config_key}' section "
                    f"(attack_model is '{self.attack_model}')."
                )

        @abstractmethod
        def run_attack(self):
            ...

        def save_predictions(self, scores_dict):
            for key, arr in scores_dict.items():
                df = pd.DataFrame(data=arr, columns=[self.membership_label_col])
                path = os.path.join(self.results_save_dir, f"{key}_predictions.csv")
                df.to_csv(path, index=False)
                print(f"  Saved → {path}")

        def evaluate_attack(self, scores_dict, labels, file_name):
            from sklearn.metrics import roc_auc_score
            rows = []
            for method, arr in scores_dict.items():
                try:
                    auc = roc_auc_score(labels, arr)
                    rows.append({"method": method, "aucroc": round(auc, 4),
                                 "ma": round(2 * auc - 1, 4)})
                    print(f"  [{method}] AUC={auc:.4f}  MA={2*auc-1:.4f}")
                except Exception as exc:
                    print(f"  [{method}] evaluation error: {exc}")
            if rows:
                path = os.path.join(self.results_save_dir, file_name)
                pd.DataFrame(rows).to_csv(path, index=False)
                print(f"  Evaluation → {path}")


# ---------------------------------------------------------------------------
# Our attack engine (import from competition_mia.py in the same zip).
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from competition_mia import (
    build_pgm_focal_points,
    encode_dataframes,
    mama_mia_score,
)


# ===========================================================================
# Data loading helpers  (match MIADataLoader in prepare_data.py exactly)
# ===========================================================================

def _load_csv(path):
    """Synthetic data: CSV, no index column."""
    return pd.read_csv(path)


def _load_tsv_transposed(path):
    """Membership test / reference: TSV, genes as rows → transpose to samples×genes."""
    return pd.read_csv(path, sep="\t", index_col=0).T


def _load_labels(path, label_col):
    """Ground-truth membership labels: CSV with index column."""
    return pd.read_csv(path, index_col=0)[label_col].values.astype(int)


# ===========================================================================
# Our MIA model class
# ===========================================================================

class MamaMIAPGMModel(BaseMIAModel):
    """MAMA-MIA membership inference attack against the DP-PGM Blue Team.

    Attack overview
    ---------------
    Private-PGM uses a *fixed* marginal structure (deterministic):
      - All 1-way singletons for every gene column
      - All 2-way (gene, target_col) pairs if target_col is set

    We reconstruct these focal points from the domain, then score each test
    sample by the MAMA-MIA likelihood-ratio sum:
        score(x) = Σ_clique  P_synth(x[clique]) / P_ref(x[clique])

    where P_synth is estimated from the Blue Team's synthetic data and
    P_ref from the reference dataset.

    Parameters read from mama_mia_pgm_config in config.yaml
    --------------------------------------------------------
    n_bins     : int   discretization bins (default 10)
    target_col : str   PGM pivot column for 2-way marginals ("" = 1-way only)
    """

    def __init__(self, config, synthetic_file, membership_test_file,
                 membership_lbl_file, mia_experiment_name,
                 reference_file=None, test_on_real=False):
        # test_on_real is passed by run_mia() but not used by our attack.
        super().__init__(config, synthetic_file, membership_test_file,
                         membership_lbl_file, mia_experiment_name, reference_file)

    def run_attack(self):
        """Execute the MAMA-MIA attack.

        Returns
        -------
        predictions : Dict[str, np.ndarray]
            {"mama_mia": scores}  – one score per test sample, higher = member.
        y_test : np.ndarray | None
            Ground-truth membership labels, or None if not provided.
        """
        n_bins     = int(self.mia_config.get("n_bins", 10))
        target_col = self.mia_config.get("target_col", "") or None

        print(f"\n[MamaMIAPGMModel] n_bins={n_bins}, target_col={target_col!r}")
        print(f"  synthetic : {self.synthetic_file}")
        print(f"  test      : {self.membership_test_file}")
        print(f"  reference : {self.reference_file}")

        # --- Load data as DataFrames (we need column names for marginals) ---
        synth_df = _load_csv(self.synthetic_file)
        test_df  = _load_tsv_transposed(self.membership_test_file)
        test_df.index = range(len(test_df))

        if self.reference_file and os.path.exists(self.reference_file):
            ref_df = _load_tsv_transposed(self.reference_file)
            ref_df.index = range(len(ref_df))
        else:
            print("  WARNING: no reference file – using synthetic data as ref (weak attack)")
            ref_df = synth_df.copy()

        # --- Ground-truth labels (None if not provided) ---
        y_test = None
        if self.membership_lbl_file and os.path.exists(self.membership_lbl_file):
            y_test = _load_labels(self.membership_lbl_file, self.membership_label_col)

        # --- Drop label column if accidentally present ---
        lbl = self.membership_label_col
        synth_df = synth_df.drop(columns=[lbl], errors="ignore")
        ref_df   = ref_df.drop(columns=[lbl], errors="ignore")
        test_df  = test_df.drop(columns=[lbl], errors="ignore")

        # --- Align columns ---
        common_cols = [c for c in synth_df.columns
                       if c in ref_df.columns and c in test_df.columns]
        synth_df = synth_df[common_cols]
        ref_df   = ref_df[common_cols]
        test_df  = test_df[common_cols]
        feature_cols = [c for c in common_cols if c != target_col]

        print(f"  Aligned on {len(common_cols)} columns, {len(test_df)} test samples")

        # --- Encode (StandardScaler → equal-depth binning) ---
        synth_enc, ref_enc, test_enc = encode_dataframes(
            synth_df, ref_df, test_df, feature_cols, n_bins, standardize=True
        )

        # --- MAMA-MIA scoring ---
        fps    = build_pgm_focal_points(common_cols, target_col)
        scores = mama_mia_score(synth_enc, ref_enc, test_enc, fps)

        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        return {"mama_mia": scores}, y_test


# ===========================================================================
# mia_classes registry  (used by the competition's dynamic import)
# ===========================================================================
mia_classes = {
    "mama_mia_pgm": MamaMIAPGMModel,
}


# ===========================================================================
# Standalone CLI  (mirrors src/mia/red_team.py so this file is self-contained)
# ===========================================================================

def _get_mia_class(name):
    if name in mia_classes:
        return mia_classes[name]
    raise ValueError(f"Unknown attack model: {name!r}. Known: {list(mia_classes)}")


@click.group()
def cli():
    pass


@cli.command("run-mia")
@click.argument("synthetic_file",    type=click.Path(exists=True))
@click.argument("mmb_test_file",     type=click.Path(exists=True))
@click.argument("mia_experiment_name", type=str, default="")
@click.option("--mmb_labels_file",  type=click.Path(), default=None)
@click.option("--test_on_real",      type=bool, default=False)
@click.option("--reference_file",   type=click.Path(), default=None)
def run_mia(synthetic_file, mmb_test_file, mia_experiment_name,
            mmb_labels_file, test_on_real, reference_file):
    """Run MAMA-MIA attack – called by competition evaluators or directly."""
    config = yaml.safe_load(open("config.yaml"))
    MIAClass = _get_mia_class(config["attack_model"])

    model = MIAClass(
        config, synthetic_file, mmb_test_file,
        mmb_labels_file, mia_experiment_name,
        reference_file, test_on_real,
    )

    predictions, y_test = model.run_attack()
    model.save_predictions(predictions)

    if y_test is not None:
        model.evaluate_attack(predictions, y_test, "evaluation_results.csv")


if __name__ == "__main__":
    cli()
