"""Health-Privacy-Challenge – Red Team entry point.

Extends the competition's red_team.py to register PrivatePGMMIAModel
(MAMA-MIA attack on Private-PGM synthetic data).

Usage
-----
  # Attack a single synthetic split:
  python red_team.py run-mia \\
      /path/to/synthetic_data_1.csv \\
      /path/to/TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes.tsv \\
      "run_1" \\
      --mmb_labels_file /path/to/labels.csv \\
      --reference_file  /path/to/reference_population.tsv

  # No ground-truth labels available (competition blind submission):
  python red_team.py run-mia \\
      /path/to/synthetic_data_1.csv \\
      /path/to/membership_test.tsv \\
      ""

Output
------
  Predictions are written to:
    {config.dir_list.home}/results/mia/{dataset}/{attack_model}/{generator}/{experiment}/
  as  {synthetic_stem}_predictions.csv  (one column: membership_label)

  Rename/copy the files to match the competition naming convention:
    synthetic_data_1_predictions.csv  …  synthetic_data_5_predictions.csv
"""

import click
import yaml
import os
import sys
import importlib

# ---------------------------------------------------------------------------
# Ensure the models/ package next to this file is importable
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

# ---------------------------------------------------------------------------
# MIA class registry
# Add new attack models here: 'config_key': ('module.path', 'ClassName')
# The config_key MUST match the attack_model value in config.yaml AND
# the <attack_model>_config section key.
# ---------------------------------------------------------------------------
mia_classes = {
    # Competition baselines (unchanged)
    'domias_baselines':    ('models.baseline',     'DOMIASBaselineModels'),
    'sc_domias_baselines': ('models.sc_baseline',  'DOMIASSingleCellBaselineModels'),

    # MAMA-MIA Private-PGM attack (this submission)
    'mama_mia_pgm':        ('models.mamamia_pgm',  'PrivatePGMMIAModel'),
}


def get_mia_class(mia_name):
    if mia_name not in mia_classes:
        raise ValueError(
            f"Unknown MIA model: '{mia_name}'. "
            f"Available: {list(mia_classes.keys())}"
        )
    module_name, class_name = mia_classes[mia_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli():
    pass


@click.command()
@click.argument('synthetic_file',     type=click.Path(exists=True))
@click.argument('mmb_test_file',      type=click.Path(exists=True))
@click.argument('mia_experiment_name', type=str, default="")
@click.option('--mmb_labels_file',    type=click.Path(exists=True), default=None,
              help='CSV with ground-truth membership labels (0/1). Optional.')
@click.option('--test_on_real',       type=bool, default=False, hidden=True)
@click.option('--reference_file',     type=click.Path(exists=True), default=None,
              help='Reference population for P_aux denominator. '
                   'If omitted, the test data is used as reference.')
def run_mia(synthetic_file:      str,
            mmb_test_file:       str,
            mia_experiment_name: str  = "",
            mmb_labels_file:     str  = None,
            test_on_real:        bool = False,
            reference_file:      str  = None):
    """Run a membership inference attack on a single synthetic dataset."""
    configfile = "config.yaml"
    config = yaml.safe_load(open(configfile))

    attack_model = config.get('attack_model')
    MIAClass     = get_mia_class(attack_model)

    mia_model = MIAClass(
        config,
        synthetic_file,
        mmb_test_file,
        mmb_labels_file,
        mia_experiment_name,
        reference_file,
        test_on_real,
    )

    predictions, y_test = mia_model.run_attack()
    mia_model.save_predictions(predictions)

    if y_test is not None:
        mia_model.evaluate_attack(predictions, y_test, "evaluation_results.csv")


@click.command()
@click.argument('synthetic_file',     type=click.Path(exists=True))
@click.argument('mmb_test_file',      type=click.Path(exists=True))
@click.argument('mia_experiment_name', type=str, default="")
@click.option('--mmb_labels_file',    type=click.Path(exists=True), default=None)
@click.option('--reference_file',     type=click.Path(exists=True), default=None)
def run_singlecell_mia(synthetic_file:      str,
                       mmb_test_file:       str,
                       mia_experiment_name: str = "",
                       mmb_labels_file:     str = None,
                       reference_file:      str = None):
    """Run a single-cell MIA with donor-level averaging."""
    configfile = "config.yaml"
    config = yaml.safe_load(open(configfile))

    attack_model = config.get('attack_model')
    MIAClass     = get_mia_class(attack_model)

    mia_model = MIAClass(
        config,
        synthetic_file,
        mmb_test_file,
        mmb_labels_file,
        mia_experiment_name,
        reference_file,
    )

    predictions, y_test = mia_model.run_attack()
    mia_model.save_predictions(predictions)

    if y_test is not None:
        grp_preds, grp_y = mia_model.perform_donor_level_avg(predictions, y_test)
        mia_model.evaluate_attack(grp_preds, grp_y, "evaluation_results.csv")


cli.add_command(run_mia)
cli.add_command(run_singlecell_mia)

if __name__ == '__main__':
    cli()
