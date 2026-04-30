from abc import ABC, abstractmethod


class BaseMIAModel(ABC):
    """Minimal stub so mia/utils/baseline.py can be imported locally.

    DOMIASBaselineModels inherits from this; we only call run_baselines()
    directly so the full competition implementation is not needed here.
    """

    def __init__(self, config, synthetic_file, membership_test_file,
                 membership_lbl_file, mia_experiment_name, reference_file=None):
        self.config = config
        self.synthetic_file = synthetic_file
        self.membership_test_file = membership_test_file
        self.membership_lbl_file = membership_lbl_file
        self.reference_file = reference_file
        self.generator_model = config.get("generator_config", {}).get("name", "")
        self.experiment_name = config.get("generator_config", {}).get("experiment_name", "")
        self.attack_model = config.get("attack_model", "")
        self.dataset_config = config.get("dataset_config", {})
        self.dataset_name = self.dataset_config.get("name", "")
        self.membership_label_col = self.dataset_config.get("membership_label_col", "membership_label")
        self.mia_config = config.get(f"{self.attack_model}_config", {})
        self.home_dir = config.get("dir_list", {}).get("home", ".")

    @abstractmethod
    def run_attack(self):
        pass
