#!/usr/bin/env bash
# build_submission.sh
# Builds the two competition zip files from pre-computed prediction CSVs.
#
# Run from the results/ppml_huskies/ directory:
#   cd results/ppml_huskies
#   bash build_submission.sh <TEAMNAME> <BRCA_PREDS_DIR> <COMBINED_PREDS_DIR>
#
# Arguments:
#   TEAMNAME          : your team name (used in zip filename)
#   BRCA_PREDS_DIR    : directory containing synthetic_data_1_predictions.csv
#                       through synthetic_data_4_predictions.csv for TCGA-BRCA
#   COMBINED_PREDS_DIR: same for TCGA-COMBINED  (5 files, splits 1-5)
#
# Example:
#   bash build_submission.sh MyTeam results/brca_attack results/combined_attack
#
# Competition format requirements:
#   redteam_{TEAMNAME}_TCGA-BRCA.zip     contains:
#     red_team.py
#     config.yaml                        (with dataset_config.name: TCGA-BRCA)
#     models/__init__.py
#     models/mamamia_pgm.py
#     synthetic_data_1_predictions.csv
#     synthetic_data_2_predictions.csv
#     synthetic_data_3_predictions.csv
#     synthetic_data_4_predictions.csv
#     environment.yaml
#
#   redteam_{TEAMNAME}_TCGA-COMBINED.zip contains the same but
#     config.yaml with dataset_config.name: TCGA-COMBINED  and 5 splits.

set -euo pipefail

TEAMNAME="${1:?Usage: $0 TEAMNAME BRCA_PREDS_DIR COMBINED_PREDS_DIR}"
BRCA_DIR="${2:?}"
COMBINED_DIR="${3:?}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------------------------------------------------------
# Helper: build one zip
# --------------------------------------------------------------------------
build_zip() {
    local DATASET="$1"        # TCGA-BRCA or TCGA-COMBINED
    local PREDS_DIR="$2"      # directory with prediction CSVs
    local N_SPLITS="$3"       # 4 for BRCA, 5 for COMBINED
    local CONFIG_DATASET="$4" # value for dataset_config.name in config.yaml

    local ZIPNAME="redteam_${TEAMNAME}_${DATASET}.zip"
    local TMPDIR="$(mktemp -d)"

    echo ""
    echo "Building ${ZIPNAME} ..."

    # --- Code files ---
    cp "${SCRIPT_DIR}/red_team.py"        "${TMPDIR}/"
    cp "${SCRIPT_DIR}/environment.yaml"   "${TMPDIR}/" 2>/dev/null || true
    mkdir -p "${TMPDIR}/models"
    cp "${SCRIPT_DIR}/models/__init__.py"   "${TMPDIR}/models/"
    cp "${SCRIPT_DIR}/models/mamamia_pgm.py" "${TMPDIR}/models/"

    # --- Config: patch dataset_config.name on the fly ---
    python3 - <<PYEOF
import yaml, re
with open("${SCRIPT_DIR}/config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["dataset_config"]["name"] = "${CONFIG_DATASET}"
with open("${TMPDIR}/config.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
PYEOF

    # --- Prediction CSVs ---
    for i in $(seq 1 "${N_SPLITS}"); do
        local src="${PREDS_DIR}/synthetic_data_${i}_predictions.csv"
        if [[ -f "${src}" ]]; then
            cp "${src}" "${TMPDIR}/"
            echo "  + synthetic_data_${i}_predictions.csv"
        else
            echo "  WARNING: ${src} not found – zip will be incomplete"
        fi
    done

    # --- Package ---
    (cd "${TMPDIR}" && zip -r "${SCRIPT_DIR}/${ZIPNAME}" .)
    rm -rf "${TMPDIR}"
    echo "  → ${SCRIPT_DIR}/${ZIPNAME}"
}

# --------------------------------------------------------------------------
# Build BRCA zip (4 splits)
# --------------------------------------------------------------------------
build_zip "TCGA-BRCA"     "${BRCA_DIR}"     4  "TCGA-BRCA"

# --------------------------------------------------------------------------
# Build COMBINED zip (5 splits)
# --------------------------------------------------------------------------
build_zip "TCGA-COMBINED" "${COMBINED_DIR}" 5  "TCGA-COMBINED"

echo ""
echo "Done.  Submission zips:"
ls -lh "${SCRIPT_DIR}/redteam_${TEAMNAME}_"*.zip
