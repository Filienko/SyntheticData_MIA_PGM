#!/usr/bin/env python3
"""write_config.py — Write a config.yaml for attack_ppml_huskies.py without re-running training.

Usage
-----
    # TCGA-COMBINED submission:
    python3 write_config.py \
        --submission-dir submission/internal/blueteam_PPML-Huskies_TCGA-COMBINED \
        --dataset TCGA-COMBINED \
        --epsilon 10.0 --num-iters 10000

    # TCGA-BRCA submission:
    python3 write_config.py \
        --submission-dir submission/internal/blueteam_PPML-Huskies_TCGA-BRCA \
        --dataset TCGA-BRCA \
        --epsilon 10.0 --num-iters 10000

The script infers subtype_col_name, count_file, and annot_file from --dataset
(override individually if your competition-repo layout differs).
"""

import argparse
import os

# Default relative paths (from competition_home) per dataset
DEFAULTS = {
    'TCGA-BRCA': {
        'subtype_col_name': 'Subtype',
        'count_file': 'data/processed/TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv',
        'annot_file': 'data/meta/TCGA-BRCA_primary_tumor_subtypes.csv',
    },
    'TCGA-COMBINED': {
        'subtype_col_name': 'cancer_type',
        'count_file': 'data/processed/TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes.tsv',
        'annot_file': 'data/meta/TCGA-COMBINED_primary_tumor_subtypes.csv',
    },
}


def main():
    p = argparse.ArgumentParser(
        description='Write config.yaml for attack_ppml_huskies.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--submission-dir', required=True,
                   help='Directory to write config.yaml into')
    p.add_argument('--dataset', required=True, choices=['TCGA-BRCA', 'TCGA-COMBINED'],
                   help='Dataset name')
    p.add_argument('--epsilon', type=float, default=10.0,
                   help='DP epsilon used during generation')
    p.add_argument('--num-iters', type=int, default=10000,
                   help='Iterations used during generation')
    p.add_argument('--subtype-col', default=None,
                   help='Override subtype column name (auto-set from --dataset)')
    p.add_argument('--count-file', default=None,
                   help='Override relative path to gene-expression TSV')
    p.add_argument('--annot-file', default=None,
                   help='Override relative path to annotation CSV')
    args = p.parse_args()

    d = DEFAULTS[args.dataset]
    subtype_col = args.subtype_col or d['subtype_col_name']
    count_file  = args.count_file  or d['count_file']
    annot_file  = args.annot_file  or d['annot_file']

    cfg = f"""\
dataset_config:
  name: "{args.dataset}"
  subtype_col_name: "{subtype_col}"
  count_file: "{count_file}"
  annot_file: "{annot_file}"

pgg_pgm_config:
  epsilon: {args.epsilon}
  iterations: {args.num_iters}
"""

    out_dir = os.path.expanduser(args.submission_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'config.yaml')
    with open(out_path, 'w') as f:
        f.write(cfg)
    print(f"Wrote {out_path}")
    print(cfg)


if __name__ == '__main__':
    main()
