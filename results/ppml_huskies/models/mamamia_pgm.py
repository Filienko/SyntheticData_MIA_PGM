"""MAMA-MIA attack on Private-PGM for the Health-Privacy-Challenge.

Implements PrivatePGMMIAModel, a concrete subclass of BaseMIAModel that
runs the MAMA-MIA likelihood-ratio membership inference attack against
a Private-PGM synthesizer.

Attack mechanics
----------------
For each target record x, the score is the weighted sum of likelihood
ratios over the fixed Private-PGM focal points:

    score(x) = Σ_{clique C}  P_synth(x[C]) / P_ref(x[C])

where:
  P_synth = empirical marginal of the synthetic dataset (numerator)
  P_ref   = empirical marginal of the reference population (denominator)
  cliques = all 1-way singletons  +  all 2-way (gene, target_col) pairs

Scores are mapped to [0,1] via a centred logistic function.

Usage
-----
  python red_team.py run-mia \\
      synthetic_data_1.csv \\
      TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes.tsv \\
      "" \\
      --mmb_labels_file membership_labels.csv \\
      --reference_file  reference_population.tsv

Config keys (under mama_mia_pgm_config in config.yaml)
-------------------------------------------------------
  n_bins              : int  – quantile bins for discretization (default 4)
  target_col          : str  – label column for 2-way marginals; "" → 1-way only
  centering_percentile: int  – sigmoid centering percentile (default 50)
                               Use 20 when ~80 % of targets are members.
"""

import os
import sys
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Import BaseMIAModel – works both inside the competition repo and standalone
# ---------------------------------------------------------------------------
try:
    from models.base import BaseMIAModel          # inside competition repo
except ImportError:
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_here, '..'))  # add src/mia to path
    from models.base import BaseMIAModel


# ===========================================================================
# Standalone helpers  (no dependency on the parent SyntheticData_MIA_PGM repo)
# ===========================================================================

def _load_data(path: str) -> pd.DataFrame:
    """Load CSV or TSV, transposing TSV files (genes×samples → samples×genes)."""
    sep = '\t' if path.endswith(('.tsv', '.txt')) else ','
    df = pd.read_csv(path, sep=sep, index_col=0)
    if sep == '\t':
        df = df.T          # genes×samples → samples×genes
        df.index.name = 'sample_id'
    return df


def _load_csv(path: str) -> pd.DataFrame:
    """Load a plain CSV (no transposing)."""
    return pd.read_csv(path)


def _is_pre_discretized(df: pd.DataFrame, cols: list, max_unique: int = 8) -> bool:
    """True when every column has ≤ max_unique distinct values.

    PRO-GENE-GEN PGM output has exactly n_bins unique float values per gene.
    """
    return all(df[col].nunique() <= max_unique for col in cols)


def _encode(synth: pd.DataFrame, ref: pd.DataFrame, targets: pd.DataFrame,
            gene_cols: list, n_bins: int, target_col: Optional[str]) -> Tuple[
                pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Discretize all DataFrames into consistent integer-bin columns.

    Case A – synthetic is already discrete (≤ n_bins+2 unique values / gene):
      • Synth  : rank unique floats → integers 0..k-1  (ordinal encoding)
      • Ref    : quantile-bin with n_bins bins, thresholds fitted on ref
      • Targets: same thresholds as ref

    Case B – synthetic is continuous:
      • Fit quantile thresholds on ref
      • Apply to synth, ref, targets
    """
    pre_disc = _is_pre_discretized(synth, gene_cols, max_unique=n_bins + 2)
    print(f"  Synth data: {'pre-discretized' if pre_disc else 'continuous'} "
          f"→ {'ordinal-rank' if pre_disc else 'quantile'} encoding with {n_bins} bins")

    synth_enc    = synth.copy()
    ref_enc      = ref.copy()
    targets_enc  = targets.copy()

    if pre_disc:
        # Synth: rank unique float values → 0, 1, …, k-1 per column
        for col in gene_cols:
            uvals = sorted(synth[col].dropna().unique())
            rank  = {v: i for i, v in enumerate(uvals)}
            synth_enc[col] = synth[col].map(rank).astype(int)

        # Ref / targets: equal-depth quantile bins fitted on ref only
        for col in gene_cols:
            boundaries = np.quantile(
                ref[col].dropna(),
                np.linspace(0, 1, n_bins + 1)[1:-1],  # n_bins-1 interior quantiles
            )
            ref_enc[col]    = np.digitize(ref[col],     boundaries).astype(int)
            targets_enc[col] = np.digitize(targets[col], boundaries).astype(int)
    else:
        # Equal-depth quantile bins fitted on ref
        for col in gene_cols:
            boundaries = np.quantile(
                ref[col].dropna(),
                np.linspace(0, 1, n_bins + 1)[1:-1],
            )
            synth_enc[col]   = np.digitize(synth[col],   boundaries).astype(int)
            ref_enc[col]     = np.digitize(ref[col],     boundaries).astype(int)
            targets_enc[col] = np.digitize(targets[col], boundaries).astype(int)

    # Encode target_col (cancer type) via sorted-label integer mapping
    if target_col:
        for df_enc, df_src in [(synth_enc, synth),
                                (ref_enc,   ref),
                                (targets_enc, targets)]:
            if target_col in df_src.columns:
                vals = sorted(df_src[target_col].dropna().unique().tolist())
                rank = {v: i for i, v in enumerate(vals)}
                df_enc[target_col] = df_src[target_col].map(rank).fillna(-1).astype(int)

    return synth_enc, ref_enc, targets_enc


def _build_focal_points(gene_cols: list,
                        target_col: Optional[str],
                        synth: pd.DataFrame,
                        targets: pd.DataFrame) -> dict:
    """Build PGM focal point dict: all 1-way + all 2-way (gene, target_col).

    2-way cliques are only added when target_col exists in both synth and targets.
    """
    fps = {(col,): 1.0 for col in gene_cols}

    if target_col:
        has_target = (target_col in synth.columns and
                      target_col in targets.columns and
                      synth[target_col].notna().any())
        if has_target:
            for col in gene_cols:
                fps[(col, target_col)] = 1.0
        else:
            print(f"  WARNING: target_col='{target_col}' not available in both "
                  "synthetic and test data → falling back to 1-way marginals only.")

    n1 = sum(1 for c in fps if len(c) == 1)
    n2 = sum(1 for c in fps if len(c) == 2)
    print(f"  Focal points: {len(fps)}  ({n1} 1-way,  {n2} 2-way)")
    return fps


def _mama_mia_score(synth_enc: pd.DataFrame,
                    ref_enc:   pd.DataFrame,
                    targets_enc: pd.DataFrame,
                    focal_points: dict) -> np.ndarray:
    """Compute weighted LR score for each target row.

    score(i) = Σ_clique  w * P_synth(target_i[clique]) / P_ref(target_i[clique])
    """
    n   = len(targets_enc)
    A   = np.zeros(n, dtype=np.float64)
    W   = np.zeros(n, dtype=np.float64)
    eps = 1e-10

    for clique, weight in focal_points.items():
        # Only use columns present in all three DataFrames
        cols = [c for c in clique
                if c in synth_enc.columns
                and c in ref_enc.columns
                and c in targets_enc.columns]
        if not cols:
            continue

        D_synth = synth_enc[cols].value_counts(normalize=True)
        D_ref   = ref_enc[cols].value_counts(normalize=True)

        for i, row in enumerate(targets_enc[cols].values):
            key = tuple(row)
            p_s = D_synth.get(key, default=eps)
            p_r = D_ref.get(key,   default=eps)
            A[i] += weight * (p_s / p_r)
            W[i] += weight

    return A / np.maximum(W, 1.0)


def _activate(raw_scores: np.ndarray, centering_percentile: int = 50) -> np.ndarray:
    """Map raw LR scores to membership probabilities via centred sigmoid.

    Centering at the P-th percentile of log-scores means that P % of targets
    receive probability < 0.5.  Use centering_percentile=20 when 80 % of
    targets are expected to be members.
    """
    logs      = np.log(np.maximum(raw_scores, 1e-300))
    threshold = np.percentile(logs, centering_percentile)
    return 1.0 / (1.0 + np.exp(-(logs - threshold)))


# ===========================================================================
# PrivatePGMMIAModel
# ===========================================================================

class PrivatePGMMIAModel(BaseMIAModel):
    """MAMA-MIA likelihood-ratio attack against a Private-PGM synthesizer.

    Registered in red_team.py as 'mama_mia_pgm'.
    Config key in config.yaml: mama_mia_pgm_config.
    """

    def __init__(self,
                 config,
                 synthetic_file:       str,
                 membership_test_file: str,
                 membership_lbl_file:  Optional[str],
                 mia_experiment_name:  str,
                 reference_file:       Optional[str] = None,
                 test_on_real:         bool = False):
        super().__init__(config,
                         synthetic_file,
                         membership_test_file,
                         membership_lbl_file,
                         mia_experiment_name,
                         reference_file)
        # Parse attack-specific config (under mama_mia_pgm_config:)
        self.n_bins               = int(self.mia_config.get('n_bins', 4))
        self.target_col           = self.mia_config.get('target_col', '') or None
        self.centering_percentile = int(self.mia_config.get('centering_percentile', 50))

    # ------------------------------------------------------------------
    # Abstract method implementation
    # ------------------------------------------------------------------

    def run_attack(self) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
        """Run the MAMA-MIA attack.

        Returns
        -------
        predictions : dict  {synth_stem: np.ndarray of membership scores in [0,1]}
        y_test      : np.ndarray | None  (ground-truth 0/1 labels, if available)
        """
        print(f"\n{'='*65}")
        print(f"MAMA-MIA Private-PGM Attack")
        print(f"  synthetic : {self.synthetic_file}")
        print(f"  test      : {self.membership_test_file}")
        print(f"  reference : {self.reference_file or '(using test data as reference)'}")
        print(f"  n_bins    : {self.n_bins}")
        print(f"  target_col: {self.target_col or '(none – 1-way only)'}")
        print(f"  centering : {self.centering_percentile}th percentile")
        print(f"{'='*65}")

        # ---- Load data ------------------------------------------------
        synth   = _load_csv(self.synthetic_file)
        targets = _load_data(self.membership_test_file)
        ref     = (_load_data(self.reference_file)
                   if self.reference_file else targets.copy())
        y_test  = self._load_labels()

        print(f"  Shapes:  synth={synth.shape}  targets={targets.shape}  ref={ref.shape}")
        if y_test is not None:
            print(f"  Labels:  {y_test.sum()} members / {(y_test==0).sum()} non-members")

        # ---- Align gene columns -------------------------------------
        gene_cols = self._align_gene_cols(synth, ref, targets)
        if not gene_cols:
            raise RuntimeError("No shared gene columns found across synth / ref / targets.")
        print(f"  Shared gene columns: {len(gene_cols)}")

        # ---- Build working DataFrames (genes + optional target_col) --
        use_cols = gene_cols[:]
        if self.target_col:
            for df in (synth, ref, targets):
                if self.target_col not in df.columns:
                    print(f"  Note: '{self.target_col}' absent from "
                          f"{type(df).__name__} – will skip 2-way marginals if needed")
            use_cols = gene_cols + [self.target_col]

        synth_w   = synth[[c for c in use_cols if c in synth.columns]].copy()
        ref_w     = ref[[c for c in use_cols if c in ref.columns]].copy()
        targets_w = targets[[c for c in use_cols if c in targets.columns]].copy()

        # ---- Encode --------------------------------------------------
        synth_enc, ref_enc, targets_enc = _encode(
            synth_w, ref_w, targets_w, gene_cols, self.n_bins, self.target_col
        )

        # ---- Focal points + raw scores ------------------------------
        fps = _build_focal_points(gene_cols, self.target_col, synth_enc, targets_enc)
        raw = _mama_mia_score(synth_enc, ref_enc, targets_enc, fps)
        print(f"  Raw score range: [{raw.min():.4f}, {raw.max():.4f}]")

        # ---- Map to [0,1] -------------------------------------------
        scores = _activate(raw, self.centering_percentile)
        print(f"  Prob  range:  [{scores.min():.4f}, {scores.max():.4f}]")

        # ---- Package predictions ------------------------------------
        # Key becomes the CSV filename stem → save_predictions() names
        # the output file <key>_predictions.csv
        synth_key  = os.path.splitext(os.path.basename(self.synthetic_file))[0]
        predictions = {synth_key: scores}

        return predictions, y_test

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_labels(self) -> Optional[np.ndarray]:
        """Load membership labels (0/1) from mmb_labels_file, if provided."""
        if not self.membership_lbl_file:
            return None
        df  = pd.read_csv(self.membership_lbl_file)
        col = self.membership_label_col
        if col in df.columns:
            return df[col].values.astype(int)
        # Fallback: use first numeric column
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                print(f"  WARNING: label column '{col}' not found; "
                      f"using '{c}' instead.")
                return df[c].values.astype(int)
        raise ValueError(f"No membership label column found in {self.membership_lbl_file}")

    @staticmethod
    def _align_gene_cols(synth: pd.DataFrame,
                         ref:   pd.DataFrame,
                         targets: pd.DataFrame) -> list:
        """Return ENSG* columns shared by all three DataFrames.

        Falls back to all numeric columns if no ENSG columns are found.
        """
        def _gene_cols(df):
            ensg = [c for c in df.columns if str(c).startswith('ENSG')]
            if ensg:
                return set(ensg)
            return set(c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))

        shared = _gene_cols(synth) & _gene_cols(ref) & _gene_cols(targets)
        # Preserve order as they appear in synth
        return [c for c in synth.columns if c in shared]
