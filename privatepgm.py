"""Private-PGM synthesizer for MAMA-MIA attack.

Adapted from:
  https://github.com/Filienko/Pro-GENE-GEN-MPC/blob/main/models/Private_PGM/model.py
  (originally from https://github.com/ryan112358/private-pgm/blob/master/examples/adult_example.py)

Unlike MST, which *adaptively* selects 2-way marginals via the exponential
mechanism, Private-PGM uses a *fixed* marginal structure:

  - All 1-way marginals  (one per column)
  - All 2-way (col, target_variable) marginals  (one per non-target column)

This makes the focal-point set deterministic: no exponential-mechanism
randomness leaks through clique selection.  The MAMA-MIA attack still applies
because:
  1. The attacker can reconstruct the exact clique set from the algorithm
     spec + domain knowledge (no shadow-model uncertainty).
  2. Weighted likelihood ratios over these fixed marginals still reveal
     whether a target record was in the training set.
"""

import sys
import numpy as np
from scipy import sparse
from scipy.optimize import bisect

# Use the same mbi path that MST and PrivBayes already rely on.
sys.path.insert(0, 'reprosyn-main/src/reprosyn/methods/mbi/')
from mbi import Dataset, Domain, FactoredInference
from cdp2adp import cdp_rho

# Generator base from reprosyn (gives us encode_ordinal / decode_ordinal).
sys.path.insert(1, 'reprosyn-main/src/reprosyn/')
from generator import PipelineBase, encode_ordinal, decode_ordinal


# ---------------------------------------------------------------------------
# Noise calibration (mirrors model.py's moments_calibration).
# ---------------------------------------------------------------------------

def _calibrate_sigma(epsilon, delta):
    """Return Gaussian noise sigma so the TWO-round query is (eps, delta)-DP.

    The reference (model.py moments_calibration(round1=1, round2=1, eps, delta))
    finds sigma such that composing TWO Gaussian mechanisms (one 1-way round,
    one 2-way round), each with sensitivity 1 and noise sigma, satisfies
    (eps, delta)-DP under RDP composition.

    Under zCDP each Gaussian mechanism with sensitivity 1 and noise sigma
    contributes rho_i = 1 / (2 * sigma^2).  Two rounds therefore compose to:
        rho_total = 1 / sigma^2

    For (eps, delta)-DP we need rho_total <= cdp_rho(eps, delta), giving:
        sigma >= 1 / sqrt(cdp_rho(eps, delta))

    Previously this returned sqrt(1 / (2*rho)), which is calibrated for only
    ONE round and is ~sqrt(2) too small, effectively spending ~2x epsilon.
    """
    if delta <= 0:
        # Pure DP via Laplace; return scale = 1/(eps * d) (model.py fallback).
        return None  # handled separately in privatepgm()
    rho = cdp_rho(epsilon, delta)
    return np.sqrt(1.0 / rho)          # two-round composition: rho_total = 1/sigma^2


# ---------------------------------------------------------------------------
# Core training / synthesis function.
# ---------------------------------------------------------------------------

def privatepgm(data, cliques, epsilon, delta, rows, num_iters=1000):
    """Train Private-PGM with fixed marginals and return synthetic data.

    Parameters
    ----------
    data      : mbi.Dataset (ordinal-encoded)
    cliques   : list of 2-way clique tuples, e.g. [('age', 'income'), ...]
                These are the (col, target_variable) pairs.
    epsilon   : privacy budget
    delta     : delta parameter (use 0 for pure DP / Laplace noise)
    rows      : number of synthetic rows to generate
    num_iters : FactoredInference mirror-descent iterations

    Returns
    -------
    (synth_dataset, selected_cliques)
        synth_dataset   : mbi.Dataset of synthetic records
        selected_cliques: list of tuples – the cliques actually measured
                          (1-way union 2-way); these become the focal points.
    """
    n = data.df.shape[0]
    measurements = []
    selected = []  # will collect every measured clique

    if delta > 0:
        sigma = _calibrate_sigma(epsilon, delta)

        # --- Round 1: all 1-way marginals ---
        d1 = len(data.domain)
        w1 = np.ones(d1) / np.sqrt(d1)   # L2-normalised weights (model.py)

        for col, wgt in zip(data.domain.attrs, w1):
            x = data.project([col]).datavector()
            I = sparse.eye(x.size)
            y = wgt * x + sigma * np.random.randn(x.size)
            # Store as (Q, noisy_marginal, effective_sigma, clique)
            measurements.append((I, y / wgt, 1.0 / wgt, (col,)))
            selected.append((col,))

        # --- Round 2: all 2-way (col, target) marginals ---
        d2 = len(cliques)
        w2 = np.ones(d2) / np.sqrt(d2)

        for cl, wgt in zip(cliques, w2):
            x = data.project(cl).datavector()
            I = sparse.eye(x.size)
            y = wgt * x + sigma * np.random.randn(x.size)
            measurements.append((I, y / wgt, 1.0 / wgt, cl))
            selected.append(cl)

    else:
        # Pure DP (Laplace noise) – mirrors model.py's else branch.
        d = len(data.domain)
        sigma_1way = 1.0 / d / 2.0

        for col in data.domain.attrs:
            x = data.project([col]).datavector()
            I = sparse.eye(x.size)
            y = x + np.random.laplace(loc=0, scale=sigma_1way, size=x.size)
            measurements.append((I, y, sigma_1way, (col,)))
            selected.append((col,))

        d2 = len(cliques)
        sigma_2way = 1.0 / d2 / 2.0

        for cl in cliques:
            x = data.project(cl).datavector()
            I = sparse.eye(x.size)
            y = x + np.random.laplace(loc=0, scale=sigma_2way, size=x.size)
            measurements.append((I, y, sigma_2way, cl))
            selected.append(cl)

    engine = FactoredInference(data.domain, iters=num_iters)
    est = engine.estimate(measurements, total=n)
    synth = est.synthetic_data(rows=rows)

    # est.cliques gives the maximal cliques in the fitted graphical model
    # (typically the 2-way ones, since they subsume the 1-way).
    # We return `selected` instead so callers know *every* measured clique.
    return synth, selected


# ---------------------------------------------------------------------------
# Reprosyn-style Pipeline wrapper (mirrors the MST class in mst.py).
# ---------------------------------------------------------------------------

def domain_from_metadata(metadata):
    """Dict of {col_name: domain_size} from reprosyn metadata list."""
    return {col["name"]: len(col["representation"]) for col in metadata}


class PRIVATEPGM(PipelineBase):
    """Generator class wrapping the Private-PGM mechanism.

    Parameters
    ----------
    epsilon        : privacy budget
    delta          : delta parameter  (default 1e-9)
    target_variable: column treated as the classification target; all 2-way
                     marginals are (col, target_variable) pairs.
                     If None, defaults to the last column.
    cliques        : explicit list of 2-way clique tuples to measure.
                     If None, auto-constructed as all (col, target_variable).
    num_iters      : FactoredInference iterations  (default 1000)
    """

    generator = staticmethod(privatepgm)

    def __init__(self, epsilon=1.0, delta=1e-9,
                 target_variable=None, cliques=None, num_iters=1000, **kw):
        parameters = {
            "epsilon": epsilon,
            "delta": delta,
            "target_variable": target_variable,
            "cliques": cliques,
            "num_iters": num_iters,
        }
        super().__init__(**kw, **parameters)

    def preprocess(self):
        self.encoded_dataset, self.encoders = encode_ordinal(self.dataset)
        self.domain = domain_from_metadata(self.dataset.metadata)
        self.encoded_dataset = Dataset(
            self.encoded_dataset, Domain.fromdict(self.domain)
        )

    def generate(self):
        target = self.params["target_variable"]
        if target is None:
            target = list(self.domain.keys())[-1]

        cliques = self.params["cliques"]
        if cliques is None:
            # Match reference order: (col, target) preserving domain iteration order.
            # The reference uses [(col, target) for col in domain if col != target].
            # sorted() was removed because it changed the clique tuple ordering
            # vs. the reference, which doesn't sort.
            cliques = [
                (col, target)
                for col in self.domain
                if col != target
            ]

        self.output = self.generator(
            self.encoded_dataset,
            cliques,
            self.params["epsilon"],
            self.params["delta"],
            self.size,
            self.params["num_iters"],
        )
        return self.output

    def postprocess(self):
        synth_dataset, selected_cliques = self.output
        self.cliques = selected_cliques
        self.output = decode_ordinal(synth_dataset.df, self.encoders)
