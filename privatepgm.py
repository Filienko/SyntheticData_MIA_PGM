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

# Generator base from reprosyn (gives us encode_ordinal / decode_ordinal).
sys.path.insert(1, 'reprosyn-main/src/reprosyn/')
from generator import PipelineBase, encode_ordinal, decode_ordinal

from scipy import optimize, sparse
import numpy as np
import sys

from utils.rdp_accountant import compute_rdp, get_privacy_spent
from mbi import Dataset, FactoredInference, Domain


class Private_PGM:
    def __init__(self, target_variable, enable_privacy, target_epsilon, target_delta):
        self.target_epsilon = target_epsilon
        self.enable_privacy = enable_privacy
        self.target_delta = target_delta
        self.target_variable = target_variable
        self.model = None

    @staticmethod
    def moments_calibration(round1, round2, eps, delta):

        orders = range(2, 4096)

        def obj(sigma):
            rdp1 = compute_rdp(1.0, sigma / round1, 1, orders)
            rdp2 = compute_rdp(1.0, sigma / round2, 1, orders)
            rdp = rdp1 + rdp2
            privacy = get_privacy_spent(orders, rdp, delta=delta)
            return privacy[0] - eps + 1e-8

        low = 1.0
        high = 1.0
        while obj(low) < 0:
            low /= 2.0
        while obj(high) > 0:
            high *= 2.0
        sigma = optimize.bisect(obj, low, high)
        assert (
            obj(sigma) - 1e-8 <= 0
        ), "not differentially private"  # true eps <= requested eps
        return sigma

    def train(self, train_df, config, cliques=None, num_iters=10000):
        domain = Domain(config.keys(), config.values())
        data = Dataset(train_df, domain)
        total = data.df.shape[0]

        if self.enable_privacy:
            if self.target_delta > 0:
                sigma = self.moments_calibration(
                    1.0, 1.0, self.target_epsilon, self.target_delta
                )
            else:
                sigma = 1.0 / len(data.domain) / 2.0
        else:
            sigma = 0.0
        print("=" * 100)
        print("sigma:", sigma)

        weights = np.ones(len(data.domain))
        weights /= np.linalg.norm(weights)  # now has L2 norm = 1

        measurements = []
        for col, wgt in zip(data.domain, weights):
            x = data.project(col).datavector()
            I = sparse.eye(x.size)
            if self.target_delta > 0:
                y = wgt * x + sigma * np.random.randn(x.size)
                measurements.append((I, y / wgt, 1.0 / wgt, (col,)))
            else:
                y = x + np.random.laplace(loc=0, scale=sigma, size=x.size)
                measurements.append((I, y, sigma, (col,)))

        # spend half of privacy budget to measure 2 way marginals with the target variable
        if cliques is None:
            cliques = []
            for col in data.domain:
                if col != self.target_variable:
                    cliques.append((col, self.target_variable))

        weights = np.ones(len(cliques))
        weights /= np.linalg.norm(weights)  # now has L2 norm = 1

        if self.target_delta == 0:
            sigma = 1.0 / len(cliques) / 2.0

        for cl, wgt in zip(cliques, weights):
            x = data.project(cl).datavector()
            I = sparse.eye(x.size)
            if self.target_delta > 0:
                y = wgt * x + sigma * np.random.randn(x.size)
                measurements.append((I, y / wgt, 1.0 / wgt, cl))
            else:
                y = x + np.random.laplace(loc=0, scale=sigma, size=x.size)
                measurements.append((I, y, sigma, cl))

        engine = FactoredInference(domain, log=True, iters=num_iters)
        self.model = engine.estimate(measurements, total=total, engine="MD")

    def generate(self, num_rows=None):
        syn_df = self.model.synthetic_data(rows=num_rows).df
        X_syn = syn_df.drop([self.target_variable], axis=1).values
        y_syn = syn_df[self.target_variable].values
        return np.concatenate([X_syn, np.expand_dims(y_syn, axis=1)], axis=1)

    def postprocess(self):
        synth_dataset, selected_cliques = self.output
        self.cliques = selected_cliques
        self.output = decode_ordinal(synth_dataset.df, self.encoders)
