"""Monkey-patch the pip-installed mbi (1.0.0) to work with JAX 0.9.x + pandas 2.x.

Import this module ONCE before any other mbi usage.  Three patches are applied:

  1. Factor._binaryop – the current implementation calls jnp.ndim(other)==0 to
     decide whether to wrap a scalar in a Factor.  Inside JAX JIT tracing that
     comparison returns a traced bool (not a Python bool), so the check is
     silently skipped and other.domain is accessed on a DynamicJaxprTracer.
     Fix: use hasattr(other, 'domain') instead of the ndim check.

  2. FactoredInference – the class was removed in mbi 1.0.0.  A shim that
     wraps estimation.mirror_descent is injected so that old mst.py /
     privatepgm.py code continues to work unchanged.

  3. MarkovRandomField.synthetic_data – uses groupby().apply() in a way that
     pandas 2.x drops grouped-by columns.  A corrected sampler is injected.

This file is idempotent: safe to import multiple times.
"""

import numpy as np
import pandas as pd
import jax
import mbi
import mbi.factor as _factor
import mbi.estimation as _est
import mbi.junction_tree as _jt
import mbi.markov_random_field as _mrf
from mbi import LinearMeasurement, Dataset, Domain

# ---------------------------------------------------------------------------
# Patch 1: Factor._binaryop – fix scalar wrapping under JIT tracing.
#
# The original uses `isinstance(other, chex.Numeric) and jnp.ndim(other)==0`
# to detect scalars.  Inside JAX JIT tracing, ndim(other)==0 produces a
# traced bool (not a Python bool), so the condition silently evaluates False
# and we fall through to `other.domain` on a DynamicJaxprTracer → crash.
# Fix: use hasattr(other, 'domain') – always a reliable Python bool.
# ---------------------------------------------------------------------------
def _binaryop_fixed(self, fn, other):
    if not hasattr(other, 'domain'):
        other = _factor.Factor(Domain([], []), other)
    newdom = self.domain.merge(other.domain)
    factor1 = self.expand(newdom)
    factor2 = other.expand(newdom)
    return _factor.Factor(newdom, fn(factor1.values, factor2.values))

_factor.Factor._binaryop = _binaryop_fixed

# ---------------------------------------------------------------------------
# Corrected synthetic_data
# ---------------------------------------------------------------------------

def _synthetic_data(model, rows):
    """Sample synthetic rows from a fitted MarkovRandomField.

    Replaces the built-in implementation whose groupby().apply() drops
    grouped-by columns under pandas 2.x.
    """
    domain = model.domain
    cols   = domain.attrs
    total  = max(1, int(rows or model.total))

    cliques      = [set(cl) for cl in model.cliques]
    _, elim_order = _jt.make_junction_tree(domain, cliques)
    order = elim_order[::-1]

    result = {}  # col -> 1-D int ndarray of length `total`

    def _draw(probs, n):
        p = np.asarray(probs, dtype=float).ravel()
        p = np.clip(p, 0, None)
        s = p.sum()
        if s <= 0:
            p = np.ones_like(p)
            s = p.sum()
        return np.random.choice(p.size, n, replace=True, p=p / s)

    for col in order:
        relevant  = [cl for cl in cliques if col in cl]
        cond_cols = tuple(
            c for c in list(set().union(*relevant))
            if c != col and c in result
        )

        if not cond_cols:
            marg = np.asarray(model.project((col,)).datavector(flatten=False))
            result[col] = _draw(marg, total)
        else:
            joint     = np.asarray(
                model.project(cond_cols + (col,)).datavector(flatten=False)
            )
            sampled   = np.empty(total, dtype=int)
            cond_vals = np.stack([result[c] for c in cond_cols], axis=1)
            unique_keys, inverse = np.unique(cond_vals, axis=0, return_inverse=True)
            for ui, key in enumerate(unique_keys):
                mask = inverse == ui
                idx  = tuple(key)
                sampled[mask] = _draw(joint[idx], mask.sum())
            result[col] = sampled

    df = pd.DataFrame({c: result[c] for c in cols})
    return Dataset(df, domain)


# ---------------------------------------------------------------------------
# FactoredInference shim
# ---------------------------------------------------------------------------

class _EstimateResult:
    def __init__(self, model):
        self._model = model

    @property
    def cliques(self):
        return list(self._model.cliques)

    def synthetic_data(self, rows=None):
        return _synthetic_data(self._model, rows)


class FactoredInference:
    """Drop-in for the old private-pgm FactoredInference class.

    Accepts the old measurement format:
        (Q, y, sigma, proj)  – Q (sparse identity) is ignored.
    Also accepts LinearMeasurement objects directly.
    """

    def __init__(self, domain, iters=1000, log=False, **kw):
        self.domain = domain
        self.iters  = iters

    def estimate(self, measurements, total=None):
        new_meas = []
        for m in measurements:
            if isinstance(m, LinearMeasurement):
                new_meas.append(m)
            else:
                _, y, sigma, proj = m
                if not isinstance(proj, tuple):
                    proj = tuple(proj)
                new_meas.append(LinearMeasurement(
                    np.asarray(y, dtype=float), proj, stddev=float(sigma)
                ))

        model = _est.mirror_descent(
            self.domain, new_meas,
            known_total=float(total) if total is not None else None,
            iters=self.iters,
        )
        return _EstimateResult(model)


# ---------------------------------------------------------------------------
# Patch 3: Replace synthetic_data on MarkovRandomField (idempotent).
# ---------------------------------------------------------------------------
_mrf.MarkovRandomField.synthetic_data = lambda self, rows=None: _synthetic_data(self, rows)

# ---------------------------------------------------------------------------
# Inject FactoredInference into the pip mbi namespace (idempotent).
# ---------------------------------------------------------------------------
if not hasattr(mbi, "FactoredInference"):
    mbi.FactoredInference = FactoredInference
