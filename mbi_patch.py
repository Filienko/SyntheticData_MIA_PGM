"""Patch the local mbi/ repo (private-pgm style) to work with pandas 3.x.

Import this module ONCE before any other mbi usage.  Two things happen:

  1. sys.path is updated so the local mbi/ directory (private-pgm, numpy-
     based, with native FactoredInference) takes precedence over any pip-
     installed mbi package.

  2. GraphicalModel.synthetic_data is replaced with a pandas-3.x-safe
     implementation.  The original uses groupby().apply() which in pandas
     2.2+ (removed in 3.0) no longer passes grouped-by columns into the
     applied function, corrupting the output.

This file is idempotent: safe to import multiple times.
"""

import os
import sys

# ---------------------------------------------------------------------------
# 1. Ensure the local mbi/ package is found before the pip-installed one.
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
import mbi
import mbi.graphical_model as _gm
from mbi import Dataset


# ---------------------------------------------------------------------------
# 2. pandas-3.x-safe synthetic_data for GraphicalModel.
#
# Replaces groupby().apply() with explicit per-group sampling so that the
# grouped-by columns are always present in the output.
# ---------------------------------------------------------------------------

def _synthetic_data(model, rows=None):
    """Sample synthetic rows from a fitted GraphicalModel (pandas-3.x safe)."""
    domain = model.domain
    cols   = domain.attrs
    total  = int(model.total) if rows is None else rows
    total  = max(1, total)

    cliques   = [set(cl) for cl in model.cliques]
    order     = model.elimination_order[::-1]
    result    = {}   # col -> 1-D int ndarray of length `total`

    def _draw(counts, n):
        p = np.asarray(counts, dtype=float).ravel()
        p = np.clip(p, 0, None)
        s = p.sum()
        if s <= 0:
            p = np.ones_like(p)
            s = float(p.sum())

        # deterministic rounding + random fill for leftover
        frac, integ = np.modf(p * n / s)
        integ = integ.astype(int)
        extra = n - integ.sum()
        if extra > 0:
            idx = np.random.choice(p.size, extra, replace=False,
                                   p=frac / frac.sum())
            integ[idx] += 1
        vals = np.repeat(np.arange(p.size), integ)
        np.random.shuffle(vals)
        return vals

    for col in order:
        relevant  = [cl for cl in cliques if col in cl]
        cond_cols = tuple(
            c for c in list(set().union(*relevant))
            if c != col and c in result
        )

        if not cond_cols:
            marg = np.asarray(model.project([col]).datavector(flatten=False))
            result[col] = _draw(marg, total)
        else:
            joint     = np.asarray(
                model.project(list(cond_cols) + [col]).datavector(flatten=False)
            )
            sampled   = np.empty(total, dtype=int)
            cond_vals = np.stack([result[c] for c in cond_cols], axis=1)
            unique_keys, inverse = np.unique(cond_vals, axis=0, return_inverse=True)
            for ui, key in enumerate(unique_keys):
                mask  = inverse == ui
                idx   = tuple(key)
                sampled[mask] = _draw(joint[idx], int(mask.sum()))
            result[col] = sampled

    df = pd.DataFrame({c: result[c] for c in cols})
    return Dataset(df, domain)


if not getattr(_gm.GraphicalModel.synthetic_data, '_patched', False):
    _gm.GraphicalModel.synthetic_data = lambda self, rows=None: _synthetic_data(self, rows)
    _gm.GraphicalModel.synthetic_data._patched = True
