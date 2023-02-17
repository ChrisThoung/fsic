# -*- coding: utf-8 -*-
"""
functions
=========
Operations (as functions) on NumPy arrays.
"""

from typing import Any

from numpy import exp, log
import numpy as np


__all__ = ['diff', 'exp', 'lag', 'lead', 'log', ]


def shift(x: np.ndarray, p: int, *, fill_value: Any = np.nan) -> np.ndarray:
    """Return `x` with elements shifted `p` places to the right."""
    if len(x.shape) != 1:
        raise NotImplementedError('`shift()` not currently implemented for non-1D arrays')

    # No shift: Just return `x`
    if p == 0:
        return x

    shifted = np.roll(x, shift=p)

    if p > 0:  # Lags
        shifted[:p] = fill_value

    elif p < 0:  # Leads
        shifted[p:] = fill_value

    return shifted

def lag(x: np.ndarray, p: int = 1, *, fill_value: Any = np.nan):
    """Return the `p`-th period lag of `x`."""
    return shift(x, p, fill_value=fill_value)

def lead(x: np.ndarray, p: int = 1, *, fill_value: Any = np.nan):
    """Return the `p`-th period lead of `x`."""
    return shift(x, -p, fill_value=fill_value)

def diff(x: np.ndarray, d: int = 1, *, fill_value: Any = np.nan) -> np.ndarray:
    """Return the `d`th difference of `x`: `x - x[-d]`."""
    if len(x.shape) != 1:
        raise NotImplementedError('`diff()` not currently implemented for non-1D arrays')

    # No differencing: Just return `x`
    if d == 0:
        return x

    elif d > 0:  # Lags
        differenced = x - lag(x, d, fill_value=fill_value)
        differenced[:d] = fill_value
        return differenced

    elif d < 0:  # Leads
        # TODO: Review
        raise NotImplementedError('`diff()` not currently(?) implemented for `d < 0` (leads)')
        differenced = x - lead(x, d, fill_value=fill_value)
        differenced[d:] = fill_value
        return differenced
