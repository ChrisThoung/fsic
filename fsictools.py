# -*- coding: utf-8 -*-
"""
fsictools
=========
Supporting tools for FSIC-based economic models.
"""

# Version number keeps track with the main `fsic` module
from fsic import __version__


def to_dataframe(model):
    """Return the values and solution information from the model as a `pandas` DataFrame."""
    from pandas import DataFrame

    df = DataFrame(model.values.T, index=model.span, columns=model.names)
    df['status'] = model.status
    df['iterations'] = model.iterations

    return df
