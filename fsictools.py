# -*- coding: utf-8 -*-
"""
fsictools
=========
Supporting tools for FSIC-based economic models. See the individual docstrings
for dependencies additional to those of `fsic`.
"""

# Version number keeps track with the main `fsic` module
from fsic import __version__

import re
from typing import List

from fsic import BaseModel, Symbol


def symbols_to_dataframe(symbols: List[Symbol]) -> 'pandas.DataFrame':
    """Convert the list of symbols to a `pandas` DataFrame. **Requires `pandas`**."""
    from pandas import DataFrame

    return DataFrame([s._asdict() for s in symbols])

def model_to_dataframe(model: BaseModel) -> 'pandas.DataFrame':
    """Return the values and solution information from the model as a `pandas` DataFrame. **Requires `pandas`**."""
    from pandas import DataFrame

    df = DataFrame(model.values.T, index=model.span, columns=model.names)
    df['status'] = model.status
    df['iterations'] = model.iterations

    return df
