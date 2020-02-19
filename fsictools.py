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
from typing import Dict, List

from fsic import BaseModel, Symbol
import fsic


def symbols_to_dataframe(symbols: List[Symbol]) -> 'pandas.DataFrame':
    """Convert the list of symbols to a `pandas` DataFrame. **Requires `pandas`**."""
    from pandas import DataFrame

    return DataFrame([s._asdict() for s in symbols])

def symbols_to_graph(symbols: List[Symbol]) -> 'networkx.DiGraph':
    """Convert the list of symbols to a NetworkX DiGraph. **Requires `networkx`."""
    import networkx as nx

    G = nx.DiGraph()

    equations = [s.equation for s in symbols if s.equation is not None]
    for e in equations:
        lhs, rhs = e.split('=', maxsplit=1)
        endogenous = [m.group(0) for m in fsic.term_re.finditer(lhs)]
        exogenous =  [m.group(0) for m in fsic.term_re.finditer(rhs)]

        # Add the equations as node properties
        G.add_nodes_from(endogenous, equation=e)

        # Add the edges
        for n in endogenous:
            for x in exogenous:
                G.add_edge(x, n)

    return G

def symbols_to_sympy(symbols: List[Symbol]) -> Dict['sympy.Symbol', 'sympy.Eq']:
    """Convert the system of equations into a dictionary of `SymPy` objects. **Requires `SymPy`**."""
    import sympy

    system = {}

    equations = [s.equation for s in symbols if s.equation is not None]
    for e in equations:
        e = e.replace('[t]', '')
        e = re.sub(r'\[t[-]([0-9]+)\]', r'_\1', e)

        lhs, rhs = map(sympy.sympify, e.split('=', maxsplit=1))
        system[lhs] = sympy.Eq(lhs, rhs)

    return system

def model_to_dataframe(model: BaseModel) -> 'pandas.DataFrame':
    """Return the values and solution information from the model as a `pandas` DataFrame. **Requires `pandas`**."""
    from pandas import DataFrame

    df = DataFrame(model.values.T, index=model.span, columns=model.names)
    df['status'] = model.status
    df['iterations'] = model.iterations

    return df
