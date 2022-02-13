# -*- coding: utf-8 -*-
"""
tools
=====
Supporting tools for FSIC-based economic models. See the individual docstrings
for dependencies additional to those of `fsic`.
"""

import re
from typing import Any, Dict, Hashable, List

from .parser import Symbol, Type
from .parser import term_re


def symbols_to_dataframe(symbols: List[Symbol]) -> 'pandas.DataFrame':
    """Convert the list of symbols to a `pandas` DataFrame. **Requires `pandas`**.

    See also
    --------
    fsic.tools.dataframe_to_symbols()
    """
    from pandas import DataFrame

    return DataFrame([s._asdict() for s in symbols])

def dataframe_to_symbols(table: 'pandas.DataFrame') -> List[Symbol]:
    """Convert a `pandas` DataFrame to a list of symbols, reversing the operation of `symbols_to_dataframe()`. **Requires `pandas`**.

    See also
    --------
    fsic.tools.symbols_to_dataframe()
    """
    symbols = []

    for _, row in table.iterrows():
        entry = dict(row)
        entry['type'] = Type(entry['type'])  # Convert to `enum`erated variable type
        symbols.append(Symbol(**entry))

    return symbols

def symbols_to_graph(symbols: List[Symbol]) -> 'networkx.DiGraph':
    """Convert the list of symbols to a NetworkX DiGraph. **Requires `networkx`."""
    import networkx as nx

    G = nx.DiGraph()

    equations = [s.equation for s in symbols if s.equation is not None]
    for e in equations:
        lhs, rhs = e.split('=', maxsplit=1)
        endogenous = [m.group(0) for m in term_re.finditer(lhs)]
        exogenous =  [m.group(0) for m in term_re.finditer(rhs)]

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
    from sympy.core.numbers import ImaginaryUnit
    from sympy.core.singleton import SingletonRegistry

    def convert(expression: str) -> Any:
        """Convert `expression` to a SymPy object."""
        # Initial conversion
        converted = sympy.sympify(expression)

        # Special treatment for:
        #  - 'I' -> ImaginaryUnit
        #  - 'S' -> SingletonRegistry
        # Need to force these to be SymPy Symbols
        if isinstance(converted, (ImaginaryUnit, SingletonRegistry)):
            converted = sympy.Symbol(expression)

        return converted

    system = {}

    equations = [s.equation for s in symbols if s.equation is not None]
    for e in equations:
        # Remove time index and append lag number if needed
        e = e.replace('[t]', '')
        e = re.sub(r'\[t[-]([0-9]+)\]', r'_\1', e)

        # Convert and store
        lhs, rhs = map(convert, map(str.strip, e.split('=', maxsplit=1)))
        system[lhs] = sympy.Eq(lhs, rhs)

    return system

def model_to_dataframe(model: 'BaseModel', *, status: bool = True, iterations: bool = True) -> 'pandas.DataFrame':
    """Return the values and solution information from the model as a `pandas` DataFrame (also available as `fsic.BaseModel.to_dataframe()` / `fsic.core.BaseModel.to_dataframe()`). **Requires `pandas`**.

    See also
    --------
    fsic.core.BaseModel.to_dataframe()
    """
    from pandas import DataFrame

    df = DataFrame({k: model[k] for k in model.names}, index=model.span)

    if status:
        df['status'] = model.status

    if iterations:
        df['iterations'] = model.iterations

    return df

def linker_to_dataframes(linker: 'BaseLinker', *, status: bool = True, iterations: bool = True) -> Dict[Hashable, 'pandas.DataFrame']:
    """Return the values and solution information from the linker and its constituent submodels as `pandas` DataFrames (also available as `fsic.BaseLinker.to_dataframes()` / `fsic.core.BaseLinker.to_dataframes()`). **Requires `pandas`**.

    See also
    --------
    fsic.core.BaseLinker.to_dataframes()
    """
    from pandas import DataFrame

    results = {linker.name: model_to_dataframe(linker, status=status, iterations=iterations)}

    for name, model in linker.submodels.items():
        results[name] = model_to_dataframe(model, status=status, iterations=iterations)

    return results
