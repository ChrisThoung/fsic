# -*- coding: utf-8 -*-
"""
Supporting tools for fsic-based economic models. See the individual docstrings
for dependencies additional to those of `fsic`.
"""

import re
from typing import Any, Dict, Hashable, List

import numpy as np

from .parser import Symbol, Type, term_re


def symbols_to_dataframe(symbols: List[Symbol]) -> 'pandas.DataFrame':  # noqa: F821
    """Convert the list of symbols to a `pandas` DataFrame. **Requires `pandas`**.

    See also
    --------
    fsic.tools.dataframe_to_symbols()
    """
    from pandas import DataFrame

    return DataFrame([s._asdict() for s in symbols])


def dataframe_to_symbols(table: 'pandas.DataFrame') -> List[Symbol]:  # noqa: F821
    """Convert a `pandas` DataFrame to a list of symbols, reversing the operation of `symbols_to_dataframe()`. **Requires `pandas`**.

    See also
    --------
    fsic.tools.symbols_to_dataframe()
    """

    def convert_row_to_symbol(entry: Dict[str, Any]) -> Symbol:
        entry['type'] = Type(entry['type'])  # Convert to `enum`erated variable type
        entry['lags']  = None if np.isnan(x := entry['lags'])  else int(x)  # fmt: skip
        entry['leads'] = None if np.isnan(x := entry['leads']) else int(x)
        entry['equation'] = None if (x := entry['equation']) in [np.nan] else x
        entry['code']     = None if (x := entry['code'])     in [np.nan] else x  # fmt: skip

        return Symbol(**entry)

    return [convert_row_to_symbol(dict(row)) for _, row in table.iterrows()]


def symbols_to_graph(symbols: List[Symbol]) -> 'networkx.DiGraph':  # noqa: F821
    """Convert the list of symbols to a NetworkX DiGraph. **Requires `networkx`."""
    import networkx as nx

    G = nx.DiGraph()

    equations = [s.equation for s in symbols if s.equation is not None]
    for e in equations:
        lhs, rhs = e.split('=', maxsplit=1)
        endogenous = [m.group(0) for m in term_re.finditer(lhs)]
        exogenous  = [m.group(0) for m in term_re.finditer(rhs)]  # fmt: skip

        # Add the equations as node properties
        G.add_nodes_from(endogenous, equation=e)

        # Add the edges
        for n in endogenous:
            for x in exogenous:
                G.add_edge(x, n)

    return G


def symbols_to_sympy(
    symbols: List[Symbol],
) -> Dict['sympy.Symbol', 'sympy.Eq']:  # noqa: F821
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


def model_to_dataframe(
    model: 'BaseModel',  # noqa: F821
    *,
    status: bool = True,
    iterations: bool = True,
    include_internal: bool = False,
) -> 'pandas.DataFrame':  # noqa: F821
    """Return the values and solution information from the model as a `pandas` DataFrame (also available as `fsic.BaseModel.to_dataframe()` / `fsic.core.BaseModel.to_dataframe()`). **Requires `pandas`**.

    See also
    --------
    fsic.core.BaseModel.to_dataframe()
    """
    from pandas import DataFrame

    names = model.names
    if not include_internal:
        names = [x for x in model.names if not x.startswith('_')]

    # NB Take variables one at a time, rather than use `model.values`. This
    #    preserves the dtypes of the individual series.
    df = DataFrame({k: model[k] for k in names}, index=model.span)

    if status:
        df['status'] = model.status

    if iterations:
        df['iterations'] = model.iterations

    return df


def linker_to_dataframes(
    linker: 'BaseLinker',  # noqa: F821
    *,
    status: bool = True,
    iterations: bool = True,  # noqa: F821
    include_internal: bool = False,
) -> Dict[Hashable, 'pandas.DataFrame']:  # noqa: F821
    """Return the values and solution information from the linker and its constituent submodels as `pandas` DataFrames (also available as `fsic.BaseLinker.to_dataframes()` / `fsic.core.BaseLinker.to_dataframes()`). **Requires `pandas`**.

    See also
    --------
    fsic.core.BaseLinker.to_dataframes()
    """
    results = {
        linker.name: linker.to_dataframe(
            status=status, iterations=iterations, include_internal=include_internal
        )
    }

    for name, model in linker.submodels.items():
        results[name] = model.to_dataframe(
            status=status, iterations=iterations, include_internal=include_internal
        )

    return results
