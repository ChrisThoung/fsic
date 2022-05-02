# -*- coding: utf-8 -*-
"""
aliasing
========
Example / test case for implementing aliases for model variables.

Aliases let you attach multiple names to the same variable. For example, Godley
and Lavoie's (2007) Model *SIM* uses 'Y' to denote national income
(GDP). Without aliases, we can only refer to this variable as 'Y':

>>> model = SIM(range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2)
>>> model.solve()
>>> model.Y[-1].round()
100.0

With aliases, we can supply a mapping that associates other names to the same
variable:

>>> class SIMAlias(AliasMixin, SIM):
...     ALIASES = {
...         'GDP': 'Y',  # Map 'GDP' to 'Y'
...     }

Now, 'Y' is accessible as 'Y' (as before):
>>> model.Y[-1].round()
100.0

but also as 'GDP':
>>> model.GDP[-1].round()
100.0

This applies to all getters and setters, and, by extension, at instantiation
and inside the class itself e.g. in `_evaluate()`.

Unresolved issues:

1. Is the implementation generalisable to `BaseModel` and `BaseLinker`?
2. If generalisable, where to put the mixin? `core`? a new `extensions` module
   or similar?
3. How to handle/resolve/catch duplicate names?
   e.g. `{'Y': 'Z'}`, where 'Y' is a model variable
4. Should the mixin allow/resolve chained aliases (and, if so, how to deal with
   this in `to_dataframe()`, which has to apply a reverse mapping)?
   e.g. `{'GDP': 'Y', 'ABMI': 'GDP'}` rather than requiring
   `{'GDP': 'Y', 'ABMI': 'Y'}`
5. Related to [4], how should `to_dataframe()` resolve multiple aliases?
   e.g. `{'GDP': 'Y', 'ABMI': 'Y'}`
6. What are the performance implications of adding an extra lookup each time to
   resolve the aliases?

Reference:

    Godley, W., Lavoie, M. (2007),
    *Monetary economics: an integrated approach to
    credit, money, income, production and wealth*,
    Palgrave Macmillan
"""

import copy
from typing import Any, Dict, Hashable, Sequence, Tuple, Union

import numpy as np

import fsic


class AliasMixin:
    """Mixin to handle variable aliases in models and linkers.

    Notes
    -----
    This mixin:
     - defines `ALIASES`, a class-level dict that maps aliases (str) to
       underlying model variable names (str)
     - at instantiation, (deep) copies `ALIASES` to an object-level attribute,
       `aliases`
     - wraps the following getters and setters to replace aliases with model
       variable names:
        - `__getattr__()`
        - `__setattr__()`
        - `__getitem__()`
        - `__setitem__()`
     - wraps `to_dataframe()` to optionally use the aliases in DataFrame column
       titles, with `use_aliases=True`

    Set up this way, aliases are available externally and internally wherever
    you might've otherwise used the original variable name. This includes
    `_evaluate()`.
    """
    # Key-value pairs mapping aliases to variable names
    ALIASES: Dict[str, str] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.__dict__['aliases'] = copy.deepcopy(self.ALIASES)
        super().__init__(*args, **{self._resolve_alias(k): v for k, v in kwargs.items()})

    def _resolve_alias(self, alias: str) -> str:
        """Return the name of the underlying model variable associated with `alias`."""
        return self.aliases.get(alias, alias)

    def __getattr__(self, name: str) -> Any:
        return super().__getattr__(self._resolve_alias(name))

    def __setattr__(self, name: str, value: Union[Any, Sequence[Any]]) -> None:
        super().__setattr__(self._resolve_alias(name), value)

    def __getitem__(self, key: Union[str, Tuple[str, Union[Hashable, slice]]]) -> Any:
        if isinstance(key, tuple):
            name, *index = key
            key = tuple([self._resolve_alias(name)] + list(index))
        else:
            key = self._resolve_alias(key)

        return super().__getitem__(key)

    def __setitem__(self, key: Union[str, Tuple[str, Union[Hashable, slice]]], value: Union[Any, Sequence[Any]]) -> None:
        if isinstance(key, tuple):
            name, *index = key
            key = tuple([self._resolve_alias(name)] + list(index))
        else:
            key = self._resolve_alias(key)

        return super().__setitem__(key, value)

    def to_dataframe(self, *, use_aliases: bool = False, **kwargs: Any) -> 'pandas.DataFrame':
        """Return the values and solution information from the model as a `pandas` DataFrame. If `use_aliases=True`, use any defined aliases as column titles. **Requires `pandas`**."""
        df = super().to_dataframe(**kwargs)

        if use_aliases:
            df = df.rename(columns={v: k for k, v in self.aliases.items()})

        return df


script = '''
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
'''

symbols = fsic.parse_model(script)
SIM = fsic.build_model(symbols)


class SIMAlias(AliasMixin, SIM):
    ALIASES = {
        'GDP': 'Y',
        'mpc_income': 'alpha_1',
        'mpc_wealth': 'alpha_2',
        'income_tax_rate': 'theta',
    }


if __name__ == '__main__':
    # Run Model *SIM*  as usual -----------------------------------------------
    model = SIM(range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4)

    model.G = 20
    model.theta = 0.2

    model.solve()

    # Run the same model with aliases -----------------------------------------
    # Aliases are completely interchangeable with the original variable names
    model_with_aliases = SIMAlias(
        range(1945, 2010 + 1),
        alpha_1=0.6,     # Parameter as listed in Godley and Lavoie (2007)
        mpc_wealth=0.4,  # Alias for `alpha_2`
    )

    model_with_aliases.G = 20  # Original variable names work as usual
    model_with_aliases.income_tax_rate = 0.2  # Aliases also work

    model_with_aliases.solve()

    # Check (show) results match ----------------------------------------------
    assert np.allclose(model.values, model_with_aliases.values)

    # Print results as DataFrames, with `use_aliases=True` to use the aliases -
    print(model_with_aliases.to_dataframe().round(2))
    print(model_with_aliases.to_dataframe(use_aliases=True).round(2))
