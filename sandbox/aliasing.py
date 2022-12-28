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

With aliases, we can supply a mapping that associates additional names (the
aliases) with the same variable:

>>> class SIMAlias(AliasMixin, SIM):
...     ALIASES = {
...         'GDP': 'Y',  # Map 'GDP' to 'Y'
...     }

The variable 'Y' remains accessible as 'Y' (as before):

>>> model.Y[-1].round()
100.0

but also as 'GDP':

>>> model.GDP[-1].round()
100.0

This applies to all getters and setters, and, by extension, at instantiation
and inside the class itself e.g. in `_evaluate()`. The code also adds aliases
when calling `dir()` and for IPython completions.

Aliases provide a many-to-one mapping of names to model variables. However,
reversing the mapping creates problems e.g. in renaming variables to aliases in
DataFrames. To resolve this, the mixin adds a further attribute,
`PREFERRED_NAMES`. This is a list of names to use in any one-to-one naming
operation (currently just `to_dataframe()`).

Unresolved issues:

1. Is the implementation generalisable to `BaseModel` and `BaseLinker`?
2. If generalisable, where to put the mixin? `core`? a new `extensions` module
   or similar (what should be the structure of that module?)?
3. Whether/how to handle/resolve/catch duplicate names?
   e.g. `{'Y': 'Z'}`, where 'Y' is a model variable
4. What are the performance implications of adding an extra lookup each time to
   resolve the aliases?

Reference:

    Godley, W., Lavoie, M. (2007),
    *Monetary economics: an integrated approach to
    credit, money, income, production and wealth*,
    Palgrave Macmillan
"""

import copy
import itertools
import math
from typing import Any, Dict, Hashable, List, Sequence, Tuple, Union
import unittest

import numpy as np

from fsic.exceptions import InitialisationError
import fsic


pandas_installed = True

try:
    import pandas as pd
except ModuleNotFoundError:
    pandas_installed = False


class AliasMixin:
    """Mixin to handle variable aliases in models and linkers.

    Notes
    -----
    This mixin:
     - defines:
        - `ALIASES`, a class-level dict that maps aliases (str) to underlying
          model variable names (str)
        - `PREFERRED_NAMES`, a class-level list of str to specify which names
          to use when printing variable names e.g. as DataFrame columns, to
          resolve aliases for the same model variable
     - at instantiation:
        - (deep) copies `ALIASES` to an object-level attribute, `aliases`
        - resolves `PREFERRED_NAMES` to an object-level copy,
          `preferred_names`, condensing chained aliases e.g. X -> Y, Y -> Z
          becomes X -> Z, Y -> Z
     - wraps the following getters and setters to replace aliases with model
       variable names:
        - `__getattr__()`
        - `__setattr__()`
        - `__getitem__()`
        - `__setitem__()`
     - wraps the following, to add the aliases for tab completion etc:
        - `__dir__()`
        - `_ipython_key_completions_()`
     - wraps `to_dataframe()` to optionally use the aliases in DataFrame column
       titles, with `use_aliases=True`

    Set up this way, aliases are available externally and internally wherever
    you might've otherwise used the original variable name. This includes
    `_evaluate()`.
    """
    # Key-value pairs mapping aliases to variable names
    ALIASES: Dict[str, str] = {}

    # List of preferred names for tabling
    PREFERRED_NAMES: Sequence[str] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Make an instance-level copy of `ALIASES` and shorten any chained
        # aliases (e.g. X -> Y -> Z) to point directly to the model variable
        # (in this example, X -> Z, Y -> Z)
        aliases = copy.deepcopy(self.ALIASES)

        while True:
            # Check for chained aliases by testing to see if there are any
            # shared names between the keys and values. If so, there is at
            # least one link that can still be shortened
            # e.g. X -> Y and Y -> Z means that Y appears in both keys and
            # values - we want to eliminate this (by remapping X -> Z)
            if len(set(aliases.keys()) & set(aliases.values())) == 0:
                break

            # Update `aliases` by renaming dictionary values. Use the old value
            # as the key to try to find a new value
            # e.g. for X -> Y, get the mapping for Y (-> Z) and replace, to
            # leave X -> Z
            # Repeating the loop carries out successive substitution
            aliases = {k: aliases.get(v, v) for k, v in aliases.items()}

        # Remove any variables that point to themselves and then store
        aliases = {k: v for k, v in aliases.items() if k != v}
        self.__dict__['aliases'] = aliases

        # Check that preferred names point uniquely to model variables with no
        # overlap
        # TODO: Silently drop duplicates or just raise an exception (as
        #       implicitly, below)?
        preferred_names = copy.deepcopy(self.PREFERRED_NAMES)

        seen = []  # Store model variables referenced in/by `preferred_names`
        for name in preferred_names:
            # If no alias, assume it's a model variable (use `name`)
            target = aliases.get(name, name)

            if target in seen:
                raise ValueError(
                    f"Name '{name}' is a duplicate reference: "
                    f"one or more other names in `PREFERRED_NAMES` "
                    f"point to the same underlying model variable")

            seen.append(target)

        self.__dict__['preferred_names'] = preferred_names

        # Instantiate the object as usual, but replace aliases with actual
        # variable names as needed, using `_resolve_alias()`
        super().__init__(*args, **{self._resolve_alias(k): v for k, v in kwargs.items()})

        # TODO: Decide whether to check for aliases that are overwriting named
        #       model variables
        # TODO: Decide whether to check that all aliases point to a defined
        #       model variable

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

    def __dir__(self) -> List[str]:
        return sorted(super().__dir__() + list(self.aliases.keys()))

    def _ipython_key_completions_(self) -> List[str]:
        return super()._ipython_key_completions_() + list(self.aliases.keys())

    def to_dataframe(self, *, use_aliases: bool = False, **kwargs: Any) -> 'pandas.DataFrame':
        """Return the values and solution information from the model as a `pandas` DataFrame. If `use_aliases=True`, use any defined aliases as column titles. **Requires `pandas`**."""
        df = super().to_dataframe(**kwargs)

        # 1. No aliases to apply
        if not use_aliases:
            return df

        # 2. No preferred names given: Rename and return
        if len(self.preferred_names) == 0:
            return df.rename(columns={v: k for k, v in self.aliases.items()})

        # 3. If here, need to resolve preferred names

        replacements = {}  # Dictionary to store variables to rename

        # Sort by value, to group aliases together if they point to the same
        # underlying model variable
        sorted_by_value = sorted(self.aliases.items(), key=lambda x: x[1])

        # Loop through the groups
        for target, group_iter in itertools.groupby(sorted_by_value, key=lambda x: x[1]):
            # Extract aliases (`x[1]` just repeats `target`)
            aliases = [x[0] for x in group_iter]

            # a. Single alias: Nothing to resolve, so just store and continue
            if len(aliases) == 1:
                # If the model variable name is preferred, do nothing
                if target in self.preferred_names:
                    continue

                # Otherwise store the single preferred alias
                replacements[target] = aliases[0]
                continue

            # b. Multiple aliases: Need to check/resolve the duplicates
            # (Add `target` to `aliases` to check for duplicates in both model
            # names and aliases)
            intersection = set(aliases + [target]) & set(self.preferred_names)

            if len(intersection) == 0:
                # No preferred names in `aliases`: Skip
                continue
            elif len(intersection) == 1:
                # One preferred name: Store
                replacements[target] = list(intersection)[0]
            else:
                # Multiple instances: Error
                names = ', '.join(f"'{x}'" for x in sorted(intersection))
                raise ValueError(
                    f"Found multiple entries in `self.preferred_names` "
                    f"(which copies `PREFERRED_NAMES`, initially) "
                    f"pointing to '{target}' "
                    f"- a unique correspondence is required: {names}")

        return df.rename(columns=replacements)


class TestAliasMixin(unittest.TestCase):

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
            'expenditure': 'output',
            'output': 'income',
            'income': 'Y',
            'mpc_income': 'alpha_1',
            'mpc_wealth': 'alpha_2',
            'income_tax_rate': 'theta',
        }

        PREFERRED_NAMES = ['GDP', 'alpha_1', 'alpha_2']

    def test_alias_mapping(self):
        # Check that alias mappings are resolved to the lowest-level variable
        model = self.SIMAlias(range(10), alpha_1=0.6, mpc_wealth=0.4)

        self.assertEqual(
            model.aliases,
            {'GDP': 'Y',
             'expenditure': 'Y',
             'output': 'Y',
             'income': 'Y',
             'mpc_income': 'alpha_1',
             'mpc_wealth': 'alpha_2',
             'income_tax_rate': 'theta', })

    def test_init(self):
        # Check __init__() works with aliases
        model = self.SIMAlias(range(10), alpha_1=0.6, mpc_wealth=0.4)

        self.assertTrue(np.allclose(model.mpc_income, np.array([0.6] * 10)))
        self.assertTrue(np.allclose(model.alpha_2, np.array([0.4] * 10)))

    def test_get_set(self):
        # Check that get and set methods work with aliases
        model = self.SIMAlias(range(-5, 5 + 1))

        self.assertTrue(np.allclose(model.Y, np.array([0.0] * 11)))

        model.Y = 50
        self.assertTrue(np.allclose(model.Y, np.array([50.0] * 11)))

        model['GDP'] = 100
        self.assertTrue(np.allclose(model.Y, np.array([100.0] * 11)))

        model.expenditure[1:3] = 75
        self.assertTrue(np.allclose(model.output, np.array([100.0, 75.0, 75.0] + [100.0] * 8)))

        model['output', 0] = 125
        self.assertTrue(np.allclose(model['income'], np.array([100.0, 75.0, 75.0, 100.0, 100.0, 125.0] +
                                                              [100.0] * 5)))

        self.assertTrue(math.isclose(model['income', 0], 125))

    def test_dir(self):
        # Check that aliases appear in the list of methods and attributes
        model = self.SIMAlias(range(5))

        full_list = set(dir(model))
        expected_to_be_present = set(['GDP', 'Y', 'expenditure', 'output', 'income',
                                      'mpc_income', 'alpha_1', 'mpc_wealth', 'alpha_2', 'income_tax_rate',
                                      'theta'])

        # Check that the intersection of the sets leaves the expected variable
        # names
        self.assertEqual(full_list & expected_to_be_present, expected_to_be_present)

    def test_ipython_key_completions(self):
        # Check that aliases appear in the list of IPython method and attribute
        # completions
        model = self.SIMAlias(range(5))

        full_list = set(model._ipython_key_completions_())
        expected_to_be_present = set(['GDP', 'Y', 'expenditure', 'output', 'income',
                                      'mpc_income', 'alpha_1', 'mpc_wealth', 'alpha_2', 'income_tax_rate',
                                      'theta'])

        # Check that the intersection of the sets leaves the expected variable
        # names
        self.assertEqual(full_list & expected_to_be_present, expected_to_be_present)

    @unittest.skipIf(not pandas_installed, 'Requires `pandas`')
    def test_to_dataframe_default(self):
        # Check that `to_dataframe()` returns, by default, the same result as
        # the base version
        model = self.SIMAlias(range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2)
        model.solve()

        pd.testing.assert_frame_equal(model.to_dataframe(),
                                      super(self.SIMAlias, model).to_dataframe())

    @unittest.skipIf(not pandas_installed, 'Requires `pandas`')
    def test_to_dataframe_aliases(self):
        # Check that `to_dataframe()` correctly resolves and renames aliases
        model = self.SIMAlias(range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2)
        model.solve()

        result = model.to_dataframe(use_aliases=True)

        default_dataframe = super(self.SIMAlias, model).to_dataframe()
        self.assertNotEqual(list(result.columns), list(default_dataframe))
        pd.testing.assert_frame_equal(result,
                                      default_dataframe.set_axis(
                                          result.columns,
                                          axis='columns'))

        self.assertEqual(list(result.columns),
                         ['C', 'YD', 'H', 'GDP', 'T',
                          'G', 'alpha_1', 'alpha_2', 'income_tax_rate', 'status',
                          'iterations'])


if __name__ == '__main__':
    # Run Model *SIM*  as usual -----------------------------------------------
    model = TestAliasMixin.SIM(range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4)

    model.G = 20
    model.theta = 0.2

    model.solve()

    # Run the same model with aliases -----------------------------------------
    # Aliases are completely interchangeable with the original variable names
    model_with_aliases = TestAliasMixin.SIMAlias(
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
