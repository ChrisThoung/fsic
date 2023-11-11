# -*- coding: utf-8 -*-
"""
Mixins compatible with both model and linker classes.
"""

import copy
import itertools
from typing import Any, Dict, Hashable, List, Sequence, Tuple, Union


class AliasMixin:
    """Mixin to add support for aliases for variable names in models and linkers.

    Aliases support many-to-one mappings of alternative names to underlying
    model names e.g. to be able to refer to both the original variable 'Y' but
    also, as needed, 'GDP'. These are available anywhere the original variable
    name is valid:
     - initialisation e.g. model = SomeModel(..., GDP=...), where GDP is an
       alias for, say, Y
     - getters and setters e.g. model.GDP, model['GDP', ...]
     - solution code e.g. self.GDP[t] = ...
     - `dir()` and IPython completions

    The mixin also adds support for aliases in `to_dataframe()`.

    Examples
    --------
    Starting with a typical model definition without aliases:

        from fsic import BaseModel

        class ExampleModel(BaseModel):
            ENDOGENOUS = ['A', 'B', 'C']
            EXOGENOUS = ['X', 'Y', 'Z']

            NAMES = ENDOGENOUS + EXOGENOUS

            def _evaluate(self, t, *args, **kwargs):
                self.A[t] = self.X[t] + 1
                self.B[t] = self.Y[t] + 2
                self.C[t] = self.Z[t] + 3

    Variable access works as usual:

    >>> model = ExampleModel(range(10))
    >>> model.A[::2] = 1
    >>> model.A
    [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.]

    Aliases preserve the above behaviour while allowing alternative names for
    the variables. This uses the `ALIASES` attribute of `AliasMixin`:

        from fsic.extensions import AliasMixin

        class ExampleModelWithAliases(AliasMixin, ExampleModel):
            ALIASES = {
                'I': 'A',  # Map 'I' as an alias for 'A'
                'J': 'A',  # Map 'J' as an alias for 'A'
                'K': 'Y',  # Map 'K' as an alias for 'Y'
                'L': 'K',  # Map 'L' to 'K', leading eventually to 'Y'
            }

    As with 'I' and 'J' above, aliases can point to the same underlying model
    variable. As with 'L', aliases can also point to other aliases.

    Now, the underlying variables are accessible by both their original names
    and their aliases:

    >>> model = ExampleModelWithAliases(range(10))

    # A is accessible as usual
    >>> model.A[::2] = 1
    >>> model.A
    [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.]

    # A is also accessible through I
    >>> model.I
    [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.]

    >>> model.I[1::2] = 2
    >>> model['I']  # Access by key (equivalent to access by attribute)
    [1. 2. 1. 2. 1. 2. 1. 2. 1. 2.]

    # By default, `to_dataframe()` works as usual, using the model's original
    # variable names
    >>> model.to_dataframe()
         A    B    C    X    Y    Z status  iterations
    0  1.0  0.0  0.0  0.0  0.0  0.0      -          -1
    1  2.0  0.0  0.0  0.0  0.0  0.0      -          -1
    2  1.0  0.0  0.0  0.0  0.0  0.0      -          -1
    3  2.0  0.0  0.0  0.0  0.0  0.0      -          -1
    4  1.0  0.0  0.0  0.0  0.0  0.0      -          -1
    5  2.0  0.0  0.0  0.0  0.0  0.0      -          -1
    6  1.0  0.0  0.0  0.0  0.0  0.0      -          -1
    7  2.0  0.0  0.0  0.0  0.0  0.0      -          -1
    8  1.0  0.0  0.0  0.0  0.0  0.0      -          -1
    9  2.0  0.0  0.0  0.0  0.0  0.0      -          -1

    # To use aliases, pass `use_aliases=True`
    >>> model.to_dataframe(use_aliases=True)
         J    B    C    X    L    Z status  iterations
    0  1.0  0.0  0.0  0.0  0.0  0.0      -          -1
    1  2.0  0.0  0.0  0.0  0.0  0.0      -          -1
    2  1.0  0.0  0.0  0.0  0.0  0.0      -          -1
    3  2.0  0.0  0.0  0.0  0.0  0.0      -          -1
    4  1.0  0.0  0.0  0.0  0.0  0.0      -          -1
    5  2.0  0.0  0.0  0.0  0.0  0.0      -          -1
    6  1.0  0.0  0.0  0.0  0.0  0.0      -          -1
    7  2.0  0.0  0.0  0.0  0.0  0.0      -          -1
    8  1.0  0.0  0.0  0.0  0.0  0.0      -          -1
    9  2.0  0.0  0.0  0.0  0.0  0.0      -          -1

    How many-to-one mappings are resolved depends on the Python version and the
    nature of its dictionary implementation. More recent versions of Python
    (3.7 onwards and, in implementation, CPython 3.6) will follow the order
    defined in `ALIASES` while older versions will generate an arbitrary order
    and resolution.

    To unambiguously resolve one-to-many reverse aliases, use the
    `PREFERRED_NAMES` attribute to state any preferred DataFrame column names,
    whether aliases or original model variables:

        class ExampleModelWithAliases2(AliasMixin, ExampleModel):
            ALIASES = {
                'I': 'A',  # Map 'I' as an alias for 'A'
                'J': 'A',  # Map 'J' as an alias for 'A'
                'K': 'Y',  # Map 'K' as an alias for 'Y'
                'L': 'K',  # Map 'L' to 'K', leading eventually to 'Y'
                'M': 'B',  # Map 'M' to 'B'
            }

            # Prefer the original 'A' to any aliases and, for 'Y', 'K' over 'L'
            # If no preference is given (e.g. for 'B'), assume the alias is
            # preferred and resolve as usual if `use_aliases=True`
            PREFERRED_NAMES = ['A', 'K']

    >>> model = ExampleModelWithAliases2(range(10))
    >>> model.to_dataframe(use_aliases=True)
         A    M    C    X    K    Z status  iterations
    0  0.0  0.0  0.0  0.0  0.0  0.0      -          -1
    1  0.0  0.0  0.0  0.0  0.0  0.0      -          -1
    2  0.0  0.0  0.0  0.0  0.0  0.0      -          -1
    3  0.0  0.0  0.0  0.0  0.0  0.0      -          -1
    4  0.0  0.0  0.0  0.0  0.0  0.0      -          -1
    5  0.0  0.0  0.0  0.0  0.0  0.0      -          -1
    6  0.0  0.0  0.0  0.0  0.0  0.0      -          -1
    7  0.0  0.0  0.0  0.0  0.0  0.0      -          -1
    8  0.0  0.0  0.0  0.0  0.0  0.0      -          -1
    9  0.0  0.0  0.0  0.0  0.0  0.0      -          -1

    Notes
    -----
    This mixin:
     - defines:
        - `ALIASES`, a class-level dict that maps aliases (str) to underlying
          model variable names (str)
        - `PREFERRED_NAMES`, a class-level list of str to specify which names
          to use when printing variable names e.g. for DataFrame columns, to
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
       titles, with `use_aliases=True` (resolving with `preferred_names` as
       needed)

    In this way, aliases are available externally and internally wherever the
    original variable name is valid. This includes `_evaluate()`.
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
                    f'one or more other names in `PREFERRED_NAMES` '
                    f'point to the same underlying model variable'
                )

            seen.append(target)

        self.__dict__['preferred_names'] = preferred_names

        # Instantiate the object as usual, but replace aliases with actual
        # variable names as needed, using `_resolve_alias()`
        super().__init__(
            *args, **{self._resolve_alias(k): v for k, v in kwargs.items()}
        )

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

    def __setitem__(
        self,
        key: Union[str, Tuple[str, Union[Hashable, slice]]],
        value: Union[Any, Sequence[Any]],
    ) -> None:
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

    def to_dataframe(
        self, *, use_aliases: bool = False, **kwargs: Any
    ) -> 'pandas.DataFrame':  # noqa: F821
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
        for target, group_iter in itertools.groupby(
            sorted_by_value, key=lambda x: x[1]
        ):
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
                    f'Found multiple entries in `self.preferred_names` '
                    f'(which copies `PREFERRED_NAMES`, initially) '
                    f"pointing to '{target}' "
                    f'- a unique correspondence is required: {names}'
                )

        return df.rename(columns=replacements)


class ProgressBarMixin:
    """Mixin to add an (optional) `tqdm` progress bar during solution. **Requires `tqdm`**

    Examples
    --------
    Add the mixin to the model class definition:

        from fsic import BaseModel
        from fsic.extensions import ProgressBarMixin

        class ExampleModel(ProgressBarMixin, BaseModel):
            ...

    By default, behaviour is unchanged:

    >>> model = ExampleModel(range(10))
    >>> model.solve()

    Passing `progress_bar=True` will print a `tqdm` progress bar to the screen:
    >>> model.solve(progress_bar=True)
    """

    def iter_periods(self, *args, progress_bar: bool = False, **kwargs):
        """Modified `iter_periods()` method: Display a `tqdm` progress bar if `progress_bar=True`. **Requires `tqdm`**"""
        # Get the original `PeriodIter` object
        period_iter = super().iter_periods(*args, **kwargs)

        # Optionally wrap with `tqdm`
        if progress_bar:
            from tqdm import tqdm

            period_iter = tqdm(period_iter)

        return period_iter
