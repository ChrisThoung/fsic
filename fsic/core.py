# -*- coding: utf-8 -*-
"""
Core `fsic` classes for defining and solving economic models. Economic models
are implemented as derived classes of those in this module, inheriting the
necessary attributes and methods that make up the API.

Base classes:

* `BaseModel`, for a single economic model
* `BaseLinker`, to store and solve multiple model instances i.e. as a
  multi-region/entity model
"""

import copy
import difflib
import enum
import re
import warnings
from collections import Counter
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .exceptions import (
    DimensionError,
    DuplicateNameError,
    InitialisationError,
    NonConvergenceError,
    SolutionError,
)
from .functions import builtins as _builtins
from .tools import linker_to_dataframes as _linker_to_dataframes
from .tools import model_to_dataframe as _model_to_dataframe

# Labelled container for vector data (1D NumPy arrays) ------------------------


class VectorContainer:
    """Labelled container for vector data (1D NumPy arrays).

    Examples
    --------
    # Initialise an object with a span that attaches labels to the vector
    # elements; here: 11-element vectors labelled [2000, 2001, ..., 2009, 2010]
    >>> container = VectorContainer(range(2000, 2010 + 1))

    # Add some data, each with a fixed dtype (as in NumPy/pandas)
    # `VectorContainer`s automatically broadcast scalar values to vectors of
    # the previously specified length
    >>> container.add_variable('A', 2)                                        # Integer
    >>> container.add_variable('B', 3.0)                                      # Float
    >>> container.add_variable('C', list(range(10 + 1)), dtype=float)         # Force as floats
    >>> container.add_variable('D', [0, 1., 2, 3., 4, 5., 6, 7., 8, 9., 10])  # Cast to float

    # View the data, whether by attribute or key
    >>> container.A                                      # By attribute
    array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

    >>> container['B']                                   # By key
    array([3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])

    >>> container.values  # Pack into a 2D array (casting to a common dtype)
    array([[ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.],
           [ 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.],
           [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.],
           [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]])

    # Vector replacement preserves both length and dtype, making it more
    # convenient than the NumPy equivalent: `container.B[:] = 4`
    >>> container.B = 4  # Standard object behaviour would assign `4` to `B`...
    >>> container.B      # ...but `VectorContainer`s broadcast and convert automatically
    array([4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.])  # Still a vector of floats

    # NumPy-style indexing, whether by attribute or key (interchangeably)
    >>> container.C[2:4]  # By attribute
    array([2., 3.])

    >>> container.B[2:4] = 5  # As above, assignment preserves original dtype
    >>> container.B
    array([4., 4., 5., 5., 4., 4., 4., 4., 4., 4., 4.])

    >>> container['D'][3:-2]  # By key
    array([3., 4., 5., 6., 7., 8.])

    >>> container['C'][5:] = 9
    >>> container.C  # `container['C']` also works
    array([0., 1., 2., 3., 4., 9., 9., 9., 9., 9., 9.])

    # pandas-like indexing using the span labels (must be by key, as a 2-tuple)
    >>> container['D', 2005:2009]  # Second element (slice) refers to the labels of the object's span
    array([5., 6., 7., 8., 9.])

    >>> container['D', 2000:2008:2] = 12
    >>> container['D']  # As previously, `container.D` also works
    array([12.,  1., 12.,  3., 12.,  5., 12.,  7., 12.,  9., 10.])
    """

    _VALID_INDEX_METHODS: List[str] = ['get_loc', 'index']

    def __init__(self, span: Sequence[Hashable], *, strict: bool = False) -> None:
        """Initialise the container with a defined and labelled `span`.

        Parameter
        ---------
        span : iterable
            Sequence of labels that defines the span of the object's variables
        strict : bool
            If `True`, the only way to add attributes to the object is with
            `add_variable()` i.e. as new container variables. Ad hoc attributes
            are expressly blocked.
            If `False`, further attributes can be added ad hoc at runtime in
            the usual way for Python objects e.g. `model.A = ...`.
        """
        # At initialisation, manually add core attributes to the object...
        self.__dict__['span'] = span
        self.__dict__['index'] = []
        self.__dict__['_strict'] = strict

        # ...alongside their accompanying entries in the attributes list
        self.__dict__['_attributes'] = ['_attributes', 'span', 'index', '_strict']

    def add_attribute(self, name: str, value: Any) -> None:
        """Add an attribute to the container."""
        if name in self.__dict__['index']:
            raise DuplicateNameError(
                f"Variable with name '{name}' already defined in current object"
            )

        if name in self.__dict__['_attributes']:
            raise DuplicateNameError(
                f"Attribute with name '{name}' already defined in current object"
            )

        super().__setattr__(name, value)
        self.__dict__['_attributes'].append(name)

    def add_variable(
        self, name: str, value: Union[Any, Sequence[Any]], *, dtype: Any = None
    ) -> None:
        """Initialise a new variable in the container, forcing a `dtype` if needed.

        Parameters
        ----------
        name : str
            The name of the new variable, which must not already exist in the
            container (raise a `DuplicateNameError` if it already exists)
        value : single value or sequence of values
            Initial value(s) for the new variable. If a sequence, must yield a
            length equal to that of `span`
        dtype : variable type
            Data type to impose on the variable. The default is to use the
            type/dtype of `value`
        """
        if name in self.__dict__['index']:
            raise DuplicateNameError(
                f"'{name}' is already defined in the current object"
            )

        # Cast to a 1D array
        if isinstance(value, Sequence) and not isinstance(value, str):
            value_as_array = np.array(value).flatten()
        else:
            value_as_array = np.full(len(self.__dict__['span']), value)

        # Impose `dtype` if needed
        if dtype is not None:
            value_as_array = value_as_array.astype(dtype)

        # Check dimensions
        if value_as_array.shape[0] != len(self.__dict__['span']):
            raise DimensionError(
                f"Invalid assignment for '{name}': "
                f"must be either a single value or "
                f"a sequence of identical length to `span`"
                f"(expected {len(self.__dict__['span'])} elements)"
            )

        self.__dict__['_' + name] = value_as_array
        self.__dict__['index'].append(name)

    @property
    def size(self) -> int:
        """Number of elements in the object's vector arrays."""
        return len(self.__dict__['index']) * len(self.__dict__['span'])

    @property
    def nbytes(self) -> int:
        """Total bytes consumed by the elements in the object's vector arrays."""
        return sum(self[k].nbytes for k in self.__dict__['index'])

    def get_closest_match(
        self,
        name: str,
        *,
        possibilities: Optional[Sequence[str]] = None,
        cutoff: Union[int, float] = 0.1,
    ) -> List[str]:
        """Return the closest match(es) to `name` among `possibilities` (by default, `self.index`).

        Parameters
        ----------
        name : str
            Variable name to attempt to match
        possibilities : sequence of str, default `None`
            List of names to try to match to. If `None`, use `self.index`.
        cutoff : numeric, default 0.1
            Cutoff score (to pass to `difflib.get_close_matches()`) above which
            to return matches

        Returns
        -------
        List[str] :
            Closest match(es) to `name` in `possibilities` (multiple matches
            arise if `possibilities` has duplicate names, once converted to
            lower case)
            If no suitable match is found, the result is an empty list
        """
        # Set default list if none provided
        if possibilities is None:
            possibilities = self.index

        # Form lookup of lowercase versions of the names
        candidates = {}
        for x in possibilities:
            candidates[x.lower()] = candidates.get(x.lower(), []) + [x]

        # Match using `difflib`
        closest_match = difflib.get_close_matches(
            name.lower(), candidates.keys(), n=1, cutoff=cutoff
        )

        # If a match has been found, return the candidate(s) from the original
        # (non-lowercase) listing
        if len(closest_match):
            return candidates[closest_match[0]]

        # Otherwise, no suitable match found
        return []

    def __getattr__(self, name: str) -> Any:
        # TODO: Update to handle `_attributes`, too?
        if name in self.__dict__['index']:
            return self.__dict__['_' + name]

        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Union[Any, Sequence[Any]]) -> None:
        # Error on attempt to add an attribute if `strict=True`
        if (
            name != 'strict'
            and self.__dict__['_strict']
            and name not in self.__dict__['index']
            and name not in self.__dict__['_attributes']  # TODO: Check inclusion here
        ):
            message = f"Unable to add new attribute '{name}' with `strict=True`"

            # Try to find a possible alternative, if `name` is a typo
            alternatives = self.get_closest_match(name)

            if len(alternatives) == 1:
                message += f". Did you mean: '{alternatives[0]}'?"
            elif len(alternatives) > 1:
                raise NotImplementedError(
                    'Handling of multiple name matches not yet implemented'
                )

            raise AttributeError(message)

        # If `name` doesn't refer to a container variable...
        if name not in self.__dict__['index']:
            # ...check for an existing attribute and modify...
            if name in self.__dict__['_attributes']:
                super().__setattr__(name, value)

            # ...otherwise, add as a new attribute
            else:
                self.add_attribute(name, value)

            return

        if isinstance(value, Sequence) and not isinstance(value, str):
            value_as_array = np.array(value, dtype=self.__dict__['_' + name].dtype)

            if value_as_array.shape[0] != len(self.__dict__['span']):
                raise DimensionError(
                    f"Invalid assignment for '{name}': "
                    f"must be either a single value or "
                    f"a sequence of identical length to `span`"
                    f"(expected {len(self.__dict__['span'])} elements)"
                )

            self.__dict__['_' + name] = value_as_array

        else:
            self.__dict__['_' + name][:] = value

    def _locate_period_in_span(self, period: Hashable) -> int:
        """Return the index position of `period` in `self.span`.

        Notes
        -----
        The class-level attribute `_VALID_INDEX_METHODS` defines recognised
        methods for matching.
        """
        for method in self._VALID_INDEX_METHODS:
            if hasattr(self.__dict__['span'], method):
                index_function = getattr(self.__dict__['span'], method)

                try:
                    return index_function(period)
                except Exception as e:
                    raise KeyError(period) from e

        raise AttributeError(
            f'Unable to find valid search method in `span`; '
            f'expected one of: {self._VALID_INDEX_METHODS}'
        )

    def _resolve_period_slice(self, index: slice) -> Tuple[int]:
        """Convert a slice into a 3-tuple of indexing information to use with `self.span`."""
        start, stop, step = index.start, index.stop, index.step

        if start is None:
            start = self.__dict__['span'][0]
        if stop is None:
            stop = self.__dict__['span'][-1]
        if step is None:
            step = 1

        start_location = self._locate_period_in_span(start)

        # Adjust for a slice as a return value e.g. from a year ('2000') in a
        # quarterly `pandas` `PeriodIndex` ('1999Q1':'2001Q4')
        if isinstance(start_location, slice):
            start_location = start_location.start

        stop_location = self._locate_period_in_span(stop)

        # Adjust for a slice (as with `start_location`)
        if isinstance(stop_location, slice):
            stop_location = stop_location.stop
        else:
            # Only extend the limit for a regular index (`pandas`, for example,
            # already adjusts for this in its own API)
            # TODO: Check how generally this treatment applies i.e. beyond
            #       `pandas`
            stop_location += 1

        return start_location, stop_location, step

    def __getitem__(self, key: Union[str, Tuple[str, Union[Hashable, slice]]]) -> Any:
        # `key` is a string (variable name): return the corresponding array
        if isinstance(key, str):
            if key not in self.__dict__['index']:
                raise KeyError(f"'{key}' not recognised as a variable name")
            return self.__getattr__(key)

        # `key` is a tuple (variable name plus index): return the selected
        # elements of the corresponding array
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError(
                    'Invalid index: must be of length one (variable name) '
                    'or length two (variable name, span index)'
                )

            # Unpack the key
            name: str
            index: Union[Hashable, slice]
            name, index = key

            # Get the full array
            if name not in self.__dict__['index']:
                raise KeyError(f"'{name}' not recognised as a variable name")
            values = self.__getattr__(name)

            # Extract and return the relevant subset
            if isinstance(index, slice):
                start_location, stop_location, step = self._resolve_period_slice(index)
                return values[start_location:stop_location:step]

            location = self._locate_period_in_span(index)
            return values[location]

        raise TypeError(f'Invalid index type ({type(key)}): `{key}`')

    def __setitem__(
        self,
        key: Union[str, Tuple[str, Union[Hashable, slice]]],
        value: Union[Any, Sequence[Any]],
    ) -> None:
        # `key` is a string (variable name): update the corresponding array in
        # its entirety
        if isinstance(key, str):
            if key not in self.__dict__['index']:
                raise KeyError(f"'{key}' not recognised as a variable name")
            self.__setattr__(key, value)
            return

        # `key` is a tuple (variable name plus index): update selected elements
        # of the corresponding array
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError(
                    'Invalid index: must be of length one (variable name) '
                    'or length two (variable name, span index)'
                )

            # Unpack the key
            name: str
            index: Union[Hashable, slice]
            name, index = key

            # Modify the relevant subset
            if isinstance(index, slice):
                start_location, stop_location, step = self._resolve_period_slice(index)
                self.__dict__['_' + name][start_location:stop_location:step] = value
                return

            location = self._locate_period_in_span(index)
            self.__dict__['_' + name][location] = value
            return

        raise TypeError(f'Invalid index type ({type(key)}): `{key}`')

    def replace_values(self, **new_values) -> None:
        """Convenience function to replace values in one or more series at once.

        Parameter
        ---------
        **new_values : key-value pairs of replacements

        Examples
        --------
        >>> container = VectorContainer(range(10))
        >>> container.add_variable('A', 3)
        >>> container.add_variable('B', range(0, 20, 2))
        >>> container.values
        array([[ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3],
               [ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18]])

        >>> container.replace_values(A=4)
        array([[ 4,  4,  4,  4,  4,  4,  4,  4,  4,  4],
               [ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18]])

        >>> container.replace_values(A=range(10), B=5)
        >>> container.values
        array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
               [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]])

        >>> container.replace_values(**{'A': 6, 'B': range(-10, 0)})
        >>> container.values
        array([[  6,   6,   6,   6,   6,   6,   6,   6,   6,   6],
               [-10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1]])
        """
        for k, v in new_values.items():
            self.__setitem__(k, v)

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__['index']

    def copy(self) -> 'VectorContainer':
        """Return a copy of the current object."""
        copied = self.__class__(span=copy.deepcopy(self.__dict__['span']))
        copied.__dict__.update({k: copy.deepcopy(v) for k, v in self.__dict__.items()})
        return copied

    __copy__ = copy

    def __deepcopy__(self, *args: Any, **kwargs: Any) -> 'VectorContainer':
        return self.copy()

    def __dir__(self) -> List[str]:
        # Return the names of the object's contents alongside the variable
        # names (in `index`) and attributes (in `_attributes`)
        return sorted(
            dir(type(self)) + self.__dict__['index'] + self.__dict__['_attributes']
        )

    def _ipython_key_completions_(self) -> List[str]:
        return self.__dict__['index']

    @property
    def strict(self) -> bool:
        """If `True`, the only way to add attributes to the object is as a variable, using `add_variable()`."""
        return self.__dict__['_strict']

    @strict.setter
    def strict(self, value: bool) -> None:
        self.__dict__['_strict'] = bool(value)

    @property
    def values(self) -> np.ndarray:
        """Container contents as a 2D (index x span) array."""
        return np.array(
            [self.__getattribute__('_' + name) for name in self.__dict__['index']]
        )

    @values.setter
    def values(self, new_values: Union[np.ndarray, Any]) -> None:
        """Replace all values (effectively, element-by-element), preserving the original data types in the container.

        Notes
        -----
        The replacement must be either:
         - a 2D array with identical dimensions to `values`, and in the right
           order: (index x span)
         - a scalar to replace all array elements

        The method coerces each row to the data type of the corresponding item
        in the container.
        """
        if isinstance(new_values, np.ndarray):
            if new_values.shape != self.values.shape:
                raise DimensionError(
                    f'Replacement array is of shape {new_values.shape} '
                    f'but expected shape is {self.values.shape}'
                )

            for name, series in zip(self.index, new_values):
                self.__setattr__(
                    name, series.astype(self.__getattribute__('_' + name).dtype)
                )

        else:
            for name in self.index:
                series = self.__getattribute__('_' + name)
                self.__setattr__(
                    name, np.full(series.shape, new_values, dtype=series.dtype)
                )

    def _resolve_expression_indexes(self, expression: str) -> str:
        """Convert an expression with backticks (denoting period labels) to one with just integer indexes."""

        def resolve_index_in_span(label: str) -> int:
            """Convert a single, possibly backticked, index/label to an integer index."""
            # If no backticks, take `label` to be a regular integer index:
            # Convert and return
            if '`' not in label:
                return int(label.strip())

            # Remove backticks
            period = label.strip().strip('`')

            # Return the index if found in `span`
            if period in self.span:
                return self._locate_period_in_span(period)

            # If not found, try to cast to an integer and check again
            # TODO: Is this too implicit?
            try:
                period = int(period)
            except ValueError:
                raise KeyError(
                    f"Unable to locate period with label '{period}' in object's span"
                )

            if period not in self.span:
                raise KeyError(
                    f"Unable to locate period with label '{period}' in object's span"
                )

            return self._locate_period_in_span(period)

        def resolve_indexes(match: re.match) -> str:
            """Convert the contents of a possibly backticked index expression to integer indexes."""
            # Treat the contents of `match` as a slice, with up to three
            # components: start, stop, step
            slice_ = match.group(1).split(':')

            if len(slice_) > 3:
                raise ValueError(
                    f"'{match.group(1)}' is invalid as a slice: Too many items"
                )

            if len(slice_) == 1:
                # One item provided: A period label as an index
                return '[' + str(resolve_index_in_span(slice_[0])) + ']'

            # Multiple items: Treat as a slice
            start, stop, *step = map(str.strip, slice_)

            # Resolve first (`start`) and second (`stop`) arguments
            if len(start):
                start = resolve_index_in_span(start)

                # Handle slices (typically from a `pandas` `PeriodIndex` or
                # similar)
                if isinstance(start, slice):
                    start = start.start

            if len(stop):
                stop = resolve_index_in_span(stop)

                # Handle slices (typically from a `pandas` `PeriodIndex` or
                # similar)
                if isinstance(stop, slice):
                    stop = stop.stop

            # Adjust for closed intervals on the right-hand side (mirroring
            # `pandas`)
            if isinstance(stop, int):
                stop += 1

            # Resolve third (`step`) argument
            if len(step) == 0:
                step = ''
            elif len(step) == 1:
                step = step[0]
            else:
                raise ValueError(
                    f"Found multiple step values in '{match.group(1)}': Expected at most one"
                )

            return f'[{start}:{stop}:{step}]'

        index_re = re.compile(r'\[\s*(.+?)?\s*\]')
        return index_re.sub(resolve_indexes, expression)

    def reindex(
        self, span: Sequence[Hashable], *, fill_value: Any = None, **fill_values: Any
    ) -> 'VectorContainer':
        """Return a copy of the current object, adjusted to match `span`. Values in overlapping periods between the old and new objects are preserved (copied over).

        TODO: Consider implementing something closer to `pandas` `reindex()`
              behaviour?

        Parameters
        ----------
        span : iterable
            Sequence of periods defining the span of the object to be returned
        fill_value :
            Default fill value for new periods
        **fill_values :
            Variable-specific fill value(s)
        """
        # Construct a mapping of:
        #  - keys: positions in `span` (the new, reindexed object)
        #  - values: the corresponding positions in `self.span` (i.e. where to
        #            get the old values from)
        positions = {}
        for i, period in enumerate(span):
            if period in self.span:
                positions[i] = self._locate_period_in_span(period)

        # Copy the current object and adjust:
        #  - the span
        #  - the individual underlying variables to conform to the new span
        reindexed = self.copy()
        reindexed.__dict__['span'] = span  # Use to bypass `strict`

        for name in reindexed.index:
            # Replace the underlying variable, to bypass dimension and type
            # checks

            # TODO: How to improve performance here?

            # Get the fill value either as:
            #  - specified (as a keyword argument in **fill_values)
            #  - the default in `fill_value`
            value = fill_values.get(name, fill_value)

            # Special handling for `bool`, `int` and `str`-like dtypes (none of
            # which have support for NaN/None)
            if np.issubdtype(self[name].dtype, bool):
                if value is None:
                    value = False
                else:
                    value = bool(value)

            elif np.issubdtype(self[name].dtype, np.integer):
                if value is None:
                    value = 0
                else:
                    value = int(value)

            elif np.issubdtype(self[name].dtype, str):
                if value is None:
                    value = ''
                else:
                    value = str(value)

            # Initialise the replacement with the correct length, and the fill
            # value
            reindexed.__dict__[f'_{name}'] = np.full(
                len(span), value, dtype=self[name].dtype
            )

            # Copy over individual values
            # TODO: Vectorise this?
            for new, old in positions.items():
                reindexed[name][new] = self[name][old]

        return reindexed

    def to_dataframe(self) -> 'pandas.DataFrame':  # noqa: F821
        """Return the contents of the container as a `pandas` DataFrame, one column per variable. **Requires `pandas`**."""
        from pandas import DataFrame

        # NB Take variables one at a time, rather than use `self.values`. This
        #    preserves the dtypes of the individual series.
        return DataFrame({k: self[k] for k in self.index}, index=self.span)

    def eval(
        self,
        expression: str,
        *,
        globals: Optional[Dict[str, Any]] = None,
        locals: Optional[Dict[str, Any]] = None,
        builtins: [Optional[Dict[str, Any]]] = None,
    ) -> Union[float, np.ndarray]:
        """Evaluate `expression` as it applies to the current object. **Uses `eval()`**.

        **Subject to change** (see Notes).

        Parameters
        ----------
        expression :
            The expression to evaluate (see Examples and Notes for further
            information)
        globals :
            `globals` argument to pass to `eval()`
        locals :
            `locals` argument to pass to `eval()`
        builtins :
            If `None` (the default), make standard operators from
            `fsic.functions` available (see Notes)
            Disable by passing `{}`

        Examples
        --------
        # Setup
        >>> container = VectorContainer(range(2000, 2010 + 1))
        >>> container.add_variable('X', 0, dtype=float)
        >>> container.add_variable('Y', 1, dtype=float)
        >>> container.add_variable('Z', 2, dtype=float)

        # Vector operations
        >>> container.eval('X + Y + Z')
        array([3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])

        # Index and slice operations
        >>> container.eval('X[0] + Y[-1]')
        1.0

        >>> container.Z[2:4] += 1
        >>> container.eval('Y[::2] + Z[:6]')
        [3. 3. 4. 4. 3. 3.]

        # Mixed index/slice-vector operations
        >>> container.X[1] = 1
        >>> container.eval('X[1] + Y')
        array([2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.])

        # To index by period rather than position, enclose values with
        # backticks (`)
        >>> container.eval('X[`2001`]')
        1.0

        >>> container.eval('X[`2001`:`2009`:3]')
        [1. 0. 0.]


        # `eval()` also works for `pandas` index objects
        >>> import pandas as pd

        # Instantiate an object covering the period 2000Q1-2002Q4
        >>> container = VectorContainer(pd.period_range(start='2000-01-01', end='2002-12-31', freq='Q'))

        >>> container.add_variable('X', range(12), dtype=float)
        >>> container.X
        [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11.]

        >>> container['X', '2001':'2002']
        [ 4.  5.  6.  7.  8.  9. 10. 11.]

        >>> container.eval('X[`2001Q2`:`2002Q3`] * 2')
        [10. 12. 14. 16. 18. 20.]

        >>> container['X', '2001']
        [4. 5. 6. 7.]

        >>> container.eval('X[`2001`] * 3')
        [12. 15. 18. 21.]

        >>> container.eval('X[:`2001:2]')
        [0. 2. 4. 6.]

        >>> container['X', '2001':'2002Q3']
        [ 4.  5.  6.  7.  8.  9. 10.]

        >>> container.eval('X[`2001`:`2002Q3`:3]')
        [ 4.  7. 10.]

        Notes
        -----
        This method carries out various pre-processing operations before
        passing the expression to Python's `eval()` function:

        1. Process `expression` to handle any period indexes e.g. to convert
           'X[`2000`]' to 'X[2]' or similar
        2. Assemble the final version of `locals` (`globals` is unchanged)

        The order for assembling `locals` is as follows (the reverse gives the
        order of precedence):

        1. `builtins`
        2. model variables
        3. the `locals` parameter passed to this method


        The main question about the overall implementation is the implicit
        casting to integer for indexes enclosed in backticks.

        For example, the expression 'X[`2000`]' first tries to match the label
        2000 as a string. If this fails, the same is tried after casting 2000
        to an integer.

        This has the convenience of not needing to keep track of label types
        (and simplifies the syntax) but may be too implicit.

        *To think on further*.

        See also
        --------
        _resolve_expression_indexes(), for the implementation set out in Notes.
        """
        # 1. Process `expression` to handle any period indexes
        if '`' in expression:
            expression = self._resolve_expression_indexes(expression)

        # 2. Assemble the final version of `locals` (`globals` is unchanged)

        # Set builtins
        if builtins is None:
            # Copy the builtins to avoid changing global package state
            # TODO: Review performance implications of copying
            builtins = copy.deepcopy(_builtins)
        locals_ = builtins

        # Update with model variables
        locals_.update({x: self[x] for x in self.index})

        # Add user locals as needed
        if locals is not None:
            locals_.update(locals)

        # Either...
        try:
            # ...return the result...
            return eval(expression, globals, locals_)

        except NameError as e:
            # ...or catch a name error (the expression contains an undefined
            # variable) and try to suggest to the user a valid alternative
            name = e.name
            suggestions = self.get_closest_match(name)

            if len(suggestions) == 0:
                raise AttributeError(
                    f"Object is empty and thus has no attribute '{name}'"
                ) from e
            else:
                raise AttributeError(
                    f"Object has no attribute '{name}'. Did you mean: '{suggestions[0]}'?"
                ) from e

    def exec(self, expression: str) -> None:
        raise NotImplementedError('`exec()` method not implemented yet')


# Model interface, wrapping the core `VectorContainer` ------------------------


class SolutionStatus(enum.Enum):
    """Enumeration to record solution status."""

    UNSOLVED = '-'

    SOLVED = '.'

    FAILED = 'F'
    ERROR = 'E'
    SKIPPED = 'S'


class ModelInterface(VectorContainer):
    NAMES: List[str] = []

    def __init__(
        self,
        span: Sequence[Hashable],
        *,
        strict: bool = False,
        dtype: Any = float,
        default_value: Union[int, float] = 0.0,
        **initial_values: Dict[str, Any],
    ) -> None:
        """Initialise model variables.

        Parameters
        ----------
        span : iterable
            Sequence of periods that defines the timespan of the model
        strict : bool
            If `True`, the only way to add attributes to the object is with
            `add_variable()` i.e. as new container variables. Ad hoc attributes
            are expressly blocked.
            If `False`, further attributes can be added ad hoc at runtime in
            the usual way for Python objects e.g. `model.A = ...`.
            On instantiation, `strict=True` also checks that any variables set
            with `initial_values` are in the class's `NAMES` attribute, raising
            an `InitialisationError` if not.
        dtype : variable type
            Default data type to impose on model variables (in NumPy arrays)
        default_value : number
            Value with which to initialise model variables
        **initial_values : keyword arguments of variable names and values
            Values with which to initialise specific named model variables
        """
        # Set up data container
        super().__init__(span, strict=strict)

        # Store the `dtype` as the default for future values e.g. when using
        # `add_variable()` after initialisation
        self.add_attribute('dtype', dtype)

        # Use the base class version of `add_variable()` because `self.names`
        # is set separately (whereas this class's version would attempt to
        # extend `self.names`)

        # Add solution tracking variables
        super().add_variable('status', SolutionStatus.UNSOLVED.value)
        super().add_variable('iterations', -1)

        # Check for duplicate names
        names = copy.deepcopy(self.NAMES)

        if len(set(names)) != len(names):
            duplicates = [k for k, v in Counter(names).items() if v > 1]
            raise DuplicateNameError(
                f"Found multiple instances of the following variable(s) in `NAMES`: {', '.join(duplicates)}"
            )

        # If `strict`, check that any variables set via `initial_values` match
        # an entry in `names` (which is itself a copy of the class's `NAMES`
        # attribute)
        if strict:
            permitted_names = set(names)
            passed_names = set(initial_values.keys())
            invalid_names = passed_names - permitted_names

            if len(invalid_names) > 0:
                message_suggestions = []
                for name in invalid_names:
                    suggestions = self.get_closest_match(name, possibilities=names)
                    if len(suggestions) == 0:
                        message_suggestions.append(
                            f' - {name} : no alternative suggestions available'
                        )
                    elif len(suggestions) == 1:
                        message_suggestions.append(
                            f" - {name} : did you mean '{suggestions[0]}'?"
                        )
                    elif len(suggestions) > 1:
                        raise NotImplementedError(
                            'Handling of multiple name matches not yet implemented'
                        )

                raise InitialisationError(
                    'Cannot add unlisted variables (i.e. variables not in `NAMES`) '
                    'when `strict=True` - found {} instance(s):\n{}'.format(
                        len(invalid_names), '\n'.join(message_suggestions)
                    )
                )

        # Add model variables
        self.add_attribute('names', names)

        for name in self.names:
            super().add_variable(
                name,
                initial_values.get(name, default_value),
                dtype=self.dtype,
            )

    def add_variable(
        self, name: str, value: Union[Any, Sequence[Any]], *, dtype: Any = None
    ) -> None:
        """Add a new variable to the model at runtime, forcing a `dtype` if needed.

        Parameters
        ----------
        name : str
            The name of the new variable, which must not already exist in the
            model (raise a `DuplicateNameError` if it already exists)
        value : single value or sequence of values
            Initial value(s) for the new variable. If a sequence, must yield a
            length equal to that of `span`
        dtype : variable type
            Data type to impose on the variable. The default is to use the
            `dtype` originally passed at initialisation. Otherwise, use this
            argument to set the type.

        Notes
        -----
        In implementation, as well as adding the data to the underlying
        container, this version of the method also extends the list of names in
        `self.names`, to keep the variable list up-to-date and for
        compatibility with, for example, `fsic.tools.model_to_dataframe()`.
        """
        # Optionally impose the `dtype`
        if dtype is None:
            dtype = self.__dict__['dtype']

        super().add_variable(name, value, dtype=dtype)
        self.__dict__['names'].append(name)

    @property
    def size(self) -> int:
        """Number of elements in the model's vector arrays."""
        return len(self.__dict__['names']) * len(self.__dict__['span'])

    def get_closest_match(
        self,
        name: str,
        *,
        possibilities: Optional[Sequence[str]] = None,
        cutoff: Union[int, float] = 0.1,
    ) -> List[str]:
        """Return the closest match(es) to `name` among `possibilities` (by default, `self.names`).

        Parameters
        ----------
        name : str
            Variable name to attempt to match
        possibilities : sequence of str, default `None`
            List of names to try to match to. If `None`, use `self.names`.
        cutoff : numeric, default 0.1
            Cutoff score (to pass to `difflib.get_close_matches()`) above which
            to return matches

        Returns
        -------
        List[str] :
            Closest match(es) to `name` in `possibilities` (multiple matches
            arise if `possibilities` has duplicate names, once converted to
            lower case)
            If no suitable match is found, the result is an empty list
        """
        # Set default list if none provided
        if possibilities is None:
            possibilities = self.names

        return super().get_closest_match(
            name, possibilities=possibilities, cutoff=cutoff
        )

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__['names']

    @property
    def values(self) -> np.ndarray:
        """Model variable values as a 2D (names x span) array."""
        return np.array([self.__getattribute__('_' + name) for name in self.names])

    @values.setter
    def values(self, new_values: Union[np.ndarray, Any]) -> None:
        """Replace all values (effectively, element-by-element), preserving the original data types in the container.

        Notes
        -----
        The replacement must be either:
         - a 2D array with identical dimensions to `values`, and in the right
           order: (index x span)
         - a scalar to replace all array elements

        The method coerces each row to the data type of the corresponding item
        in the container.
        """
        if isinstance(new_values, np.ndarray):
            if new_values.shape != self.values.shape:
                raise DimensionError(
                    f'Replacement array is of shape {new_values.shape} '
                    f'but expected shape is {self.values.shape}'
                )

            for name, series in zip(self.names, new_values):
                self.__setattr__(
                    name, series.astype(self.__getattribute__('_' + name).dtype)
                )

        else:
            for name in self.names:
                series = self.__getattribute__('_' + name)
                self.__setattr__(
                    name, np.full(series.shape, new_values, dtype=series.dtype)
                )


# Mixin to define (but not fully implement) solver behaviour ------------------


class PeriodIter:
    """Iterator of (index, label) pairs returned by `SolverMixin.iter_periods()`. Compatible with `len()`."""

    def __init__(self, *args: Any):
        self._length = len(args[0])
        self._iter = list(zip(*args))

    def __iter__(self):
        yield from self._iter

    def __next__(self):
        return next(self._iter)

    def __len__(self):
        return self._length


class SolverMixin:
    """Mixin to define (but not fully implement) solver behaviour.

    Requires `self.span` (a `Sequence` of `Hashable`s) as an attribute.
    """

    LAGS: int = 0
    LEADS: int = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Set up the object and copy class-level lags and leads to object-level attributes."""
        super().__init__(*args, **kwargs)

        self.add_attribute('lags', self.LAGS)
        self.add_attribute('leads', self.LEADS)

    def iter_periods(
        self,
        *,
        start: Optional[Hashable] = None,
        end: Optional[Hashable] = None,
        **kwargs: Any,
    ) -> Iterable[Tuple[int, Hashable]]:
        """Return pairs of period indexes and labels.

        Parameters
        ----------
        start : element in the model's `span`
            First period to return. If not given, defaults to the first
            solvable period, taking into account any lags in the model's
            equations
        end : element in the model's `span`
            Last period to return. If not given, defaults to the last solvable
            period, taking into account any leads in the model's equations
        **kwargs : not used
            Absorbs arguments passed from `solve()`. Available should the user
            want to over-ride this method in their own derived class.

        Returns
        -------
        A `PeriodIter` object, which is iterable and returns the period (index,
        label) pairs in sequence.
        """
        if len(self.span) == 0:
            raise SolutionError('Object `span` is empty: No periods to solve')

        # Default start and end periods
        if start is None:
            start = self.span[self.lags]
        if end is None:
            end = self.span[-1 - self.leads]

        # Convert to an integer range
        indexes = range(
            self._locate_period_in_span(start), self._locate_period_in_span(end) + 1
        )

        return PeriodIter(indexes, self.span[indexes.start : indexes.stop])

    def solve(
        self,
        *,
        start: Optional[Hashable] = None,
        end: Optional[Hashable] = None,
        min_iter: int = 0,
        max_iter: int = 100,
        tol: Union[int, float] = 1e-10,
        offset: int = 0,
        failures: str = 'raise',
        errors: str = 'raise',
        catch_first_error: bool = True,
        **kwargs: Any,
    ) -> Tuple[List[Hashable], List[int], List[bool]]:
        """Solve the model. Use default periods if none provided.

        Parameters
        ----------
        start : element in the model's `span`
            First period to solve. If not given, defaults to the first solvable
            period, taking into account any lags in the model's equations
        end : element in the model's `span`
            Last period to solve. If not given, defaults to the last solvable
            period, taking into account any leads in the model's equations
        min_iter : int
            Minimum number of iterations to solution each period
        max_iter : int
            Maximum number of iterations to solution each period
        tol : int or float
            Tolerance for convergence
        offset : int
            If non-zero, copy an initial set of endogenous values from the
            relative period described by `offset`. For example, `offset=-1`
            initialises each period's solution with the values from the
            previous period.
        failures : str, one of {'raise', 'ignore'}
            Action should the solution fail to converge in a period (by
            reaching the maximum number of iterations, `max_iter`)
             - 'raise' (default): stop immediately and raise a
                                  `NonConvergenceError`
             - 'ignore': continue to the next period
        errors : str, one of {'raise', 'skip', 'ignore', 'replace'}
            User-specified treatment on encountering numerical solution errors
            e.g. NaNs and infinities
             - 'raise' (default): stop immediately and raise a `SolutionError`
                                  [set current period solution status to 'E']
             - 'skip': stop solving the current period and move to the next one
                       [set current period solution status to 'S']
             - 'ignore': continue solving, with no special treatment or action
                         [period solution statuses as usual i.e. '.' or 'F']
             - 'replace': each iteration, replace NaNs and infinities with
                          zeroes
                          [period solution statuses as usual i.e. '.' or 'F']
        catch_first_error : bool
            If `True` (default) and `errors='raise'`, raise an exception
            (`SolutionError`) on the first numerical error/warning during
            solution. This identifies the problem statement in the stack trace
            without modifying any values at this point.
            If `False`, only check for errors after completing an iteration,
            raising an exception (`SolutionError`) after the fact. This allows
            numerical errors (NaNs, Infs) to propagate through the solution
            before raising the exception.
        **kwargs :
            Further keyword arguments to pass on to other methods:
             - `iter_periods()`
             - `solve_t()`

        Returns
        -------
        Three lists, each of length equal to the number of periods to be
        solved:
         - the names of the periods to be solved, as they appear in the model's
           span
         - integers: the index positions of the above periods in the model's
           span
         - bools, one per period: `True` if the period solved successfully;
           `False` otherwise

        Notes
        -----
        With `catch_first_error=True` and `errors='raise'`, solution
        immediately halts on the first error, throwing an exception up the call
        stack. This identifies the problem statement in the stack trace (which
        is helpful for debugging) but also prevents a NaN/Inf entering the
        solution (which may be useful for inspection) and, were there a
        NaN/Inf, from it being propagated (it's not immediately obvious if this
        has any use, though).
        """
        # Error if `min_iter` exceeds `max_iter`
        if min_iter > max_iter:
            raise ValueError(
                f'Value of `min_iter` ({min_iter}) cannot '
                f'exceed value of `max_iter` ({max_iter})'
            )

        # Catch invalid `start` and `end` periods here e.g. to avoid later
        # problems with indexing a year against a `pandas` `PeriodIndex`
        if start is not None and not isinstance(
            self._locate_period_in_span(start), int
        ):
            raise KeyError(start)

        if end is not None and not isinstance(self._locate_period_in_span(end), int):
            raise KeyError(end)

        period_iter = self.iter_periods(start=start, end=end, **kwargs)

        # fmt: off
        indexes = [None] * len(period_iter)
        labels  = [None] * len(period_iter)
        solved  = [None] * len(period_iter)
        # fmt: on

        for i, (t, period) in enumerate(period_iter):
            indexes[i] = t
            labels[i] = period
            solved[i] = self.solve_t(
                t,
                min_iter=min_iter,
                max_iter=max_iter,
                tol=tol,
                offset=offset,
                failures=failures,
                errors=errors,
                catch_first_error=catch_first_error,
                **kwargs,
            )

        return labels, indexes, solved

    def solve_period(
        self,
        period: Hashable,
        *,
        min_iter: int = 0,
        max_iter: int = 100,
        tol: Union[int, float] = 1e-10,
        offset: int = 0,
        failures: str = 'raise',
        errors: str = 'raise',
        catch_first_error: bool = True,
        **kwargs: Any,
    ) -> bool:
        """Solve a single period.

        Parameters
        ----------
        period : element in the model's `span`
            Named period to solve
        min_iter : int
            Minimum number of iterations to solution each period
        max_iter : int
            Maximum number of iterations to solution each period
        tol : int or float
            Tolerance for convergence
        offset : int
            If non-zero, copy an initial set of endogenous values from the
            relative period described by `offset`. For example, `offset=-1`
            initialises each period's solution with the values from the
            previous period.
        failures : str, one of {'raise', 'ignore'}
            Action should the solution fail to converge (by reaching the
            maximum number of iterations, `max_iter`)
             - 'raise' (default): stop immediately and raise a
                                  `NonConvergenceError`
             - 'ignore': do nothing
        errors : str, one of {'raise', 'skip', 'ignore', 'replace'}
            User-specified treatment on encountering numerical solution errors
            e.g. NaNs and infinities
             - 'raise' (default): stop immediately and raise a `SolutionError`
                                  [set period solution status to 'E']
             - 'skip': stop solving the current period
                       [set period solution status to 'S']
             - 'ignore': continue solving, with no special treatment or action
                         [period solution statuses as usual i.e. '.' or 'F']
             - 'replace': each iteration, replace NaNs and infinities with
                          zeroes
                          [period solution statuses as usual i.e. '.' or 'F']
        catch_first_error : bool
            If `True` (default) and `errors='raise'`, raise an exception
            (`SolutionError`) on the first numerical error/warning during
            solution. This identifies the problem statement in the stack trace
            without modifying any values at this point.
            If `False`, only check for errors after completing an iteration,
            raising an exception (`SolutionError`) after the fact. This allows
            numerical errors (NaNs, Infs) to propagate through the solution
            before raising the exception.
        **kwargs :
            Further keyword arguments to pass to the solution routines

        Returns
        -------
        `True` if the model solved for the current period; `False` otherwise.

        Notes
        -----
        With `catch_first_error=True` and `errors='raise'`, solution
        immediately halts on the first error, throwing an exception up the call
        stack. This identifies the problem statement in the stack trace (which
        is helpful for debugging) but also prevents a NaN/Inf entering the
        solution (which may be useful for inspection) and, were there a
        NaN/Inf, from it being propagated (it's not immediately obvious if this
        has any use, though).
        """
        t = self._locate_period_in_span(period)

        if not isinstance(t, int):
            raise KeyError(
                f'Invalid `period` argument: unable to convert to an integer '
                f'(a single location in `self.span`). '
                f'`period` resolved to type {type(t)} with value {t}'
            )

        return self.solve_t(
            t,
            min_iter=min_iter,
            max_iter=max_iter,
            tol=tol,
            offset=offset,
            failures=failures,
            errors=errors,
            catch_first_error=catch_first_error,
            **kwargs,
        )

    def solve_t(
        self,
        t: int,
        *,
        min_iter: int = 0,
        max_iter: int = 100,
        tol: Union[int, float] = 1e-10,
        offset: int = 0,
        failures: str = 'raise',
        errors: str = 'raise',
        catch_first_error: bool = True,
        **kwargs: Any,
    ) -> bool:
        """Solve for the period at integer position `t` in the model's `span`.

        Parameters
        ----------
        t : int
            Position in `span` of the period to solve
        min_iter : int
            Minimum number of iterations to solution each period
        max_iter : int
            Maximum number of iterations to solution each period
        tol : int or float
            Tolerance for convergence
        offset : int
            If non-zero, copy an initial set of endogenous values from the
            period at position `t + offset`. For example, `offset=-1` copies
            the values from the previous period.
        failures : str, one of {'raise', 'ignore'}
            Action should the solution fail to converge (by reaching the
            maximum number of iterations, `max_iter`)
             - 'raise' (default): stop immediately and raise a
                                  `NonConvergenceError`
             - 'ignore': do nothing
        errors : str, one of {'raise', 'skip', 'ignore', 'replace'}
            User-specified treatment on encountering numerical solution errors
            e.g. NaNs and infinities
             - 'raise' (default): stop immediately and raise a `SolutionError`
                                  [set period solution status to 'E']
             - 'skip': stop solving the current period
                       [set period solution status to 'S']
             - 'ignore': continue solving, with no special treatment or action
                         [period solution statuses as usual i.e. '.' or 'F']
             - 'replace': each iteration, replace NaNs and infinities with
                          zeroes
                          [period solution statuses as usual i.e. '.' or 'F']
        catch_first_error : bool
            If `True` (default) and `errors='raise'`, raise an exception
            (`SolutionError`) on the first numerical error/warning during
            solution. This identifies the problem statement in the stack trace
            without modifying any values at this point.
            If `False`, only check for errors after completing an iteration,
            raising an exception (`SolutionError`) after the fact. This allows
            numerical errors (NaNs, Infs) to propagate through the solution
            before raising the exception.
        **kwargs :
            Further keyword arguments to pass to the solution routines

        Returns
        -------
        `True` if the model solved for the current period; `False` otherwise.

        Notes
        -----
        With `catch_first_error=True` and `errors='raise'`, solution
        immediately halts on the first error, throwing an exception up the call
        stack. This identifies the problem statement in the stack trace (which
        is helpful for debugging) but also prevents a NaN/Inf entering the
        solution (which may be useful for inspection) and, were there a
        NaN/Inf, from it being propagated (it's not immediately obvious if this
        has any use, though).
        """


# Base class for individual models --------------------------------------------
class BaseModel(SolverMixin, ModelInterface):
    """Base class for economic models."""

    ENDOGENOUS: List[str] = []
    EXOGENOUS: List[str] = []

    PARAMETERS: List[str] = []
    ERRORS: List[str] = []

    NAMES: List[str] = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
    CHECK: List[str] = ENDOGENOUS

    LAGS: int = 0
    LEADS: int = 0

    CODE: Optional[str] = None

    def __init__(
        self,
        span: Sequence[Hashable],
        *,
        engine: str = 'python',
        strict: bool = False,
        dtype: Any = float,
        default_value: Union[int, float] = 0.0,
        **initial_values: Dict[str, Any],
    ) -> None:
        """Initialise model variables.

        Parameters
        ----------
        span : iterable
            Sequence of periods that defines the timespan of the model
        engine : str
            Signal of the (expected) underlying solution method/implementation
        strict : bool
            If `True`, the only way to add attributes to the object is with
            `add_variable()` i.e. as new container variables. Ad hoc attributes
            are expressly blocked.
            If `False`, further attributes can be added ad hoc at runtime in
            the usual way for Python objects e.g. `model.A = ...`.
            On instantiation, `strict=True` also checks that any variables set
            with `initial_values` are in the class's `NAMES` attribute, raising
            an `InitialisationError` if not.
        dtype : variable type
            Data type to impose on model variables (in NumPy arrays)
        default_value : number
            Value with which to initialise model variables
        **initial_values : keyword arguments of variable names and values
            Values with which to initialise specific named model variables
        """
        super().__init__(
            span=span,
            strict=strict,
            dtype=dtype,
            default_value=default_value,
            **initial_values,
        )

        self.add_attribute('engine', engine)

    @classmethod
    def from_dataframe(
        cls: 'BaseModel',
        data: 'pandas.DataFrame',  # noqa: F821
        *args: Any,
        **kwargs: Any,
    ) -> 'BaseModel':
        """Initialise the model by taking the index and values from a `pandas` DataFrame(-like).

        TODO: Consider keyword argument to control type conversion of the index

        Parameters
        ----------
        data : `pandas` DataFrame(-like)
            The DataFrame's index (once converted to a list) becomes the
            model's `span`.
            The DataFrame's contents set the model's values, as with
            `**initial_values` in the class `__init__()` method. Default values
            continue to be set for any variables not included in the DataFrame.
        *args, **kwargs : further arguments to the class `__init__()` method
        """
        # TODO: Consider a more general way to deal with this
        from pandas import DatetimeIndex, MultiIndex, PeriodIndex, TimedeltaIndex

        index = data.index

        if not isinstance(
            index, (DatetimeIndex, MultiIndex, PeriodIndex, TimedeltaIndex)
        ):
            index = list(index)

        return cls(index, *args, **{k: v.values for k, v in data.items()}, **kwargs)

    def reindex(
        self, span: Sequence[Hashable], *, fill_value: Any = None, **fill_values: Any
    ) -> 'BaseModel':
        """Return a copy of the current object, adjusted to match `span`. Values in overlapping periods between the old and new objects are preserved (copied over).

        Parameters
        ----------
        span : iterable
            Sequence of periods defining the span of the object to be returned
        fill_value :
            Default fill value for new periods
        **fill_values :
            Variable-specific fill value(s)

        Notes
        -----
        These attributes have the following defaults (which can be over-ridden
        in the usual way, using `fill_values`):
         - status: '-' (`SolutionStatus.UNSOLVED.value`)
         - iterations: -1

        TODO: Consider how to generalise defaults to support extensions
        """
        fill_values['status'] = fill_values.get('status', SolutionStatus.UNSOLVED.value)
        fill_values['iterations'] = fill_values.get('iterations', -1)

        return super().reindex(span, fill_value=fill_value, **fill_values)

    def to_dataframe(
        self, *, status: bool = True, iterations: bool = True
    ) -> 'pandas.DataFrame':  # noqa: F821
        """Return the values and solution information from the model as a `pandas` DataFrame. **Requires `pandas`**."""
        return _model_to_dataframe(self, status=status, iterations=iterations)

    def solve_t(
        self,
        t: int,
        *,
        min_iter: int = 0,
        max_iter: int = 100,
        tol: Union[int, float] = 1e-10,
        offset: int = 0,
        failures: str = 'raise',
        errors: str = 'raise',
        catch_first_error: bool = True,
        **kwargs: Any,
    ) -> bool:
        """Solve for the period at integer position `t` in the model's `span`.

        Parameters
        ----------
        t : int
            Position in `span` of the period to solve
        min_iter : int
            Minimum number of iterations to solution each period
        max_iter : int
            Maximum number of iterations to solution each period
        tol : int or float
            Tolerance for convergence
        offset : int
            If non-zero, copy an initial set of endogenous values from the
            period at position `t + offset`. For example, `offset=-1` copies
            the values from the previous period.
        failures : str, one of {'raise', 'ignore'}
            Action should the solution fail to converge (by reaching the
            maximum number of iterations, `max_iter`)
             - 'raise' (default): stop immediately and raise a
                                  `NonConvergenceError`
             - 'ignore': do nothing
        errors : str, one of {'raise', 'skip', 'ignore', 'replace'}
            User-specified treatment on encountering numerical solution errors
            e.g. NaNs and infinities
             - 'raise' (default): stop immediately and raise a `SolutionError`
                                  [set period solution status to 'E']
             - 'skip': stop solving the current period
                       [set period solution status to 'S']
             - 'ignore': continue solving, with no special treatment or action
                         [period solution statuses as usual i.e. '.' or 'F']
             - 'replace': each iteration, replace NaNs and infinities with
                          zeroes
                          [period solution statuses as usual i.e. '.' or 'F']
        catch_first_error : bool
            If `True` (default) and `errors='raise'`, raise an exception
            (`SolutionError`) on the first numerical error/warning during
            solution. This identifies the problem statement in the stack trace
            without modifying any values at this point.
            If `False`, only check for errors after completing an iteration,
            raising an exception (`SolutionError`) after the fact. This allows
            numerical errors (NaNs, Infs) to propagate through the solution
            before raising the exception.
        **kwargs :
            Further keyword arguments to pass to the solution routines

        Returns
        -------
        `True` if the model solved for the current period; `False` otherwise.

        Notes
        -----
        With `catch_first_error=True` and `errors='raise'`, solution
        immediately halts on the first error, throwing an exception up the call
        stack. This identifies the problem statement in the stack trace (which
        is helpful for debugging) but also prevents a NaN/Inf entering the
        solution (which may be useful for inspection) and, were there a
        NaN/Inf, from it being propagated (it's not immediately obvious if this
        has any use, though).
        """

        def get_check_values() -> np.ndarray:
            """Return a 1D NumPy array of variable values for checking in the current period."""
            return np.array([self.__dict__['_' + name][t] for name in self.CHECK])

        # Error if `min_iter` exceeds `max_iter`
        if min_iter > max_iter:
            raise ValueError(
                f'Value of `min_iter` ({min_iter}) '
                f'cannot exceed value of `max_iter` ({max_iter})'
            )

        # Optionally copy initial values from another period
        if offset:
            t_check = t
            if t_check < 0:
                t_check += len(self.span)

            # Error if `offset` points prior to the current model span
            if t_check + offset < 0:
                raise IndexError(
                    f'`offset` argument ({offset}) for position `t` ({t}) '
                    f'implies a period before the span of the current model instance: '
                    f'{offset} + {t} -> position {offset + t_check} < 0'
                )

            # Error if `offset` points beyond the current model span
            if t_check + offset >= len(self.span):
                raise IndexError(
                    f'`offset` argument ({offset}) for position `t` ({t}) '
                    f'implies a period beyond the span of the current model instance: '
                    f'{offset} + {t} -> position {offset + t_check} >= {len(self.span)} periods in span'
                )

            for name in self.ENDOGENOUS:
                self.__dict__['_' + name][t] = self.__dict__['_' + name][t + offset]

        status = SolutionStatus.UNSOLVED.value
        current_values = get_check_values()

        # Raise an exception if there are pre-existing NaNs or infinities, and
        # error checking is at its strictest ('raise')
        if errors == 'raise' and np.any(~np.isfinite(current_values)):
            raise SolutionError(
                f'Pre-existing NaNs or infinities found '
                f'in one or more `CHECK` variables '
                f'in period with label: {self.span[t]} (index: {t})'
            )

        # Run any code prior to solution
        with warnings.catch_warnings(record=True) as w:
            if errors == 'raise' and catch_first_error:
                warnings.simplefilter('error')
            else:
                warnings.simplefilter('always')

            try:
                self.solve_t_before(
                    t,
                    errors=errors,
                    catch_first_error=catch_first_error,
                    iteration=0,
                    **kwargs,
                )
            except Exception as e:
                raise SolutionError(
                    f'Error in `solve_t_before()` '
                    f'in period with label: {self.span[t]} (index: {t})'
                ) from e

        for iteration in range(1, max_iter + 1):
            previous_values = current_values.copy()

            with warnings.catch_warnings(record=True) as w:
                if errors == 'raise' and catch_first_error:
                    # Immediately raise an exception in the event of a
                    # numerical solution error
                    warnings.simplefilter('error')
                else:
                    warnings.simplefilter('always')

                try:
                    self._evaluate(
                        t,
                        errors=errors,
                        catch_first_error=catch_first_error,
                        iteration=iteration,
                        **kwargs,
                    )
                except Exception as e:
                    if errors == 'raise':
                        self.status[t] = SolutionStatus.ERROR.value
                        self.iterations[t] = iteration

                    raise SolutionError(
                        f'Error after {iteration} iterations(s) '
                        f'in period with label: {self.span[t]} (index: {t})'
                    ) from e

            current_values = get_check_values()

            # It's possible that the current iteration generated no NaNs or
            # infinities, but the previous one did: check and continue if
            # needed
            if np.any(~np.isfinite(previous_values)):
                continue

            if np.any(~np.isfinite(current_values)):
                if errors == 'raise':
                    self.status[t] = SolutionStatus.ERROR.value
                    self.iterations[t] = iteration

                    raise SolutionError(
                        f'Numerical solution error after {iteration} iteration(s). '
                        f'Non-finite values (NaNs, Infs) generated '
                        f'in period with label: {self.span[t]} (index: {t})'
                    )

                elif errors == 'skip':
                    status = SolutionStatus.SKIPPED.value
                    break

                elif errors == 'ignore':
                    if iteration == max_iter:
                        status = SolutionStatus.FAILED.value
                        break
                    continue

                elif errors == 'replace':
                    if iteration == max_iter:
                        status = SolutionStatus.FAILED.value
                        break
                    else:
                        current_values[~np.isfinite(current_values)] = 0.0
                        continue

                else:
                    raise ValueError(f'Invalid `errors` argument: {errors}')

            if iteration < min_iter:
                continue

            diff = current_values - previous_values

            if np.all(np.abs(diff) < tol):
                with warnings.catch_warnings(record=True) as w:  # noqa: F841
                    if errors == 'raise' and catch_first_error:
                        warnings.simplefilter('error')
                    else:
                        warnings.simplefilter('always')

                    try:
                        self.solve_t_after(
                            t,
                            errors=errors,
                            catch_first_error=catch_first_error,
                            iteration=iteration,
                            **kwargs,
                        )
                    except Exception as e:
                        raise SolutionError(
                            f'Error in `solve_t_after()` '
                            f'in period with label: {self.span[t]} (index: {t})'
                        ) from e

                status = SolutionStatus.SOLVED.value
                break
        else:
            status = SolutionStatus.FAILED.value

        self.status[t] = status
        self.iterations[t] = iteration

        if status == SolutionStatus.FAILED.value and failures == 'raise':
            raise NonConvergenceError(
                f'Solution failed to converge after {iteration} iterations(s) '
                f'in period with label: {self.span[t]} (index: {t})'
            )

        return status == SolutionStatus.SOLVED.value

    def solve_t_before(
        self,
        t: int,
        *,
        errors: str = 'raise',
        catch_first_error: bool = True,
        iteration: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Pre-solution method: This runs each period, before the iterative solution. Over-ride to implement custom behaviour."""

    def solve_t_after(
        self,
        t: int,
        *,
        errors: str = 'raise',
        catch_first_error: bool = True,
        iteration: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Post-solution method: This runs each period, after the iterative solution. Over-ride to implement custom behaviour."""

    def _evaluate(
        self,
        t: int,
        *,
        errors: str = 'raise',
        catch_first_error: bool = True,
        iteration: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Evaluate the system of equations for the period at integer position `t` in the model's `span`.

        Parameters
        ----------
        t : int
            Position in `span` of the period to solve
        errors : str
            User-specified treatment on encountering numerical solution
            errors. Note that it is up to the user's over-riding code to decide
            how to handle this.
        catch_first_error : bool
            If `True` (default) and `errors='raise'`, raise an exception
            (`SolutionError`) on the first numerical error/warning during
            solution. This identifies the problem statement in the stack trace
            without modifying any values at this point.
            If `False`, only check for errors after completing an iteration,
            raising an exception (`SolutionError`) after the fact. This allows
            numerical errors (NaNs, Infs) to propagate through the solution
            before raising the exception.
        iteration : int
            The current iteration count. This is not guaranteed to take a
            non-`None` value if the user has over-ridden the default calling
            `solve_t()` method. Note that it is up to the user's over-riding
            code to decide how to handle this.
        **kwargs :
            Further keyword arguments for solution
        """


# Base class to link models ---------------------------------------------------


class BaseLinker(SolverMixin, ModelInterface):
    """Base class to link economic models."""

    ENDOGENOUS: List[str] = []
    EXOGENOUS: List[str] = []

    PARAMETERS: List[str] = []
    ERRORS: List[str] = []

    NAMES: List[str] = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
    CHECK: List[str] = ENDOGENOUS

    def __init__(
        self,
        submodels: Optional[Dict[Hashable, BaseModel]] = None,
        *,
        span: Optional[Sequence[Hashable]] = None,
        name: Hashable = '_',
        dtype: Any = float,
        default_value: Union[int, float] = 0.0,
        **initial_values: Dict[str, Any],
    ) -> None:
        """Initialise linker with constituent submodels and core model variables.

        Parameters
        ----------
        submodels : dict (or `None`)
            Mapping of submodel identifiers (keys) to submodel instances
            (values)
        span : iterable
            Optional sequence of periods to set the timespan of the linked
            model (rather than infer it from the submodels)
        name :
            Identifier for the model embedded in the linker (in the same way
            that the submodels each have an identifier/key, as in `submodels`)
        dtype : variable type
            Data type to impose on core model variables (in NumPy arrays)
        default_value : number
            Value with which to initialise core model variables
        **initial_values : keyword arguments of variable names and values
            Values with which to initialise specific named core model variables

        Notes
        -----
        For now, the submodels must have identical `span` attributes. Method
        raises an `InitialisationError` if not.
        """
        if submodels is None or len(submodels) == 0:
            submodels = {}

        self.__dict__['submodels'] = submodels
        self.__dict__['name'] = name

        if len(submodels):
            # Get a list of submodel IDs and pop the first submodel as the one
            # to compare all the others against
            identifiers = iter(self.__dict__['submodels'])

            base_name = next(identifiers)
            base = self.__dict__['submodels'][base_name]

            if span is None:
                span = copy.deepcopy(base.span)
            else:
                raise NotImplementedError('Custom `span` handling not yet implemented')

            lags = base.LAGS
            leads = base.LEADS

            # Check for a common span across submodels (error if not) and work
            # out the longest lags and leads
            for id_ in identifiers:
                # Check spans are identical
                comparator = self.__dict__['submodels'][id_]
                if comparator.span != base.span:
                    raise InitialisationError(
                        f'''\
Spans of submodels differ:
 - '{base_name}', {len(base.span):,} period(s): {base.span}
 - '{id_}', {len(comparator.span):,} period(s): {comparator.span}'''
                    )  # fmt: skip

                # Update longest lags and leads
                # fmt: off
                lags =  max(lags, comparator.LAGS)
                leads = max(leads, comparator.LEADS)
                # fmt: on

        else:
            if span is None:
                span = []

            lags = leads = 0

        # Store lags and leads
        self.__dict__['_LAGS'] = lags
        self.__dict__['_LEADS'] = leads

        # Initialise internal store for the core of the model (which is
        # separate from the individual submodels)
        super().__init__(
            span=span, dtype=dtype, default_value=default_value, **initial_values
        )

    @property
    def sizes(self) -> Dict[Hashable, int]:
        """Dictionary of the number of elements in the linker and each model's vector arrays."""
        return dict(
            **{self.__dict__['name']: super().size},
            **{k: v.size for k, v in self.__dict__['submodels'].items()},
        )

    @property
    def size(self) -> int:
        """Total number of elements in the linker and each model's vector arrays."""
        return sum(self.sizes.values())

    @property
    def nbytes(self) -> int:
        """Total bytes consumed by the elements in the linker and each model's vector arrays."""
        return sum(
            [super().nbytes] + [v.nbytes for v in self.__dict__['submodels'].values()]
        )

    @property
    def LAGS(self) -> int:
        """Longest lag among submodels."""
        return self.__dict__['_LAGS']

    @property
    def LEADS(self) -> int:
        """Longest lead among submodels."""
        return self.__dict__['_LEADS']

    def copy(self) -> 'BaseLinker':
        """Return a copy of the current object."""
        copied = self.__class__(
            submodels={
                copy.deepcopy(k): copy.deepcopy(v)
                for k, v in self.__dict__['submodels'].items()
            }
        )

        copied.__dict__.update(
            {
                k: copy.deepcopy(v)
                for k, v in self.__dict__.items()
                if k not in ['submodels']
            }
        )

        return copied

    __copy__ = copy

    def __deepcopy__(self, *args: Any, **kwargs: Any) -> 'BaseLinker':
        return self.copy()

    def reindex(
        self, span: Sequence[Hashable], *args: Any, **kwargs: Any
    ) -> 'BaseLinker':
        # TODO: Still to consider design of the `BaseLinker` equivalent to the
        #       (now implemented) `VectorContainer` and `BaseModel` versions
        raise NotImplementedError(
            '`reindex()` method not yet implemented in `BaseLinker`'
        )

    def to_dataframe(
        self, *, status: bool = True, iterations: bool = True
    ) -> 'pandas.DataFrame':  # noqa: F821
        """Return the values and solution information from the linker as a `pandas` DataFrame. **Requires `pandas`**."""
        return _model_to_dataframe(self, status=status, iterations=iterations)

    def to_dataframes(
        self, *, status: bool = True, iterations: bool = True
    ) -> Dict[Hashable, 'pandas.DataFrame']:  # noqa: F821
        """Return the values and solution information from the linker and its constituent submodels as `pandas` DataFrames. **Requires `pandas`**."""
        return _linker_to_dataframes(self, status=status, iterations=iterations)

    def solve(
        self,
        *,
        start: Optional[Hashable] = None,
        end: Optional[Hashable] = None,
        submodels: Optional[Sequence[Hashable]] = None,
        min_iter: int = 0,
        max_iter: int = 100,
        tol: Union[int, float] = 1e-10,
        offset: int = 0,
        failures: str = 'raise',
        errors: str = 'raise',
        catch_first_error: bool = True,
        **kwargs: Any,
    ) -> Tuple[List[Hashable], List[int], List[bool]]:
        """Solve the linker and its constituent submodels. Use default periods if none provided.

        Parameters
        ----------
        start : element in the model's `span`
            First period to solve. If not given, defaults to the first solvable
            period, taking into account any lags in the model's equations
        end : element in the model's `span`
            Last period to solve. If not given, defaults to the last solvable
            period, taking into account any leads in the model's equations
        submodels : sequence of submodel identifiers (as in `self.submodels.keys()`), default `None`
            Submodels to solve for. If `None` (the default), solve them all.
        min_iter : int
            Minimum number of iterations to solution each period
        max_iter : int
            Maximum number of iterations to solution each period
        tol : int or float
            Tolerance for convergence
        offset : int
            If non-zero, copy an initial set of endogenous values from the
            relative period described by `offset`. For example, `offset=-1`
            initialises each period's solution with the values from the
            previous period.
        failures : str, one of {'raise', 'ignore'}
            Action should the solution fail to converge in a period (by
            reaching the maximum number of iterations, `max_iter`)
             - 'raise' (default): stop immediately and raise a
                                  `NonConvergenceError`
             - 'ignore': continue to the next period
        errors : str, one of {'raise', 'skip', 'ignore', 'replace'}
            User-specified treatment on encountering numerical solution errors
            e.g. NaNs and infinities
             - 'raise' (default): stop immediately and raise a `SolutionError`
                                  [set current period solution status to 'E']
             - 'skip': stop solving the current period and move to the next one
                       [set current period solution status to 'S']
             - 'ignore': continue solving, with no special treatment or action
                         [period solution statuses as usual i.e. '.' or 'F']
             - 'replace': each iteration, replace NaNs and infinities with
                          zeroes
                          [period solution statuses as usual i.e. '.' or 'F']
        catch_first_error : bool
            If `True` (default) and `errors='raise'`, raise an exception
            (`SolutionError`) on the first numerical error/warning during
            solution. This identifies the problem statement in the stack trace
            without modifying any values at this point.
            If `False`, only check for errors after completing an iteration,
            raising an exception (`SolutionError`) after the fact. This allows
            numerical errors (NaNs, Infs) to propagate through the solution
            before raising the exception.
        **kwargs :
            Further keyword arguments to pass on to other methods:
             - `iter_periods()`
             - `solve_t()`

        Returns
        -------
        Three lists, each of length equal to the number of periods to be
        solved:
         - the names of the periods to be solved, as they appear in the model's
           span
         - integers: the index positions of the above periods in the model's
           span
         - bools, one per period: `True` if the period solved successfully;
           `False` otherwise

        Notes
        -----
        With `catch_first_error=True` and `errors='raise'`, solution
        immediately halts on the first error, throwing an exception up the call
        stack. This identifies the problem statement in the stack trace (which
        is helpful for debugging) but also prevents a NaN/Inf entering the
        solution (which may be useful for inspection) and, were there a
        NaN/Inf, from it being propagated (it's not immediately obvious if this
        has any use, though).
        """
        # Error if `min_iter` exceeds `max_iter`
        if min_iter > max_iter:
            raise ValueError(
                f'Value of `min_iter` ({min_iter}) '
                f'cannot exceed value of `max_iter` ({max_iter})'
            )

        period_iter = self.iter_periods(start=start, end=end, **kwargs)

        # fmt: off
        indexes = [None] * len(period_iter)
        labels  = [None] * len(period_iter)
        solved  = [None] * len(period_iter)
        # fmt: on

        for i, (t, period) in enumerate(period_iter):
            indexes[i] = t
            labels[i] = period
            solved[i] = self.solve_t(
                t,
                submodels=submodels,
                min_iter=min_iter,
                max_iter=max_iter,
                tol=tol,
                offset=offset,
                failures=failures,
                errors=errors,
                catch_first_error=catch_first_error,
                **kwargs,
            )

        return labels, indexes, solved

    def solve_t(
        self,
        t: int,
        *,
        submodels: Optional[Sequence[Hashable]] = None,
        min_iter: int = 0,
        max_iter: int = 100,
        tol: Union[int, float] = 1e-10,
        offset: int = 0,
        failures: str = 'raise',
        errors: str = 'raise',
        catch_first_error: bool = True,
        **kwargs: Any,
    ) -> bool:
        """Solve for the period at integer position `t` in the linker's `span`.

        Parameters
        ----------
        t : int
            Position in `span` of the period to solve
        submodels : sequence of submodel identifiers (as in `self.submodels.keys()`), default `None`
            Submodels to solve for. If `None` (the default), solve them all.
        min_iter : int
            Minimum number of iterations to solution each period
        max_iter : int
            Maximum number of iterations to solution each period
        tol : int or float
            Tolerance for convergence
        offset : int
            If non-zero, copy an initial set of endogenous values from the
            period at position `t + offset`. For example, `offset=-1` copies
            the values from the previous period.
        failures : str, one of {'raise', 'ignore'}
            Action should the solution fail to converge (by reaching the
            maximum number of iterations, `max_iter`)
             - 'raise' (default): stop immediately and raise a
                                  `NonConvergenceError`
             - 'ignore': do nothing
        errors : str, one of {'raise', 'skip', 'ignore', 'replace'}
            User-specified treatment on encountering numerical solution errors
            e.g. NaNs and infinities
             - 'raise' (default): stop immediately and raise a `SolutionError`
                                  [set period solution status to 'E']
             - 'skip': stop solving the current period
                       [set period solution status to 'S']
             - 'ignore': continue solving, with no special treatment or action
                         [period solution statuses as usual i.e. '.' or 'F']
             - 'replace': each iteration, replace NaNs and infinities with
                          zeroes
                          [period solution statuses as usual i.e. '.' or 'F']
        catch_first_error : bool
            If `True` (default) and `errors='raise'`, raise an exception
            (`SolutionError`) on the first numerical error/warning during
            solution. This identifies the problem statement in the stack trace
            without modifying any values at this point.
            If `False`, only check for errors after completing an iteration,
            raising an exception (`SolutionError`) after the fact. This allows
            numerical errors (NaNs, Infs) to propagate through the solution
            before raising the exception.
        **kwargs :
            Further keyword arguments to pass to the solution routines

        Returns
        -------
        `True` if the linker solved for the current period; `False` otherwise.

        Notes
        -----
        With `catch_first_error=True` and `errors='raise'`, solution
        immediately halts on the first error, throwing an exception up the call
        stack. This identifies the problem statement in the stack trace (which
        is helpful for debugging) but also prevents a NaN/Inf entering the
        solution (which may be useful for inspection) and, were there a
        NaN/Inf, from it being propagated (it's not immediately obvious if this
        has any use, though).
        """
        if submodels is None:
            submodels = list(self.__dict__['submodels'].keys())

        def get_check_values() -> Dict[Hashable, np.ndarray]:
            """Return NumPy arrays of variable values for the current period, for checking."""
            check_values = {
                '_': np.array([self.__dict__['_' + name][t] for name in self.CHECK]),
            }

            for k, submodel in self.submodels.items():
                if k in submodels:
                    check_values[k] = np.array(
                        [submodel[name][t] for name in submodel.CHECK]
                    )

            return check_values

        status = SolutionStatus.UNSOLVED.value
        current_values = get_check_values()

        # Set iteration counters to zero for submodels to solve (also use this
        # as an opportunity to make sure specified submodels are valid)
        for name in submodels:
            try:
                submodel = self.__dict__['submodels'][name]
            except KeyError:
                raise KeyError(f"'{name}' not found in list of submodels")

            submodel.iterations[t] = 0

        # Run any code prior to solution
        self.solve_t_before(
            t,
            submodels=submodels,
            errors=errors,
            catch_first_error=catch_first_error,
            iteration=0,
            **kwargs,
        )

        for iteration in range(1, max_iter + 1):
            previous_values = copy.deepcopy(current_values)

            self.evaluate_t_before(
                t,
                submodels=submodels,
                errors=errors,
                catch_first_error=catch_first_error,
                iteration=iteration,
                **kwargs,
            )
            self.evaluate_t(
                t,
                submodels=submodels,
                errors=errors,
                catch_first_error=catch_first_error,
                iteration=iteration,
                **kwargs,
            )
            self.evaluate_t_after(
                t,
                submodels=submodels,
                errors=errors,
                catch_first_error=catch_first_error,
                iteration=iteration,
                **kwargs,
            )

            current_values = get_check_values()

            if iteration < min_iter:
                continue

            diff = {k: current_values[k] - previous_values[k] for k in current_values}
            diff_squared = {k: v**2 for k, v in diff.items()}

            if all(np.all(v < tol) for v in diff_squared.values()):
                status = SolutionStatus.SOLVED.value
                self.solve_t_after(
                    t,
                    submodels=submodels,
                    errors=errors,
                    catch_first_error=catch_first_error,
                    iteration=iteration,
                    **kwargs,
                )
                break

        else:
            status = SolutionStatus.FAILED.value

        self.status[t] = status
        self.iterations[t] = iteration

        for name in submodels:
            submodel = self.__dict__['submodels'][name]
            submodel.status[t] = status

        if status == SolutionStatus.FAILED.value and failures == 'raise':
            raise NonConvergenceError(
                f'Solution failed to converge after {iteration} iterations(s) '
                f'in period with label: {self.span[t]} (index: {t})'
            )

        return status == SolutionStatus.SOLVED.value

    def evaluate_t(
        self,
        t: int,
        *,
        submodels: Optional[Sequence[Hashable]] = None,
        errors: str = 'raise',
        catch_first_error: bool = True,
        iteration: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Evaluate the system of equations for the period at integer position `t` in the linker's `span`.

        Parameters
        ----------
        t : int
            Position in `span` of the period to solve
        submodels : sequence of submodel identifiers (as in `self.submodels.keys()`), default `None`
            Submodels to evaluate. If `None` (the default), evaluate them all.
        errors : str
            User-specified treatment on encountering numerical solution
            errors, as passed to the individual submodels.
        catch_first_error : bool
            If `True` (default) and `errors='raise'`, raise an exception
            (`SolutionError`) on the first numerical error/warning during
            solution. This identifies the problem statement in the stack trace
            without modifying any values at this point.
            If `False`, only check for errors after completing an iteration,
            raising an exception (`SolutionError`) after the fact. This allows
            numerical errors (NaNs, Infs) to propagate through the solution
            before raising the exception.        iteration : int
            The current iteration count. This is not guaranteed to take a
            non-`None` value if the user has over-ridden the default calling
            `solve_t()` method. Note that it is up to the individual submodels'
            over-riding code to decide how to handle this.
        **kwargs :
            Further keyword arguments for solution

        Notes
        -----
        With `catch_first_error=True` and `errors='raise'`, solution
        immediately halts on the first error, throwing an exception up the call
        stack. This identifies the problem statement in the stack trace (which
        is helpful for debugging) but also prevents a NaN/Inf entering the
        solution (which may be useful for inspection) and, were there a
        NaN/Inf, from it being propagated (it's not immediately obvious if this
        has any use, though).
        """
        if submodels is None:
            submodels = list(self.__dict__['submodels'].keys())

        for name in submodels:
            submodel = self.__dict__['submodels'][name]

            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                submodel._evaluate(
                    t,
                    errors=errors,
                    catch_first_error=catch_first_error,
                    iteration=iteration,
                    **kwargs,
                )

            submodel.iterations[t] += 1

    def solve_t_before(
        self,
        t: int,
        *,
        submodels: Optional[Sequence[Hashable]] = None,
        errors: str = 'raise',
        catch_first_error: bool = True,
        iteration: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Pre-solution method: This runs each period, before the iterative solution. Over-ride to implement custom linker behaviour."""

    def solve_t_after(
        self,
        t: int,
        *,
        submodels: Optional[Sequence[Hashable]] = None,
        errors: str = 'raise',
        catch_first_error: bool = True,
        iteration: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Post-solution method: This runs each period, after the iterative solution. Over-ride to implement custom linker behaviour."""

    def evaluate_t_before(
        self,
        t: int,
        *,
        submodels: Optional[Sequence[Hashable]] = None,
        errors: str = 'raise',
        catch_first_error: bool = True,
        iteration: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Evaluate any linker equations before solving the individual submodels. Over-ride to implement custom linker behaviour."""

    def evaluate_t_after(
        self,
        t: int,
        *,
        submodels: Optional[Sequence[Hashable]] = None,
        errors: str = 'raise',
        catch_first_error: bool = True,
        iteration: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Evaluate any linker equations after solving the individual submodels. Over-ride to implement custom linker behaviour."""
