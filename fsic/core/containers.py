# -*- coding: utf-8 -*-
"""
Core `VectorContainer` class to handle collections of one-dimensional data.
"""

import copy
import difflib
import re
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..exceptions import (
    DimensionError,
    DuplicateNameError,
)
from ..functions import builtins as _builtins


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

    @staticmethod
    def _locate_period_in_span_fallback(period: Hashable, span: np.ndarray) -> int:
        """Fallback (static) location method, should other `span`-indexing methods fail."""
        # Convert `span` to a NumPy array of type `object` and locate matches
        locations = np.asarray(np.asarray(span, dtype=object) == period).nonzero()

        # For now(?), only support one-dimensional array-likes
        assert len(locations) == 1
        positions = locations[0]  # Take first (sole) set of axis indexes only

        # No matches: `period` not defined
        if len(positions) == 0:
            raise KeyError(period)

        # One match: Unpack and return
        if len(positions) == 1:
            return positions[0]

        raise NotImplementedError('Multiple matches not supported')

    _VALID_INDEX_METHODS: List[Union[str, Callable]] = [
        'get_loc',
        'index',
        _locate_period_in_span_fallback,
    ]

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
                    f"a sequence of identical length to `span` "
                    f"(expected {len(self.__dict__['span'])} element[s] "
                    f"but found {value_as_array.shape[0]})"
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
        for i, method in enumerate(self._VALID_INDEX_METHODS):
            if isinstance(method, str):
                if hasattr(self.__dict__['span'], method):
                    index_function = getattr(self.__dict__['span'], method)

                    try:
                        return index_function(period)
                    except Exception as e:
                        raise KeyError(period) from e

            elif isinstance(method, Callable):
                try:
                    return method(period, self.__dict__['span'])
                except Exception as e:
                    raise KeyError(period) from e

            else:
                raise TypeError(
                    f'Unrecognised type ({type(method)}) of search method '
                    f'with index {i} in `self._VALID_INDEX_METHODS`: {method}'
                )

        raise AttributeError(
            f'Unable to find valid search method in `span`; '
            f'expected one of the following, as listed in `self._VALID_INDEX_METHODS`: '
            f'{self._VALID_INDEX_METHODS}'
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
        self,
        span: Sequence[Hashable],
        *,
        fill_value: Any = None,
        strict: Optional[bool] = None,
        **fill_values: Any,
    ) -> 'VectorContainer':
        """Return a copy of the current object, adjusted to match `span`. Values in overlapping periods between the old and new objects are preserved (copied over).

        Parameters
        ----------
        span : iterable
            Sequence of periods defining the span of the object to be returned
        fill_value :
            Default fill value for new periods
        strict : bool
            If `True`, raise a `KeyError` if `fill_values` refers to variables
            not defined in the current object. Ignore if `False`.
            If `None`, use the current value of the object's `strict`
            attribute.
        **fill_values :
            Variable-specific fill value(s)
        """
        if strict is None:
            strict = self.strict

        if strict:
            # Check for variables in `fill_values` but not in the object index
            undefined_variables = set(fill_values.keys()) - set(self.index)
            if undefined_variables:
                raise KeyError(
                    f"Found {len(undefined_variables)} undefined variable(s) "
                    f"with `strict=True`: {', '.join(sorted(undefined_variables))}"
                )

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
