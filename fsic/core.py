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

from collections import Counter
import copy
import enum
from typing import Any, Dict, Hashable, Iterator, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np

from .exceptions import DimensionError, DuplicateNameError, EvalError, InitialisationError, NonConvergenceError, SolutionError
from .tools import model_to_dataframe as _model_to_dataframe
from .tools import linker_to_dataframes as _linker_to_dataframes


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
        self.__dict__['span'] = span
        self.__dict__['index'] = []
        self.__dict__['_strict'] = strict

    def add_variable(self, name: str, value: Union[Any, Sequence[Any]], *, dtype: Any = None) -> None:
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
                "'{}' is already defined in the current object".format(name))

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
            raise DimensionError("Invalid assignment for '{}': "
                                 "must be either a single value or "
                                 "a sequence of identical length to `span`"
                                 "(expected {} elements)".format(
                                     name, len(self.__dict__['span'])))

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

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__['index']:
            return self.__dict__['_' + name]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Union[Any, Sequence[Any]]) -> None:
        # Error on attempt to add an attribute if `strict=True`
        if name != 'strict' and self.__dict__['_strict'] and name not in self.__dict__['index']:
            raise AttributeError("Unable to add new attribute '{}' with `strict=True`".format(name))

        if name not in self.__dict__['index']:
            super.__setattr__(self, name, value)
            return

        elif isinstance(value, Sequence) and not isinstance(value, str):
            value_as_array = np.array(value,
                                      dtype=self.__dict__['_' + name].dtype)

            if value_as_array.shape[0] != len(self.__dict__['span']):
                raise DimensionError("Invalid assignment for '{}': "
                                     "must be either a single value or "
                                     "a sequence of identical length to `span`"
                                     "(expected {} elements)".format(
                                         name, len(self.__dict__['span'])))

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
                return index_function(period)
        else:
            raise AttributeError(
                f'Unable to find valid search method in `span`; '
                f'expected one of: {self._VALID_INDEX_METHODS}')

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
            # `pandas`
            stop_location += 1

        return start_location, stop_location, step

    def __getitem__(self, key: Union[str, Tuple[str, Union[Hashable, slice]]]) -> Any:
        # `key` is a string (variable name): return the corresponding array
        if isinstance(key, str):
            if key not in self.__dict__['index']:
                raise KeyError("'{}' not recognised as a variable name".format(key))
            return self.__getattr__(key)

        # `key` is a tuple (variable name plus index): return the selected
        # elements of the corresponding array
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError(
                    'Invalid index: must be of length one (variable name) '
                    'or length two (variable name, span index)')

            # Unpack the key
            name: str
            index: Union[Hashable, slice]
            name, index = key

            # Get the full array
            if name not in self.__dict__['index']:
                raise KeyError("'{}' not recognised as a variable name".format(name))
            values = self.__getattr__(name)

            # Extract and return the relevant subset
            if isinstance(index, slice):
                start_location, stop_location, step = self._resolve_period_slice(index)
                return values[start_location:stop_location:step]

            else:
                location = self._locate_period_in_span(index)
                return values[location]

        raise TypeError('Invalid index type ({}): `{}`'.format(type(key), key))

    def __setitem__(self, key: Union[str, Tuple[str, Union[Hashable, slice]]], value: Union[Any, Sequence[Any]]) -> None:
        # `key` is a string (variable name): update the corresponding array in
        # its entirety
        if isinstance(key, str):
            if key not in self.__dict__['index']:
                raise KeyError("'{}' not recognised as a variable name".format(key))
            self.__setattr__(key, value)
            return

        # `key` is a tuple (variable name plus index): update selected elements
        # of the corresponding array
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError(
                    'Invalid index: must be of length one (variable name) '
                    'or length two (variable name, span index)')

            # Unpack the key
            name: str
            index: Union[Hashable, slice]
            name, index = key

            # Modify the relevant subset
            if isinstance(index, slice):
                start_location, stop_location, step = self._resolve_period_slice(index)
                self.__dict__['_' + name][start_location:stop_location:step] = value
                return

            else:
                location = self._locate_period_in_span(index)
                self.__dict__['_' + name][location] = value
                return

        raise TypeError('Invalid index type ({}): `{}`'.format(type(key), key))

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
        copied.__dict__.update(
            {k: copy.deepcopy(v)
             for k, v in self.__dict__.items()})
        return copied

    __copy__ = copy

    def __deepcopy__(self, *args, **kwargs) -> 'VectorContainer':
        return self.copy()

    def __dir__(self) -> List[str]:
        return sorted(
            dir(type(self)) + self.__dict__['index'] + ['span', 'index'])

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
        return np.array([
            self.__getattribute__('_' + name) for name in self.__dict__['index']
        ])

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
                    'Replacement array is of shape {} but expected shape is {}'
                    .format(new_values.shape, self.values.shape))

            for name, series in zip(self.index, new_values):
                self.__setattr__(name, series.astype(self.__getattribute__('_' + name).dtype))

        else:
             for name in self.index:
                 series = self.__getattribute__('_' + name)
                 self.__setattr__(name, np.full(series.shape, new_values, dtype=series.dtype))

    def eval(self, expression: str) -> np.ndarray:
        raise NotImplementedError('`eval()` method not implemented yet')

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

    def __init__(self, span: Sequence[Hashable], *, strict: bool = False, dtype: Any = float, default_value: Union[int, float] = 0.0, **initial_values: Dict[str, Any]) -> None:
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

        # Store the `dtype` as the default for future values e.g. using
        # `add_variable()` after initialisation
        self.__dict__['dtype'] = dtype

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
                'Found multiple instances of the following variable(s) in `NAMES`: {}'
                .format(', '.join(duplicates)))

        # If `strict`, check that any variables set via `initial_values` match
        # an entry in `names` (which is itself a copy of the class's `NAMES`
        # attribute)
        if strict:
            permitted_names = set(names)
            passed_names = set(initial_values.keys())
            invalid_names = passed_names - permitted_names

            if len(invalid_names) > 0:
                raise InitialisationError(
                    'Cannot add unlisted variables (i.e. variables not in `NAMES`) '
                    'when `strict=True` - found {} instance(s): {}'
                    .format(len(invalid_names), ', '.join(invalid_names)))

        # Add model variables
        self.__dict__['names'] = names

        for name in self.__dict__['names']:
            super().add_variable(name,
                                 initial_values.get(name, default_value),
                                 dtype=self.__dict__['dtype'])

    def add_variable(self, name: str, value: Union[Any, Sequence[Any]], *, dtype: Any = None) -> None:
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
        self.__dict__['names'] += name

    @property
    def size(self) -> int:
        """Number of elements in the model's vector arrays."""
        return len(self.__dict__['names']) * len(self.__dict__['span'])

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__['names']

    def __dir__(self) -> List[str]:
        return sorted(super().__dir__() + ['names'])

    @property
    def values(self) -> np.ndarray:
        """Model variable values as a 2D (names x span) array."""
        return np.array([
            self.__getattribute__('_' + name) for name in self.names
        ])

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
                    'Replacement array is of shape {} but expected shape is {}'
                    .format(new_values.shape, self.values.shape))

            for name, series in zip(self.names, new_values):
                self.__setattr__(name, series.astype(self.__getattribute__('_' + name).dtype))

        else:
            for name in self.names:
                series = self.__getattribute__('_' + name)
                self.__setattr__(name, np.full(series.shape, new_values, dtype=series.dtype))


# Mixin to define (but not fully implement) solver behaviour ------------------

class PeriodIter:
    """Iterator of (index, label) pairs returned by `BaseModel.iter_periods()`. Compatible with `len()`."""

    def __init__(self, *args):
        self._length = len(args[0])
        self._iter = list(zip(*args))

    def __iter__(self):
        yield from self._iter

    def __next__(self):
        return next(self._iter)

    def __len__(self):
        return self._length

class SolverMixin:
    """Requires `self.span` (a `Sequence` of `Hashable`s) as an attribute."""

    LAGS: int = 0
    LEADS: int = 0

    def iter_periods(self, *, start: Optional[Hashable] = None, end: Optional[Hashable] = None, **kwargs: Any) -> Iterator[Tuple[int, Hashable]]:
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
        # Default start and end periods
        if start is None:
            start = self.span[self.LAGS]
        if end is None:
            end = self.span[-1 - self.LEADS]

        # Convert to an integer range
        indexes = range(self._locate_period_in_span(start),
                        self._locate_period_in_span(end) + 1)

        return PeriodIter(indexes, self.span[indexes.start:indexes.stop])

    def solve(self, *, start: Optional[Hashable] = None, end: Optional[Hashable] = None, min_iter: int = 0, max_iter: int = 100, tol: Union[int, float] = 1e-10, offset: int = 0, failures: str = 'raise', errors: str = 'raise', catch_first_error: bool = True, **kwargs: Dict[str, Any]) -> Tuple[List[Hashable], List[int], List[bool]]:
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
        kwargs :
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
        Before version 0.8.0, error handling worked as follows during solution:

          1. Call `_evaluate()` (i.e. solve one iteration), temporarily
             suppressing all warnings (numerical solution errors). Results
             arising from warnings (NaNs, Infs) enter the solution at this
             point and propagate if these results feed into other equations
             during the iteration.
          2. If any warnings came up during the call to `_evaluate()`, handle
             according to the value of `errors`:
              - 'raise':   raise an exception (`SolutionError`)
              - 'skip':    move immediately to the next period, retaining the
                           NaNs/Infs
              - 'ignore':  proceed to the next iteration (in the hope that
                           NaNs/Infs are eventually replaced with finite
                           values)
              - 'replace': replace NaNs/Infs with zeroes before proceeding to
                           the next iteration (in the hope that this is enough
                           to eventually generate finite values throughout the
                           solution)

        However, this behaviour (even if straightforward to replicate in
        Fortran) comes at the expense of knowing which equation(s) led to an
        error. All we can see is how non-finite values propagated through the
        solution from a single pass/iteration. Moreover, by allowing the entire
        system to run through each time, there's no guarantee that 'ignore' or
        'replace' will help to solve the model, should the same pattern of NaNs
        and Infs repeat each iteration.

        Consequently, the above treatment is no longer the default behaviour in
        version 0.8.0, which introduces the keyword argument
        `catch_first_error` (default `True`).

        With `catch_first_error=True` and `errors='raise'`, solution
        immediately halts on the first error, throwing an exception up the call
        stack. This identifies the problem statement in the stack trace (which
        is helpful for debugging) but also prevents a NaN/Inf entering the
        solution (which may be useful for inspection) and, were there a
        NaN/Inf, from it being propagated (it's not immediately obvious if this
        has any use, though).

        With `catch_first_error=False`, the behaviour returns to the pre-0.8.0
        treatment.
        """
        # Error if `min_iter` exceeds `max_iter`
        if min_iter > max_iter:
            raise ValueError(
                'Value of `min_iter` ({}) cannot exceed value of `max_iter` ({})'.format(
                    min_iter, max_iter))

        period_iter = self.iter_periods(start=start, end=end, **kwargs)

        indexes = [None] * len(period_iter)
        labels  = [None] * len(period_iter)
        solved  = [None] * len(period_iter)

        for i, (t, period) in enumerate(period_iter):
            indexes[i] = t
            labels[i] = period
            solved[i] = self.solve_t(t, min_iter=min_iter, max_iter=max_iter, tol=tol, offset=offset,
                                     failures=failures, errors=errors, catch_first_error=catch_first_error, **kwargs)

        return labels, indexes, solved

    def solve_period(self, period: Hashable, *, min_iter: int = 0, max_iter: int = 100, tol: Union[int, float] = 1e-10, offset: int = 0, failures: str = 'raise', errors: str = 'raise', catch_first_error: bool = True, **kwargs: Dict[str, Any]) -> bool:
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
        kwargs :
            Further keyword arguments to pass to the solution routines

        Returns
        -------
        `True` if the model solved for the current period; `False` otherwise.

        Notes
        -----
        Before version 0.8.0, error handling worked as follows during solution:

          1. Call `_evaluate()` (i.e. solve one iteration), temporarily
             suppressing all warnings (numerical solution errors). Results
             arising from warnings (NaNs, Infs) enter the solution at this
             point and propagate if these results feed into other equations
             during the iteration.
          2. If any warnings came up during the call to `_evaluate()`, handle
             according to the value of `errors`:
              - 'raise':   raise an exception (`SolutionError`)
              - 'skip':    move immediately to the next period, retaining the
                           NaNs/Infs
              - 'ignore':  proceed to the next iteration (in the hope that
                           NaNs/Infs are eventually replaced with finite
                           values)
              - 'replace': replace NaNs/Infs with zeroes before proceeding to
                           the next iteration (in the hope that this is enough
                           to eventually generate finite values throughout the
                           solution)

        However, this behaviour (even if straightforward to replicate in
        Fortran) comes at the expense of knowing which equation(s) led to an
        error. All we can see is how non-finite values propagated through the
        solution from a single pass/iteration. Moreover, by allowing the entire
        system to run through each time, there's no guarantee that 'ignore' or
        'replace' will help to solve the model, should the same pattern of NaNs
        and Infs repeat each iteration.

        Consequently, the above treatment is no longer the default behaviour in
        version 0.8.0, which introduces the keyword argument
        `catch_first_error` (default `True`).

        With `catch_first_error=True` and `errors='raise'`, solution
        immediately halts on the first error, throwing an exception up the call
        stack. This identifies the problem statement in the stack trace (which
        is helpful for debugging) but also prevents a NaN/Inf entering the
        solution (which may be useful for inspection) and, were there a
        NaN/Inf, from it being propagated (it's not immediately obvious if this
        has any use, though).

        With `catch_first_error=False`, the behaviour returns to the pre-0.8.0
        treatment.
        """
        t = self._locate_period_in_span(period)

        if not isinstance(t, int):
            raise IndexError(
                f'Unable to convert `period` to a valid integer '
                f'(a single location in `self.span`). '
                f'`period` resolved to type {type(t)} with value {t}')

        return self.solve_t(t, min_iter=min_iter, max_iter=max_iter, tol=tol, offset=offset, failures=failures, errors=errors, catch_first_error=catch_first_error, **kwargs)

    def solve_t(self, t: int, *, min_iter: int = 0, max_iter: int = 100, tol: Union[int, float] = 1e-10, offset: int = 0, failures: str = 'raise', errors: str = 'raise', catch_first_error: bool = True, **kwargs: Dict[str, Any]) -> bool:
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
        kwargs :
            Further keyword arguments to pass to the solution routines

        Returns
        -------
        `True` if the model solved for the current period; `False` otherwise.

        Notes
        -----
        Before version 0.8.0, error handling worked as follows during solution:

          1. Call `_evaluate()` (i.e. solve one iteration), temporarily
             suppressing all warnings (numerical solution errors). Results
             arising from warnings (NaNs, Infs) enter the solution at this
             point and propagate if these results feed into other equations
             during the iteration.
          2. If any warnings came up during the call to `_evaluate()`, handle
             according to the value of `errors`:
              - 'raise':   raise an exception (`SolutionError`)
              - 'skip':    move immediately to the next period, retaining the
                           NaNs/Infs
              - 'ignore':  proceed to the next iteration (in the hope that
                           NaNs/Infs are eventually replaced with finite
                           values)
              - 'replace': replace NaNs/Infs with zeroes before proceeding to
                           the next iteration (in the hope that this is enough
                           to eventually generate finite values throughout the
                           solution)

        However, this behaviour (even if straightforward to replicate in
        Fortran) comes at the expense of knowing which equation(s) led to an
        error. All we can see is how non-finite values propagated through the
        solution from a single pass/iteration. Moreover, by allowing the entire
        system to run through each time, there's no guarantee that 'ignore' or
        'replace' will help to solve the model, should the same pattern of NaNs
        and Infs repeat each iteration.

        Consequently, the above treatment is no longer the default behaviour in
        version 0.8.0, which introduces the keyword argument
        `catch_first_error` (default `True`).

        With `catch_first_error=True` and `errors='raise'`, solution
        immediately halts on the first error, throwing an exception up the call
        stack. This identifies the problem statement in the stack trace (which
        is helpful for debugging) but also prevents a NaN/Inf entering the
        solution (which may be useful for inspection) and, were there a
        NaN/Inf, from it being propagated (it's not immediately obvious if this
        has any use, though).

        With `catch_first_error=False`, the behaviour returns to the pre-0.8.0
        treatment.
        """
        raise NotImplementedError('Method must be over-ridden by a child class')


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

    def __init__(self, span: Sequence[Hashable], *, engine: str = 'python', strict: bool = False, dtype: Any = float, default_value: Union[int, float] = 0.0, **initial_values: Dict[str, Any]) -> None:
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
        self.__dict__['engine'] = engine

        super().__init__(span=span,
                         strict=strict,
                         dtype=dtype,
                         default_value=default_value,
                         **initial_values)

    @classmethod
    def from_dataframe(cls: 'BaseModel', data: 'pandas.DataFrame', *args, **kwargs) -> 'BaseModel':
        """Initialise the model by taking the index and values from a `pandas` DataFrame(-like).

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
        return cls(list(data.index),
                   *args,
                   **{k: v.values for k, v in data.items()},
                   **kwargs)

    def to_dataframe(self, *, status: bool = True, iterations: bool = True) -> 'pandas.DataFrame':
        """Return the values and solution information from the model as a `pandas` DataFrame. **Requires `pandas`**."""
        return _model_to_dataframe(self, status=status, iterations=iterations)

    def solve_t(self, t: int, *, min_iter: int = 0, max_iter: int = 100, tol: Union[int, float] = 1e-10, offset: int = 0, failures: str = 'raise', errors: str = 'raise', catch_first_error: bool = True, **kwargs: Dict[str, Any]) -> bool:
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
        kwargs :
            Further keyword arguments to pass to the solution routines

        Returns
        -------
        `True` if the model solved for the current period; `False` otherwise.

        Notes
        -----
        As of version 0.3.0, fsic provides (some) support (escape hatches) for
        numerical errors in solution.

        For example, there may be an equation that involves a division
        operation but the equation that determines the divisor follows
        later. If that divisor was initialised to zero, this leads to a
        divide-by-zero operation that NumPy evaluates to a NaN. This becomes
        problematic if the NaNs then propagate through the solution. Similar
        problems come about with infinities e.g. from log(0).

        The `solve_t()` method now catches such operations (after a full pass
        through / iteration over the system of equations).


        Before version 0.8.0, error handling worked as follows during solution:

          1. Call `_evaluate()` (i.e. solve one iteration), temporarily
             suppressing all warnings (numerical solution errors). Results
             arising from warnings (NaNs, Infs) enter the solution at this
             point and propagate if these results feed into other equations
             during the iteration.
          2. If any warnings came up during the call to `_evaluate()`, handle
             according to the value of `errors`:
              - 'raise':   raise an exception (`SolutionError`)
              - 'skip':    move immediately to the next period, retaining the
                           NaNs/Infs
              - 'ignore':  proceed to the next iteration (in the hope that
                           NaNs/Infs are eventually replaced with finite
                           values)
              - 'replace': replace NaNs/Infs with zeroes before proceeding to
                           the next iteration (in the hope that this is enough
                           to eventually generate finite values throughout the
                           solution)

        However, this behaviour (even if straightforward to replicate in
        Fortran) comes at the expense of knowing which equation(s) led to an
        error. All we can see is how non-finite values propagated through the
        solution from a single pass/iteration. Moreover, by allowing the entire
        system to run through each time, there's no guarantee that 'ignore' or
        'replace' will help to solve the model, should the same pattern of NaNs
        and Infs repeat each iteration.

        Consequently, the above treatment is no longer the default behaviour in
        version 0.8.0, which introduces the keyword argument
        `catch_first_error` (default `True`).

        With `catch_first_error=True` and `errors='raise'`, solution
        immediately halts on the first error, throwing an exception up the call
        stack. This identifies the problem statement in the stack trace (which
        is helpful for debugging) but also prevents a NaN/Inf entering the
        solution (which may be useful for inspection) and, were there a
        NaN/Inf, from it being propagated (it's not immediately obvious if this
        has any use, though).

        With `catch_first_error=False`, the behaviour returns to the pre-0.8.0
        treatment.
        """
        def get_check_values() -> np.ndarray:
            """Return a 1D NumPy array of variable values for checking in the current period."""
            return np.array([
                self.__dict__['_' + name][t] for name in self.CHECK
            ])

        # Error if `min_iter` exceeds `max_iter`
        if min_iter > max_iter:
            raise ValueError(
                'Value of `min_iter` ({}) cannot exceed value of `max_iter` ({})'.format(
                    min_iter, max_iter))

        # Optionally copy initial values from another period
        if offset:
            t_check = t
            if t_check < 0:
                t_check += len(self.span)

            # Error if `offset` points prior to the current model span
            if t_check + offset < 0:
                raise IndexError(
                    '`offset` argument ({}) for position `t` ({}) '
                    'implies a period before the span of the current model instance: '
                    '{} + {} -> position {} < 0'.format(
                        offset, t, offset, t, offset + t_check))

            # Error if `offset` points beyond the current model span
            if t_check + offset >= len(self.span):
                raise IndexError(
                    '`offset` argument ({}) for position `t` ({}) '
                    'implies a period beyond the span of the current model instance: '
                    '{} + {} -> position {} >= {} periods in span'.format(
                        offset, t, offset, t, offset + t_check, len(self.span)))

            for name in self.ENDOGENOUS:
                self.__dict__['_' + name][t] = self.__dict__['_' + name][t + offset]

        status = SolutionStatus.UNSOLVED.value
        current_values = get_check_values()

        # Raise an exception if there are pre-existing NaNs or infinities, and
        # error checking is at its strictest ('raise')
        if errors == 'raise' and np.any(~np.isfinite(current_values)):
            raise SolutionError(
                'Pre-existing NaNs or infinities found '
                'in one or more `CHECK` variables '
                'in period with label: {} (index: {})'
                .format(self.span[t], t))

        # Run any code prior to solution
        with warnings.catch_warnings(record=True) as w:
            if errors == 'raise' and catch_first_error:
                warnings.simplefilter('error')
            else:
                warnings.simplefilter('always')

            try:
                self.solve_t_before(t, errors=errors, catch_first_error=catch_first_error, iteration=0, **kwargs)
            except Exception as e:
                raise SolutionError(
                    'Error in `solve_t_before()` '
                    'in period with label: {} (index: {})'
                    .format(self.span[t], t)) from e

        for iteration in range(1, max_iter + 1):
            previous_values = current_values.copy()

            with warnings.catch_warnings(record=True) as w:
                if errors == 'raise' and catch_first_error:
                    # New in version 0.8.0:
                    # Immediately raise an exception in the event of a
                    # numerical solution error
                    warnings.simplefilter('error')
                else:
                    warnings.simplefilter('always')

                try:
                    self._evaluate(t, errors=errors, catch_first_error=catch_first_error, iteration=iteration, **kwargs)
                except Exception as e:
                    if errors == 'raise':
                        self.status[t] = SolutionStatus.ERROR.value
                        self.iterations[t] = iteration

                    raise SolutionError(
                        'Error after {} iterations(s) '
                        'in period with label: {} (index: {})'
                        .format(iteration, self.span[t], t)) from e

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
                        'Numerical solution error after {} iteration(s). '
                        'Non-finite values (NaNs, Infs) generated '
                        'in period with label: {} (index: {})'
                        .format(iteration, self.span[t], t))

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
                    raise ValueError('Invalid `errors` argument: {}'.format(errors))

            if iteration < min_iter:
                continue

            diff = current_values - previous_values

            if np.all(np.abs(diff) < tol):
                with warnings.catch_warnings(record=True) as w:
                    if errors == 'raise' and catch_first_error:
                        warnings.simplefilter('error')
                    else:
                        warnings.simplefilter('always')

                    try:
                        self.solve_t_after(t, errors=errors, catch_first_error=catch_first_error, iteration=iteration, **kwargs)
                    except Exception as e:
                        raise SolutionError(
                            'Error in `solve_t_after()` '
                            'in period with label: {} (index: {})'
                            .format(self.span[t], t)) from e

                status = SolutionStatus.SOLVED.value
                break
        else:
            status = SolutionStatus.FAILED.value

        self.status[t] = status
        self.iterations[t] = iteration

        if status == SolutionStatus.FAILED.value and failures == 'raise':
            raise NonConvergenceError(
                'Solution failed to converge after {} iterations(s) '
                'in period with label: {} (index: {})'
                .format(iteration, self.span[t], t))

        return status == SolutionStatus.SOLVED.value

    def solve_t_before(self, t: int, *, errors: str = 'raise', catch_first_error: bool = True, iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
        """Pre-solution method: This runs each period, before the iterative solution. Over-ride to implement custom behaviour."""
        pass

    def solve_t_after(self, t: int, *, errors: str = 'raise', catch_first_error: bool = True, iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
        """Post-solution method: This runs each period, after the iterative solution. Over-ride to implement custom behaviour."""
        pass

    def _evaluate(self, t: int, *, errors: str = 'raise', catch_first_error: bool = True, iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
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
        kwargs :
            Further keyword arguments for solution
        """
        raise NotImplementedError('Method must be over-ridden by a child class')


# Base class to link models ---------------------------------------------------

class BaseLinker(SolverMixin, ModelInterface):
    """Base class to link economic models."""

    ENDOGENOUS: List[str] = []
    EXOGENOUS: List[str] = []

    PARAMETERS: List[str] = []
    ERRORS: List[str] = []

    NAMES: List[str] = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
    CHECK: List[str] = ENDOGENOUS

    def __init__(self, submodels: Dict[Hashable, BaseModel], *, name: Hashable = '_', dtype: Any = float, default_value: Union[int, float] = 0.0, **initial_values: Dict[str, Any]) -> None:
        """Initialise linker with constituent submodels and core model variables.

        Parameters
        ----------
        submodels : dict
            Mapping of submodel identifiers (keys) to submodel instances
            (values)
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
        self.__dict__['submodels'] = submodels
        self.__dict__['name'] = name

        # Get a list of submodel IDs and pop the first submodel as the one to
        # compare all the others against
        identifiers = iter(self.__dict__['submodels'])

        base_name = next(identifiers)
        base = self.__dict__['submodels'][base_name]

        lags = base.LAGS
        leads = base.LEADS

        # Check for a common span across submodels (error if not) and work out
        # the longest lags and leads
        for name in identifiers:
            # Check spans are identical
            comparator = self.__dict__['submodels'][name]
            if comparator.span != base.span:
                raise InitialisationError('''\
Spans of submodels differ:
 - '{}', {:,} period(s): {}
 - '{}', {:,} period(s): {}'''.format(
     base_name, len(base.span), base.span,
     name, len(comparator.span), comparator.span))

            # Update longest lags and leads
            lags = max(lags, comparator.LAGS)
            leads = max(leads, comparator.LEADS)

        # Store lags and leads
        self.__dict__['_LAGS'] = lags
        self.__dict__['_LEADS'] = leads

        # Initialise internal store for the core of the model (which is
        # separate from the individual submodels)
        super().__init__(span=base.span,
                         dtype=dtype,
                         default_value=default_value,
                         **initial_values)

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
            [super().nbytes] +
            [v.nbytes for v in self.__dict__['submodels'].values()]
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
            submodels={copy.deepcopy(k): copy.deepcopy(v)
                       for k, v in self.__dict__['submodels'].items()}
        )

        copied.__dict__.update(
            {k: copy.deepcopy(v)
             for k, v in self.__dict__.items()
             if k not in ['submodels']}
        )

        return copied

    __copy__ = copy

    def __deepcopy__(self, *args, **kwargs) -> 'BaseLinker':
        return self.copy()

    def to_dataframes(self, *, status: bool = True, iterations: bool = True) -> Dict[Hashable, 'pandas.DataFrame']:
        """Return the values and solution information from the linker and its constituent submodels as `pandas` DataFrames. **Requires `pandas`**."""
        return _linker_to_dataframes(self, status=status, iterations=iterations)

    def solve(self, *, start: Optional[Hashable] = None, end: Optional[Hashable] = None, submodels: Optional[Sequence[Hashable]] = None, min_iter: int = 0, max_iter: int = 100, tol: Union[int, float] = 1e-10, offset: int = 0, failures: str = 'raise', errors: str = 'raise', catch_first_error: bool = True, **kwargs: Dict[str, Any]) -> Tuple[List[Hashable], List[int], List[bool]]:
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
        kwargs :
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
        Before version 0.8.0, error handling worked as follows during solution:

          1. Call `_evaluate()` (i.e. solve one iteration), temporarily
             suppressing all warnings (numerical solution errors). Results
             arising from warnings (NaNs, Infs) enter the solution at this
             point and propagate if these results feed into other equations
             during the iteration.
          2. If any warnings came up during the call to `_evaluate()`, handle
             according to the value of `errors`:
              - 'raise':   raise an exception (`SolutionError`)
              - 'skip':    move immediately to the next period, retaining the
                           NaNs/Infs
              - 'ignore':  proceed to the next iteration (in the hope that
                           NaNs/Infs are eventually replaced with finite
                           values)
              - 'replace': replace NaNs/Infs with zeroes before proceeding to
                           the next iteration (in the hope that this is enough
                           to eventually generate finite values throughout the
                           solution)

        However, this behaviour (even if straightforward to replicate in
        Fortran) comes at the expense of knowing which equation(s) led to an
        error. All we can see is how non-finite values propagated through the
        solution from a single pass/iteration. Moreover, by allowing the entire
        system to run through each time, there's no guarantee that 'ignore' or
        'replace' will help to solve the model, should the same pattern of NaNs
        and Infs repeat each iteration.

        Consequently, the above treatment is no longer the default behaviour in
        version 0.8.0, which introduces the keyword argument
        `catch_first_error` (default `True`).

        With `catch_first_error=True` and `errors='raise'`, solution
        immediately halts on the first error, throwing an exception up the call
        stack. This identifies the problem statement in the stack trace (which
        is helpful for debugging) but also prevents a NaN/Inf entering the
        solution (which may be useful for inspection) and, were there a
        NaN/Inf, from it being propagated (it's not immediately obvious if this
        has any use, though).

        With `catch_first_error=False`, the behaviour returns to the pre-0.8.0
        treatment.
        """
        # Error if `min_iter` exceeds `max_iter`
        if min_iter > max_iter:
            raise ValueError(
                'Value of `min_iter` ({}) cannot exceed value of `max_iter` ({})'.format(
                    min_iter, max_iter))

        period_iter = self.iter_periods(start=start, end=end, **kwargs)

        indexes = [None] * len(period_iter)
        labels  = [None] * len(period_iter)
        solved  = [None] * len(period_iter)

        for i, (t, period) in enumerate(period_iter):
            indexes[i] = t
            labels[i] = period
            solved[i] = self.solve_t(t,
                                     submodels=submodels,
                                     min_iter=min_iter, max_iter=max_iter, tol=tol, offset=offset,
                                     failures=failures, errors=errors, catch_first_error=catch_first_error, **kwargs)

        return labels, indexes, solved

    def solve_t(self, t: int, *, submodels: Optional[Sequence[Hashable]] = None, min_iter: int = 0, max_iter: int = 100, tol: Union[int, float] = 1e-10, offset: int = 0, failures: str = 'raise', errors: str = 'raise', catch_first_error: bool = True, **kwargs: Dict[str, Any]) -> bool:
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
        kwargs :
            Further keyword arguments to pass to the solution routines

        Returns
        -------
        `True` if the linker solved for the current period; `False` otherwise.

        Notes
        -----
        Before version 0.8.0, error handling worked as follows during solution:

          1. Call `_evaluate()` (i.e. solve one iteration), temporarily
             suppressing all warnings (numerical solution errors). Results
             arising from warnings (NaNs, Infs) enter the solution at this
             point and propagate if these results feed into other equations
             during the iteration.
          2. If any warnings came up during the call to `_evaluate()`, handle
             according to the value of `errors`:
              - 'raise':   raise an exception (`SolutionError`)
              - 'skip':    move immediately to the next period, retaining the
                           NaNs/Infs
              - 'ignore':  proceed to the next iteration (in the hope that
                           NaNs/Infs are eventually replaced with finite
                           values)
              - 'replace': replace NaNs/Infs with zeroes before proceeding to
                           the next iteration (in the hope that this is enough
                           to eventually generate finite values throughout the
                           solution)

        However, this behaviour (even if straightforward to replicate in
        Fortran) comes at the expense of knowing which equation(s) led to an
        error. All we can see is how non-finite values propagated through the
        solution from a single pass/iteration. Moreover, by allowing the entire
        system to run through each time, there's no guarantee that 'ignore' or
        'replace' will help to solve the model, should the same pattern of NaNs
        and Infs repeat each iteration.

        Consequently, the above treatment is no longer the default behaviour in
        version 0.8.0, which introduces the keyword argument
        `catch_first_error` (default `True`).

        With `catch_first_error=True` and `errors='raise'`, solution
        immediately halts on the first error, throwing an exception up the call
        stack. This identifies the problem statement in the stack trace (which
        is helpful for debugging) but also prevents a NaN/Inf entering the
        solution (which may be useful for inspection) and, were there a
        NaN/Inf, from it being propagated (it's not immediately obvious if this
        has any use, though).

        With `catch_first_error=False`, the behaviour returns to the pre-0.8.0
        treatment.
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
                    check_values[k] = np.array([submodel[name][t] for name in submodel.CHECK])

            return check_values


        status = SolutionStatus.UNSOLVED.value
        current_values = get_check_values()

        # Set iteration counters to zero for submodels to solve (also use this
        # as an opportunity to make sure specified submodels are valid)
        for name in submodels:
            try:
                submodel = self.__dict__['submodels'][name]
            except KeyError:
                raise KeyError("'{}' not found in list of submodels".format(name))

            submodel.iterations[t] = 0

        # Run any code prior to solution
        self.solve_t_before(t, submodels=submodels, errors=errors, catch_first_error=catch_first_error, iteration=0, **kwargs)

        for iteration in range(1, max_iter + 1):
            previous_values = copy.deepcopy(current_values)

            self.evaluate_t_before(t, submodels=submodels, errors=errors, catch_first_error=catch_first_error, iteration=iteration, **kwargs)
            self.evaluate_t(       t, submodels=submodels, errors=errors, catch_first_error=catch_first_error, iteration=iteration, **kwargs)
            self.evaluate_t_after( t, submodels=submodels, errors=errors, catch_first_error=catch_first_error, iteration=iteration, **kwargs)

            current_values = get_check_values()

            if iteration < min_iter:
                continue

            diff = {k: current_values[k] - previous_values[k]
                    for k in current_values.keys()}
            diff_squared = {k: v ** 2 for k, v in diff.items()}

            if all(np.all(v < tol) for v in diff_squared.values()):
                status = SolutionStatus.SOLVED.value
                self.solve_t_after(t, submodels=submodels, errors=errors, catch_first_error=catch_first_error, iteration=iteration, **kwargs)
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
                'Solution failed to converge after {} iterations(s) '
                'in period with label: {} (index: {})'
                .format(iteration, self.span[t], t))

        return status == SolutionStatus.SOLVED.value

    def evaluate_t(self, t: int, *, submodels: Optional[Sequence[Hashable]] = None, errors: str = 'raise', catch_first_error: bool = True, iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
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
        kwargs :
            Further keyword arguments for solution

        Notes
        -----
        Before version 0.8.0, error handling worked as follows during solution:

          1. Call `_evaluate()` (i.e. solve one iteration), temporarily
             suppressing all warnings (numerical solution errors). Results
             arising from warnings (NaNs, Infs) enter the solution at this
             point and propagate if these results feed into other equations
             during the iteration.
          2. If any warnings came up during the call to `_evaluate()`, handle
             according to the value of `errors`:
              - 'raise':   raise an exception (`SolutionError`)
              - 'skip':    move immediately to the next period, retaining the
                           NaNs/Infs
              - 'ignore':  proceed to the next iteration (in the hope that
                           NaNs/Infs are eventually replaced with finite
                           values)
              - 'replace': replace NaNs/Infs with zeroes before proceeding to
                           the next iteration (in the hope that this is enough
                           to eventually generate finite values throughout the
                           solution)

        However, this behaviour (even if straightforward to replicate in
        Fortran) comes at the expense of knowing which equation(s) led to an
        error. All we can see is how non-finite values propagated through the
        solution from a single pass/iteration. Moreover, by allowing the entire
        system to run through each time, there's no guarantee that 'ignore' or
        'replace' will help to solve the model, should the same pattern of NaNs
        and Infs repeat each iteration.

        Consequently, the above treatment is no longer the default behaviour in
        version 0.8.0, which introduces the keyword argument
        `catch_first_error` (default `True`).

        With `catch_first_error=True` and `errors='raise'`, solution
        immediately halts on the first error, throwing an exception up the call
        stack. This identifies the problem statement in the stack trace (which
        is helpful for debugging) but also prevents a NaN/Inf entering the
        solution (which may be useful for inspection) and, were there a
        NaN/Inf, from it being propagated (it's not immediately obvious if this
        has any use, though).

        With `catch_first_error=False`, the behaviour returns to the pre-0.8.0
        treatment.
        """
        if submodels is None:
            submodels = list(self.__dict__['submodels'].keys())

        for name in submodels:
            submodel = self.__dict__['submodels'][name]

            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                submodel._evaluate(t,
                                   errors=errors, catch_first_error=catch_first_error,
                                   iteration=iteration,
                                   **kwargs)

            submodel.iterations[t] += 1

    def solve_t_before(self, t: int, *, submodels: Optional[Sequence[Hashable]] = None, errors: str = 'raise', catch_first_error: bool = True, iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
        """Pre-solution method: This runs each period, before the iterative solution. Over-ride to implement custom linker behaviour."""
        pass

    def solve_t_after(self, t: int, *, submodels: Optional[Sequence[Hashable]] = None, errors: str = 'raise', catch_first_error: bool = True, iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
        """Post-solution method: This runs each period, after the iterative solution. Over-ride to implement custom linker behaviour."""
        pass

    def evaluate_t_before(self, t: int, *, submodels: Optional[Sequence[Hashable]] = None, errors: str = 'raise', catch_first_error: bool = True, iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
        """Evaluate any linker equations before solving the individual submodels. Over-ride to implement custom linker behaviour."""
        pass

    def evaluate_t_after(self, t: int, *, submodels: Optional[Sequence[Hashable]] = None, errors: str = 'raise', catch_first_error: bool = True, iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
        """Evaluate any linker equations after solving the individual submodels. Over-ride to implement custom linker behaviour."""
        pass
