# -*- coding: utf-8 -*-
"""
Intermediate classes to bridge data management in `VectorContainer` with fully
fledged model classes, as in `BaseModel` and `BaseLinker`.
"""

import copy
import enum
from collections import Counter
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .containers import VectorContainer
from ..exceptions import (
    DimensionError,
    DuplicateNameError,
    InitialisationError,
    SolutionError,
)


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
                f'Found multiple instances of the following variable(s) in `NAMES`: {", ".join(duplicates)}'
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
