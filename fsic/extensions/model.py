# -*- coding: utf-8 -*-
"""
Mixins to extend the functionality of the core model class.
"""

from typing import Any, Hashable, Optional, Sequence, Union

import numpy as np

from ..exceptions import DimensionError


class Trace:
    """Object to store step-by-step results for a single period.

    Attributes
    ----------
    names : Sequence[str]
        Names of the variables
    index : List[Any]
        Labels for each set of results
    values : NumPy array
        (N x I) array of results, where N is the number of variables (as in
        `names`) and I is the number of sets of results so far
    """

    def __init__(self, names: Sequence[str]) -> 'Trace':
        """Initialise an empty object with variable names `names`."""
        self.names: Sequence[str] = names

        # Initialise other attributes as empty variables
        self.index: List[Any] = []
        self.values: np.ndarray = np.array([])

    def is_empty(self) -> bool:
        """Return `True` if the current object contains values (the size of `self.values` is non-zero); `False` otherwise."""
        return self.values.shape == (0,)

    def append(self, label: Any, values: np.ndarray) -> None:
        """Add `values` (a vector-like array) to the object, with accompanying label `label`."""
        # Check for no more than two dimensions
        if values.ndim > 2:
            raise DimensionError(
                f'`values` cannot have more than two dimensions but '
                f'argument passed has shape {values.shape} ({values.ndim} dimensions)'
            )

        # If two dimensions, check for a vector-like shape i.e. that one of the
        # dimension lengths is 1
        if values.ndim == 2:
            if values.shape[0] != 1 and values.shape[1] != 1:
                raise DimensionError(
                    f'`values` can only have two dimensions '
                    f'if one of the dimensions is of length 1 but '
                    f'argument passed has shape {values.shape}'
                )

        # Store the arguments
        self.index.append(label)

        if self.is_empty():
            # Save `values` as a column vector
            self.values = values.reshape((-1, 1))
        else:
            # Horizontally join `values` as a column vector to the existing
            # array
            self.values = np.hstack([self.values, values.reshape((-1, 1))])

    def to_dataframe(self) -> 'DataFrame':
        """Return a `DataFrame` of the current object's contents."""
        from pandas import DataFrame

        if self.is_empty():
            return DataFrame()

        return DataFrame(self.values.T, index=self.index, columns=self.names)


class TracerMixin:
    """Mixin to store step-by-step values to `Trace` objects, one per period.

    For each period, the trace is an (N x I) NumPy array where:
    * N is the number of variables stored (which the user can set)
    * I is the number of iterations (so far), with up to four extra sets of
      values (see table in Notes below)

    Examples
    --------
    Add the mixin to an existing model class:

        from fsic import BaseModel
        from fsic.extensions.model import TracerMixin

        class ExampleModel(BaseModel):
            ENDOGENOUS = ['Y', 'C']
            EXOGENOUS = ['G']

            NAMES = ENDOGENOUS + EXOGENOUS

            CHECK = ['Y', 'C']

            def _evaluate(self, t, *args, **kwargs):
                self.Y[t] = self.C[t] + self.G[t]
                self.C[t] = 0.1 * self.Y[t]

        class ExampleModelWithTracer(TracerMixin, ExampleModel):
            pass

    With nothing over-ridden, instances of this class will:
     - add a new entry to the container with the name 'trace' (as in
       `TRACE_NAME`) to store (initially empty) `Trace` objects, one per period
     - by default, store *all* model variables to the `Trace` objects (the
       result of `TRACE_VARIABLES=None`)

    # Models extended with a tracer gain a `trace` attribute stored in the same
    # way as `status` and `iterations`
    >>> model = ExampleModelWithTracer(range(1990, 1995), G=10)
    >>> model.trace  # `model['trace']` works, too
    [<fsic.extensions.model.Trace object at ...>
     <fsic.extensions.model.Trace object at ...>
     <fsic.extensions.model.Trace object at ...>
     <fsic.extensions.model.Trace object at ...>
     <fsic.extensions.model.Trace object at ...>]

    # On instantiation, `Trace` objects have an empty NumPy array for the
    # `values` attribute
    >>> model['trace', 1991].values
    []

    # With no changes to the default settings, turning the trace on during
    # solution updates the `Trace` object for each period, storing results for
    # *all* model variables
    >>> model.solve(trace=True)

    # The `values` attribute then stores the step-by-step (mostly
    # iteration-by-iteration) calculations each period, with the variable order
    # by row following that in `NAMES`
    >>> model['trace', 1991].values
    [[ 0.          0.          0.         10.         11.         11.1        11.11
      11.111      11.1111     11.11111    11.111111   11.1111111  11.11111111 11.11111111
      11.11111111 11.11111111 11.11111111]
     [ 0.          0.          0.          1.          1.1         1.11        1.111
       1.1111      1.11111     1.111111    1.1111111   1.11111111  1.11111111  1.11111111
       1.11111111  1.11111111  1.11111111]
     [10.         10.         10.         10.         10.         10.         10.
      10.         10.         10.         10.         10.         10.         10.
      10.         10.         10.        ]]

    # These are more easily viewed as a `pandas` DataFrame
    >>> model['trace', 1991].to_dataframe()
                    Y         C     G
    start    0.000000  0.000000  10.0
    before   0.000000  0.000000  10.0
    0        0.000000  0.000000  10.0
    1       10.000000  1.000000  10.0
    2       11.000000  1.100000  10.0
    3       11.100000  1.110000  10.0
    4       11.110000  1.111000  10.0
    5       11.111000  1.111100  10.0
    6       11.111100  1.111110  10.0
    7       11.111110  1.111111  10.0
    8       11.111111  1.111111  10.0
    9       11.111111  1.111111  10.0
    10      11.111111  1.111111  10.0
    11      11.111111  1.111111  10.0
    12      11.111111  1.111111  10.0
    13      11.111111  1.111111  10.0
    end     11.111111  1.111111  10.0

    To set (limit) the variables in the trace, you can over-ride the
    class-level `TRACE_VARIABLES` attribute (the default listing when
    `trace=True` in solution):

        class ExampleModelWithTracer2(TracerMixin, ExampleModel):
            TRACE_VARIABLES = ['C', 'Y']  # The default `Trace` will follow this order

    Following the same steps as above:

    >>> model = ExampleModelWithTracer2(range(1990, 1995), G=10)
    >>> model.solve(trace=True)
    >>> model['trace', 1991].to_dataframe()
                   C          Y
    start   0.000000   0.000000
    before  0.000000   0.000000
    0       0.000000   0.000000
    1       1.000000  10.000000
    2       1.100000  11.000000
    3       1.110000  11.100000
    4       1.111000  11.110000
    5       1.111100  11.111000
    6       1.111110  11.111100
    7       1.111111  11.111110
    8       1.111111  11.111111
    9       1.111111  11.111111
    10      1.111111  11.111111
    11      1.111111  11.111111
    12      1.111111  11.111111
    13      1.111111  11.111111
    end     1.111111  11.111111

    Alternatively, on a per-period basis, pass the desired list of variables as
    the `trace` argument:

    >>> model = ExampleModelWithTracer(range(1990, 1995), G=10)
    >>> model.solve(trace=['Y', 'C'])
    >>> model['trace', 1991].to_dataframe()
                    Y         C
    start    0.000000  0.000000
    before   0.000000  0.000000
    0        0.000000  0.000000
    1       10.000000  1.000000
    2       11.000000  1.100000
    3       11.100000  1.110000
    4       11.110000  1.111000
    5       11.111000  1.111100
    6       11.111100  1.111110
    7       11.111110  1.111111
    8       11.111111  1.111111
    9       11.111111  1.111111
    10      11.111111  1.111111
    11      11.111111  1.111111
    12      11.111111  1.111111
    13      11.111111  1.111111
    end     11.111111  1.111111

    As well as `solve()`, this mixin makes the `trace` control / keyword
    argument available in all `solve_` methods:

    * `solve()` - as above
    * `solve_period()`
    * `solve_t()`

    Notes
    -----
    This mixin:
     - defines:
        - `TRACE_NAME`, a class-level str that sets the name of the variable to
          store `Trace` objects in individual model instances (default:
          'trace')
        - `TRACE_VARIABLES`, a(n optional) class-level sequence of str to set
          the default list of variables to include in a `Trace` (default:
          `None`, storing *all* model variables)
     - at instantiation:
        - adds a new entry to the container (using the name in `TRACE_NAME`
          above) made up of empty `Trace` objects
     - implements the following methods to store results to `Trace` objects:
        - `trace_t()`, to store the current results for the period with index
          `t`
        - `trace_period()`, to store the current results for the period with
          label `period` (calling `trace_t()`)
     - wraps the following solution methods to store results to `Trace` objects
       each period using `trace_t()`:
        - `solve_t()`, to store the results before any solution takes place
          (before calling the parent class version)
        - `solve_t_before()`, to store the results before and after calling the
          parent class version
        - `_evaluate()`, to call the parent class version and then store the
          results after each iteration
        - `solve_t_after()`, to call the parent class version and then store
          the final results after any post-solution calculations

    The default labels store the results as follows:

    | Label  | Result                                              | Stored in method   |
    |--------+-----------------------------------------------------+--------------------|
    |  start | Starting values before any solution                 | `solve_t()`        |
    | before | Pre-solution values (e.g. with `offset` applied)    | `solve_t_before()` |
    |      0 | Values just before iterative solution               | `solve_t_before()` |
    |      . | Values each iteration                               | `_evaluate()`      |
    |      N | Values after final (hopefully convergent) iteration | `_evaluate()`      |
    |    end | Final solution values after any final calculations  | `solve_t_after()`  |
    """

    # Variable name to store `Trace` objects
    # Available to change should the model have a variable already called
    # 'trace' (raises a `DuplicateNameError` if there is a name collision)
    TRACE_NAME: str = 'trace'

    # Default variables to store to `Trace` objects; if `None`, store all
    # variables
    TRACE_VARIABLES: Optional[Sequence[str]] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Instantiate object using parent class method
        super().__init__(*args, **kwargs)

        # Check that the name under which to store the traces is free to be
        # used
        if self.TRACE_NAME in self.index:
            raise DuplicateNameError(
                f"Unable to set trace variable name '{self.TRACE_NAME}' "
                "(in `self.TRACE_NAME`). "
                "Name is already defined in the current object."
            )

        # Add `Trace` objects to the model instance
        self.__dict__['index'].append(self.TRACE_NAME)
        self.__dict__['_' + self.TRACE_NAME] = np.array([Trace([]) for _ in self.span])

    def trace_period(
        self,
        period: Hashable,
        label: Any,
        *args,
        trace: Optional[Union[bool, str, Sequence[str]]] = None,
        reset: bool = False,
        **kwargs,
    ):
        """Store current results for the period with label `period`.

        Parameters
        ----------
        t : int
            Position in `span` of the period results to store
        label : Any
            Label to attach to this set of results
        trace : bool, str, or list of str; default `None`
            Variables to track, depending on the argument value:
             - `None` or `False`: No tracing
             - `True`: Use the variables in `TRACE_VARIABLES` (all variables if
               `TRACE_VARIABLES` is `None`)
             - str or Sequence[str]: List of variables to track
        reset : bool; default `False`
            If `True`, remove any prior contents of the `Trace` this period,
            replacing with the current results
        *args, **kwargs :
            No effect. These just absorb arguments intended for other in
            solution code, to avoid errors.
        """
        t = self._locate_period_in_span(period)

        if not isinstance(t, int):
            raise KeyError(
                f'Invalid `period` argument: unable to convert to an integer '
                f'(a single location in `self.span`). '
                f'`period` resolved to type {type(t)} with value {t}'
            )

        self.trace_t(t, label, *args, trace=trace, reset=reset, **kwargs)

    def trace_t(
        self,
        t: int,
        label: Any,
        *args,
        trace: Optional[Union[bool, str, Sequence[str]]] = None,
        reset: bool = False,
        **kwargs,
    ):
        """Store current results for the period at integer position `t` in the model's `span`.

        Parameters
        ----------
        t : int
            Position in `span` of the period results to store
        label : Any
            Label to attach to this set of results
        trace : bool, str, or list of str; default `None`
            Variables to track, depending on the argument value:
             - `None` or `False`: No tracing
             - `True`: Use the variables in `TRACE_VARIABLES` (all variables if
               `TRACE_VARIABLES` is `None`)
             - str or Sequence[str]: List of variables to track
        reset : bool; default `False`
            If `True`, remove any prior contents of the `Trace` this period,
            replacing with the current results
        *args, **kwargs :
            No effect. These just absorb arguments intended for other in
            solution code, to avoid errors.
        """

        # Convert strings to lists of strings
        if isinstance(trace, str):
            trace = [trace]

        # Set the list of variable names
        if isinstance(trace, Sequence):
            # User-supplied list
            names = trace
        else:
            # Otherwise use the defaults in the class-level `TRACE_VARIABLES`
            # attribute
            names = self.TRACE_VARIABLES
            if names is None:
                names = self.names  # `None` sets *all* variables

        # Extract results as a column vector
        results = np.array([[self[x][t]] for x in names])

        # If starting from an empty `Trace` or `reset`ting the `Trace`,
        # re-initialise the variable
        if self[self.TRACE_NAME][t].is_empty() or reset:
            self[self.TRACE_NAME][t] = Trace(names)

        # Add the results to the `Trace`
        self[self.TRACE_NAME][t].append(label, results)

    def solve_t(
        self,
        t: int,
        *args,
        trace: Optional[Union[bool, str, Sequence[str]]] = None,
        reset: bool = False,
        **kwargs,
    ) -> bool:
        """Solve for the period at integer position `t` in the model's `span`.

        If `trace` is enabled (is truthy), save the results before any
        computation (label: 'start').
        """
        # Store results before calling the parent method i.e. to get the
        # results before any calculation
        if trace:
            self.trace_t(t, 'start', *args, trace=trace, reset=reset, **kwargs)

        return super().solve_t(t, *args, trace=trace, reset=reset, **kwargs)

    def solve_t_before(
        self,
        t: int,
        *args: Any,
        trace: Optional[Union[bool, str, Sequence[str]]] = None,
        reset: bool = False,
        iteration: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Pre-solution method: This runs each period, before the iterative solution.

        If `trace` is enabled (is truthy), save the results before calling the
        parent class method (label: 'before').
        """
        # Store results before calling the parent class method i.e. after
        # preliminary calculations in `solve_t()`
        if trace:
            self.trace_t(t, 'before', *args, trace=trace, reset=reset, **kwargs)

        super().solve_t_before(
            t, *args, trace=trace, reset=reset, iteration=iteration, **kwargs
        )

        # Store results after calling the parent class method i.e. just before
        # the start of the iterative solution
        if trace:
            self.trace_t(t, 0, *args, trace=trace, reset=reset, **kwargs)

    def solve_t_after(
        self,
        t: int,
        *args: Any,
        trace: Optional[Union[bool, str, Sequence[str]]] = None,
        reset: bool = False,
        iteration: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Post-solution method: This runs each period, before the iterative solution.

        If `trace` is enabled (is truthy), save the results after calling the
        parent class method (label: 'end').
        """
        super().solve_t_after(
            t, *args, trace=trace, reset=reset, iteration=iteration, **kwargs
        )

        # Store final results
        if trace:
            self.trace_t(t, 'end', *args, trace=trace, reset=reset, **kwargs)

    def _evaluate(
        self,
        t: int,
        *args: Any,
        trace: Optional[Union[bool, str, Sequence[str]]] = None,
        reset: bool = False,
        iteration: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Evaluate the system of equations for the period at integer position `t` in the model's `span`.

        If `trace` is enabled (is truthy), save the results after calling the
        parent class method (label: the value of `iteration`).
        """
        super()._evaluate(
            t, *args, trace=trace, reset=reset, iteration=iteration, **kwargs
        )

        # Store results *after* each iteration
        if trace:
            self.trace_t(t, iteration, *args, trace=trace, reset=reset, **kwargs)
