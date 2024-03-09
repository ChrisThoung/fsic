# -*- coding: utf-8 -*-
"""
Core `BaseModel` class for a single economic model. Economic models are
implemented as derived classes, inheriting the necessary attributes and methods
that make up the API.
"""

import warnings
from typing import Any, Dict, Hashable, List, Optional, Sequence, Union

import numpy as np

from .interfaces import ModelInterface, SolverMixin, SolutionStatus
from ..exceptions import (
    NonConvergenceError,
    SolutionError,
)
from ..tools import model_to_dataframe as _model_to_dataframe


class BaseModel(SolverMixin, ModelInterface):
    """Base class for individual economic models."""

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

        self.add_attribute('endogenous', self.ENDOGENOUS)
        self.add_attribute('check', self.CHECK)

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
        self,
        span: Sequence[Hashable],
        *,
        fill_value: Any = None,
        strict: Optional[bool] = None,
        **fill_values: Any,
    ) -> 'BaseModel':
        """Return a copy of the current object, adjusted to match `span`. Values in overlapping periods between the old and new objects are preserved (copied over).

        Parameters
        ----------
        span : iterable
            Sequence of periods defining the span of the object to be returned
        strict : bool
            If `True`, raise a `KeyError` if `fill_values` refers to variables
            not defined in the current object. Ignore if `False`.
            If `None`, use the current value of the object's `strict`
            attribute.
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

        return super().reindex(
            span, fill_value=fill_value, strict=strict, **fill_values
        )

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
            return np.array([self.__dict__['_' + name][t] for name in self.check])

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

            for name in self.endogenous:
                self.__dict__['_' + name][t] = self.__dict__['_' + name][t + offset]

        status = SolutionStatus.UNSOLVED.value
        current_values = get_check_values()

        # Raise an exception if there are pre-existing NaNs or infinities, and
        # error checking is at its strictest ('raise')
        if errors == 'raise' and np.any(~np.isfinite(current_values)):
            raise SolutionError(
                f'Pre-existing NaNs or infinities found '
                f'in one or more `check` variables '
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
