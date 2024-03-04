# -*- coding: utf-8 -*-
"""
Core `BaseLinker` class to store and solve multiple model instances i.e. as a
multi-region/entity model. Economic models are implemented as derived classes,
inheriting the necessary attributes and methods that make up the API.
"""

import copy
import warnings
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .interfaces import ModelInterface, SolverMixin, SolutionStatus
from .models import BaseModel
from ..exceptions import (
    InitialisationError,
    NonConvergenceError,
)
from ..tools import linker_to_dataframes as _linker_to_dataframes
from ..tools import model_to_dataframe as _model_to_dataframe


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

        self.add_attribute('endogenous', self.ENDOGENOUS)
        self.add_attribute('check', self.CHECK)

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
        self,
        *,
        status: bool = True,
        iterations: bool = True,
        include_internal: bool = False,
    ) -> 'pandas.DataFrame':  # noqa: F821
        """Return the values and solution information from the linker as a `pandas` DataFrame. **Requires `pandas`**."""
        return _model_to_dataframe(
            self,
            status=status,
            iterations=iterations,
            include_internal=include_internal,
        )

    def to_dataframes(
        self,
        *,
        status: bool = True,
        iterations: bool = True,
        include_internal: bool = False,
    ) -> Dict[Hashable, 'pandas.DataFrame']:  # noqa: F821
        """Return the values and solution information from the linker and its constituent submodels as `pandas` DataFrames. **Requires `pandas`**."""
        return _linker_to_dataframes(
            self,
            status=status,
            iterations=iterations,
            include_internal=include_internal,
        )

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
                '_': np.array([self.__dict__['_' + name][t] for name in self.check]),
            }

            for k, submodel in self.submodels.items():
                if k in submodels:
                    check_values[k] = np.array(
                        [submodel[name][t] for name in submodel.check]
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
