# -*- coding: utf-8 -*-
"""
Tools to speed up fsic-based economic models by generating F2PY-compatible
Fortran code.
"""

from fsic import __version__

import itertools
import re
import textwrap
from types import ModuleType
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .core import BaseModel, SolutionStatus
from .exceptions import FortranEngineError, InitialisationError, NonConvergenceError, SolutionError
from .parser import Symbol, Type


# Class implementing interface to Fortran code --------------------------------

class FortranEngine:
    """Subclass for derivatives of fsic `BaseModel` to speed up model solution by calling compiled Fortran code."""

    ENGINE: Optional[ModuleType] = None

    _FAILURE_OPTIONS: Dict[str, int] = {
        'raise':   0,
        'ignore':  2,
    }

    _ERROR_OPTIONS: Dict[str, int] = {
        'raise':   0,
        'skip':    1,
        'ignore':  2,
        'replace': 3,
    }

    def __init__(self, span: Sequence[Hashable], *, engine: str = 'fortran', strict: bool = False, dtype: Any = float, default_value: Union[int, float] = 0.0, **initial_values: Dict[str, Any]) -> None:
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
        dtype : variable type
            Data type to impose on model variables (in NumPy arrays)
        default_value : number
            Value with which to initialise model variables
        **initial_values : keyword arguments of variable names and values
            Values with which to initialise specific named model variables
        """
        if engine == 'fortran' and self.ENGINE is None:
            raise InitialisationError(
                "`engine` argument is '{}' but class `ENGINE` attribute is `{}`. "
                "Check that `ENGINE` has an assigned value (i.e. a module of solution routines); "
                "this typically uses a subclass".format(engine, self.ENGINE))

        super().__init__(span=span,
                         engine=engine,
                         strict=strict,
                         dtype=dtype,
                         default_value=default_value,
                         **initial_values)

    def solve(self, *, start: Optional[Hashable] = None, end: Optional[Hashable] = None, min_iter: int = 0, max_iter: int = 100, tol: Union[int, float] = 1e-10, offset: int = 0, failures: str = 'raise', errors: str = 'raise', **kwargs: Dict[str, Any]) -> Tuple[List[Hashable], List[int], List[bool]]:
        """Solve the model. Use default periods if none provided.

        This method is part of the `FortranEngine` class and wraps a Fortran
        subroutine for faster solution.

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
        **kwargs : not used
            Retained for consistency with the wider fsic API

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
        Unlike the `BaseModel` version of `solve()` (which in turn inherits
        from `SolverMixin`), this implementation does not call
        `iter_periods()`.
        """
        # Error if `min_iter` exceeds `max_iter`
        if min_iter > max_iter:
            raise ValueError(
                'Value of `min_iter` ({}) cannot exceed value of `max_iter` ({})'.format(
                    min_iter, max_iter))

        # Form lists of period information (avoid using `iter_periods()` in case
        # this is being over-ridden elsewhere)

        # Set default start and end periods if no others supplied
        if start is None:
            start = self.span[self.LAGS]
        if end is None:
            end = self.span[-1 - self.LEADS]

        # Convert to an integer range and assemble accompanying list of labels
        indexes = list(range(self._locate_period_in_span(start),
                             self._locate_period_in_span(end) + 1))
        labels = [self.span[t] for t in indexes]

        # Solve: Add 1 to `indexes` to go from zero-based (Python) to one-based
        #        (Fortran) indexing
        solved_values, convergences, iterations, error_codes = (
            self.ENGINE.solve(self.values.astype(float),
                              [t + 1 for t in indexes],
                              min_iter,
                              max_iter,
                              tol,
                              offset,
                              [self.names.index(x) for x in self.CHECK],
                              self._FAILURE_OPTIONS[failures],
                              self._ERROR_OPTIONS[errors], )
        )

        # Store the values back to this Python instance
        self.values = solved_values

        # Loop through results information and update object and return values
        solved = [None] * len(indexes)

        for i, (t, period, converged, iteration, error_code) in enumerate(zip(indexes, labels, convergences, iterations, error_codes)):

            # Converged
            if converged:
                self.status[t] = SolutionStatus.SOLVED.value
                self.iterations[t] = iteration
                solved[i] = True
                continue

            # Failed to converge but no errors
            elif not converged and error_code == 0:
                self.status[t] = SolutionStatus.FAILED.value
                self.iterations[t] = iteration
                solved[i] = False

                # Raise Exception as required
                if failures == 'raise':
                    raise NonConvergenceError(
                        'Solution failed to converge after {} iterations(s) '
                        'in period with label: {} (index: {})'
                        .format(iteration, period, t))

            # Numerical solution error: Raise
            elif error_code == 21 and errors == 'raise':
                self.status[t] = SolutionStatus.ERROR.value
                self.iterations[t] = iteration
                solved[i] = False

                raise SolutionError(
                    'Numerical solution error after {} iterations(s) '
                    'in period with label: {} (index: {})'
                    .format(iteration, period, t))

            # Found pre-existing NaN or infinity prior to solution
            elif error_code == 31 and errors == 'raise':
                raise SolutionError(
                    'Pre-existing NaNs or infinities found '
                    'in period with label: {} (index: {})'
                    .format(period, t))

            # Error if `offset` points prior to the current model span
            elif error_code == 41:
                t_check = t
                if t_check < 0:
                    t_check += len(self.span)

                raise IndexError(
                    '`offset` argument ({}) for position `t` ({}) '
                    'implies a period before the span of the current model instance: '
                    '{} + {} -> position {} < 0'.format(
                        offset, t, offset, t, offset + t_check))

            # Error if `offset` points beyond the current model span
            elif error_code == 42:
                t_check = t
                if t_check < 0:
                    t_check += len(self.span)

                raise IndexError(
                    '`offset` argument ({}) for position `t` ({}) '
                    'implies a period beyond the span of the current model instance: '
                    '{} + {} -> position {} >= {} periods in span'.format(
                        offset, t, offset, t, offset + t_check, len(self.span)))

            # Some other error but `errors='skip'`
            elif error_code == 22 and errors == 'skip':
                self.status[t] = SolutionStatus.SKIPPED.value
                self.iterations[t] = iteration
                solved[i] = False

            # Any uncaught errors
            else:
                raise FortranEngineError(
                    'Failed to solve model in period with label {} (index: {}), '
                    'with uncaught error code {} after {} iteration(s)'
                    .format(period, t, error_code, iteration))

        return labels, indexes, solved

    def solve_t(self, t: int, *, min_iter: int = 0, max_iter: int = 100, tol: Union[int, float] = 1e-10, offset: int = 0, failures: str = 'raise', errors: str = 'raise', **kwargs: Dict[str, Any]) -> bool:
        """Solve for the period at integer position `t` in the model's `span`.

        This method is part of the `FortranEngine` class and wraps a Fortran
        subroutine for faster solution.

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

        if errors not in self._ERROR_OPTIONS:
            raise ValueError('Invalid `errors` argument: {}'.format(errors))

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
                'in period with label: {} (index: {})'
                .format(self.span[t], t))

        # Solve: Add 1 to `t` to go from zero-based (Python) to one-based
        #        (Fortran) indexing
        solved_values, converged, iteration, error_code = (
            self.ENGINE.solve_t(self.values.astype(float),
                                t + 1,
                                min_iter,
                                max_iter,
                                tol,
                                offset,
                                [self.names.index(x) for x in self.CHECK],
                                self._ERROR_OPTIONS[errors]))

        converged = bool(converged)

        # Store the values back to this Python instance
        self.values = solved_values

        # Check solution result and error code
        if error_code == 0:
            if converged:
                status = SolutionStatus.SOLVED.value
            else:
                status = SolutionStatus.FAILED.value

        elif error_code == 21 and errors == 'raise':
            self.status[t] = SolutionStatus.ERROR.value
            self.iterations[t] = iteration

            raise SolutionError(
                'Numerical solution error after {} iterations(s) '
                'in period with label: {} (index: {})'
                .format(iteration, self.span[t], t))

        elif error_code == 22 and errors == 'skip':
            status = SolutionStatus.SKIPPED.value

        else:
            raise FortranEngineError(
                'Failed to solve model in period with label {} (index: {}), '
                'with uncaught error code {} after {} iteration(s)'
                .format(self.span[t], t, error_code, iteration))

        self.status[t] = status
        self.iterations[t] = iteration

        if status == SolutionStatus.FAILED.value and failures == 'raise':
            raise NonConvergenceError(
                'Solution failed to converge after {} iterations(s) '
                'in period with label: {} (index: {})'
                .format(iteration, self.span[t], t))

        return status == SolutionStatus.SOLVED.value

    def _evaluate(self, t: int, *, errors: str = 'raise', iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
        """Evaluate the system of equations for the period at integer position `t` in the model's `span`.

        This method is part of the `FortranEngine` class and wraps a Fortran
        subroutine for faster solution.

        Parameters
        ----------
        t : int
            Position in `span` of the period to solve
        errors : str
            ** Does nothing: defined for compatibility with the base Python API
               only **
        iteration : int
            ** Does nothing: defined for compatibility with the base Python API
               only **
        kwargs :
            ** Does nothing: defined for compatibility with the base Python API
               only **
            Further keyword arguments for solution are *not* passed to the
            Fortran routines

        Notes
        -----
        Calling this method directly is not guaranteed to confer a speed
        gain. Solution speed may actually be impaired. This is because of the
        overhead of packing and unpacking the results either side of the call
        to the Fortran code.
        """
        # Add 1 to `t` to go from zero-based (Python) to one-based (Fortran) indexing
        solved_values, error_code = self.ENGINE.evaluate(self.values.astype(float), t + 1)

        # The Fortran code should fail silently but assign an error code
        # according to the outcome
        if error_code != 0:
            # `t` is out of bounds, even accounting for reverse indexing
            if error_code in (11, 12, 13, 14):
                raise IndexError(
                    'index {} is out of bounds for axis 0 with size {}'.format(
                        t, len(self.span)))
            else:
                raise SolutionError(
                    'Failed to evaluate the system of equations at index `t` = {}, '
                    'with unidentified error code {}'.format(t, error_code))

        # If here, store the values back to this Python instance
        self.values = solved_values


# Fortran code generator ------------------------------------------------------

fortran_template = '''\
! This file of Fortran code has been generated programmatically by
! `fsic.fortran`, a module of `fsic` (Version {version}).
!
! This module embeds the following equation(s) in the `evaluate()` subroutine:
!
{system}

module structure
  implicit none

  integer :: lags = {lags}, leads = {leads}

  ! Index numbers of different variable types
{endogenous}
{exogenous}
{parameters}
{errors}

end module structure


module failure_codes
  implicit none

  ! Failure control options, to match Python implementation
  integer :: failure_control_raise = 0
  integer :: failure_control_ignore = 2

end module failure_codes

module error_codes
  implicit none

  ! Error control options, to match Python implementation
  integer :: error_control_raise = 0
  integer :: error_control_skip = 1
  integer :: error_control_ignore = 2
  integer :: error_control_replace = 3

  ! Indexing / array bounds errors
  integer :: index_error_below = 11, index_error_above = 12
  integer :: index_error_lags = 13, index_error_leads = 14

  ! Numerical solution errors
  integer :: numerical_error_raise = 21
  integer :: numerical_error_skip = 22
  integer :: numerical_error_ignore = 23
  integer :: numerical_error_replace = 24

  ! Convergence check errors
  integer :: pre_existing_non_finite_value = 31

  ! Offset errors
  integer :: offset_predates_span = 41, offset_postdates_span = 42

end module error_codes


subroutine evaluate(initial_values, t, solved_values, error_code, nrows, ncols)
  use structure
  use error_codes
  implicit none

  ! `nrows` is the number of variables
  ! `ncols` is the full number of periods in the current model instance
  integer, intent(in) :: nrows, ncols

  real(8), dimension(nrows, ncols), intent(in) :: initial_values
  integer, intent(in) :: t

  real(8), dimension(nrows, ncols), intent(out) :: solved_values
  integer, intent(out) :: error_code

  integer :: index

  ! Copy the initial values before evaluation
  ! (copy all values to avoid having to know which elements will change)
  solved_values = initial_values

  ! Initialise error code to -1 to indicate not yet resolved
  error_code = -1

  ! Reproduce the behaviour in the original Python version of `_evaluate()` to
  ! allow reverse indexing
  index = t
  if(index < 1) then
     index = index + ncols
  end if

  ! Error if `index` is still out of bounds
  if(index < 1) then
     error_code = index_error_below
     return
  else if(index > ncols) then
     error_code = index_error_above
     return
  end if

  ! Check that `index` allows for enough lags and leads
  if(index <= lags) then
     error_code = index_error_lags
     return
  else if(index > (ncols - leads)) then
     error_code = index_error_leads
     return
  end if

  ! ---------------------------------------------------------------------------
{equations}
  ! ---------------------------------------------------------------------------

  ! If here, evaluation ran through seemingly without hitch: Return 0
  error_code = 0

end subroutine evaluate


subroutine solve_t(initial_values, t, min_iter, max_iter, tol, offset, convergence_variables, error_control,  &
                &  solved_values, converged, iteration, error_code,                                           &
                &  nrows, ncols, nvars)
  use, intrinsic :: ieee_arithmetic
  use structure
  use error_codes
  implicit none

  ! `nrows` is the number of variables
  ! `ncols` is the number of periods
  integer, intent(in) :: nrows, ncols

  ! `nvars` is the number of variables to check for convergence (indexes set in
  ! `convergence_variables`)
  integer, intent(in) :: nvars

  real(8), dimension(nrows, ncols), intent(in) :: initial_values
  integer, intent(in) :: t, min_iter, max_iter
  real(8), intent(in) :: tol
  integer, intent(in) :: offset
  integer, dimension(nvars), intent(in) :: convergence_variables
  integer, intent(in) :: error_control

  real(8), dimension(nrows, ncols), intent(out) :: solved_values
  logical, intent(out) :: converged
  integer, intent(out) :: iteration, error_code

  real(8), dimension(nrows, ncols) :: previous_values
  real(8), dimension(nvars) :: current_check, previous_check, diff

  integer :: index, offset_location, i

  ! Copy the initial values before solution
  ! (copy all values to avoid having to know which elements will change)
  solved_values = initial_values
  converged = .false.

  ! Initialise error code to -1 to indicate not yet resolved
  error_code = -1

  ! Reproduce the behaviour in the original Python version of `_evaluate()` to
  ! allow reverse indexing
  index = t
  if(index < 1) then
     index = index + ncols
  end if

  ! Error if `index` is still out of bounds
  if(index < 1) then
     error_code = index_error_below
     return
  else if(index > ncols) then
     error_code = index_error_above
     return
  end if

  ! Check that `index` allows for enough lags and leads
  if(index <= lags) then
     error_code = index_error_lags
     return
  else if(index > (ncols - leads)) then
     error_code = index_error_leads
     return
  end if

  ! Optionally copy initial values from another period
  if(offset /= 0) then

     offset_location = index + offset

     if(offset_location < 1) then
        error_code = offset_predates_span
        return
     else if(offset_location > ncols) then
        error_code = offset_postdates_span
        return
     end if

     solved_values(endogenous, index) = solved_values(endogenous, offset_location)

  end if

  ! Array of variable values to check for convergence
  current_check = solved_values(convergence_variables, index)

  ! Check for pre-existing NaNs or infinities
  if(error_control == error_control_raise .and. any(.not. ieee_is_finite(current_check))) then
     error_code = pre_existing_non_finite_value
     return
  end if

  ! Solve
  do iteration = 1, max_iter

     ! Save the values of the convergence variables
     previous_check = current_check

     ! Evaluate the system of equations
     previous_values = solved_values
     call evaluate(previous_values, index, solved_values, error_code, nrows, ncols)

     ! Get the new values of the convergence variables
     current_check = solved_values(convergence_variables, index)

     ! Check error code and return if a problem is found
     ! (indicated by a non-zero error code)
     if(error_code /= 0) then
        return
     end if

     ! Check for numerical errors
     if(any(.not. ieee_is_finite(solved_values(endogenous, index)))) then

        if(error_control == error_control_raise) then
           error_code = numerical_error_raise
           return

        else if(error_control == error_control_skip) then
           error_code = numerical_error_skip
           return

        else if(error_control == error_control_ignore) then
           cycle

        else if(error_control == error_control_replace) then
           ! Only replace values if there are still iterations to go
           ! i.e. a chance for the solution to resolve itself
           if(iteration < max_iter) then

              do i = 1, size(endogenous)
                 if(.not. ieee_is_finite(solved_values(endogenous(i), index))) then
                    solved_values(endogenous(i), index) = 0.0
                 end if
              end do

           end if

           cycle

        end if

     end if

     if(iteration < min_iter) then
        cycle
     end if

     ! Test for convergence
     diff = current_check - previous_check

     if(all(abs(diff) < tol)) then
        converged = .true.
        exit
     end if

  end do

  if(.not. converged) then
     iteration = iteration - 1
  end if

end subroutine solve_t

subroutine solve(initial_values, indexes,                                                                 &
              &  min_iter, max_iter, tol, offset, convergence_variables, failure_control, error_control,  &
              &  solved_values, convergence_results, iterations, solution_error_codes,                    &
              &  nrows, ncols, nvars, nperiods)
  use, intrinsic :: ieee_arithmetic
  use structure
  use failure_codes
  use error_codes
  implicit none

  ! `nrows` is the number of variables
  ! `ncols` is the full number of periods in the current model instance
  integer, intent(in) :: nrows, ncols

  ! `nvars` is the number of variables to check for convergence (indexes set in
  ! `convergence_variables`)
  integer, intent(in) :: nvars

  ! `nperiods` is the number of periods to solve
  integer, intent(in) :: nperiods

  real(8), dimension(nrows, ncols), intent(in) :: initial_values
  integer, dimension(nperiods), intent(in) :: indexes
  integer, intent(in) :: min_iter, max_iter
  real(8), intent(in) :: tol
  integer, intent(in) :: offset
  integer, dimension(nvars), intent(in) :: convergence_variables
  integer, intent(in) :: failure_control, error_control

  real(8), dimension(nrows, ncols), intent(out) :: solved_values
  logical, dimension(nperiods), intent(out) :: convergence_results
  integer, dimension(nperiods), intent(out) :: iterations, solution_error_codes

  integer :: i
  real(8), dimension(nrows, ncols) :: previous_values
  logical :: converged
  integer :: iteration, error_code

  ! Copy the initial values before solution
  solved_values = initial_values

  ! Initialise other results variables to -1 (not yet resolved)
  convergence_results = .false.
  iterations = -1
  solution_error_codes = -1

  do i = 1, nperiods

     previous_values = solved_values

     call solve_t(previous_values, indexes(i), min_iter, max_iter, tol, offset, convergence_variables, error_control,  &
               &  solved_values, converged, iteration, error_code,                                                     &
               &  nrows, ncols, nvars)

     iterations(i) = iteration
     solution_error_codes(i) = error_code

     ! No errors: Check if converged or not
     if(error_code == 0) then

        if(converged) then
           ! Converged: Store and continue
           convergence_results(i) = .true.

        else if(failure_control == failure_control_raise) then
           ! Failed to converge: Raise an error as required
           return
        end if

     ! Errors: Raise as required
     else if(error_control == error_control_raise) then
        return
     end if

  end do

end subroutine solve
'''

def build_fortran_definition(symbols: List[Symbol], *, lags: Optional[int] = None, leads: Optional[int] = None, min_lags: int = 0, min_leads: int = 0, wrap_width: int = 100) -> str:
    """Return a string of Fortran code that embeds the equations in `symbols`.

    Parameters
    ----------
    symbols : list of fsic Symbol objects
        Constituent symbols of the model
    lags, leads : int, default `None`
        If passed (i.e. not `None`), impose the specified number of lags or
        leads on the model
    min_lags, min_leads : int, default 0
        Set the minimum lag or lead length on the model
    wrap_width : int
        Wrap lines to be at most `wrap_width` characters in length
    """
    # Separate variable names according to variable type
    endogenous = [s.name for s in symbols if s.type == Type.ENDOGENOUS]
    exogenous  = [s.name for s in symbols if s.type == Type.EXOGENOUS]
    parameters = [s.name for s in symbols if s.type == Type.PARAMETER]
    errors     = [s.name for s in symbols if s.type == Type.ERROR]

    # Set longest lag and lead
    non_indexed_symbols = [s for s in symbols
                           if s.type not in (Type.FUNCTION, Type.KEYWORD, Type.VERBATIM)]

    if lags is None:
        if len(non_indexed_symbols) > 0:
            lags =  abs(min(s.lags  for s in non_indexed_symbols))
        else:
            lags = 0

        lags = max(lags, min_lags)

    if leads is None:
        if len(non_indexed_symbols) > 0:
            leads = abs(max(s.leads for s in non_indexed_symbols))
        else:
            leads = 0

        leads = max(leads, min_leads)

    # Map variable names to numbers (index positions)
    variables_to_numbers = {
        x: i
        for i, x in enumerate(itertools.chain(endogenous, exogenous, parameters, errors),
                              start=1)
    }

    # Generate code block of equations
    equation_code = []     # Code to insert into the Fortran subroutine
    equation_summary = []  # Summary for the header at the top of the file

    symbols_with_code = filter(lambda x: x.type == Type.ENDOGENOUS and x.equation is not None,
                               symbols)

    for s in symbols_with_code:
        equation = s.equation

        pattern = re.compile(r'([_A-Za-z][_A-Za-z0-9]*)\[(.*?)\]')
        code = equation
        for match in reversed(list(pattern.finditer(equation))):
            start, end = match.span()
            variable = 'solved_values({}, {})'.format(variables_to_numbers[match[1]], match[2].replace('t', 'index'))
            code = code[:start] + variable + code[end:]

        block = '! {}\n{}'.format(equation, '  &\n&  '.join(textwrap.wrap(code, width=wrap_width)))
        equation_code.append(textwrap.indent(block, '  '))

        equation_summary.append(equation)

    # Fill in template
    def create_integer_array_definition(variable_names: Sequence[str], name: str) -> str:
        """Return a Fortran definition with `name`, containing the index numbers in `variable_names`.

        Examples
        --------
        >>> variables_to_numbers = {'A': 1, 'B': 2, 'C': 3, }
        >>> create_integer_array_definition(['A', 'C'], 'endogenous')
        'integer, dimension(2) :: endogenous = (/ 1, 3 /)'

        >>> create_integer_array_definition([], 'errors')
        'integer, dimension(0) :: errors
        """
        indexes = [variables_to_numbers[k] for k in variable_names]

        definition = 'integer, dimension({}) :: {}'.format(len(indexes), name)
        if len(indexes) > 0:
            definition += ' = (/ {} /)'.format(', '.join(map(str, indexes)))

        # Line wrap as needed
        blocks = textwrap.wrap(definition, width=wrap_width)
        definition = textwrap.indent('  &\n&  '.join(blocks), '  ')

        return definition

    fortran_definition_string = fortran_template.format(
        system='\n'.join('!   ' + x for x in equation_summary),
        equations='\n\n'.join(equation_code),
        lags=lags, leads=leads,
        version=__version__,

        endogenous=create_integer_array_definition(endogenous, 'endogenous'),
        exogenous=create_integer_array_definition(exogenous, 'exogenous'),
        parameters=create_integer_array_definition(parameters, 'parameters'),
        errors=create_integer_array_definition(errors, 'errors'),
    )

    return fortran_definition_string
