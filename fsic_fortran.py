# -*- coding: utf-8 -*-
"""
fsic_fortran
============
Tools to speed up FSIC-based economic models by generating F2PY-compatible
Fortran code.
"""

# Version number keeps track with the main `fsic` module
from fsic import __version__

import itertools
import re
import textwrap
from typing import Any, Dict, Hashable, List, Sequence, Union

import numpy as np

from fsic import FSICError, InitialisationError, NonConvergenceError, SolutionError
from fsic import Symbol, Type


class FortranEngineError(FSICError):
    pass


# Class implementing interface to Fortran code --------------------------------

class FortranEngine:
    """Subclass for derivatives of FSIC `BaseModel` to speed up model solution by calling compiled Fortran code."""

    ENGINE = None

    def __init__(self, span: Sequence[Hashable], *, engine: str = 'fortran', dtype: Any = float, default_value: Union[int, float] = 0.0, **initial_values: Dict[str, Any]) -> None:
        """Initialise model variables.

        Parameters
        ----------
        span : iterable
            Sequence of periods that define the timespan of the model
        engine : str
            Signal of the (expected) underlying solution method/implementation
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
                         dtype=dtype, default_value=default_value, **initial_values)

    def solve_t(self, t: int, *, max_iter: int = 100, tol: Union[int, float] = 1e-10, offset: int = 0, failures: str = 'raise', errors: str = 'raise', **kwargs: Dict[str, Any]) -> bool:
        """Solve for the period at integer position `t` in the model's `span`.

        Parameters
        ----------
        t : int
            Position in `span` of the period to solve
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
            e.g. NaNs
             - 'raise' (default): stop immediately and raise a `SolutionError`
                                  [set period solution status to 'E']
             - 'skip': stop solving the current period
                       [set period solution status to 'S']
             - 'ignore': continue solving, with no special treatment or action
                         [period solution statuses as usual i.e. '.' or 'F']
             - 'replace': each iteration, replace NaNs with zeroes
                         [period solution statuses as usual i.e. '.' or 'F']
        kwargs :
            Further keyword arguments to pass to the solution routines

        Returns
        -------
        `True` if the model solved for the current period; `False` otherwise.

        Notes
        -----
        As of version 0.3.0, FSIC provides (some) support (escape hatches) for
        numerical errors in solution.

        For example, there may be an equation that involves a division
        operation but the equation that determines the divisor follows
        later. If that divisor was initialised to zero, this leads to a
        divide-by-zero operation that NumPy evaluates to a NaN. This becomes
        problematic if the NaNs then propagate through the solution.

        The `solve_t()` method now catches such operations (after a full pass
        through / iteration over the system of equations).
        """
        def get_check_values() -> np.ndarray:
            """Return a 1D NumPy array of variable values for checking in the current period."""
            return np.array([
                self.__dict__['_' + name][t] for name in self.CHECK
            ])

        error_options = {
            'raise':   0,
            'skip':    1,
            'ignore':  2,
            'replace': 3,
        }

        if errors not in error_options:
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

        status = '-'
        current_values = get_check_values()

        # Raise an exception if there are pre-existing NaNs and error checking
        # is at its strictest ('raise')
        if errors == 'raise' and np.any(~np.isfinite(current_values)):
            raise SolutionError(
                'Pre-existing NaNs found '
                'in period with label: {} (index: {})'
                .format(self.span[t], t))

        # Solve: Add 1 to `t` to go from zero-based (Python) to one-based
        #        (Fortran) indexing
        solved_values, converged, iteration, error_code = (
            self.ENGINE.solve_t(self.values.astype(float),
                                t + 1,
                                max_iter,
                                tol,
                                [self.names.index(x) for x in self.CHECK],
                                error_options[errors]))

        converged = bool(converged)

        # Store the values back to this Python instance
        self.values = solved_values

        # Check solution result and error code
        if error_code == 0:
            if converged:
                status = '.'
            else:
                status = 'F'
        else:
            raise FortranEngineError(
                'Failed to solve model in period with label {} (index: {}), '
                'with uncaught error code {} after {} iteration(s)'
                .format(self.span[t], t, error_code, iteration))

        self.status[t] = status
        self.iterations[t] = iteration

        if status == 'F' and failures == 'raise':
            raise NonConvergenceError(
                'Solution failed to converge after {} iterations(s) '
                'in period with label: {} (index: {})'
                .format(iteration, self.span[t], t))

        if status == '.':
            return True
        else:
            return False

    def _evaluate(self, t: int, **kwargs: Dict[str, Any]) -> None:
        """Evaluate the system of equations for the period at integer position `t` in the model's `span`.

        Parameters
        ----------
        t : int
            Position in `span` of the period to solve
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
! `fsic_fortran`, a module of `fsic` (Version {version}).
!
! This module embeds the following equations in the `evaluate()` subroutine:
!
{system}

module structure
  implicit none

  integer :: lags = {lags}, leads = {leads}

end module structure

module error_codes
  implicit none

  integer :: success = 0

  ! Indexing / array bounds errors
  integer :: index_error_below = 11, index_error_above = 12
  integer :: index_error_lags = 13, index_error_leads = 14

end module error_codes


subroutine evaluate(initial_values, t, solved_values, error_code, nrows, ncols)
  use structure
  use error_codes
  implicit none

  ! `nrows` is the number of variables
  ! `ncols` is the number of periods
  integer, intent(in) :: nrows, ncols

  real(8), dimension(nrows, ncols), intent(in) :: initial_values
  integer, intent(in) :: t

  real(8), dimension(nrows, ncols), intent(out) :: solved_values
  integer, intent(out) :: error_code

  integer :: index

  ! Copy the initial values before evaluation
  ! (copy all values to avoid having to know which elements will change)
  solved_values = initial_values

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
  error_code = success

end subroutine evaluate


subroutine solve_t(initial_values, t, max_iter, tol, convergence_variables, error_control,  &
                &  solved_values, converged, iteration, error_code,                         &
                &  nrows, ncols, nvars)
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
  integer, intent(in) :: t, max_iter
  real(8), intent(in) :: tol
  integer, dimension(nvars), intent(in) :: convergence_variables
  integer, intent(in) :: error_control

  real(8), dimension(nrows, ncols), intent(out) :: solved_values
  logical, intent(out) :: converged
  integer, intent(out) :: iteration, error_code

  real(8), dimension(nrows, ncols) :: previous_values
  real(8), dimension(nvars) :: current_check, previous_check, diff_squared

  ! Copy the initial values before solution
  ! (copy all values to avoid having to know which elements will change)
  solved_values = initial_values
  converged = .false.

  ! Array of variable values to check for convergence
  current_check = solved_values(convergence_variables, t)

  ! Solve
  do iteration = 1, max_iter

     ! Save the values of the convergence variables
     previous_check = current_check

     ! Evaluate the system of equations
     previous_values = solved_values
     call evaluate(previous_values, t, solved_values, error_code, nrows, ncols)

     ! Get the new values of the convergence variables
     current_check = solved_values(convergence_variables, t)

     ! Check error code and return if a problem is found
     ! (indicated by a non-zero error code)
     if(error_code /= 0) then
        return
     end if

     ! Test for convergence
     diff_squared = (current_check - previous_check) ** 2

     if(all(diff_squared < tol)) then
        converged = .true.
        exit
     end if

  end do

  if(.not. converged) then
     iteration = iteration - 1
  end if

end subroutine solve_t
'''

def build_fortran_definition(symbols: List[Symbol], *, wrap_width: int = 100) -> str:
    """Return a string of Fortran code that embeds the equations in `symbols`."""
    # Separate variable names according to variable type
    endogenous = [s.name for s in symbols if s.type == Type.ENDOGENOUS]
    exogenous  = [s.name for s in symbols if s.type == Type.EXOGENOUS]
    parameters = [s.name for s in symbols if s.type == Type.PARAMETER]
    errors     = [s.name for s in symbols if s.type == Type.ERROR]

    # Set longest lag and lead
    non_function_symbols = [s for s in symbols if s.type != Type.FUNCTION]

    if len(non_function_symbols):
        lags =  abs(min(s.lags  for s in non_function_symbols))
        leads = abs(max(s.leads for s in non_function_symbols))
    else:
        lags = leads = 0

    # Map variable names to numbers (index positions)
    variables_to_numbers = {
        x: i
        for i, x in enumerate(itertools.chain(endogenous, exogenous, parameters, errors),
                              start=1)
    }

    # Generate code block of equations
    equation_code = []     # Code to insert into the Fortran subroutine
    equation_summary = []  # Summary for the header at the top of the file

    for s in filter(lambda x: x.type == Type.ENDOGENOUS, symbols):
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
    fortran_definition_string = fortran_template.format(
        system='\n'.join('!   ' + x for x in equation_summary),
        equations='\n\n'.join(equation_code),
        lags=lags, leads=leads,
        version=__version__,
    )

    return fortran_definition_string
