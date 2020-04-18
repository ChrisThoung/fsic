# -*- coding: utf-8 -*-
"""
fsic
====
Tools for macroeconomic modelling in Python.
"""

__version__ = '0.5.2.dev'


import copy
import enum
import itertools
import keyword
from numbers import Number
import re
from typing import Any, Dict, Hashable, Iterator, List, Match, NamedTuple, Optional, Sequence, Tuple, Union
import warnings

import numpy as np


# Custom exceptions -----------------------------------------------------------

class FSICError(Exception):
    pass


class BuildError(FSICError):
    pass

class DimensionError(FSICError):
    pass

class NonConvergenceError(FSICError):
    pass

class ParserError(FSICError):
    pass

class SolutionError(FSICError):
    pass


# Compiled regular expressions ------------------------------------------------

# Equations are Python-like statements that can take up a single line:
#
#     C = {alpha_1} * YD + {alpha_2} * H[-1]
#
# or span multiple lines using parentheses:
#
#     C = ({alpha_1} * YD +
#          {alpha_2} * H[-1])
#
#     (C =
#          {alpha_1} * YD +
#          {alpha_2} * H[-1])

equation_re = re.compile(
    r'''
        (?: ^ \( .*?      [=]        .*? \) (?= \s* ) $ )|  # Brackets beginning on the left-hand side
        (?: ^    \S+? \s* [=] \s* \( .*? \) (?= \s* ) $ )|  # Brackets beginning on the right-hand side
        (?: ^    \S+? \s* [=] \s*    .*?    (?= \s* ) $ )   # Equation on a single line
    ''',
    re.DOTALL | re.MULTILINE | re.VERBOSE
)

# Terms are valid Python identifiers. If they aren't functions, they may also
# have an index. For example:
#
#     C
#     H_d
#     V[-1]
#     exp()
#
# Enclose parameters in braces: {alpha_1}
#
# and errors in angle brackets: <epsilon>

term_re = re.compile(
    # Match Python keywords but without attaching a name (to filter out later)
    r'(?: \b (?: {} ) \b )|'.format('|'.join(keyword.kwlist)) +

    # Valid terms for the parser
    r'''
        (?: (?P<_FUNCTION> [_A-Za-z][_A-Za-z0-9]* ) \s* (?= \( ) )|

        (?:
            (?: \{ \s* (?P<_PARAMETER> [_A-Za-z][_A-Za-z0-9]* ) \s* \} )|
            (?: \< \s* (?P<_ERROR>     [_A-Za-z][_A-Za-z0-9]* ) \s* \> )|
            (?:        (?P<_VARIABLE>  [_A-Za-z][_A-Za-z0-9]* )        )
        )
        (?: \[ \s* (?P<INDEX> .*? ) \s* \] )?
    ''',
    re.VERBOSE
)


# Containers for term and symbol information ----------------------------------

class Type(enum.IntEnum):
    """Enumeration to track term/symbol types."""
    VARIABLE = enum.auto()
    EXOGENOUS = enum.auto()
    ENDOGENOUS = enum.auto()

    PARAMETER = enum.auto()
    ERROR = enum.auto()

    FUNCTION = enum.auto()


# Replacement functions for model solution
replacement_function_names = {
    'exp': 'np.exp',
    'log': 'np.log',
    'max': 'max',
    'min': 'min',
}

class Term(NamedTuple):
    """Container for information about a single term of an equation."""
    name: str
    type: Type
    index_: Optional[int]

    def __str__(self) -> str:
        """Standardised representation of the term."""
        expression = self.name

        # If not a function, add the index
        if self.type != Type.FUNCTION:
            assert self.index_ is not None

            if self.index_ > 0:
                index = '[t+{}]'.format(self.index_)
            elif self.index_ == 0:
                index = '[t]'
            else:
                index = '[t{}]'.format(self.index_)

            expression += index

        return expression

    @property
    def code(self) -> str:
        """Representation of the term in code for model solution."""
        code = str(self)

        # If a function, update the name; otherwise amend to be an object
        # attribute
        if self.type == Type.FUNCTION:
            for k, v in replacement_function_names.items():
                code = code.replace(k, v)
        else:
            code = 'self._' + code

        return code


class Symbol(NamedTuple):
    """Container for information about a single symbol of an equation or model."""
    name: str
    type: Type
    lags: Optional[int]
    leads: Optional[int]
    equation: Optional[str]
    code: Optional[str]

    def combine(self, other: 'Symbol') -> 'Symbol':
        """Combine the attributes of two symbols."""

        def resolve_strings(old: Optional[str], new: Optional[str]) -> Optional[str]:
            resolved = old

            if old is not None and new is not None:
                if old != new:
                    raise ParserError(
                        "Endogenous variable '{}' defined twice:\n    {}\n    {}"
                        .format(self.name, old, new))

            elif old is None and new is not None:
                resolved = new

            return resolved

        assert self.name == other.name

        combined_type = self.type
        if self.type != other.type:
            assert self.type in (Type.VARIABLE, Type.EXOGENOUS, Type.ENDOGENOUS)
            assert other.type in (Type.VARIABLE, Type.EXOGENOUS, Type.ENDOGENOUS)
            combined_type = max(self.type, other.type)

        if self.lags is None:
            assert other.lags is None
            lags = None
        else:
            lags = min(self.lags, other.lags, 0)

        if self.leads is None:
            assert other.leads is None
            leads = None
        else:
            leads = max(self.leads, other.leads, 0)

        equation = resolve_strings(self.equation, other.equation)
        code = resolve_strings(self.code, other.code)

        return Symbol(name=self.name,
                      type=combined_type,
                      lags=lags,
                      leads=leads,
                      equation=equation,
                      code=code)


# Parser functions to convert strings to objects ------------------------------

def split_equations_iter(model: str) -> Iterator[str]:
    """Return the equations of `model` as an iterator of strings."""

    def strip_comments(line: str) -> str:
        hash_position = line.find('#')
        if hash_position == -1:
            return line
        else:
            return line[:hash_position].rstrip()

    # Counter: Increment for every opening bracket and decrement for every
    # closing one. When zero, all pairs have been matched
    unmatched_parentheses = 0

    buffer = []

    for line in map(strip_comments, model.splitlines()):
        buffer.append(line)

        # Count unmatched parentheses
        for char in line:
            if char == '(':
                unmatched_parentheses += 1
            elif char == ')':
                unmatched_parentheses -= 1

        # If complete, combine and yield
        if unmatched_parentheses == 0:
            equation = '\n'.join(buffer)
            if equation.strip():
                assert equation_re.search(equation)
                yield equation
            buffer = []

def split_equations(model: str) -> List[str]:
    """Return the equations of `model` as a list of strings."""
    return list(split_equations_iter(model))


def parse_terms(expression: str) -> List[Term]:
    """Return the terms of `expression` as a list of Term objects."""

    def process_term_match(match: Match[str]) -> Term:
        """Return the contents of `match` as a Term object."""
        groupdict = match.groupdict()

        # Filter to the named group that matched for this term
        type_key_list = [k for k, v in groupdict.items()
                         if k.startswith('_') and v is not None]
        assert len(type_key_list) == 1
        type_key = type_key_list[0]

        # Add implicit index (0) to non-function terms
        index: Optional[int] = None

        index_ = groupdict['INDEX']
        if type_key != '_FUNCTION':
            if index_ is None:
                index_ = '0'
            index = int(index_)

        return Term(name=groupdict[type_key],
                    type=Type[type_key[1:]],
                    index_=index)

    return [process_term_match(m)
            for m in term_re.finditer(expression)
            # Python keywords are unnamed groups: Filter these out
            if any(m.groups())]

def parse_equation_terms(equation: str) -> List[Term]:
    """Return the terms of `equation` as a list of Term objects."""

    def replace_type(term: Term, new_type: Type) -> Term:
        if term.type == Type.VARIABLE:
            term = term._replace(type=new_type)
        return term

    left, right = equation.split('=', maxsplit=1)

    lhs_terms = [replace_type(t, Type.ENDOGENOUS) for t in parse_terms(left)]
    rhs_terms = [replace_type(t, Type.EXOGENOUS) for t in parse_terms(right)]

    return lhs_terms + rhs_terms

def parse_equation(equation: str) -> List[Symbol]:
    """Return the symbols of `equation` as a list of Symbol objects."""
    terms = parse_equation_terms(equation)

    # Construct standardised and code representations of the equation
    template = equation
    for match in reversed(list(term_re.finditer(equation))):
        # Skip Python keywords, which yield unnamed groups
        if not any(match.groups()):
            continue

        start, end = match.span()
        template = '{}{}{}'.format(template[:start], '{}', template[end:])

    template = re.sub(r'\s+',   ' ', template)  # Remove repeated whitespace
    template = re.sub(r'\(\s+', '(', template)  # Remove space after opening brackets
    template = re.sub(r'\s+\)', ')', template)  # Remove space before closing brackets

    equation = template.format(*[str(t) for t in terms])
    code = template.format(*[t.code for t in terms])

    # `symbols` stores the final symbols and is successively updated in the
    # loop below
    symbols: Dict[str, Symbol] = {}

    # `functions` keeps track of functions seen, to avoid duplicating entries
    # in `symbols`
    functions: Dict[str, Symbol] = {}

    for term in terms:
        symbol = Symbol(name=term.name,
                        type=term.type,
                        lags=term.index_,
                        leads=term.index_,
                        equation=None,
                        code=None)

        name = symbol.name

        if symbol.type == Type.FUNCTION:
            # Function previously encountered: Test for equality against the
            # previous entry
            if name in functions:
                assert symbol == functions[name]
            # Otherwise, store
            else:
                symbols[name] = symbol
                functions[name] = symbol
            continue

        # Update endogenous variables with the equation and code information
        # from above
        if symbol.type == Type.ENDOGENOUS:
            symbol = symbol._replace(equation=equation, code=code)

        symbols[name] = symbols.get(name, symbol).combine(symbol)

    return list(symbols.values())

def parse_model(model: str, *, check_syntax: bool = True) -> List[Symbol]:
    """Return the symbols of `model` as a list of Symbol objects."""
    # Symbols: one list per model equation
    symbols_by_equation: List[List[Symbol]] = []

    # Store any statements that fail the (optional) syntax check as 2-tuples of
    #  - int: location of statement in `model`
    #  - str: the statement itself
    problem_statements: List[Tuple[int, str, str]] = []

    # Parse each statement (equation), optionally checking the syntax
    for i, statement in enumerate(split_equations_iter(model)):
        equation_symbols = parse_equation(statement)

        if check_syntax:
            equations = [s.equation for s in equation_symbols if s.equation is not None]

            for e in equations:
                try:
                    exec(e)
                except NameError:  # Ignore name errors (undefined variables)
                    pass
                except SyntaxError:
                    problem_statements.append((i, statement, e))
                    break

        symbols_by_equation.append(equation_symbols)

    # Error if any problem statements found
    if problem_statements:
        # Construct report for each statement: number, original statement and
        # attempted equation
        lines = []
        for s in problem_statements:
            line = '    {}:  {}\n'.format(*s[:2])
            line += ' ' * line.index(':')
            line += '-> ' + s[-1]
            lines.append(line)

        raise ParserError(
            'Failed to parse the following statements:\n' +
            '\n'.join(lines))

    # Store combined symbols to a dictionary, successively combining in the
    # loop below
    symbols: Dict[str, Symbol] = {}
    for symbol in itertools.chain(*symbols_by_equation):
        name = symbol.name
        symbols[name] = symbols.get(name, symbol).combine(symbol)

    return list(symbols.values())


# Base class for individual models --------------------------------------------

class BaseModel:
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

    def __init__(self, span: Sequence[Hashable], *, dtype: Any = float, default_value: Union[int, float] = 0.0, **initial_values: Dict[str, Any]) -> None:
        """Initialise model variables.

        Parameters
        ----------
        span : iterable
            Sequence of periods that define the timespan of the model
        dtype : variable type
            Data type to impose on model variables (in NumPy arrays)
        default_value : number
            Value with which to initialise model variables
        **initial_values : keyword arguments of variable names and values
            Values with which to initialise specific named model variables
        """
        # Initialise model attributes
        self.__dict__['span'] = span
        self.__dict__['names'] = self.NAMES

        self.__dict__['_status'] = np.full(len(self.span), '-', dtype=str)
        self.__dict__['_iterations'] = np.full(len(self.span), -1, dtype=int)

        self.__dict__['_dtype'] = dtype

        # Initialise model variables with a leading underscore in the
        # name. `__getattr__()` and `__setattr__()` provide indirect access via
        # the non-underscored names
        for name in self.names:
            value = initial_values.get(name, default_value)

            if isinstance(value, Sequence):
                value_as_array = np.array(value).flatten()
            else:
                value_as_array = np.full(len(self.span), value, dtype=self._dtype)

            if value_as_array.shape[0] != len(self.span):
                raise DimensionError("Invalid assignment for '{}': "
                                     "must be either a number or "
                                     "a sequence of identical length "
                                     "(expect {} elements)".format(
                                         name, len(self.span)))

            self.__setattr__('_' + name, value_as_array)

    def __getattr__(self, name: str) -> Any:
        """Over-riding method to provide indirect access to internal model
        variables as needed."""
        if (name in self.__dict__['names'] or
            name in ['status', 'iterations']):
            return self.__dict__['_' + name]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Over-riding method to provide indirect access to internal model
        variables as needed."""
        if (name in self.__dict__['names'] or
            name == 'iterations'):
            if isinstance(value, Number):
                self.__dict__['_' + name][:] = value
            elif len(self.__dict__['_' + name]) == len(value):
                self.__dict__['_' + name] = np.array(
                    value,
                    dtype=self.__dict__['_' + name].dtype)
            else:
                raise DimensionError("Invalid assignment for '{}': "
                                     "must be either a number or "
                                     "a sequence of identical length "
                                     "(expect {} elements)".format(
                                         name, len(self.span)))

        elif name == 'status':
            if isinstance(value, str):
                self.__dict__['_' + name][:] = value
            elif len(self.__dict__['_' + name]) == len(value):
                self.__dict__['_' + name] = np.array(
                    value,
                    dtype=self.__dict__['_' + name].dtype)
            else:
                raise DimensionError("Invalid assignment for '{}': "
                                     "must be either a string or "
                                     "a sequence of identical length "
                                     "(expect {} elements)".format(
                                         name, len(self.span)))

        else:
            super.__setattr__(self, name, value)

    def __getitem__(self, key: str) -> Any:
        if isinstance(key, str):
            if key in self.names:
                return self.__getattr__(key)
            else:
                raise KeyError("'{}' not a model variable".format(key))
        else:
            assert len(key) == 2
            name, index = key
            values = self.__getattr__(name)

            if isinstance(index, slice):
                start, stop, step = index.start, index.stop, index.step

                if start is None:
                    start = self.span[0]
                if stop is None:
                    stop = self.span[-1]
                if step is None:
                    step = 1

                start_location = self.span.index(start)
                stop_location = self.span.index(stop) + 1

                return values[start_location:stop_location:step]
            else:
                location = self.span.index(index)
                return values[location]

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(key, str):
            self.__setattr__(key, value)
        else:
            assert len(key) == 2
            name, index = key

            if isinstance(index, slice):
                start, stop, step = index.start, index.stop, index.step

                if start is None:
                    start = self.span[0]
                if stop is None:
                    stop = self.span[-1]
                if step is None:
                    step = 1

                start_location = self.span.index(start)
                stop_location = self.span.index(stop) + 1

                self.__dict__['_' + name][start_location:stop_location:step] = value
            else:
                location = self.span.index(index)
                self.__dict__['_' + name][location] = value

    def copy(self) -> 'BaseModel':
        """Return a copy of the current (state of the) model instance."""
        copied = self.__class__(span=copy.deepcopy(self.span))
        copied.__dict__.update(
            {k: copy.deepcopy(v)
             for k, v in self.__dict__.items()})
        return copied

    __copy__ = copy
    __deepcopy__ = copy

    def __dir__(self) -> List[str]:
        return sorted(
            dir(type(self)) +
            self.__dict__['names'] +
            ['span', 'names', 'status', 'iterations'])

    def _ipython_key_completions_(self) -> List[str]:
        return self.__dict__['names']

    @property
    def values(self) -> np.ndarray:
        """Model variable values as a 2D (variables x time) array."""
        return np.array([
            self.__getattribute__('_' + name) for name in self.names
        ])

    def solve(self, *, start: Optional[Hashable] = None, end: Optional[Hashable] = None, max_iter: int = 100, tol: Union[int, float] = 1e-10, offset: int = 0, failures: str = 'raise', errors: str = 'raise', **kwargs: Dict[str, Any]) -> Tuple[List[Hashable], List[int], List[bool]]:
        """Solve the model. Use default periods if none provided.

        Parameters
        ----------
        start : element in the model's `span`
            First period to solve. If not given, defaults to the first solvable
            period, taking into account any lags in the model's equations
        end : element in the model's `span`
            Last period to solve. If not given, defaults to the last solvable
            period, taking into account any leads in the model's equations
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
            e.g. NaNs
             - 'raise' (default): stop immediately and raise a `SolutionError`
                                  [set current period solution status to 'E']
             - 'skip': stop solving the current period and move to the next one
                       [set current period solution status to 'S']
             - 'ignore': continue solving, with no special treatment or action
                         [period solution statuses as usual i.e. '.' or 'F']
             - 'replace': each iteration, replace NaNs with zeroes
                         [period solution statuses as usual i.e. '.' or 'F']
        kwargs :
            Further keyword arguments to pass to the solution routines

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
        """
        # Default start and end periods
        if start is None:
            start = self.span[self.LAGS]
        if end is None:
            end = self.span[-1 - self.LEADS]

        # Convert to integer positions
        start_t = self.span.index(start)
        end_t = self.span.index(end)

        # Initialise lists of return values
        indexes = list(range(start_t, end_t + 1))
        labels = [self.span[i] for i in indexes]
        solved = [False] * len(indexes)

        # Solve
        for i, t in enumerate(indexes):
            solved[i] = self.solve_t(t, max_iter=max_iter, tol=tol, offset=offset,
                                     failures=failures, errors=errors, **kwargs)

        return labels, indexes, solved

    def solve_period(self, period: Hashable, *, max_iter: int = 100, tol: Union[int, float] = 1e-10, offset: int = 0, failures: str = 'raise', errors: str = 'raise', **kwargs: Dict[str, Any]) -> bool:
        """Solve a single period.

        Parameters
        ----------
        period : element in the model's `span`
            Named period to solve
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
        """
        t = self.span.index(period)
        return self.solve_t(t, max_iter=max_iter, tol=tol, offset=offset, failures=failures, errors=errors, **kwargs)

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
        if errors == 'raise' and np.any(np.isnan(current_values)):
            raise SolutionError(
                'Pre-existing NaNs found '
                'in period with label: {} (index: {})'
                .format(self.span[t], t))

        for iteration in range(1, max_iter + 1):
            previous_values = current_values.copy()

            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                self._evaluate(t, **kwargs)

            current_values = get_check_values()

            # It's possible that the current iteration generated no NaNs, but
            # the previous one did: check and continue if needed
            if np.any(np.isnan(previous_values)):
                continue

            if np.any(np.isnan(current_values)):

                if errors == 'raise':
                    self.status[t] = 'E'
                    self.iterations[t] = iteration

                    raise SolutionError(
                        'Numerical solution error after {} iterations(s) '
                        'in period with label: {} (index: {})'
                        .format(iteration, self.span[t], t))

                elif errors == 'skip':
                    status = 'S'
                    break

                elif errors == 'ignore':
                    if iteration == max_iter:
                        status = 'F'
                        break
                    continue

                elif errors == 'replace':
                    if iteration == max_iter:
                        status = 'F'
                        break
                    else:
                        current_values[np.isnan(current_values)] = 0.0
                        continue

                else:
                    raise ValueError('Invalid `errors` argument: {}'.format(errors))

            diff = current_values - previous_values
            diff_squared = diff ** 2

            if np.all(diff_squared < tol):
                status = '.'
                break
        else:
            status = 'F'

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
            Further keyword arguments for solution
        """
        raise NotImplementedError('Method must be over-ridden by a child class')


# Model class generator -------------------------------------------------------

model_template = '''\
class Model(BaseModel):
    ENDOGENOUS = {endogenous}
    EXOGENOUS = {exogenous}

    PARAMETERS = {parameters}
    ERRORS = {errors}

    NAMES = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
    CHECK = ENDOGENOUS

    LAGS = {lags}
    LEADS = {leads}

    def _evaluate(self, t):
{equations}\
'''

def build_model_definition(symbols: List[Symbol]) -> str:
    """Return a model class definition string from the contents of `symbols`."""
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

    # Generate code block of equations
    expressions = [s.code for s in symbols if s.type == Type.ENDOGENOUS]
    equations = '\n'.join(['        {}'.format(e) for e in expressions])

    # If there are no equations, insert `pass` instead
    if not len(equations):
        equations = '        pass'

    # Fill in class template
    model_definition_string = model_template.format(
        endogenous=endogenous,
        exogenous=exogenous,

        parameters=parameters,
        errors=errors,

        lags=lags,
        leads=leads,

        equations=equations,
    )

    return model_definition_string

def build_model(symbols: List[Symbol]) -> Any:
    """Return a model class definition from the contents of `symbols`. **Uses `exec()`.**

    Notes
    -----
    After constructing the class definition (as a string), this function calls
    `exec()` to define the class. In the event of a `SyntaxError`, the function
    loops through the `Symbol`s with defined equations and tries to construct
    and execute a class definition for each one, individually. The function
    then raises a `BuildError`, printing any `Symbol`s that failed to
    `exec`ute.
    """
    # Construct the class definition
    model_definition_string = build_model_definition(symbols)

    # Execute the class definition code
    try:
        exec(model_definition_string)
    except SyntaxError:
        failed_execs: List[str] = []

        symbols_with_equations = list(filter(lambda x: x.equation is not None, symbols))
        for s in symbols_with_equations:
            try:
                exec(build_model_definition([s]))
            except SyntaxError:
                failed_execs.append(str(s))

        if failed_execs:
            raise BuildError(
                'Failed to `exec`ute the following `Symbol` object(s):\n' +
                '\n'.join('    {}'.format(x) for x in failed_execs))

    # Otherwise, if here, assign the original code to an attribute and return
    # the class
    locals()['Model'].CODE = model_definition_string
    return locals()['Model']
