# -*- coding: utf-8 -*-
"""
Tools to generate a model class (which derives from `fsic.BaseModel`) from a
Python-like model script that describes the equations.

Key functions:

* `parse_model()` to convert a model script (string) to an intermediate
  representation: a list of `fsic` `Symbol` objects
    - the `fsic.tools` module includes functions to analyse these objects
* `build_model_definition()` to convert the list of `Symbol`s to a string of
  Python code
* `build_model()`, a convenience function to call `build_model_definition()`
  and `exec()` the code, returning an immediately useable working class
  definition

The parser provides a convenient way to specify a model as something like (a
string):

    'Y = C + I + G'

and convert it to the corresponding Python code:

    class Model(BaseModel):
        ENDOGENOUS: List[str] = ['Y']
        EXOGENOUS: List[str] = ['C', 'I', 'G']

        PARAMETERS: List[str] = []
        ERRORS: List[str] = []

        NAMES: List[str] = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
        CHECK: List[str] = ENDOGENOUS

        LAGS: int = 0
        LEADS: int = 0

        def _evaluate(self, t: int, *, errors: str = 'raise', iteration: Optional[int] = None, **kwargs: Any) -> None:
            # Y[t] = C[t] + I[t] + G[t]
            self._Y[t] = self._C[t] + self._I[t] + self._G[t]

The parser identifies:

* variables, both endogenous and exogenous
* parameters, as variables enclosed in braces e.g. '{alpha_1}'
* errors/residuals, as variables enclosed in angled brackets e.g. '<e>'
* the longest lags and leads in the system of equations, to prevent any
  attempts to solve for an infeasible period

and generates the necessary code for the `_evaluate()` method (as above).

Typical usage is to convert the string into a list of `fsic` `Symbol` objects
using `parse_model()`:

    >>> import fsic

    >>> symbols = fsic.parse_model('Y = C + I + G')
    >>> print(symbols)

    [Symbol(name='Y', type=<Type.ENDOGENOUS: 3>, lags=0, leads=0, equation='Y[t] = C[t] + I[t] + G[t]', code='self._Y[t] = self._C[t] + self._I[t] + self._G[t]'),
     Symbol(name='C', type=<Type.EXOGENOUS: 2>, lags=0, leads=0, equation=None, code=None),
     Symbol(name='I', type=<Type.EXOGENOUS: 2>, lags=0, leads=0, equation=None, code=None),
     Symbol(name='G', type=<Type.EXOGENOUS: 2>, lags=0, leads=0, equation=None, code=None)]

and to then convert those symbols into the final class definition, whether:

* directly, by returning a class definition, using `build_model()`
* by returning the Python code as a string using `build_model_definition()`
  e.g. to `exec()` manually or to print to a text file

    # Use `build_model()` to return a class definition
    >>> Model = fsic.build_model(symbols)     # Return the model class definition
    >>> model = Model(range(1945, 2010 + 1))  # Instantiate a model instance

    # Use `build_model_definition()` to return the code that defines the class
    # definition
    >>> model_definition = fsic.build_model_definition(symbols)
    >>> print(model_definition)

    class Model(BaseModel):
        ENDOGENOUS: List[str] = ['Y']
        EXOGENOUS: List[str] = ['C', 'I', 'G']

        PARAMETERS: List[str] = []
        ERRORS: List[str] = []

        NAMES: List[str] = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
        CHECK: List[str] = ENDOGENOUS

        LAGS: int = 0
        LEADS: int = 0

        def _evaluate(self, t: int, *, errors: str = 'raise', iteration: Optional[int] = None, **kwargs: Any) -> None:
            # Y[t] = C[t] + I[t] + G[t]
            self._Y[t] = self._C[t] + self._I[t] + self._G[t]

If using `build_model()` to generate and `exec()` the class definition, the
original code is accessible using the `CODE` attribute
(e.g. `Model.CODE`). Otherwise, the attribute is `None`.
"""

# `noqa` comments below avoid linters mistakenly finding imports needed for
# `exec` calls in this module
import enum
import itertools
import keyword
import re
import textwrap
import warnings
from typing import (
    Any,         # noqa: F401
    Callable,
    Dict,
    Iterator,
    List,
    Match,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)  # fmt: skip

import numpy as np  # noqa: F401

from .exceptions import BuildError, ParserError, SymbolError

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
        (?: ^ [`]{3,}\n .*? [`]{3,} $ )|  # Three (or more) backticks enclose multiline code to be inserted verbatim

        # Current implementation nests the single-line verbatim case in the below
        (?: ^ \( .*?      [=]        .*? \) (?= \s* ) $ )|  # Brackets beginning on the left-hand side
        (?: ^    \S+? \s* [=] \s* \( .*? \) (?= \s* ) $ )|  # Brackets beginning on the right-hand side
        (?: ^    \S+? \s* [=] \s*    .*?    (?= \s* ) $ )   # Equation on a single line
    ''',
    re.DOTALL | re.MULTILINE | re.VERBOSE,
)  # fmt: skip

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

keyword_list = '|'.join(keyword.kwlist)

term_re = re.compile(
    # Match verbatim code enclosed in backticks
    r'(?: (?P<_VERBATIM> [`] (.+?) [`]) )|'
    # Match attempts to use reserved Python keywords as variable names (to be
    # raised as errors elsewhere)
    rf'(?: (?P<_INVALID> (?: {keyword_list}) \s* \[ .*? \]) )|' +
    # Match Python keywords
    rf'(?: \b (?P<_KEYWORD> {keyword_list} ) \b )|'
    +
    # Valid terms for the parser
    r'''
        (?: (?P<_FUNCTION> [_A-Za-z][_A-Za-z0-9.]*[_A-Za-z0-9]* ) \s* (?= \( ) )|

        (?:
            (?: \{ \s* (?P<_PARAMETER> [_A-Za-z][_A-Za-z0-9]* ) \s* \} )|
            (?: \< \s* (?P<_ERROR>     [_A-Za-z][_A-Za-z0-9]* ) \s* \> )|
            (?:        (?P<_VARIABLE>  [_A-Za-z][_A-Za-z0-9]* )        )
        )
        (?: \[ \s* (?P<INDEX> .*? ) \s* \] )?
    ''',
    re.VERBOSE,
)  # fmt: skip


# Containers for term and symbol information ----------------------------------


class Type(enum.IntEnum):
    """Enumeration to track term/symbol types."""

    VARIABLE = enum.auto()
    EXOGENOUS = enum.auto()
    ENDOGENOUS = enum.auto()

    PARAMETER = enum.auto()
    ERROR = enum.auto()

    FUNCTION = enum.auto()
    KEYWORD = enum.auto()

    VERBATIM = enum.auto()
    INVALID = enum.auto()


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
    index_: Optional[Union[int, str]]

    def __str__(self) -> str:
        """Standardised representation of the term."""
        # If a function, keyword or verbatim block, just return the name
        if self.type in (Type.FUNCTION, Type.KEYWORD, Type.VERBATIM):
            return self.name

        # Otherwise, process and add the index
        if isinstance(self.index_, int):
            if self.index_ > 0:
                index = f'[t+{self.index_}]'
            elif self.index_ == 0:
                index = '[t]'
            else:
                index = f'[t{self.index_}]'

            return self.name + index

        if isinstance(self.index_, str):
            return f'{self.name}[{self.index_}]'

        raise TypeError(
            f'Type of `self.index_` is {type(self.index_)} '
            'but expected either `int` or `str`'
        )

    @property
    def code(self) -> str:
        """Representation of the term in code for model solution."""
        code = str(self)

        # If a function or a keyword, update the name; otherwise amend to be an
        # object attribute
        if self.type in (Type.FUNCTION, Type.KEYWORD):
            return replacement_function_names.get(code, code)

        # If a block of verbatim code, drop the enclosing backticks
        if self.type in (Type.VERBATIM,):
            return code.strip('`')

        # If the index is a string, treat as a named period
        if isinstance(self.index_, str):
            return f"self['{self.name}', {self.index_}]"

        # Otherwise, access as a regular internal variable
        return 'self._' + code


class Symbol(NamedTuple):
    """Container for information about a single symbol of an equation or model."""

    name: Optional[str]
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
                        f"Endogenous variable '{self.name}' defined twice:"
                        f'\n    {old}\n    {new}'
                    )

            elif old is None and new is not None:
                resolved = new

            return resolved

        assert self.name == other.name

        combined_type = self.type
        if self.type != other.type:
            if self.type not in (
                Type.VARIABLE,
                Type.EXOGENOUS,
                Type.ENDOGENOUS,
            ) or other.type not in (Type.VARIABLE, Type.EXOGENOUS, Type.ENDOGENOUS):
                raise SymbolError(
                    f'''\
Unable to combine the following pair of symbols:
 - {self}
 - {other}

Variables cannot appear in the input script as both endogenous/exogenous
variables and parameters/errors etc.'''
                )  # fmt: skip

            combined_type = max(self.type, other.type)

        def resolve_by_type_pair(this, that, function) -> Optional[Union[int, str]]:
            """Resolve the longest lag or lead according to the types of `this` and `that` (using `function()` if both are integers)."""
            types = type(this), type(that)

            if types == (type(None), type(None)):
                outcome = None
            elif types == (int, int):
                outcome = function(this, that, 0)
            elif types == (str, str):
                outcome = 0
            elif types == (int, str):
                outcome = this
            elif types == (str, int):
                outcome = that
            else:
                raise TypeError(f'Unhandled pair of types: {str(types)}')

            return outcome

        lags = resolve_by_type_pair(self.lags, other.lags, min)
        leads = resolve_by_type_pair(self.leads, other.leads, max)

        equation = resolve_strings(self.equation, other.equation)
        code = resolve_strings(self.code, other.code)

        return Symbol(
            name=self.name,
            type=combined_type,
            lags=lags,
            leads=leads,
            equation=equation,
            code=code,
        )


# Parser functions to convert strings to objects ------------------------------


def split_equations_iter(model: str) -> Iterator[str]:
    """Return the equations of `model` as an iterator of strings."""

    def strip_comments(line: str) -> str:
        hash_position = line.find('#')
        if hash_position == -1:
            return line

        return line[:hash_position].rstrip()

    # Conditions for extracting a complete set of statements:
    #  - unmatched_parentheses = 0 : no incomplete bracket pairs (typically for continuation lines)
    #  - complete_verbatim_block = True : no incomplete pairs of fences for verbatim code

    # Counter: Increment for every opening bracket and decrement for every
    # closing one. When zero, all pairs have been matched
    unmatched_parentheses = 0

    # If we encounter a fenced code block (three or more backticks), set
    # `complete_verbatim_block` to `False` until we find the corresponding
    # closing backticks
    complete_verbatim_block = True

    buffer = []

    for line in map(strip_comments, model.splitlines()):
        buffer.append(line)

        if line.startswith('```'):
            if len(buffer) == 1:  # Opening code fence
                complete_verbatim_block = False
                continue

            # Closing code fence
            complete_verbatim_block = True

        # Count unmatched parentheses
        for char in line:
            if char == '(':
                unmatched_parentheses += 1
            elif char == ')':
                unmatched_parentheses -= 1

            if unmatched_parentheses < 0:
                raise ParserError(
                    'Found closing bracket before opening bracket '
                    'while attempting to read the following: {}'.format(
                        '\n'.join(buffer)
                    )
                )

        # If complete, combine and yield
        if unmatched_parentheses == 0 and complete_verbatim_block:
            # Combine into a single string
            equation = '\n'.join(buffer)

            if equation.strip():  # Skip pure whitespace
                # Check that the string is a valid equation by testing against
                # the regular expression
                match = equation_re.search(equation)

                if not match:
                    # A failed match could occur because of unnecessary leading
                    # whitespace in the equation expression. Check for this and
                    # raise a slightly more helpful error if so
                    match = equation_re.search(equation.strip())

                    # Not compatible with Python 3.6: `if isinstance(match, re.Match):`
                    if match is not None:
                        raise IndentationError(
                            f"Found unnecessary leading whitespace in equation: '{equation}'"
                        )

                    # Otherwise, raise the general error
                    raise ParserError(f"Failed to parse equation: '{equation}'")

                yield equation

            # Reset the buffer to collect another equation
            buffer = []

    # If `unmatched_parentheses` is non-zero at this point, there must have
    # been an error in the input script's syntax. Throw an error
    if unmatched_parentheses != 0:
        raise ParserError(
            'Failed to identify any equations in the following, '
            'owing to unmatched brackets: ' + '\n'.join(buffer)
        )


def split_equations(model: str) -> List[str]:
    """Return the equations of `model` as a list of strings."""
    return list(split_equations_iter(model))


def parse_terms(expression: str) -> List[Term]:
    """Return the terms of `expression` as a list of Term objects."""

    def process_term_match(match: Match[str]) -> Term:
        """Return the contents of `match` as a Term object."""
        groupdict = match.groupdict()

        # Filter to the named group that matched for this term
        type_key_list = [
            k for k, v in groupdict.items() if k.startswith('_') and v is not None
        ]
        assert len(type_key_list) == 1
        type_key = type_key_list[0]

        # Add implicit index (0) to non-function terms
        index: Optional[Union[int, str]] = None

        index_ = groupdict['INDEX']
        if type_key not in ('_FUNCTION', '_KEYWORD'):
            # 1. No index (e.g. 'C'): Assume 0 (contemporaneous)
            if index_ is None:
                index = 0

            # 2. Quoted period (e.g. 'C['2000']): Leave unchanged (as a string)
            elif (index_.startswith("'") and index_.endswith("'")) or (
                index_.startswith('"') and index_.endswith('"')
            ):
                index = index_

            # 3. Verbatim index (e.g. 'C[`2000`], to handle an integer period):
            #    Strip backticks but retain as a string
            elif index_.startswith('`') and index_.endswith('`'):
                index = index_[1:-1]

            # 4. Anything else: Convert to `int`
            else:
                try:
                    index = int(index_)
                except ValueError:
                    raise ParserError(
                        f"Unable to parse index '{index_}' of "
                        f"'{match.group(0)}' in '{expression}'"
                    )

        return Term(name=groupdict[type_key], type=Type[type_key[1:]], index_=index)

    return [
        process_term_match(m)
        for m in term_re.finditer(expression)
        # Python keywords are unnamed groups: Filter these out
        if any(m.groups())
    ]


def parse_equation_terms(equation: str) -> List[Term]:
    """Return the terms of `equation` as a list of Term objects."""

    def replace_type(term: Term, new_type: Type) -> Term:
        if term.type == Type.VARIABLE:
            term = term._replace(type=new_type)
        return term

    left, right = equation.split('=', maxsplit=1)

    try:
        lhs_terms = [replace_type(t, Type.ENDOGENOUS) for t in parse_terms(left)]
    except ParserError as e:
        # Catch any parser errors at a term level and raise a further exception
        # to print the expression that failed
        raise ParserError(f"Failed to parse left-hand side of: '{equation}'") from e

    try:
        rhs_terms = [replace_type(t, Type.EXOGENOUS) for t in parse_terms(right)]
    except ParserError as e:
        # Catch any parser errors at a term level and raise a further exception
        # to print the expression that failed
        raise ParserError(f"Failed to parse right-hand side of: '{equation}'") from e

    if any(filter(lambda x: x.type == Type.KEYWORD, lhs_terms)) or any(
        filter(lambda x: x.type == Type.INVALID, rhs_terms)
    ):
        raise ParserError(
            'Equation uses one or more reserved Python keywords as variable '
            f'names - these keywords are invalid for this purpose: `{equation}`'
        )

    return lhs_terms + rhs_terms


def parse_equation(equation: str) -> List[Symbol]:
    """Return the symbols of `equation` as a list of Symbol objects."""
    # Return an empty list if the string is empty / pure whitespace
    if len(equation.strip()) == 0:
        return []

    # Use `split_equations()` to check that the expression is valid and only
    # contains one equation
    equations = split_equations(equation)
    if len(equations) != 1:
        raise ParserError(
            f'`parse_equation()` expects a string that defines a single equation '
            f'but found {len(equations)} instead'
        )

    # Insert verbatim code straight into a Symbol object and return
    if equation.startswith('`') and equation.endswith('`'):
        return [
            Symbol(
                name=None,
                type=Type.VERBATIM,
                lags=None,
                leads=None,
                equation=equation,
                code=equation.strip('`\r\n'),
            )
        ]

    # Check for complete bracket pairs
    bracket_pairs = [
        ('{', '}'),
    ]

    for opening, closing in bracket_pairs:
        count = 0

        for character in equation:
            if character == opening:
                count += 1

            elif character == closing:
                count -= 1

        if count != 0:
            raise ParserError(
                f"Found incomplete brackets ('{opening}', '{closing}') "
                f'in equation: {equation}'
            )

    # Extract the terms from the equation
    terms = parse_equation_terms(equation)

    # Construct standardised and code representations of the equation
    template = equation
    for match in reversed(list(term_re.finditer(equation))):
        # Skip Python keywords, which yield unnamed groups
        if not any(match.groups()):
            continue

        start, end = match.span()
        template = f'{template[:start]}{{}}{template[end:]}'

    # fmt: off
    template = re.sub(r'\s+',   ' ', template)  # Remove repeated whitespace
    template = re.sub(r'\(\s+', '(', template)  # Remove space after opening brackets
    template = re.sub(r'\s+\)', ')', template)  # Remove space before closing brackets
    # fmt: on

    equation = template.format(*[str(t) for t in terms])
    code = template.format(*[t.code for t in terms])

    # `symbols` stores the final symbols and is successively updated in the
    # loop below
    symbols: Dict[str, Symbol] = {}

    # `functions` keeps track of functions seen, to avoid duplicating entries
    # in `symbols`
    functions: Dict[str, Symbol] = {}

    for term in terms:
        # Skip verbatim terms: these shouldn't be converted to individual
        # symbols
        if term.type == Type.VERBATIM:
            continue

        symbol = Symbol(
            name=term.name,
            type=term.type,
            lags=term.index_,
            leads=term.index_,
            equation=None,
            code=None,
        )

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
    """Return the symbols of `model` as a list of Symbol objects.

    Notes
    -----
    A list of `Symbol` objects is an intermediate representation of a system of
    equations, identifying the constituent terms.

    `build_model()` and `build_model_definition()` convert these symbols to
    working Python code.

    The `fsic.tools` module provides functions to analyse the system from these
    symbols.

    See also
    --------
    fsic.build_model()
    fsic.build_model_definition()

    fsic.tools.symbols_to_dataframe()
    fsic.tools.symbols_to_graph()
    fsic.tools.symbols_to_sympy()
    """
    # Symbols: one list per model equation
    symbols_by_equation: List[List[Symbol]] = []

    # Store any statements that fail the (optional) syntax check as 2-tuples of
    #  - int: location of statement in `model`
    #  - str: the original statement
    #  - str: the (attempted) translation of the statement
    problem_statements: List[Tuple[int, str, str]] = []

    # Parse each statement (equation), optionally checking the syntax
    for i, statement in enumerate(split_equations_iter(model)):
        equation_symbols = parse_equation(statement)

        if check_syntax:
            equation_code = [s.code for s in equation_symbols if s.code is not None]

            for e in equation_code:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')

                    # Check for exceptions when trying to run the current equation
                    try:
                        exec(e)
                    except NameError:  # Ignore name errors (undefined variables)
                        pass
                    except SyntaxError:
                        problem_statements.append((i, statement, e))
                        break

                    # Check for warnings and treat them as errors
                    if len(w) == 0:
                        pass
                    elif len(w) == 1:
                        if issubclass(w[0].category, SyntaxWarning):
                            problem_statements.append((i, statement, e))
                            break
                        else:
                            raise ParserError(
                                f'Unexpected warning (error) when parsing: '
                                f'{statement}\n    {w[0]}'
                            )
                    else:
                        raise ParserError(
                            f'Unexpected number of warnings (errors) '
                            f'when parsing: {statement}'
                        )

        symbols_by_equation.append(equation_symbols)

    # Error if any problem statements found
    if problem_statements:
        # Construct report for each statement: number, original statement and
        # attempted equation
        lines = []
        for s in problem_statements:
            line = f'    {s[0]}:  {[1]}\n'
            line += ' ' * line.index(':')
            line += '-> ' + s[-1]
            lines.append(line)

        raise ParserError(
            f'Failed to parse the following {len(lines)} statement(s):\n'
            + '\n'.join(lines)
        )

    # Store combined symbols to a:
    #  - dictionary for regular symbols (to track replacements)
    #  - list for verbatim code
    symbols: Dict[str, Symbol] = {}
    verbatim: List[str] = []

    for symbol in itertools.chain(*symbols_by_equation):
        name = symbol.name

        if name is None:
            verbatim.append(symbol)
        else:
            symbols[name] = symbols.get(name, symbol).combine(symbol)

    return list(symbols.values()) + verbatim


# Model class generator -------------------------------------------------------

model_template_typed = '''\
class Model(BaseModel):
    ENDOGENOUS: List[str] = {endogenous}
    EXOGENOUS: List[str] = {exogenous}

    PARAMETERS: List[str] = {parameters}
    ERRORS: List[str] = {errors}

    NAMES: List[str] = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
    CHECK: List[str] = ENDOGENOUS

    LAGS: int = {lags}
    LEADS: int = {leads}

    def solve_t_before(self, t: int, *, errors: str = 'raise', catch_first_error: bool = True, iteration: Optional[int] = None, **kwargs: Any) -> None:
        """Pre-solution method: This runs each period, before the iterative solution. Over-ride to implement custom behaviour."""
        pass

    def solve_t_after(self, t: int, *, errors: str = 'raise', catch_first_error: bool = True, iteration: Optional[int] = None, **kwargs: Any) -> None:
        """Post-solution method: This runs each period, after the iterative solution. Over-ride to implement custom behaviour."""
        pass

    def _evaluate(self, t: int, *, errors: str = 'raise', catch_first_error: bool = True, iteration: Optional[int] = None, **kwargs: Any) -> None:
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
{equations}\
'''

model_template_untyped = '''\
class Model(BaseModel):
    ENDOGENOUS = {endogenous}
    EXOGENOUS = {exogenous}

    PARAMETERS = {parameters}
    ERRORS = {errors}

    NAMES = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
    CHECK = ENDOGENOUS

    LAGS = {lags}
    LEADS = {leads}

    def solve_t_before(self, t, *, errors='raise', catch_first_error=True, iteration=None, **kwargs):
        """Pre-solution method: This runs each period, before the iterative solution. Over-ride to implement custom behaviour."""
        pass

    def solve_t_after(self, t, *, errors='raise', catch_first_error=True, iteration=None, **kwargs):
        """Post-solution method: This runs each period, after the iterative solution. Over-ride to implement custom behaviour."""
        pass

    def _evaluate(self, t, *, errors='raise', catch_first_error=True, iteration=None, **kwargs):
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
{equations}\
'''


def build_model_definition(
    symbols: List[Symbol],
    *,
    lags: Optional[int] = None,
    leads: Optional[int] = None,
    min_lags: int = 0,
    min_leads: int = 0,
    converter: Optional[Callable[[Symbol], str]] = None,
    with_type_hints: bool = True,
) -> str:
    """Return a model class definition string from the contents of `symbols` (with type hints, by default).

    Parameters
    ----------
    symbols : list of fsic Symbol objects
        Constituent symbols of the model
    lags, leads : int, default `None`
        If passed (i.e. not `None`), impose the specified number of lags or
        leads on the model
    min_lags, min_leads : int, default 0
        Set the minimum lag or lead length on the model
    converter : (optional) callable, default `None`
        Mechanism for customising the Python code generator. If `None`,
        generates the equation as a comment followed by the equivalent Python
        code on the next line (see examples below). If a callable, takes a
        Symbol object as an input and must return a string of Python code for
        insertion into the model template.
    with_type_hints : bool
        If `True`, use the version of the model template with type hints. If
        `False`, exclude type hints (which may be convenient if looking to
        manipulate and/or execute the code directly).

    Notes
    -----
    The `converter` argument provides a way to alter the code generated at an
    equation level. The default converter just takes a Symbol object of type
    `ENDOGENOUS` and prints both the standardised equation as a comment, and
    the accompanying code. For example:

        # Y[t] = C[t] + I[t] + G[t] + X[t] - M[t]
        self._Y[t] = self._C[t] + self._I[t] + self._G[t] + self._X[t] - self._M[t]

    By passing a custom callable, you could generate something like the
    following instead, using the contents of the Symbol object:

        def custom_converter(symbol):
            lhs, rhs = map(str.strip, symbol.code.split('=', maxsplit=1))
            return '''\
# {}
_ = {}
if _ > 0:  # Ignore negative values
    {} = _'''.format(symbol.equation, rhs, lhs)

    Which would generate:

        # Y[t] = C[t] + I[t] + G[t] + X[t] - M[t]
        _ = self._C[t] + self._I[t] + self._G[t] + self._X[t] - self._M[t]
        if _ > 0:  # Ignore negative values
            self._Y[t] = _

    Examples
    --------
    >>> import fsic

    >>> symbols = fsic.parse_model('Y = C + I + G + X - M')
    >>> symbols
    [Symbol(name='Y', type=<Type.ENDOGENOUS: 3>, lags=0, leads=0, equation='Y[t] = C[t] + I[t] + G[t] + X[t] - M[t]', code='self._Y[t] = self._C[t] + self._I[t] + self._G[t] + self._X[t] - self._M[t]'),
     Symbol(name='C', type=<Type.EXOGENOUS: 2>, lags=0, leads=0, equation=None, code=None),
     Symbol(name='I', type=<Type.EXOGENOUS: 2>, lags=0, leads=0, equation=None, code=None),
     Symbol(name='G', type=<Type.EXOGENOUS: 2>, lags=0, leads=0, equation=None, code=None),
     Symbol(name='X', type=<Type.EXOGENOUS: 2>, lags=0, leads=0, equation=None, code=None),
     Symbol(name='M', type=<Type.EXOGENOUS: 2>, lags=0, leads=0, equation=None, code=None)]

    # With type hints (default)
    >>> print(fsic.build_model_definition(symbols))
    class Model(BaseModel):
        ENDOGENOUS: List[str] = ['Y']
        EXOGENOUS: List[str] = ['C', 'I', 'G', 'X', 'M']

        PARAMETERS: List[str] = []
        ERRORS: List[str] = []

        NAMES: List[str] = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
        CHECK: List[str] = ENDOGENOUS

        LAGS: int = 0
        LEADS: int = 0

        def _evaluate(self, t: int, *, errors: str = 'raise', iteration: Optional[int] = None, **kwargs: Any) -> None:
            # Y[t] = C[t] + I[t] + G[t] + X[t] - M[t]
            self._Y[t] = self._C[t] + self._I[t] + self._G[t] + self._X[t] - self._M[t]

    # Without type hints
    >>> print(fsic.build_model_definition(symbols, with_type_hints=False))
    class Model(BaseModel):
        ENDOGENOUS = ['Y']
        EXOGENOUS = ['C', 'I', 'G', 'X', 'M']

        PARAMETERS = []
        ERRORS = []

        NAMES = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
        CHECK = ENDOGENOUS

        LAGS = 0
        LEADS = 0

        def _evaluate(self, t, *, errors='raise', iteration=None, **kwargs):
            # Y[t] = C[t] + I[t] + G[t] + X[t] - M[t]
            self._Y[t] = self._C[t] + self._I[t] + self._G[t] + self._X[t] - self._M[t]

    # Custom code generator
    >>> def custom_converter(symbol):
    ...     lhs, rhs = map(str.strip, symbol.code.split('=', maxsplit=1))
    ...     return '''\
    ... # {}
    ... _ = {}
    ... if _ > 0:  # Ignore negative values
    ...     {} = _'''.format(symbol.equation, rhs, lhs)

    >>> print(fsic.build_model_definition(symbols, converter=custom_converter))
    class Model(BaseModel):
        ENDOGENOUS: List[str] = ['Y']
        EXOGENOUS: List[str] = ['C', 'I', 'G', 'X', 'M']

        PARAMETERS: List[str] = []
        ERRORS: List[str] = []

        NAMES: List[str] = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
        CHECK: List[str] = ENDOGENOUS

        LAGS: int = 0
        LEADS: int = 0

        def _evaluate(self, t: int, *, errors: str = 'raise', iteration: Optional[int] = None, **kwargs: Any) -> None:
            # Y[t] = C[t] + I[t] + G[t] + X[t] - M[t]
            _ = self._C[t] + self._I[t] + self._G[t] + self._X[t] - self._M[t]
            if _ > 0:  # Ignore negative values
                self._Y[t] = _
    """

    def default_converter(symbol: Symbol) -> str:
        """Return Python code for the current equation as an `exec`utable string."""
        return '''\
{}
{}'''.format(
            '\n'.join('# ' + x for x in symbol.equation.splitlines()), symbol.code
        )  # fmt: skip

    if converter is None:
        converter = default_converter

    if with_type_hints:
        model_template = model_template_typed
    else:
        model_template = model_template_untyped

    # Separate variable names according to variable type
    # fmt: off
    endogenous = [s.name for s in symbols if s.type == Type.ENDOGENOUS]
    exogenous  = [s.name for s in symbols if s.type == Type.EXOGENOUS]
    parameters = [s.name for s in symbols if s.type == Type.PARAMETER]
    errors     = [s.name for s in symbols if s.type == Type.ERROR]
    # fmt: on

    # Set longest lag and lead
    non_indexed_symbols = [
        s for s in symbols if s.type not in (Type.FUNCTION, Type.KEYWORD, Type.VERBATIM)
    ]

    # TODO: Force to int here?
    if lags is None:
        if len(non_indexed_symbols) > 0:
            lags = abs(min(s.lags for s in non_indexed_symbols))
        else:
            lags = 0

        lags = max(lags, min_lags)

    # TODO: Force to int here?
    if leads is None:
        if len(non_indexed_symbols) > 0:
            leads = abs(max(s.leads for s in non_indexed_symbols))
        else:
            leads = 0

        leads = max(leads, min_leads)

    # Generate code block of equations
    expressions = [
        converter(s)
        for s in symbols
        # Only convert symbols that are endogenous or verbatim *and*
        # include the corresponding code
        if s.type in (Type.ENDOGENOUS, Type.VERBATIM)
        and s.equation is not None
        and s.code is not None
    ]
    equations = '\n\n'.join(textwrap.indent(e, '        ') for e in expressions)

    # If there are no equations, insert `pass` instead
    if len(equations) == 0:
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


def build_model(
    symbols: List[Symbol],
    *,
    lags: Optional[int] = None,
    leads: Optional[int] = None,
    min_lags: int = 0,
    min_leads: int = 0,
    converter: Optional[Callable[[Symbol], str]] = None,
    with_type_hints: bool = True,
) -> 'BaseModel':  # noqa: F821
    """Return a model class definition from the contents of `symbols`. **Uses `exec()`.**

    Parameters
    ----------
    symbols : list of fsic Symbol objects
        Constituent symbols of the model
    lags, leads : int, default `None`
        If passed (i.e. not `None`), impose the specified number of lags or
        leads on the model
    min_lags, min_leads : int, default 0
        Set the minimum lag or lead length on the model
    converter : (optional) callable, default `None`
        Mechanism for customising the Python code generator. If `None`,
        generates the equation as a comment followed by the equivalent Python
        code on the next line. If a callable, takes a Symbol object as an input
        and must return a string of Python code for insertion into the model
        template.
        See the docstring for `build_model_definition()` for further details.
    with_type_hints : bool
        If `True`, use the version of the model template with type hints. If
        `False`, exclude type hints (which may be convenient if looking to
        manipulate and/or execute the code directly).

    Notes
    -----
    After constructing the class definition (as a string), this function calls
    `exec()` to define the class. In the event of a `SyntaxError`, the function
    loops through the `Symbol`s with defined equations and tries to construct
    and execute a class definition for each one, individually. The function
    then raises a `BuildError`, printing any `Symbol`s that failed to
    `exec`ute.
    """
    from .core import BaseModel  # noqa: F401

    # Construct the class definition
    model_definition_string = build_model_definition(
        symbols,
        lags=lags,
        leads=leads,
        min_lags=min_lags,
        min_leads=min_leads,
        converter=converter,
        with_type_hints=with_type_hints,
    )

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
                'Failed to `exec`ute the following `Symbol` object(s):\n'
                + '\n'.join('    {x}' for x in failed_execs)
            )

    # Otherwise, if here, assign the original code to an attribute and return
    # the class
    locals()['Model'].CODE = model_definition_string
    return locals()['Model']
