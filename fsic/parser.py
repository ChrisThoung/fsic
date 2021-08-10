# -*- coding: utf-8 -*-
"""
parser
======
"""

import enum
import itertools
import keyword
import re
import textwrap
from typing import Any, Callable, Dict, Iterator, List, Match, NamedTuple, Optional, Tuple
import warnings

import numpy as np

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
    # Match attempts to use reserved Python keywords as variable names (to be
    # raised as errors elsewhere)
    r'(?: (?P<_INVALID> (?: {}) \s* \[ .*? \]) )|'.format('|'.join(keyword.kwlist)) +

    # Match Python keywords
    r'(?: \b (?P<_KEYWORD> {} ) \b )|'.format('|'.join(keyword.kwlist)) +

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
    KEYWORD = enum.auto()

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
    index_: Optional[int]

    def __str__(self) -> str:
        """Standardised representation of the term."""
        expression = self.name

        # If neither a function nor a keyword, add the index
        if self.type not in (Type.FUNCTION, Type.KEYWORD):
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

        # If a function or a keyword, update the name; otherwise amend to be an
        # object attribute
        if self.type in (Type.FUNCTION, Type.KEYWORD):
            code = replacement_function_names.get(code, code)
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
            if (self.type  not in (Type.VARIABLE, Type.EXOGENOUS, Type.ENDOGENOUS) or
                other.type not in (Type.VARIABLE, Type.EXOGENOUS, Type.ENDOGENOUS)):
                raise SymbolError('''\
Unable to combine the following pair of symbols:
 - {}
 - {}

Variables cannot appear in the input script as both endogenous/exogenous
variables and parameters/errors etc.'''.format(self, other))

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

            if unmatched_parentheses < 0:
                raise ParserError('Found closing bracket before opening bracket '
                                  'while attempting to read the following: {}'.format('\n'.join(buffer)))

        # If complete, combine and yield
        if unmatched_parentheses == 0:
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
                            "Found unnecessary leading whitespace in equation: '{}'"
                            .format(equation))

                    # Otherwise, raise the general error
                    raise ParserError("Failed to parse equation: '{}'".format(equation))

                yield equation

            # Reset the buffer to collect another equation
            buffer = []

    # If `unmatched_parentheses` is non-zero at this point, there must have
    # been an error in the input script's syntax. Throw an error
    if unmatched_parentheses != 0:
        raise ParserError('Failed to identify any equations in the following, '
                          'owing to unmatched brackets: {}'.format('\n'.join(buffer)))

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
        if type_key not in ('_FUNCTION', '_KEYWORD'):
            if index_ is None:
                index_ = '0'

            try:
                index = int(index_)
            except ValueError:
                raise ParserError("Unable to parse index '{}' of '{}' in '{}'".format(
                    index_, match.group(0), expression))

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

    try:
        lhs_terms = [replace_type(t, Type.ENDOGENOUS) for t in parse_terms(left)]
    except ParserError:
        # Catch any parser errors at a term level and raise a further exception
        # to print the expression that failed
        raise ParserError("Failed to parse left-hand side of: '{}'".format(equation))

    try:
        rhs_terms = [replace_type(t, Type.EXOGENOUS) for t in parse_terms(right)]
    except ParserError:
        # Catch any parser errors at a term level and raise a further exception
        # to print the expression that failed
        raise ParserError("Failed to parse right-hand side of: '{}'".format(equation))

    if (any(filter(lambda x: x.type == Type.KEYWORD, lhs_terms)) or
        any(filter(lambda x: x.type == Type.INVALID, rhs_terms))):
        raise ParserError(
            'Equation uses one or more reserved Python keywords as variable '
            'names - these keywords are invalid for this purpose: `{}`'
            .format(equation))

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
            '`parse_equation()` expects a string that defines a single equation '
            'but found {} instead'.format(len(equations)))

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
    #  - str: the original statement
    #  - str: the (attempted) translation of the statement
    problem_statements: List[Tuple[int, str, str]] = []

    # Parse each statement (equation), optionally checking the syntax
    for i, statement in enumerate(split_equations_iter(model)):
        equation_symbols = parse_equation(statement)

        if check_syntax:
            equations = [s.equation for s in equation_symbols if s.equation is not None]

            for e in equations:
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
                                'Unexpected warning (error) when parsing: {}\n    {}'.format(statement, w[0]))
                    else:
                        raise ParserError(
                            'Unexpected number of warnings (errors) when parsing: {}'.format(statement))

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
            'Failed to parse the following {} statement(s):\n'.format(len(lines)) +
            '\n'.join(lines))

    # Store combined symbols to a dictionary, successively combining in the
    # loop below
    symbols: Dict[str, Symbol] = {}
    for symbol in itertools.chain(*symbols_by_equation):
        name = symbol.name
        symbols[name] = symbols.get(name, symbol).combine(symbol)

    return list(symbols.values())


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

    def _evaluate(self, t: int, *, errors: str = 'raise', iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
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

    def _evaluate(self, t, *, errors='raise', iteration=None, **kwargs):
{equations}\
'''

def build_model_definition(symbols: List[Symbol], *, converter: Optional[Callable[[Symbol], str]] = None, with_type_hints: bool = True) -> str:
    """Return a model class definition string from the contents of `symbols` (with type hints, by default).

    Parameters
    ----------
    symbols : list of fsic Symbol objects
        Constituent symbols of the model
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

        def _evaluate(self, t: int, *, errors: str = 'raise', iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
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

        def _evaluate(self, t: int, *, errors: str = 'raise', iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
            # Y[t] = C[t] + I[t] + G[t] + X[t] - M[t]
            _ = self._C[t] + self._I[t] + self._G[t] + self._X[t] - self._M[t]
            if _ > 0:  # Ignore negative values
                self._Y[t] = _
    """

    def default_converter(symbol: Symbol) -> str:
        """Return Python code for the current equation as an `exec`utable string."""
        return '''\
# {}
{}'''.format(symbol.equation, symbol.code)

    if converter is None:
        converter = default_converter

    if with_type_hints:
        model_template = model_template_typed
    else:
        model_template = model_template_untyped

    # Separate variable names according to variable type
    endogenous = [s.name for s in symbols if s.type == Type.ENDOGENOUS]
    exogenous  = [s.name for s in symbols if s.type == Type.EXOGENOUS]
    parameters = [s.name for s in symbols if s.type == Type.PARAMETER]
    errors     = [s.name for s in symbols if s.type == Type.ERROR]

    # Set longest lag and lead
    non_indexed_symbols = [s for s in symbols
                           if s.type not in (Type.FUNCTION, Type.KEYWORD)]

    if len(non_indexed_symbols) > 0:
        lags =  abs(min(s.lags  for s in non_indexed_symbols))
        leads = abs(max(s.leads for s in non_indexed_symbols))
    else:
        lags = leads = 0

    # Generate code block of equations
    expressions = [converter(s) for s in symbols if s.type == Type.ENDOGENOUS]
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

def build_model(symbols: List[Symbol], *, converter: Optional[Callable[[Symbol], str]] = None, with_type_hints: bool = True) -> 'BaseModel':
    """Return a model class definition from the contents of `symbols`. **Uses `exec()`.**

    Parameters
    ----------
    symbols : list of fsic Symbol objects
        Constituent symbols of the model
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
    from .core import BaseModel

    # Construct the class definition
    model_definition_string = build_model_definition(symbols,
                                                     converter=converter,
                                                     with_type_hints=with_type_hints)

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
