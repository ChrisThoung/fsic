# -*- coding: utf-8 -*-
"""
test_core
=========
Test suite for core fsic code.

Example equations/models are adapted from:

    Almon, C. (2017)
    *The craft of economic modeling*,
    Third, enlarged edition, *Inforum*
    http://www.inforum.umd.edu/papers/TheCraft.html

    Godley, W., Lavoie, M. (2007)
    *Monetary economics: An integrated approach to credit, money, income, production and wealth*,
    Palgrave Macmillan

Godley and Lavoie (2007) implementation also informed by Gennaro Zezza's EViews
programs:

    http://gennaro.zezza.it/software/eviews/glch03.php
"""

import copy
import keyword
import math
import sys
import unittest

import numpy as np

from fsic.exceptions import SolutionError
import fsic


pandas_installed = True

try:
    import pandas as pd
except ModuleNotFoundError:
    pandas_installed = False


class TestRegexes(unittest.TestCase):
    def test_equation_re(self):
        # Test that the equation regex correctly identifies individual
        # equations in a system
        script = """
(C =
     {alpha_1} * YD +
     {alpha_2} * H[-1])
YD = (Y -
      T)
Y = C + G
T = {theta} * Y
H = (
     H[-1] + YD - C
)

`self.Y[t] = self.C[t] + self.G[t]`
`self._Y[t] = self._C[t] + self._G[t]`

```
self.Y[t] = (self.C[t] +
             self.G[t])
```

```
self._Y[t] = (self._C[t] +
              self._G[t])
```
"""
        expected = [
            '(C =\n     {alpha_1} * YD +\n     {alpha_2} * H[-1])',
            'YD = (Y -\n      T)',
            'Y = C + G',
            'T = {theta} * Y',
            'H = (\n     H[-1] + YD - C\n)',
            '`self.Y[t] = self.C[t] + self.G[t]`',
            '`self._Y[t] = self._C[t] + self._G[t]`',
            """```
self.Y[t] = (self.C[t] +
             self.G[t])
```""",
            """```
self._Y[t] = (self._C[t] +
              self._G[t])
```""",
        ]

        self.assertEqual(
            [match.group(0) for match in fsic.parser.equation_re.finditer(script)],
            expected,
        )


class TestTerm(unittest.TestCase):
    def test_str(self):
        # Check that `str` representations are as expected
        self.assertEqual(
            str(fsic.parser.Term('C', fsic.parser.Type.VARIABLE, -1)), 'C[t-1]'
        )
        self.assertEqual(
            str(fsic.parser.Term('C', fsic.parser.Type.VARIABLE, 0)), 'C[t]'
        )
        self.assertEqual(
            str(fsic.parser.Term('C', fsic.parser.Type.VARIABLE, 1)), 'C[t+1]'
        )


class TestParsers(unittest.TestCase):
    def test_split_equations(self):
        # Test that `split_equations()` correctly identifies individual
        # equations in a system
        script = """
(C =
     {alpha_1} * YD +
     {alpha_2} * H[-1])
YD = (Y -
      T)
Y = C + G
T = {theta} * Y
H = (
     H[-1] + YD - C
)

# Duplicates of the household wealth equation spanning multiple lines (using
# brackets)
H = H[-1] + (  # Also use to check comments handling
    YD - C)
H = (H[-1] +
     YD) - C
"""
        expected = [
            '(C =\n     {alpha_1} * YD +\n     {alpha_2} * H[-1])',
            'YD = (Y -\n      T)',
            'Y = C + G',
            'T = {theta} * Y',
            'H = (\n     H[-1] + YD - C\n)',
            'H = H[-1] + (\n    YD - C)',
            'H = (H[-1] +\n     YD) - C',
        ]

        self.assertEqual(fsic.parser.split_equations(script), expected)

    def test_parse_terms(self):
        # Test that `parse_terms()` correctly identifies individual terms in an
        # expression
        expression = 'C = exp({alpha_1} * log(YD) + {alpha_2} * log(H[-1]) + <epsilon>)'
        expected = [
            fsic.parser.Term(name='C', type=fsic.parser.Type.VARIABLE, index_=0),
            fsic.parser.Term(name='exp', type=fsic.parser.Type.FUNCTION, index_=None),
            fsic.parser.Term(name='alpha_1', type=fsic.parser.Type.PARAMETER, index_=0),
            fsic.parser.Term(name='log', type=fsic.parser.Type.FUNCTION, index_=None),
            fsic.parser.Term(name='YD', type=fsic.parser.Type.VARIABLE, index_=0),
            fsic.parser.Term(name='alpha_2', type=fsic.parser.Type.PARAMETER, index_=0),
            fsic.parser.Term(name='log', type=fsic.parser.Type.FUNCTION, index_=None),
            fsic.parser.Term(name='H', type=fsic.parser.Type.VARIABLE, index_=-1),
            fsic.parser.Term(name='epsilon', type=fsic.parser.Type.ERROR, index_=0),
        ]

        self.assertEqual(fsic.parser.parse_terms(expression), expected)

    def test_parse_equation_terms(self):
        # Test that `parse_equation_terms()` correctly identifies individual
        # terms in an equation
        equation = 'C = exp({alpha_1} * log(YD) + {alpha_2} * log(H[-1]) + <epsilon>)'
        expected = [
            fsic.parser.Term(name='C', type=fsic.parser.Type.ENDOGENOUS, index_=0),
            fsic.parser.Term(name='exp', type=fsic.parser.Type.FUNCTION, index_=None),
            fsic.parser.Term(name='alpha_1', type=fsic.parser.Type.PARAMETER, index_=0),
            fsic.parser.Term(name='log', type=fsic.parser.Type.FUNCTION, index_=None),
            fsic.parser.Term(name='YD', type=fsic.parser.Type.EXOGENOUS, index_=0),
            fsic.parser.Term(name='alpha_2', type=fsic.parser.Type.PARAMETER, index_=0),
            fsic.parser.Term(name='log', type=fsic.parser.Type.FUNCTION, index_=None),
            fsic.parser.Term(name='H', type=fsic.parser.Type.EXOGENOUS, index_=-1),
            fsic.parser.Term(name='epsilon', type=fsic.parser.Type.ERROR, index_=0),
        ]

        self.assertEqual(fsic.parser.parse_equation_terms(equation), expected)

    def test_parse_equation(self):
        # Test that `parse_equation()` correctly identifies the symbols that
        # make up an equation
        equation = 'C = exp({alpha_1} * log(YD) + {alpha_2} * log(H[-1]) + <epsilon>)'
        expected = [
            fsic.parser.Symbol(
                name='C',
                type=fsic.parser.Type.ENDOGENOUS,
                lags=0,
                leads=0,
                equation='C[t] = exp(alpha_1[t] * log(YD[t]) + '
                'alpha_2[t] * log(H[t-1]) + '
                'epsilon[t])',
                code='self._C[t] = np.exp(self._alpha_1[t] * np.log(self._YD[t]) + '
                'self._alpha_2[t] * np.log(self._H[t-1]) + '
                'self._epsilon[t])',
            ),
            fsic.parser.Symbol(
                name='exp',
                type=fsic.parser.Type.FUNCTION,
                lags=None,
                leads=None,
                equation=None,
                code=None,
            ),
            fsic.parser.Symbol(
                name='alpha_1',
                type=fsic.parser.Type.PARAMETER,
                lags=0,
                leads=0,
                equation=None,
                code=None,
            ),
            fsic.parser.Symbol(
                name='log',
                type=fsic.parser.Type.FUNCTION,
                lags=None,
                leads=None,
                equation=None,
                code=None,
            ),
            fsic.parser.Symbol(
                name='YD',
                type=fsic.parser.Type.EXOGENOUS,
                lags=0,
                leads=0,
                equation=None,
                code=None,
            ),
            fsic.parser.Symbol(
                name='alpha_2',
                type=fsic.parser.Type.PARAMETER,
                lags=0,
                leads=0,
                equation=None,
                code=None,
            ),
            fsic.parser.Symbol(
                name='H',
                type=fsic.parser.Type.EXOGENOUS,
                lags=-1,
                leads=0,
                equation=None,
                code=None,
            ),
            fsic.parser.Symbol(
                name='epsilon',
                type=fsic.parser.Type.ERROR,
                lags=0,
                leads=0,
                equation=None,
                code=None,
            ),
        ]

        self.assertEqual(fsic.parser.parse_equation(equation), expected)

    def test_parse_equation_verbatim_single_line(self):
        # Check that `parse_equation()` correctly handles a single-line piece
        # of verbatim code
        equation = '`self.Y[t] = self.C[t] + self.G[t]`'
        expected = [
            fsic.parser.Symbol(
                name=None,
                type=fsic.parser.Type.VERBATIM,
                lags=None,
                leads=None,
                equation='`self.Y[t] = self.C[t] + self.G[t]`',
                code='self.Y[t] = self.C[t] + self.G[t]',
            )
        ]

        self.assertEqual(fsic.parser.parse_equation(equation), expected)

    def test_parse_equation_verbatim_partial(self):
        # Check that `parse_equation()` correctly leaves a verbatim excerpt of
        # code unchanged in an equation
        equation = "Cp = C / `self['C', 1960]`"
        expected = [
            fsic.parser.Symbol(
                name='Cp',
                type=fsic.parser.Type.ENDOGENOUS,
                lags=0,
                leads=0,
                equation="Cp[t] = C[t] / `self['C', 1960]`",
                code="self._Cp[t] = self._C[t] / self['C', 1960]",
            ),
            fsic.parser.Symbol(
                name='C',
                type=fsic.parser.Type.EXOGENOUS,
                lags=0,
                leads=0,
                equation=None,
                code=None,
            ),
        ]

        self.assertEqual(fsic.parser.parse_equation(equation), expected)

    def test_parse_equation_empty(self):
        # Check that `parse_equation()` returns an empty list if the equation
        # string is empty
        self.assertEqual(fsic.parser.parse_equation(''), [])
        self.assertEqual(fsic.parser.parse_equation(' '), [])

    def test_parse_equation_multiple_equations_error(self):
        # Check that `parse_equation()` raises a `ParserError` if a string
        # doesn't define a single equation
        with self.assertRaises(fsic.exceptions.ParserError):
            fsic.parser.parse_equation('A = B\nC = D')

    def test_parse_equation_indentation_error(self):
        # Check that `parse_equation()` behaves identically to `parse_model()`
        # and raises an `IndentationError` if there's any leading whitespace in
        # an equation string
        with self.assertRaises(IndentationError):
            fsic.parser.parse_equation(' Y = C + G')

    def test_parse_equation_function_replacement(self):
        # Check that function replacement only applies to functions explicitly
        # defined in `fsic.replacement_function_names`. For example:
        #  - log -> np.log     (replaced)
        #  - np.log -> np.log  (unchanged)
        equation = 'C = log(A) + np.log(B)'
        expected = [
            fsic.parser.Symbol(
                name='C',
                type=fsic.parser.Type.ENDOGENOUS,
                lags=0,
                leads=0,
                equation='C[t] = log(A[t]) + np.log(B[t])',
                code='self._C[t] = np.log(self._A[t]) + np.log(self._B[t])',
            ),
            fsic.parser.Symbol(
                name='log',
                type=fsic.parser.Type.FUNCTION,
                lags=None,
                leads=None,
                equation=None,
                code=None,
            ),
            fsic.parser.Symbol(
                name='A',
                type=fsic.parser.Type.EXOGENOUS,
                lags=0,
                leads=0,
                equation=None,
                code=None,
            ),
            fsic.parser.Symbol(
                name='np.log',
                type=fsic.parser.Type.FUNCTION,
                lags=None,
                leads=None,
                equation=None,
                code=None,
            ),
            fsic.parser.Symbol(
                name='B',
                type=fsic.parser.Type.EXOGENOUS,
                lags=0,
                leads=0,
                equation=None,
                code=None,
            ),
        ]

        self.assertEqual(fsic.parser.parse_equation(equation), expected)

    def test_parse_equation_namespace_functions(self):
        # Check that functions in namespaces (below, `mean` in `np` [NumPy])
        # are left unchanged
        equation = 'Y = np.mean(X)'
        expected = [
            fsic.parser.Symbol(
                name='Y',
                type=fsic.parser.Type.ENDOGENOUS,
                lags=0,
                leads=0,
                equation='Y[t] = np.mean(X[t])',
                code='self._Y[t] = np.mean(self._X[t])',
            ),
            fsic.parser.Symbol(
                name='np.mean',
                type=fsic.parser.Type.FUNCTION,
                lags=None,
                leads=None,
                equation=None,
                code=None,
            ),
            fsic.parser.Symbol(
                name='X',
                type=fsic.parser.Type.EXOGENOUS,
                lags=0,
                leads=0,
                equation=None,
                code=None,
            ),
        ]

        self.assertEqual(fsic.parser.parse_equation(equation), expected)

    def test_parse_model(self):
        # Test that `parse_model()` correctly identifies the symbols that make
        # up a system of equations
        model = """
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
"""
        expected = [
            fsic.parser.Symbol(
                name='C',
                type=fsic.parser.Type.ENDOGENOUS,
                lags=0,
                leads=0,
                equation='C[t] = alpha_1[t] * YD[t] + alpha_2[t] * H[t-1]',
                code='self._C[t] = self._alpha_1[t] * self._YD[t] + self._alpha_2[t] * self._H[t-1]',
            ),
            fsic.parser.Symbol(
                name='alpha_1',
                type=fsic.parser.Type.PARAMETER,
                lags=0,
                leads=0,
                equation=None,
                code=None,
            ),
            fsic.parser.Symbol(
                name='YD',
                type=fsic.parser.Type.ENDOGENOUS,
                lags=0,
                leads=0,
                equation='YD[t] = Y[t] - T[t]',
                code='self._YD[t] = self._Y[t] - self._T[t]',
            ),
            fsic.parser.Symbol(
                name='alpha_2',
                type=fsic.parser.Type.PARAMETER,
                lags=0,
                leads=0,
                equation=None,
                code=None,
            ),
            fsic.parser.Symbol(
                name='H',
                type=fsic.parser.Type.ENDOGENOUS,
                lags=-1,
                leads=0,
                equation='H[t] = H[t-1] + YD[t] - C[t]',
                code='self._H[t] = self._H[t-1] + self._YD[t] - self._C[t]',
            ),
            fsic.parser.Symbol(
                name='Y',
                type=fsic.parser.Type.ENDOGENOUS,
                lags=0,
                leads=0,
                equation='Y[t] = C[t] + G[t]',
                code='self._Y[t] = self._C[t] + self._G[t]',
            ),
            fsic.parser.Symbol(
                name='T',
                type=fsic.parser.Type.ENDOGENOUS,
                lags=0,
                leads=0,
                equation='T[t] = theta[t] * Y[t]',
                code='self._T[t] = self._theta[t] * self._Y[t]',
            ),
            fsic.parser.Symbol(
                name='G',
                type=fsic.parser.Type.EXOGENOUS,
                lags=0,
                leads=0,
                equation=None,
                code=None,
            ),
            fsic.parser.Symbol(
                name='theta',
                type=fsic.parser.Type.PARAMETER,
                lags=0,
                leads=0,
                equation=None,
                code=None,
            ),
        ]

        self.assertEqual(fsic.parse_model(model), expected)

    def test_parse_model_verbatim(self):
        # Check that the parser can extract a block of verbatim code from a
        # string
        script = """
```
self.T[t] = self.theta[t] * self.Y[t]
```
"""
        self.assertEqual(
            fsic.parse_model(script),
            [
                fsic.parser.Symbol(
                    name=None,
                    type=fsic.parser.Type.VERBATIM,
                    lags=None,
                    leads=None,
                    equation="""```
self.T[t] = self.theta[t] * self.Y[t]
```""",
                    code='self.T[t] = self.theta[t] * self.Y[t]',
                )
            ],
        )

    def test_parser_no_lhs(self):
        # Check that invalid equations (missing a left-hand side expression)
        # lead to a `ParserError`
        with self.assertRaises(fsic.exceptions.ParserError):
            fsic.parse_model('A + B + C')

    def test_accidental_int_call(self):
        # Check that a missing operator between an integer and a variable is
        # caught properly
        with self.assertRaises(fsic.exceptions.ParserError):
            fsic.parse_model('3(A)')

    def test_accidental_float_call(self):
        # Check that a missing operator between a float and a variable is
        # caught properly
        with self.assertRaises(fsic.exceptions.ParserError):
            fsic.parse_model('3.0(A)')

    def test_missing_operator_int(self):
        # Check that a missing operator between an integer and a variable is
        # caught properly
        with self.assertRaises(fsic.exceptions.ParserError):
            fsic.parse_model('3A')

    def test_missing_operator_float(self):
        # Check that a missing operator between a float and a variable is
        # caught properly
        with self.assertRaises(fsic.exceptions.ParserError):
            fsic.parse_model('3.0A')

    def test_inconsistent_variable_type_left(self):
        # Check for an error if a variable is defined in multiple ways e.g. as
        # both an endogenous/exogenous variable and a parameter
        with self.assertRaises(fsic.exceptions.SymbolError):
            fsic.parse_model(
                """
A = f(B, {C})  # C is a parameter
B = f(C, D)    # But here, C is an exogenous variable
"""
            )

    def test_inconsistent_variable_type_right(self):
        # Check for an error if a variable is defined in multiple ways e.g. as
        # both an endogenous/exogenous variable and a parameter
        with self.assertRaises(fsic.exceptions.SymbolError):
            fsic.parse_model(
                """
A = f(B, C, D)  # C is an exogenous variable
B = f(<C>, D)   # But here, C is an error
"""
            )

    def test_leading_whitespace(self):
        # Check that an equation with unnecessary leading whitespace raises an
        # `IndentationError`
        with self.assertRaises(IndentationError):
            fsic.parse_model(' Y = C + I + G + X - M')

    def test_incomplete_braces_error(self):
        # Check that an incomplete set of braces raises a `ParserError`
        with self.assertRaises(fsic.exceptions.ParserError):
            # Missing closing brace after alpha_1
            fsic.parse_model('C = {alpha_0} + {alpha_1 * Y')

        with self.assertRaises(fsic.exceptions.ParserError):
            # Missing opening brace before alpha_1
            fsic.parse_model('C = {alpha_0} + alpha_1} * Y')


class TestVectorContainer(unittest.TestCase):
    def test_add_attribute(self):
        # Check that attribute creation works as expected
        container = fsic.core.VectorContainer(range(5 + 1))

        container.A = False
        container.B = 1
        container.C = 2.0

        self.assertTrue(isinstance(container.A, bool))
        self.assertEqual(container.A, False)

        self.assertTrue(isinstance(container.B, int))
        self.assertEqual(container.B, 1)

        self.assertTrue(isinstance(container.C, float))
        self.assertEqual(container.C, 2.0)

    def test_add_attribute_strict(self):
        # Check that attribute creation works as expected with `strict=True`
        container = fsic.core.VectorContainer(range(5 + 1), strict=True)

        container.add_attribute('A', False)
        container.add_attribute('B', 1)
        container.add_attribute('C', 2.0)

        self.assertTrue(isinstance(container.A, bool))
        self.assertEqual(container.A, False)

        self.assertTrue(isinstance(container.B, int))
        self.assertEqual(container.B, 1)

        self.assertTrue(isinstance(container.C, float))
        self.assertEqual(container.C, 2.0)

        with self.assertRaises(AttributeError):
            container.D = 3.0

        self.assertNotIn('D', container)

    def test_set_sequence_dimension_error(self):
        # Check that mismatched array lengths are caught as exceptions
        container = fsic.core.VectorContainer(range(5 + 1))
        for i, a in enumerate('ABC'):
            container.add_variable(a, i, dtype=float)

        with self.assertRaises(
            fsic.exceptions.DimensionError,
            msg="Invalid assignment for 'B': must be either a single value "
            'or a sequence of identical length to `span` '
            '(expected 6 element[s] but found 9)',
        ):
            container.B = [0] * 10

    def test_locate_period_in_span_error(self):
        # Check that non-existent index keys raise an error
        container = fsic.core.VectorContainer(range(5 + 1))
        for i, a in enumerate('ABC', start=1):
            container.add_variable(a, list(range(0, (5 + 1) * i, i)), dtype=float)

        # These should work fine
        self.assertEqual(container['A', 5], 5)
        self.assertEqual(container['B', 4], 8)
        self.assertEqual(container['C', 3], 9)

        with self.assertRaises(KeyError):
            container['A', 6]

    def test_locate_period_in_span_numpy_array(self):
        # Check that non-existent index keys raise an error
        container = fsic.core.VectorContainer(np.arange(5 + 1, dtype=int))
        for i, a in enumerate('ABC', start=1):
            container.add_variable(a, list(range(0, (5 + 1) * i, i)), dtype=float)

        # These should work fine
        self.assertEqual(container['A', 5], 5)
        self.assertEqual(container['B', 4], 8)
        self.assertEqual(container['C', 3], 9)

        with self.assertRaises(KeyError):
            container['A', 6]

    @unittest.skipIf(not pandas_installed, 'Requires `pandas`')
    def test_locate_period_in_span_pandas(self):
        # Check that non-existent index keys raise an error
        container = fsic.core.VectorContainer(
            pd.period_range(start='2000-01-01', end='2005-12-31', freq='Q')
        )
        for i, a in enumerate('ABC', start=1):
            container.add_variable(a, list(range(1, (24 * i) + 1, i)), dtype=float)

        # These should work fine
        self.assertTrue(np.allclose(container['A', '2005'], np.array([21, 22, 23, 24])))
        self.assertTrue(np.allclose(container['B', '2004'], np.array([33, 35, 37, 39])))
        self.assertTrue(np.allclose(container['C', '2003'], np.array([37, 40, 43, 46])))

        with self.assertRaises(KeyError):
            container['A', '2006']

    def test_size(self):
        # Check that `size` returns the total number of array elements
        container = fsic.core.VectorContainer(range(20, 30))
        for i, a in enumerate('ABC'):
            container.add_variable(a, i, dtype=float)

        self.assertEqual(container.size, 30)

    def test_nbytes(self):
        # Check that `nbytes` returns the total bytes consumed by the array
        # elements
        container = fsic.core.VectorContainer(range(20, 30))
        for i, a in enumerate('ABC'):
            container.add_variable(a, i, dtype=float)

        self.assertEqual(container.nbytes, 30 * 8)

    def test_replace(self):
        # Check multiple-replacement method
        container = fsic.core.VectorContainer(range(-10, 10))
        for i, a in enumerate('ABC'):
            container.add_variable(a, i)

        self.assertTrue(
            np.allclose(container.values, np.array([[0] * 20, [1] * 20, [2] * 20]))
        )

        container.replace_values(A=5, C=list(range(20)))

        self.assertTrue(
            np.allclose(
                container.values, np.array([[5] * 20, [1] * 20, list(range(20))])
            )
        )

        container.replace_values(**{'B': 12, 'C': -1})

        self.assertTrue(
            np.allclose(container.values, np.array([[5] * 20, [12] * 20, [-1] * 20]))
        )

    def test_strict(self):
        # Check that a `VectorContainer` object raises an exception if
        # attempting to add a new non-variable attribute with `strict=True`

        # Set up a container
        container = fsic.core.VectorContainer(range(10))

        container.add_variable('A', 0, dtype=int)
        container.add_variable('B', 1, dtype=int)
        container.add_variable('C', 2, dtype=int)

        self.assertEqual(container.values.shape, (3, 10))

        # By default, adding new attributes should work fine...
        container.D = 'This should work fine'
        self.assertEqual(container.D, 'This should work fine')

        # ...but won't expand the contents of the container
        self.assertEqual(container.values.shape, (3, 10))

        # With `strict=True`, the object should now block attempts to add new
        # attributes that aren't new container variables
        container.strict = True

        # This should still work
        container.add_variable('E', 4, dtype=int)

        # Contents expand accordingly
        self.assertEqual(container.values.shape, (4, 10))

        # But this should now raise an error
        with self.assertRaises(AttributeError):
            container.F = 5

        # Undo `strict=True` and now attempt to add the attribute
        container.strict = False

        container.G = 'This should work fine'
        self.assertEqual(container.G, 'This should work fine')

        self.assertEqual(container.values.shape, (4, 10))

    def test_get_closest_match(self):
        # Check that `get_closest_match()` suggests reasonable (expected)
        # alternatives to undefined variable names

        # Keys: Actual variable names
        # Values: Undefined names (to see if the container can suggest the key)
        inputs = {
            'A':    ('a',),
            'B':    ('b',),
            'C':    ('c',),

            'AB':   ('Ab',),
            'BC':   ('bC',),

            'ABC':  ('abc', 'AbC', 'aBc',),
            'ABCD': ('ABCd', 'ABcd', 'Abcd',),

            'XYZ':  ('xyz', 'XyZ', 'xYz',),
        }  # fmt: skip

        # Add variables to the container (from the keys above)
        container = fsic.core.VectorContainer(range(1995, 2005 + 1))
        for x in inputs:
            container.add_variable(x, 0, dtype=float)

        # Loop by key and then possible names to see if the key is correctly
        # suggested each time
        for expected, undefined_names in inputs.items():
            for x in undefined_names:
                with self.subTest(undefined_name=x):
                    self.assertEqual(container.get_closest_match(x), [expected])

    def test_values_replacement_array(self):
        # Check that the `values` property can be replaced by an array of
        # identical shape
        container = fsic.core.VectorContainer(range(10))
        container.add_variable('A', 0.0)
        container.add_variable('B', 1.0)
        container.add_variable('C', 2.0)

        self.assertTrue(np.allclose(container.A, 0))
        self.assertTrue(np.allclose(container.B, 1))
        self.assertTrue(np.allclose(container.C, 2))

        container.values = np.full((3, 10), 3)

        self.assertTrue(np.allclose(container.A, 3))
        self.assertTrue(np.allclose(container.B, 3))
        self.assertTrue(np.allclose(container.C, 3))

        self.assertTrue(np.allclose(container.values, 3))

    def test_values_replacement_single_value(self):
        # Check that the `values` property can be replaced by a single value
        container = fsic.core.VectorContainer(range(10))
        container.add_variable('A', 0.0)
        container.add_variable('B', 1.0)
        container.add_variable('C', 2.0)

        self.assertTrue(np.allclose(container.A, 0))
        self.assertTrue(np.allclose(container.B, 1))
        self.assertTrue(np.allclose(container.C, 2))

        container.values = 3

        self.assertTrue(np.allclose(container.A, 3))
        self.assertTrue(np.allclose(container.B, 3))
        self.assertTrue(np.allclose(container.C, 3))

        self.assertTrue(np.allclose(container.values, 3))

    def test_values_replacement_dimension_error(self):
        # Check that dimension mismatches raise a `DimensionError`
        container = fsic.core.VectorContainer(range(10))
        container.add_variable('A', 0.0)
        container.add_variable('B', 1.0)
        container.add_variable('C', 2.0)

        with self.assertRaises(fsic.exceptions.DimensionError):
            container.values = np.zeros((3, 4))

    def test_reindex_defaults(self):
        # Check that the `reindex()` method correctly reshapes the data
        container = fsic.core.VectorContainer(range(1995, 2005 + 1))
        container.add_variable('X', False, dtype=bool)
        container.add_variable('Y', 1, dtype=int)
        container.add_variable('Z', 2.0, dtype=float)

        self.assertEqual(container.span, range(1995, 2005 + 1))
        self.assertEqual(container.values.shape, (3, 11))

        # Reindex the container to a new span and check its contents
        reindexed_container = container.reindex(range(2000, 2015 + 1))
        self.assertFalse(reindexed_container.strict)

        self.assertEqual(reindexed_container.span, range(2000, 2015 + 1))
        self.assertEqual(reindexed_container.values.shape, (3, 16))

        # Check original container is unchanged
        self.assertEqual(container.span, range(1995, 2005 + 1))
        self.assertEqual(container.values.shape, (3, 11))

        # Check the contents for the overlapping part match
        self.assertTrue(np.allclose(container['X'][6:], reindexed_container['X'][:5]))
        self.assertTrue(np.allclose(container['Y'][6:], reindexed_container['Y'][:5]))
        self.assertTrue(np.allclose(container['Z'][6:], reindexed_container['Z'][:5]))

        # Check the non-overlapping part of the reindexed container is filled
        # correctly (bool: False; int: 0; anything else: NaN)
        self.assertFalse(reindexed_container['X'][6:].any())
        self.assertTrue((reindexed_container['Y'][6:] == 0).all())
        self.assertTrue(np.isnan(reindexed_container['Z'][6:]).all())

    def test_reindex_defaults_strict(self):
        # Check that the `reindex()` method correctly reshapes the data with
        # `strict=True`
        container = fsic.core.VectorContainer(range(1995, 2005 + 1), strict=True)
        container.add_variable('X', False, dtype=bool)
        container.add_variable('Y', 1, dtype=int)
        container.add_variable('Z', 2.0, dtype=float)

        self.assertEqual(container.span, range(1995, 2005 + 1))
        self.assertEqual(container.values.shape, (3, 11))

        # Reindex the container to a new span and check its contents
        reindexed_container = container.reindex(range(2000, 2015 + 1))
        self.assertTrue(reindexed_container.strict)

        self.assertEqual(reindexed_container.span, range(2000, 2015 + 1))
        self.assertEqual(reindexed_container.values.shape, (3, 16))

        # Check original container is unchanged
        self.assertEqual(container.span, range(1995, 2005 + 1))
        self.assertEqual(container.values.shape, (3, 11))

        # Check the contents for the overlapping part match
        self.assertTrue(np.allclose(container['X'][6:], reindexed_container['X'][:5]))
        self.assertTrue(np.allclose(container['Y'][6:], reindexed_container['Y'][:5]))
        self.assertTrue(np.allclose(container['Z'][6:], reindexed_container['Z'][:5]))

        # Check the non-overlapping part of the reindexed container is filled
        # correctly (bool: False; int: 0; anything else: NaN)
        self.assertFalse(reindexed_container['X'][6:].any())
        self.assertTrue((reindexed_container['Y'][6:] == 0).all())
        self.assertTrue(np.isnan(reindexed_container['Z'][6:]).all())

    def test_reindex_fill_values(self):
        # Check that the `reindex()` method correctly reshapes the data,
        # filling with user-specified values
        container = fsic.core.VectorContainer(range(1995, 2005 + 1))
        container.add_variable('X', False, dtype=bool)
        container.add_variable('Y', 1, dtype=int)
        container.add_variable('Z', 2.0, dtype=float)

        self.assertEqual(container.span, range(1995, 2005 + 1))
        self.assertEqual(container.values.shape, (3, 11))

        # Reindex the container to a new span and check its contents
        reindexed_container = container.reindex(range(2000, 2015 + 1), X=-1, Y=-1, Z=0)
        self.assertFalse(reindexed_container.strict)

        self.assertEqual(reindexed_container.span, range(2000, 2015 + 1))
        self.assertEqual(reindexed_container.values.shape, (3, 16))

        # Check original container is unchanged
        self.assertEqual(container.span, range(1995, 2005 + 1))
        self.assertEqual(container.values.shape, (3, 11))

        # Check the contents for the overlapping part match
        self.assertTrue(np.allclose(container['X'][6:], reindexed_container['X'][:5]))
        self.assertTrue(np.allclose(container['Y'][6:], reindexed_container['Y'][:5]))
        self.assertTrue(np.allclose(container['Z'][6:], reindexed_container['Z'][:5]))

        # Check the non-overlapping part of the reindexed container is filled
        # correctly (bool: False; int: 0; anything else: NaN)
        self.assertTrue(reindexed_container['X'][6:].all())
        self.assertTrue((reindexed_container['Y'][6:] == -1).all())
        self.assertTrue(np.allclose(reindexed_container['Z'][6:], 0.0))

    def test_reindex_fill_values_strict(self):
        # Check that the `reindex()` method correctly reshapes the data with
        # `strict=True`, filling with user-specified values
        container = fsic.core.VectorContainer(range(1995, 2005 + 1), strict=True)
        container.add_variable('X', False, dtype=bool)
        container.add_variable('Y', 1, dtype=int)
        container.add_variable('Z', 2.0, dtype=float)

        self.assertEqual(container.span, range(1995, 2005 + 1))
        self.assertEqual(container.values.shape, (3, 11))

        # Reindex the container to a new span and check its contents
        reindexed_container = container.reindex(range(2000, 2015 + 1), X=-1, Y=-1, Z=0)
        self.assertTrue(reindexed_container.strict)

        self.assertEqual(reindexed_container.span, range(2000, 2015 + 1))
        self.assertEqual(reindexed_container.values.shape, (3, 16))

        # Check original container is unchanged
        self.assertEqual(container.span, range(1995, 2005 + 1))
        self.assertEqual(container.values.shape, (3, 11))

        # Check the contents for the overlapping part match
        self.assertTrue(np.allclose(container['X'][6:], reindexed_container['X'][:5]))
        self.assertTrue(np.allclose(container['Y'][6:], reindexed_container['Y'][:5]))
        self.assertTrue(np.allclose(container['Z'][6:], reindexed_container['Z'][:5]))

        # Check the non-overlapping part of the reindexed container is filled
        # correctly (bool: False; int: 0; anything else: NaN)
        self.assertTrue(reindexed_container['X'][6:].all())
        self.assertTrue((reindexed_container['Y'][6:] == -1).all())
        self.assertTrue(np.allclose(reindexed_container['Z'][6:], 0.0))

    def test_reindex_fill_values_strict_keyword(self):
        # Check that the `reindex()` method correctly reshapes the data with
        # `strict=True` at the keyword level, filling with user-specified values
        container = fsic.core.VectorContainer(range(1995, 2005 + 1), strict=False)
        container.add_variable('X', False, dtype=bool)
        container.add_variable('Y', 1, dtype=int)
        container.add_variable('Z', 2.0, dtype=float)

        # This should work fine, even though 'A' is missing from the object
        container.reindex(range(2003, 2000, -1), X=2, Y=3, A=4)

        # Whereas this should fail with `strict=True`
        with self.assertRaises(KeyError):
            container.reindex(range(2003, 2000, -1), X=2, Y=3, A=4, strict=True)

        # Switch `strict` to `True`
        container.strict = True

        # This should now fail
        with self.assertRaises(KeyError):
            container.reindex(range(2003, 2000, -1), X=2, Y=3, A=4)

        # While this should now work
        container.reindex(range(2003, 2001), X=2, Y=3, A=4, strict=False)

    def test_eval_index(self):
        # Check `eval()` method with integer indexes
        container = fsic.core.VectorContainer(range(1995, 2005 + 1))
        container.add_variable('X', 0, dtype=float)
        container.add_variable('Y', 1, dtype=float)
        container.add_variable('Z', 2, dtype=float)

        self.assertTrue(np.allclose(container.X, 0))
        self.assertTrue(np.allclose(container.Y, 1))
        self.assertTrue(np.allclose(container.Z, 2))

        # Test vector operations
        self.assertTrue(
            np.allclose(container.eval('X + Y + Z'), np.array([3] * 11, dtype=float))
        )
        self.assertTrue(
            np.allclose(container.eval('X + Y - Z'), np.array([-1] * 11, dtype=float))
        )

        # Test index operations
        container.X += range(-5, 5 + 1)
        self.assertTrue(math.isclose(container.eval('X[0]'), -5))
        self.assertTrue(math.isclose(container.eval('X[-1]'), 5))

        # Test slice operations
        self.assertTrue(
            np.allclose(
                container.eval('X[0:2] + Y[-2:]'), np.array([-4, -3], dtype=float)
            )
        )
        self.assertTrue(
            np.allclose(
                container.eval('X[:-1:2] + Y[1::2]'),
                np.array([-4, -2, 0, 2, 4], dtype=float),
            )
        )

        # Test mixed operations
        self.assertTrue(
            np.allclose(
                container.eval('(X[0] + Y[-1::2]) * Z'),
                np.array([-8] * 11, dtype=float),
            )
        )

        # Test external functions
        self.assertTrue(
            math.isclose(
                container.eval('math.exp(X[0])', locals={'math': math}), math.exp(-5)
            )
        )
        self.assertTrue(
            np.allclose(
                container.eval('np.exp(Y)', locals={'math': math}),
                np.array([math.exp(1)] * 11),
            )
        )

    def test_eval_label(self):
        # Check `eval()` method with label indexes
        container = fsic.core.VectorContainer(range(1995, 2005 + 1))
        container.add_variable('X', 0, dtype=float)
        container.add_variable('Y', 1, dtype=float)
        container.add_variable('Z', 2, dtype=float)

        self.assertTrue(np.allclose(container.X, 0))
        self.assertTrue(np.allclose(container.Y, 1))
        self.assertTrue(np.allclose(container.Z, 2))

        # Test index operations
        container.X += range(-5, 5 + 1)
        self.assertTrue(math.isclose(container.eval('X[`1995`]'), -5))
        self.assertTrue(math.isclose(container.eval('X[`2005`]'), 5))

        # Test slice operations
        self.assertTrue(
            np.allclose(
                container.eval('X[`1995`:`1996`] + Y[`2004`:]'),
                np.array([-4, -3], dtype=float),
            )
        )
        self.assertTrue(
            np.allclose(
                container.eval('X[:`2004`:2] + Y[`1996`::2]'),
                np.array([-4, -2, 0, 2, 4], dtype=float),
            )
        )

        # Test mixed operations
        self.assertTrue(
            np.allclose(
                container.eval('(X[`1995`] + Y[`2004`::2]) * Z'),
                np.array([-8] * 11, dtype=float),
            )
        )

        # Test external functions
        self.assertTrue(
            math.isclose(
                container.eval('math.exp(X[`1995`])', locals={'math': math}),
                math.exp(-5),
            )
        )

    @unittest.skipIf(not pandas_installed, 'Requires `pandas`')
    def test_eval_label_periodindex(self):
        # Check `eval()` method on pandas PeriodIndex objects
        container = fsic.core.VectorContainer(
            pd.period_range(start='2000-01-01', end='2005-12-31', freq='Q')
        )
        container.add_variable('X', 0, dtype=float)
        container.add_variable('Y', 1, dtype=float)
        container.add_variable('Z', 2, dtype=float)

        self.assertTrue(np.allclose(container.X, 0))
        self.assertTrue(np.allclose(container.Y, 1))
        self.assertTrue(np.allclose(container.Z, 2))

        # Test index operations
        container.X[-1] = 5
        self.assertTrue(math.isclose(container.eval('X[`2005Q4`]'), 5))

        container.Y[-8:-4] = 10
        self.assertTrue(
            np.allclose(container.eval('Y[`2004`]'), np.array([10] * 4, dtype=float))
        )

        # Test slice operations
        container.Z[1:6] = range(1, 6)
        self.assertTrue(
            np.allclose(
                container.eval('Z[`2000Q2`:`2001Q2`] * 2'),
                np.arange(2, 11, 2, dtype=float),
            )
        )
        self.assertTrue(
            np.allclose(
                container.eval('Z[`2001`:`2002`] + 1'),
                np.array([5, 6, 3, 3, 3, 3, 3, 3], dtype=float),
            )
        )

        self.assertTrue(
            np.allclose(
                container.eval('X[`2005Q2`:] + 3'), np.array([3, 3, 8], dtype=float)
            )
        )
        self.assertTrue(
            np.allclose(
                container.eval('X[`2004`:] * 3'),
                np.array([0, 0, 0, 0, 0, 0, 0, 15], dtype=float),
            )
        )

        self.assertTrue(
            np.allclose(
                container.eval('Z[:`2001Q2`] - 3'),
                np.array([-1, -2, -1, 0, 1, 2], dtype=float),
            )
        )
        self.assertTrue(
            np.allclose(
                container.eval('Z[:`2001`] / 3'),
                np.array([2, 1, 2, 3, 4, 5, 2, 2], dtype=float) / 3,
            )
        )

        self.assertTrue(
            np.allclose(
                container.eval('Z[`2000Q2`:`2001Q2`:2] * 2'),
                np.arange(2, 11, 4, dtype=float),
            )
        )
        self.assertTrue(
            np.allclose(
                container.eval('Z[`2001`:`2002`:2] + 1'),
                np.array([5, 3, 3, 3], dtype=float),
            )
        )

        self.assertTrue(
            np.allclose(
                container.eval('X[`2005Q2`::2] + 3'), np.array([3, 8], dtype=float)
            )
        )
        self.assertTrue(
            np.allclose(
                container.eval('X[`2004`::2] * 3'), np.array([0, 0, 0, 0], dtype=float)
            )
        )

        self.assertTrue(
            np.allclose(
                container.eval('Z[:`2001Q2`:2] - 3'), np.array([-1, -1, 1], dtype=float)
            )
        )
        self.assertTrue(
            np.allclose(
                container.eval('Z[:`2001`:2] / 3'),
                np.array([2, 2, 4, 2], dtype=float) / 3,
            )
        )

        # # Test mixed operations
        self.assertTrue(
            np.allclose(
                container.eval('(Z[:`2001`:2] + X[-1]) / 3'),
                np.array([7, 7, 9, 7], dtype=float) / 3,
            )
        )

        # # Test external functions
        self.assertTrue(
            np.allclose(
                container.eval('(Z[:`2001`:2] + X[-1]) / np.exp(3)', locals={'np': np}),
                np.array([7, 7, 9, 7], dtype=float) / np.exp(3),
            )
        )

    def test_eval_undefined_variable_error(self):
        # Check that `eval()` returns a suggested alternative in the event of a
        # `NameError`
        container = fsic.core.VectorContainer(range(1995, 2005 + 1))
        container.add_variable('X', 0, dtype=float)

        with self.assertRaises(
            AttributeError, msg="Object has no attribute 'X'. Did you mean: 'x'?"
        ):
            container.eval('x * 2')

    def test_eval_undefined_variable_error_empty_container(self):
        # Check that `eval()` raises an undefined variable error when trying to
        # suggest an alternative
        container = fsic.core.VectorContainer(range(1995, 2005 + 1))

        with self.assertRaises(
            AttributeError,
            msg="Object is empty and thus has no attribute with name 'x'",
        ):
            container.eval('x * 2')

    @unittest.expectedFailure
    def test_exec(self):
        # Check `exec()` method
        container = fsic.core.VectorContainer(range(1995, 2005 + 1))
        container.add_variable('X', 0, dtype=float)
        container.add_variable('Y', 1, dtype=float)
        container.add_variable('Z', 2, dtype=float)

        self.assertTrue(np.allclose(container.X, 0))
        self.assertTrue(np.allclose(container.Y, 1))
        self.assertTrue(np.allclose(container.Z, 2))

        container.exec('Z = X * Y')
        self.assertTrue(np.allclose(container.X, 0))
        self.assertTrue(np.allclose(container.Y, 1))
        self.assertTrue(np.allclose(container.Z, 0))

    @unittest.skipIf(not pandas_installed, 'Requires `pandas`')
    def test_to_dataframe(self):
        # Check that `VectorContainer`s correctly return a DataFrame of values
        from pandas import DataFrame
        from pandas.testing import assert_frame_equal

        container = fsic.core.VectorContainer(range(1995, 2005 + 1), strict=True)
        container.add_variable('X', False, dtype=bool)
        container.add_variable('Y', 1, dtype=int)
        container.add_variable('Z', 2.0, dtype=float)

        assert_frame_equal(
            container.to_dataframe(),
            DataFrame({'X': False, 'Y': 1, 'Z': 2.0}, index=range(1995, 2005 + 1)),
        )


class TestInit(unittest.TestCase):
    SCRIPT = """
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
"""
    SYMBOLS = fsic.parse_model(SCRIPT)

    def setUp(self):
        self.Model = fsic.build_model(self.SYMBOLS)

    def test_init_with_duplicate_names_error(self):
        # Check for a `DuplicateNameError` if the model repeats any variable
        # names in its `NAMES` attribute
        self.Model.NAMES += ['G']  # Add in a second instance of 'G'

        with self.assertRaises(fsic.exceptions.DuplicateNameError):
            self.Model(range(5))

    def test_init_with_arrays(self):
        model = self.Model(range(5), G=np.arange(0, 10, 2), alpha_1=[0.6] * 5)
        self.assertEqual(model.values.shape, (9, 5))
        self.assertTrue(
            np.allclose(
                model.values,
                np.array(
                    [
                        [0.0] * 5,
                        [0.0] * 5,
                        [0.0] * 5,
                        [0.0] * 5,
                        [0.0] * 5,
                        [0.0, 2.0, 4.0, 6.0, 8.0],
                        [0.6] * 5,
                        [0.0] * 5,
                        [0.0] * 5,
                    ]
                ),
            )
        )

    def test_init_dimension_error(self):
        with self.assertRaises(fsic.exceptions.DimensionError):
            # C is invalid because it has the wrong shape
            self.Model(range(10), C=[0, 0])

    def test_init_strict(self):
        # Check that `strict=True` raises an exception on passing a variable
        # not listed in the model's `NAMES` attribute
        with self.assertRaises(fsic.exceptions.InitialisationError):
            self.Model(range(5), strict=True, A=5)

    @unittest.skipIf(not pandas_installed, 'Requires `pandas`')
    def test_from_dataframe(self):
        # Test instantiation from a `pandas` DataFrame (if installed)
        from pandas import DataFrame

        # Input data with the span (index) and values
        data = DataFrame(
            {
                'alpha_1': 0.6,
                'alpha_2': 0.4,
                'G': 20,
                'theta': 0.2,
            },
            index=range(-5, 10),
        )

        model = self.Model.from_dataframe(data)

        # Check span matches
        self.assertEqual(model.span, list(range(-5, 10)))

        # Check specified values match
        self.assertTrue(np.allclose(model.alpha_1, 0.6))
        self.assertTrue(np.allclose(model.alpha_2, 0.4))
        self.assertTrue(np.allclose(model.G, 20))
        self.assertTrue(np.allclose(model.theta, 0.2))

        # All other values should be zero
        for k in model.names:
            if k not in ['alpha_1', 'alpha_2', 'G', 'theta']:
                with self.subTest(variable=k):
                    self.assertTrue(np.allclose(model[k], 0.0))

    @unittest.skipIf(not pandas_installed, 'Requires `pandas`')
    def test_from_dataframe_timeseries_indexes(self):
        # Test instantiation from a `pandas` DataFrame (if installed),
        # preserving time-series indexes as `pandas` objects
        import pandas as pd
        from pandas import DataFrame

        # TODO: Review compatibility check - compare against `pandas` version
        #       instead?
        monthly_frequency = 'ME' if sys.version_info[:2] > (3, 8) else 'M'

        test_cases = {
            'DatetimeIndex': pd.date_range(
                start='01/01/2000', periods=269, freq=monthly_frequency
            ),
            'MultiIndex.from_tuples': pd.MultiIndex.from_tuples(
                [
                    (year, term)
                    for year in range(2000, 2005 + 1)
                    for term in range(1, 3 + 1)
                ],
                names=['year', 'term'],
            ),
            'MultiIndex.from_product': pd.MultiIndex.from_product(
                [range(2000, 2005 + 1), range(1, 3 + 1)], names=['year', 'term']
            ),
            'PeriodIndex': pd.period_range(start='1990Q1', end='2000Q4', freq='Q'),
            'TimedeltaIndex': pd.timedelta_range(start='1 day', periods=50),
        }

        for name, span in test_cases.items():
            with self.subTest(index=name):
                data = DataFrame(
                    {
                        'alpha_1': 0.6,
                        'alpha_2': 0.4,
                        'G': 20,
                        'theta': 0.2,
                    },
                    index=span,
                )

                model = self.Model.from_dataframe(data)

                # Check span matches
                self.assertEqual(type(model.span), type(span))
                self.assertTrue((model.span == span).all())

                # Check specified values match
                self.assertTrue(np.allclose(model.alpha_1, 0.6))
                self.assertTrue(np.allclose(model.alpha_2, 0.4))
                self.assertTrue(np.allclose(model.G, 20))
                self.assertTrue(np.allclose(model.theta, 0.2))

                # All other values should be zero
                for k in model.names:
                    if k not in ['alpha_1', 'alpha_2', 'G', 'theta']:
                        with self.subTest(variable=k):
                            self.assertTrue(np.allclose(model[k], 0.0))


class TestInterface(unittest.TestCase):
    SCRIPT = """
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
"""
    SYMBOLS = fsic.parse_model(SCRIPT)

    def setUp(self):
        self.Model = fsic.build_model(self.SYMBOLS)

    def test_dir(self):
        # Check that `dir(model)` includes the model's variable names
        model = self.Model(range(10))

        for name in model.names + ['span', 'names', 'status', 'iterations']:
            self.assertIn(name, dir(model))

    def test_modify_iterations(self):
        # Check that the `iterations` attribute stays as a NumPy array
        model = self.Model(range(10))

        iterations = np.full(10, -1, dtype=int)

        self.assertEqual(model.iterations.shape, iterations.shape)
        self.assertTrue(np.all(model.iterations == iterations))

        # Set all values to 1
        model.iterations = 1
        iterations[:] = 1

        self.assertEqual(model.iterations.shape, iterations.shape)
        self.assertTrue(np.all(model.iterations == iterations))

        # Assign a sequence
        model.iterations = range(0, 20, 2)
        iterations = np.arange(0, 20, 2)

        self.assertEqual(model.iterations.shape, iterations.shape)
        self.assertTrue(np.all(model.iterations == iterations))

    def test_modify_iterations_errors(self):
        # Check that `iterations` assignment errors are as expected
        model = self.Model(range(10))

        with self.assertRaises(fsic.exceptions.DimensionError):
            model.iterations = [0, 1]  # Incompatible dimensions

    def test_modify_status(self):
        # Check that the `status` attribute stays as a NumPy array
        model = self.Model(range(10))

        status = np.full(10, '-')

        self.assertEqual(model.status.shape, status.shape)
        self.assertTrue(np.all(model.status == status))

        # Set all values to '.'
        model.status = '.'
        status[:] = '.'

        self.assertEqual(model.status.shape, status.shape)
        self.assertTrue(np.all(model.status == status))

        # Assign a sequence
        model.status = ['-', '.'] * 5
        status = np.array(['-', '.'] * 5)

        self.assertEqual(model.status.shape, status.shape)
        self.assertTrue(np.all(model.status == status))

    def test_modify_status_errors(self):
        # Check that `status` assignment errors are as expected
        model = self.Model(range(10))

        with self.assertRaises(fsic.exceptions.DimensionError):
            model.status = ['-', '.']  # Incompatible dimensions

    def test_iter_periods(self):
        # Check properties of the iterator returned by `iter_periods()`
        model = self.Model(range(2000, 2009 + 1))

        self.assertEqual(len(model.iter_periods()), 9)
        self.assertEqual(len(model.iter_periods(start=2005)), 5)
        self.assertEqual(len(model.iter_periods(end=2006)), 6)
        self.assertEqual(len(model.iter_periods(start=2001, end=2007)), 7)

        self.assertEqual(
            list(model.iter_periods()), [(i, 2000 + i) for i in range(1, 9 + 1)]
        )

    def test_iter_periods_reuse(self):
        # Check that `PeriodIter` objects are reusable
        model = self.Model(
            [
                '{}Q{}'.format(year, quarter)
                for year in range(2000, 2005 + 1)
                for quarter in range(1, 4 + 1)
            ]
        )

        period_iter = model.iter_periods()

        list_1 = list(period_iter)
        list_2 = list(period_iter)

        self.assertEqual(list_1, list_2)


class TestModelContainerMethods(unittest.TestCase):
    SCRIPT = 'C = {alpha_1} * YD + {alpha_2} * H[-1]'
    SYMBOLS = fsic.parse_model(SCRIPT)

    def setUp(self):
        self.Model = fsic.build_model(self.SYMBOLS)

    def test_size(self):
        # Check that `size` returns the total number of array elements,
        # excluding `status` and `iterations`
        model = self.Model(range(20, 30))

        self.assertNotEqual(
            model.size, 70
        )  # If `status` and `iterations` were included
        self.assertEqual(model.size, 50)

    def test_nbytes(self):
        # Check that `nbytes` returns the total bytes consumed by the array
        # elements
        model = self.Model(range(20, 30))
        self.assertEqual(
            model.nbytes,
            (50 * 8)     # Array elements
            + (10 * 8)   # Iterations
            + (10 * 4),  # Status
        )  # fmt: skip

    def test_add_variable(self):
        # Check that `add_variable()` extends the model object's store (both
        # values and the accompanying key) while applying the correct type,
        # whether default or explicitly specified
        model = self.Model(range(10))

        # Check list of names is unchanged
        self.assertEqual(model.names, model.NAMES)
        self.assertEqual(model.names, ['C', 'YD', 'H', 'alpha_1', 'alpha_2'])

        self.assertEqual(model.values.shape, (5, 10))

        # Add new variables of various types
        # fmt: off
        model.add_variable('I', 0, dtype=int)       # Impose int
        model.add_variable('J', 0)                  # float, by default
        model.add_variable('K', 0, dtype=float)     # Impose float
        model.add_variable('L', False, dtype=bool)  # Import bool
        # fmt: on

        # Check list of names is now changed
        self.assertEqual(model.names, model.NAMES + ['I', 'J', 'K', 'L'])
        self.assertEqual(
            model.names, ['C', 'YD', 'H', 'alpha_1', 'alpha_2', 'I', 'J', 'K', 'L']
        )

        self.assertEqual(model.values.shape, (9, 10))

        self.assertEqual(model.I.dtype, int)
        self.assertEqual(model['I'].dtype, int)

        self.assertEqual(model.J.dtype, float)
        self.assertEqual(model['J'].dtype, float)

        self.assertEqual(model.K.dtype, float)
        self.assertEqual(model['K'].dtype, float)

        self.assertEqual(model.L.dtype, bool)
        self.assertEqual(model['L'].dtype, bool)

        # Check `add_variable()` for names longer than one character
        # fmt: off
        model.add_variable('AB', 0, dtype=int)       # Impose int
        model.add_variable('CD', 0)                  # float, by default
        model.add_variable('EF', 0, dtype=float)     # Impose float
        model.add_variable('GH', False, dtype=bool)  # Import bool
        # fmt: on

        # Check list of names is now changed
        self.assertEqual(
            model.names, model.NAMES + ['I', 'J', 'K', 'L', 'AB', 'CD', 'EF', 'GH']
        )
        self.assertEqual(
            model.names,
            [
                'C',
                'YD',
                'H',
                'alpha_1',
                'alpha_2',
                'I',
                'J',
                'K',
                'L',
                'AB',
                'CD',
                'EF',
                'GH',
            ],
        )

        self.assertEqual(model.values.shape, (13, 10))

        self.assertEqual(model.AB.dtype, int)
        self.assertEqual(model['AB'].dtype, int)

        self.assertEqual(model.CD.dtype, float)
        self.assertEqual(model['CD'].dtype, float)

        self.assertEqual(model.EF.dtype, float)
        self.assertEqual(model['EF'].dtype, float)

        self.assertEqual(model.GH.dtype, bool)
        self.assertEqual(model['GH'].dtype, bool)

    def test_add_variable_strict(self):
        # Check that `add_variable()` extends the model object's store (both
        # values and the accompanying key) while applying the correct type,
        # whether default or explicitly specified
        # This version augmented to test for `strict=True`
        model = self.Model(range(10), strict=True)

        # Check list of names is unchanged
        self.assertEqual(model.names, model.NAMES)
        self.assertEqual(model.names, ['C', 'YD', 'H', 'alpha_1', 'alpha_2'])

        self.assertEqual(model.values.shape, (5, 10))

        # Add new variables of various types
        # fmt: off
        model.add_variable('I', 0, dtype=int)       # Impose int
        model.add_variable('J', 0)                  # float, by default
        model.add_variable('K', 0, dtype=float)     # Impose float
        model.add_variable('L', False, dtype=bool)  # Import bool
        # fmt: on

        # Check list of names is now changed
        self.assertEqual(model.names, model.NAMES + ['I', 'J', 'K', 'L'])
        self.assertEqual(
            model.names, ['C', 'YD', 'H', 'alpha_1', 'alpha_2', 'I', 'J', 'K', 'L']
        )

        self.assertEqual(model.values.shape, (9, 10))

        self.assertEqual(model.I.dtype, int)
        self.assertEqual(model['I'].dtype, int)

        self.assertEqual(model.J.dtype, float)
        self.assertEqual(model['J'].dtype, float)

        self.assertEqual(model.K.dtype, float)
        self.assertEqual(model['K'].dtype, float)

        self.assertEqual(model.L.dtype, bool)
        self.assertEqual(model['L'].dtype, bool)

        with self.assertRaises(AttributeError):
            model.M = 'Should raise an `AttributeError`'

    def test_get_closest_match(self):
        # Check that `get_closest_match()` suggests reasonable (expected)
        # alternatives to undefined variable names

        # Keys: Actual variable names
        # Values: Undefined names (to see if the model can suggest the key)
        inputs = {
            'A':    ('a',),
            'B':    ('b',),
            # 'C':    ('c',),  # Exclude from test to avoid collision with model definition of C

            'AB':   ('Ab',),
            'BC':   ('bC',),

            'ABC':  ('abc', 'AbC', 'aBc',),
            'ABCD': ('ABCd', 'ABcd', 'Abcd',),

            'XYZ':  ('xyz', 'XyZ', 'xYz',),
        }  # fmt: skip

        # Add variables to the model (from the keys above)
        model = self.Model(range(1995, 2005 + 1))
        for x in inputs:
            model.add_variable(x, 0, dtype=float)

        # Loop by key and then possible names to see if the key is correctly
        # suggested each time
        for expected, undefined_names in inputs.items():
            for x in undefined_names:
                with self.subTest(undefined_name=x):
                    self.assertEqual(model.get_closest_match(x), [expected])

    def test_get_closest_match_multiple(self):
        # Check that `get_closest_match()` handles multiple match candidates
        # correctly (currently, by raising a `NotImplementedError`)

        # yd_r/YD_r problem uncovered while attempting to implement Model
        # *INSOUT* from Chapter 10 of Godley and Lavoie (2007)

        # Use `strict=True` to test by failed attribute assignment
        model = self.Model(range(1995, 2005 + 1), strict=True)
        model.add_variable('YD_r', 0)
        model.add_variable('yd_r', 0)

        with self.assertRaises(
            NotImplementedError,
            msg='Handling of multiple name matches not yet implemented',
        ):
            model.yd_k_r = 1

        # Reverse the order of variable addition and test the same again (an
        # earlier bug meant this behaved differently to the order above)
        model = self.Model(range(1995, 2005 + 1), strict=True)
        model.add_variable('yd_r', 0)
        model.add_variable('YD_r', 0)

        with self.assertRaises(
            NotImplementedError,
            msg='Handling of multiple name matches not yet implemented',
        ):
            model.yd_k_r = 1

    def test_getitem_by_name(self):
        # Test variable access by name
        model = self.Model(
            [
                '{}Q{}'.format(y, q)
                for y in range(1990, 1995 + 1)
                for q in range(1, 4 + 1)
            ]
        )
        model.YD = np.arange(len(model.span))

        self.assertTrue(np.allclose(model['YD'], np.arange(len(model.span))))

    def test_getitem_by_name_error(self):
        # Test that variable access raises a KeyError if the name isn't a model
        # variable
        model = self.Model(
            [
                '{}Q{}'.format(y, q)
                for y in range(1990, 1995 + 1)
                for q in range(1, 4 + 1)
            ]
        )
        model.YD = np.arange(len(model.span))

        with self.assertRaises(KeyError):
            model['ABC']

    def test_getitem_by_name_and_index(self):
        # Test simultaneous variable and single period access
        model = self.Model(
            [
                '{}Q{}'.format(y, q)
                for y in range(1990, 1995 + 1)
                for q in range(1, 4 + 1)
            ]
        )
        model.YD = np.arange(len(model.span))

        self.assertEqual(model['YD', '1991Q1'], 4.0)

    def test_getitem_by_name_and_slice_to(self):
        # Test simultaneous variable and slice access
        model = self.Model(
            [
                '{}Q{}'.format(y, q)
                for y in range(1990, 1995 + 1)
                for q in range(1, 4 + 1)
            ]
        )
        model.YD = np.arange(len(model.span))

        self.assertTrue(np.allclose(model['YD', :'1990Q4'], np.arange(3 + 1)))

    def test_getitem_by_name_and_slice_from(self):
        # Test simultaneous variable and slice access
        model = self.Model(
            [
                '{}Q{}'.format(y, q)
                for y in range(1990, 1995 + 1)
                for q in range(1, 4 + 1)
            ]
        )
        model.YD = np.arange(len(model.span))

        self.assertTrue(np.allclose(model['YD', '1995Q1':], np.arange(20, 23 + 1)))

    def test_getitem_by_name_and_slice_step(self):
        # Test simultaneous variable and slice access
        model = self.Model(
            [
                '{}Q{}'.format(y, q)
                for y in range(1990, 1995 + 1)
                for q in range(1, 4 + 1)
            ]
        )
        model.YD = np.arange(len(model.span))

        self.assertTrue(np.allclose(model['YD', ::4], np.arange(0, 23 + 1, 4)))

    def test_setitem_by_name_with_number(self):
        # Test variable assignment by name, setting all elements to a single
        # value
        model = self.Model(
            [
                '{}Q{}'.format(y, q)
                for y in range(1990, 1995 + 1)
                for q in range(1, 4 + 1)
            ]
        )
        model.YD = np.arange(len(model.span))

        model['YD'] = 10
        self.assertTrue(
            np.allclose(model['YD'], np.full(len(model.span), 10, dtype=float))
        )

    def test_setitem_by_name_with_array(self):
        # Test variable assignment by name, replacing with a new array
        model = self.Model(
            [
                '{}Q{}'.format(y, q)
                for y in range(1990, 1995 + 1)
                for q in range(1, 4 + 1)
            ]
        )
        model.YD = np.arange(len(model.span))

        model['YD'] = np.arange(0, len(model.span) * 2, 2)
        self.assertTrue(np.allclose(model['YD'], np.arange(0, len(model.span) * 2, 2)))

    def test_setitem_by_name_and_index(self):
        # Test variable assignment by name and single period
        model = self.Model(
            [
                '{}Q{}'.format(y, q)
                for y in range(1990, 1995 + 1)
                for q in range(1, 4 + 1)
            ]
        )
        model.YD = np.arange(len(model.span))

        model['YD', '1990Q1'] = 100

        expected = np.arange(len(model.span))
        expected[0] = 100

        self.assertTrue(np.allclose(model['YD'], expected))

    def test_setitem_by_name_and_slice_to(self):
        # Test variable assignment by name and slice
        model = self.Model(
            [
                '{}Q{}'.format(y, q)
                for y in range(1990, 1995 + 1)
                for q in range(1, 4 + 1)
            ]
        )
        model.YD = np.arange(len(model.span))

        model['YD', :'1990Q4'] = 100

        expected = np.arange(len(model.span))
        expected[:4] = 100

        self.assertTrue(np.allclose(model['YD'], expected))

    def test_setitem_by_name_and_slice_from(self):
        # Test variable assignment by name and slice
        model = self.Model(
            [
                '{}Q{}'.format(y, q)
                for y in range(1990, 1995 + 1)
                for q in range(1, 4 + 1)
            ]
        )
        model.YD = np.arange(len(model.span))

        model['YD', '1995Q1':] = 100

        expected = np.arange(len(model.span))
        expected[-4:] = 100

        self.assertTrue(np.allclose(model['YD'], expected))

    def test_setitem_dimension_error(self):
        # Test check for misaligned dimensions at assignment
        model = self.Model(
            [
                '{}Q{}'.format(y, q)
                for y in range(1990, 1995 + 1)
                for q in range(1, 4 + 1)
            ]
        )
        model.YD = np.arange(len(model.span))

        with self.assertRaises(fsic.exceptions.DimensionError):
            model.C = [0, 0]

    def test_contains(self):
        # Test `in` (membership) operator
        model = self.Model(range(10))

        model_variables = ['C', 'YD', 'H', 'alpha_1', 'alpha_2']
        for name in model_variables:
            with self.subTest(name=name):
                self.assertIn(name, model)

        self.assertNotIn('G', model)

        self.assertNotIn('status', model)
        self.assertNotIn('iterations', model)

    def test_values_replacement_array(self):
        # Check that the `values` property can be replaced by an array of
        # identical shape
        model = self.Model(range(10))

        self.assertTrue(np.allclose(model.values, 0))

        model.values = np.full((5, 10), 1)

        self.assertTrue(np.allclose(model.C, 1))
        self.assertTrue(np.allclose(model.alpha_1, 1))
        self.assertTrue(np.allclose(model.YD, 1))
        self.assertTrue(np.allclose(model.alpha_2, 1))
        self.assertTrue(np.allclose(model.H, 1))

        self.assertTrue(np.allclose(model.values, 1))

    def test_values_replacement_single_value(self):
        # Check that the `values` property can be replaced by a single value
        model = self.Model(range(10))

        self.assertTrue(np.allclose(model.values, 0))

        model.values = 1

        self.assertTrue(np.allclose(model.C, 1))
        self.assertTrue(np.allclose(model.alpha_1, 1))
        self.assertTrue(np.allclose(model.YD, 1))
        self.assertTrue(np.allclose(model.alpha_2, 1))
        self.assertTrue(np.allclose(model.H, 1))

        self.assertTrue(np.allclose(model.values, 1))

    def test_values_replacement_dimension_error(self):
        # Check that dimension mismatches raise a `DimensionError`
        model = self.Model(range(10))

        self.assertTrue(np.allclose(model.values, 0))

        with self.assertRaises(fsic.exceptions.DimensionError):
            model.values = np.zeros((3, 4))

    def test_reindex_defaults(self):
        # Check that `reindex()` works as expected, filling the 'status' and
        # 'iterations' attributes correctly
        model = self.Model(range(-5, 5 + 1))
        self.assertFalse(model.strict)

        self.assertEqual(model.values.shape, (5, 11))

        self.assertTrue(np.allclose(model.C, 0.0))
        self.assertTrue(np.allclose(model.YD, 0.0))
        self.assertTrue(np.allclose(model.H, 0.0))
        self.assertTrue(np.allclose(model.alpha_1, 0.0))
        self.assertTrue(np.allclose(model.alpha_2, 0.0))

        self.assertTrue((model.status == '-').all())
        self.assertTrue((model.iterations == -1).all())

        reindexed_model = model.reindex(range(0, 15 + 1))
        self.assertFalse(model.strict)

        self.assertEqual(reindexed_model.values.shape, (5, 16))

        self.assertTrue(np.allclose(reindexed_model['C', 0:5], 0.0))
        self.assertTrue(np.allclose(reindexed_model['YD', 0:5], 0.0))
        self.assertTrue(np.allclose(reindexed_model['H', 0:5], 0.0))
        self.assertTrue(np.allclose(reindexed_model['alpha_1', 0:5], 0.0))
        self.assertTrue(np.allclose(reindexed_model['alpha_2', 0:5], 0.0))

        self.assertTrue(np.isnan(reindexed_model['C', 6:]).all())
        self.assertTrue(np.isnan(reindexed_model['YD', 6:]).all())
        self.assertTrue(np.isnan(reindexed_model['H', 6:]).all())
        self.assertTrue(np.isnan(reindexed_model['alpha_1', 6:]).all())
        self.assertTrue(np.isnan(reindexed_model['alpha_2', 6:]).all())

        self.assertTrue((reindexed_model.status == '-').all())
        self.assertTrue((reindexed_model.iterations == -1).all())

    def test_reindex_defaults_strict(self):
        # Check that `reindex()` works as expected, filling the 'status' and
        # 'iterations' attributes correctly with `strict=True`
        model = self.Model(range(-5, 5 + 1), strict=True)
        self.assertTrue(model.strict)

        self.assertEqual(model.values.shape, (5, 11))

        self.assertTrue(np.allclose(model.C, 0.0))
        self.assertTrue(np.allclose(model.YD, 0.0))
        self.assertTrue(np.allclose(model.H, 0.0))
        self.assertTrue(np.allclose(model.alpha_1, 0.0))
        self.assertTrue(np.allclose(model.alpha_2, 0.0))

        self.assertTrue((model.status == '-').all())
        self.assertTrue((model.iterations == -1).all())

        reindexed_model = model.reindex(range(0, 15 + 1))
        self.assertTrue(model.strict)

        self.assertEqual(reindexed_model.values.shape, (5, 16))

        self.assertTrue(np.allclose(reindexed_model['C', 0:5], 0.0))
        self.assertTrue(np.allclose(reindexed_model['YD', 0:5], 0.0))
        self.assertTrue(np.allclose(reindexed_model['H', 0:5], 0.0))
        self.assertTrue(np.allclose(reindexed_model['alpha_1', 0:5], 0.0))
        self.assertTrue(np.allclose(reindexed_model['alpha_2', 0:5], 0.0))

        self.assertTrue(np.isnan(reindexed_model['C', 6:]).all())
        self.assertTrue(np.isnan(reindexed_model['YD', 6:]).all())
        self.assertTrue(np.isnan(reindexed_model['H', 6:]).all())
        self.assertTrue(np.isnan(reindexed_model['alpha_1', 6:]).all())
        self.assertTrue(np.isnan(reindexed_model['alpha_2', 6:]).all())

        self.assertTrue((reindexed_model.status == '-').all())
        self.assertTrue((reindexed_model.iterations == -1).all())


class TestBuild(unittest.TestCase):
    def test_no_equations(self):
        # Test that a set of symbols with no endogenous variables still
        # successfully generates a class
        expected = '''class Model(BaseModel):
    ENDOGENOUS: List[str] = []
    EXOGENOUS: List[str] = ['C', 'G']

    PARAMETERS: List[str] = []
    ERRORS: List[str] = []

    NAMES: List[str] = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
    CHECK: List[str] = ENDOGENOUS

    LAGS: int = 0
    LEADS: int = 0

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
        pass'''

        symbols = fsic.parse_model('Y = C + G')
        # Delete the symbol for the endogenous variable
        del symbols[0]

        code = fsic.parser.build_model_definition(symbols)
        self.assertEqual(code, expected)

    def test_no_type_hints(self):
        # Check that `build_model_definition()` can generate a code template
        # without type hints
        expected = '''class Model(BaseModel):
    ENDOGENOUS = ['Y']
    EXOGENOUS = ['C', 'I', 'G', 'X', 'M']

    PARAMETERS = []
    ERRORS = []

    NAMES = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
    CHECK = ENDOGENOUS

    LAGS = 0
    LEADS = 0

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
        # Y[t] = C[t] + I[t] + G[t] + X[t] - M[t]
        self._Y[t] = self._C[t] + self._I[t] + self._G[t] + self._X[t] - self._M[t]'''

        symbols = fsic.parse_model('Y = C + I + G + X - M')
        code = fsic.parser.build_model_definition(symbols, with_type_hints=False)
        self.assertEqual(code, expected)

    def test_no_symbols(self):
        # Test that empty input generates an empty model template
        expected = '''class Model(BaseModel):
    ENDOGENOUS: List[str] = []
    EXOGENOUS: List[str] = []

    PARAMETERS: List[str] = []
    ERRORS: List[str] = []

    NAMES: List[str] = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
    CHECK: List[str] = ENDOGENOUS

    LAGS: int = 0
    LEADS: int = 0

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
        pass'''

        symbols = fsic.parse_model('')
        code = fsic.parser.build_model_definition(symbols)
        self.assertEqual(code, expected)

    def test_conditional_expression(self):
        # Test that the parser can handle certain conditional expressions
        expected = '''class Model(BaseModel):
    ENDOGENOUS: List[str] = ['Y']
    EXOGENOUS: List[str] = ['X', 'Z']

    PARAMETERS: List[str] = []
    ERRORS: List[str] = []

    NAMES: List[str] = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
    CHECK: List[str] = ENDOGENOUS

    LAGS: int = 0
    LEADS: int = 0

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
        # Y[t] = X[t] if X[t] > Z[t] else Z[t]
        self._Y[t] = self._X[t] if self._X[t] > self._Z[t] else self._Z[t]'''

        symbols = fsic.parse_model('Y = X if X > Z else Z')
        code = fsic.build_model(symbols).CODE
        self.assertEqual(code, expected)

    def test_multiple_verbatim_statements(self):
        # Check that the parser can generate code from a block that includes
        # verbatim code
        # Test multiple statements to check that the code doesn't incorrectly
        # raise collisions/duplicates
        expected = '''class Model(BaseModel):
    ENDOGENOUS: List[str] = []
    EXOGENOUS: List[str] = []

    PARAMETERS: List[str] = []
    ERRORS: List[str] = []

    NAMES: List[str] = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
    CHECK: List[str] = ENDOGENOUS

    LAGS: int = 0
    LEADS: int = 0

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
        # `self.Y[t] = self.C[t] + self.G[t]`
        self.Y[t] = self.C[t] + self.G[t]

        # `self.T[t] = self.theta[t] * self.Y[t]`
        self.T[t] = self.theta[t] * self.Y[t]'''

        symbols = fsic.parse_model(
            """
`self.Y[t] = self.C[t] + self.G[t]`
`self.T[t] = self.theta[t] * self.Y[t]`
"""
        )
        code = fsic.build_model(symbols).CODE
        self.assertEqual(code, expected)

    def test_verbatim_block(self):
        # Check that a verbatim block is correctly inserted into the model
        # template
        expected = '''class Model(BaseModel):
    ENDOGENOUS: List[str] = []
    EXOGENOUS: List[str] = []

    PARAMETERS: List[str] = []
    ERRORS: List[str] = []

    NAMES: List[str] = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
    CHECK: List[str] = ENDOGENOUS

    LAGS: int = 0
    LEADS: int = 0

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
        # ```
        # self.T[t] = self.theta[t] * self.Y[t]
        # ```
        self.T[t] = self.theta[t] * self.Y[t]'''

        symbols = fsic.parse_model(
            """
```
self.T[t] = self.theta[t] * self.Y[t]
```
"""
        )
        code = fsic.build_model_definition(symbols)
        self.assertEqual(code, expected)

    def test_custom_converter(self):
        # Test that a custom function can be passed into the model builder
        expected = '''class Model(BaseModel):
    ENDOGENOUS: List[str] = ['Y']
    EXOGENOUS: List[str] = ['X']

    PARAMETERS: List[str] = []
    ERRORS: List[str] = []

    NAMES: List[str] = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
    CHECK: List[str] = ENDOGENOUS

    LAGS: int = 0
    LEADS: int = 0

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
        # Y[t] = X[t]
        _ = self._X[t]
        if np.isfinite(_):
            self._Y[t] = _'''

        symbols = fsic.parse_model('Y = X')

        def converter(symbol):
            lhs, rhs = map(str.strip, symbol.code.split('=', maxsplit=1))
            return """\
# {}
_ = {}
if np.isfinite(_):
    {} = _""".format(symbol.equation, rhs, lhs)

        code = fsic.build_model(symbols, converter=converter).CODE
        self.assertEqual(code, expected)

    def test_empty_endogenous_symbols(self):
        # Check that `build_model()` can generate a model from an endogenous
        # symbol with no accompanying equation
        Model = fsic.build_model(
            [
                fsic.parser.Symbol(
                    # Note how `equation` and `code` are `None`
                    name='C',
                    type=fsic.parser.Type.ENDOGENOUS,
                    lags=0,
                    leads=0,
                    equation=None,
                    code=None,
                )
            ]
        )

        self.assertEqual(Model.ENDOGENOUS, ['C'])
        self.assertEqual(Model.EXOGENOUS, [])
        self.assertEqual(Model.PARAMETERS, [])
        self.assertEqual(Model.ERRORS, [])

        self.assertEqual(Model.NAMES, ['C'])
        self.assertEqual(Model.CHECK, ['C'])

        self.assertEqual(Model.LAGS, 0)
        self.assertEqual(Model.LEADS, 0)

    def test_lags(self):
        # Check that the `lags` keyword argument imposes a lag length on the
        # final model

        # No symbols: Take specified lag length
        self.assertEqual(fsic.build_model([], lags=2).LAGS, 2)

        # Symbol with lag of 1: Take specified lag length (impose 2)
        self.assertEqual(
            fsic.build_model(
                [
                    fsic.parser.Symbol(
                        name='C',
                        type=fsic.parser.Type.ENDOGENOUS,
                        lags=-1,
                        leads=0,
                        equation=None,
                        code=None,
                    )
                ],
                lags=2,
            ).LAGS,
            2,
        )

        # Symbol with lag of 3: Take specified lag length (impose 2)
        self.assertEqual(
            fsic.build_model(
                [
                    fsic.parser.Symbol(
                        name='C',
                        type=fsic.parser.Type.ENDOGENOUS,
                        lags=-3,
                        leads=0,
                        equation=None,
                        code=None,
                    )
                ],
                lags=2,
            ).LAGS,
            2,
        )

    def test_leads(self):
        # Check that the `leads` keyword argument imposes a lead length on the
        # final model

        # No symbols: Take specified lead length
        self.assertEqual(fsic.build_model([], leads=2).LEADS, 2)

        # Symbol with lead of 1: Take specified lead length (impose 2)
        self.assertEqual(
            fsic.build_model(
                [
                    fsic.parser.Symbol(
                        name='C',
                        type=fsic.parser.Type.ENDOGENOUS,
                        lags=0,
                        leads=1,
                        equation=None,
                        code=None,
                    )
                ],
                leads=2,
            ).LEADS,
            2,
        )

        # Symbol with lead of 3: Take specified lead length (impose 2)
        self.assertEqual(
            fsic.build_model(
                [
                    fsic.parser.Symbol(
                        name='C',
                        type=fsic.parser.Type.ENDOGENOUS,
                        lags=0,
                        leads=3,
                        equation=None,
                        code=None,
                    )
                ],
                leads=2,
            ).LEADS,
            2,
        )

    def test_min_lags(self):
        # Check that the `min_lags` keyword argument ensures a minimum lag
        # length

        # No symbols: Take specified minimum lag length
        self.assertEqual(fsic.build_model([], min_lags=2).LAGS, 2)

        # Symbol with lag of 1: Take minimum lag length (impose 2)
        self.assertEqual(
            fsic.build_model(
                [
                    fsic.parser.Symbol(
                        name='C',
                        type=fsic.parser.Type.ENDOGENOUS,
                        lags=-1,
                        leads=0,
                        equation=None,
                        code=None,
                    )
                ],
                min_lags=2,
            ).LAGS,
            2,
        )

        # Symbol with lag of 3: Ignore minimum lag length (set to 3)
        self.assertEqual(
            fsic.build_model(
                [
                    fsic.parser.Symbol(
                        name='C',
                        type=fsic.parser.Type.ENDOGENOUS,
                        lags=-3,
                        leads=0,
                        equation=None,
                        code=None,
                    )
                ],
                min_lags=2,
            ).LAGS,
            3,
        )

    def test_min_leads(self):
        # Check that the `min_leads` keyword argument ensures a minimum lead
        # length

        # No symbols: Take specified minimum lead length
        self.assertEqual(fsic.build_model([], min_leads=2).LEADS, 2)

        # Symbol with lead of 1: Take minimum lead length (impose 2)
        self.assertEqual(
            fsic.build_model(
                [
                    fsic.parser.Symbol(
                        name='C',
                        type=fsic.parser.Type.ENDOGENOUS,
                        lags=0,
                        leads=-1,
                        equation=None,
                        code=None,
                    )
                ],
                min_leads=2,
            ).LEADS,
            2,
        )

        # Symbol with lead of 3: Ignore minimum lead length (set to 3)
        self.assertEqual(
            fsic.build_model(
                [
                    fsic.parser.Symbol(
                        name='C',
                        type=fsic.parser.Type.ENDOGENOUS,
                        lags=0,
                        leads=-3,
                        equation=None,
                        code=None,
                    )
                ],
                min_leads=2,
            ).LEADS,
            3,
        )


class TestBuildAndSolve(unittest.TestCase):
    # Tolerance for absolute differences to be considered almost equal
    DELTA = 0.05

    def test_gl2007_sim(self):
        # Test that `build_model()` correctly constructs a working model
        # definition from a simplified version of Godley and Lavoie's (2007)
        # Model SIM, and that the model solves as expected
        model = """
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
"""
        symbols = fsic.parse_model(model)
        SIM = fsic.build_model(symbols)

        # Initialise a new model instance, set values and solve
        sim = SIM(range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4)

        sim.G[15:] = 20
        sim.theta[15:] = 0.2

        sim.solve()

        # Check final period (steady-state) results
        self.assertAlmostEqual(sim.C[-1], 80.0, delta=self.DELTA)
        self.assertAlmostEqual(sim.YD[-1], 80.0, delta=self.DELTA)
        self.assertAlmostEqual(sim.H[-1], 80.0, delta=self.DELTA)
        self.assertAlmostEqual(sim.Y[-1], 100.0, delta=self.DELTA)
        self.assertAlmostEqual(sim.T[-1], 20.0, delta=self.DELTA)
        self.assertAlmostEqual(sim.G[-1], 20.0, delta=self.DELTA)

    def test_almon_ami(self):
        # Test that `build_model()` correctly constructs a working model
        # definition from Almon's (2017) AMI model, and that the model solves
        # as expected
        model = """
C = 0.60 * Y[-1] + 0.35 * Y[-2]
I = (R +
     1.0 * (PQ[-1] - PQ[-2]) +
     1.0 * (PQ[-2] - PQ[-3]) +
     0.1 * Q[-1])
PQ = max(Q, PQ[-1])
M = -380 + 0.2 * (C + I + X)
Q = C + I + G + X - M
Y = 0.72 * Q
"""
        symbols = fsic.parse_model(model)
        AMI = fsic.build_model(symbols)

        # Initialise a new model instance, set values and solve
        ami = AMI(range(1, 24 + 1))

        ami.G = [714.9, 718.5, 722.2, 725.9, 729.5,
                 733.2, 736.9, 740.5, 744.2, 747.8,
                 751.5, 755.2, 758.8, 762.5, 766.1,
                 769.8, 773.5, 777.1, 780.8, 784.5,
                 788.1, 791.8, 795.4, 799.1]  # fmt: skip
        ami.R = [518.7, 522.5, 526.2, 530.0, 533.7,
                 537.5, 541.2, 545.0, 548.7, 552.5,
                 556.2, 560.0, 563.7, 567.5, 571.2,
                 575.0, 578.7, 582.5, 586.2, 590.0,
                 593.7, 597.5, 601.2, 605.0]  # fmt: skip
        ami.X = [303.9, 308.8, 313.8, 318.7, 323.6,
                 328.5, 333.5, 338.4, 343.3, 348.3,
                 353.2, 358.1, 363.0, 368.0, 372.9,
                 377.8, 382.8, 387.7, 392.6, 397.6,
                 402.5, 407.4, 412.3, 417.3]  # fmt: skip

        ami.C[:6] = [2653.6, 2696.7, 2738.8, 2769.0, 2785.3, 2784.8]
        ami.I[:6] = [631.9, 653.6, 648.2, 628.1, 588.0, 553.7]
        ami.M[:6] = [337.9, 351.8, 361.8, 363.1, 359.4, 353.4]
        ami.Q[:7] = [3966.6, 4026.0, 4061.2, 4078.6, 4067.2, 4047.0, 4033.7]
        ami.PQ[:7] = [3966.6, 4026.0, 4061.2, 4078.6, 4078.6, 4078.6, 4078.6]
        ami.Y[:7] = [2855.9, 2898.7, 2924.0, 2936.6, 2928.4, 2913.8, 2904.2]

        ami.solve(start=7)

        # Check final period results
        self.assertAlmostEqual(ami.C[-1], 3631.3, delta=self.DELTA)
        self.assertAlmostEqual(ami.I[-1], 1136.2, delta=self.DELTA)
        self.assertAlmostEqual(ami.PQ[-1], 6394.5, delta=self.DELTA)
        self.assertAlmostEqual(ami.M[-1], 657.0, delta=self.DELTA)
        self.assertAlmostEqual(ami.Q[-1], 5327.0, delta=self.DELTA)
        self.assertAlmostEqual(ami.Y[-1], 3835.4, delta=self.DELTA)


class TestCopy(unittest.TestCase):
    SCRIPT = """
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
"""
    SYMBOLS = fsic.parse_model(SCRIPT)

    def setUp(self):
        self.Model = fsic.build_model(self.SYMBOLS)

    def test_copy_method(self):
        model = self.Model(range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, theta=0.2)
        model.G = 20

        duplicate_model = model.copy()

        # Values should be identical at this point
        self.assertTrue(np.allclose(model.values, duplicate_model.values))

        # The solved model should have different values to the duplicate
        model.solve()
        self.assertFalse(np.allclose(model.values, duplicate_model.values))

        # The solved duplicate should match the original again
        duplicate_model.solve()
        self.assertTrue(np.allclose(model.values, duplicate_model.values))

    def test_copy_dunder_method(self):
        model = self.Model(range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, theta=0.2)
        model.G = 20

        duplicate_model = copy.copy(model)

        # Values should be identical at this point
        self.assertTrue(np.allclose(model.values, duplicate_model.values))

        # The solved model should have different values to the duplicate
        model.solve()
        self.assertFalse(np.allclose(model.values, duplicate_model.values))

        # The solved duplicate should match the original again
        duplicate_model.solve()
        self.assertTrue(np.allclose(model.values, duplicate_model.values))

    def test_deepcopy_dunder_method(self):
        model = self.Model(range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, theta=0.2)
        model.G = 20

        duplicate_model = copy.deepcopy(model)

        # Values should be identical at this point
        self.assertTrue(np.allclose(model.values, duplicate_model.values))

        # The solved model should have different values to the duplicate
        model.solve()
        self.assertFalse(np.allclose(model.values, duplicate_model.values))

        # The solved duplicate should match the original again
        duplicate_model.solve()
        self.assertTrue(np.allclose(model.values, duplicate_model.values))


class TestSolve(unittest.TestCase):
    SCRIPT = """
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
"""
    SYMBOLS = fsic.parse_model(SCRIPT)

    # Tolerance for absolute differences to be considered almost equal
    DELTA = 0.05

    def setUp(self):
        self.Model = fsic.build_model(self.SYMBOLS)

    def test_solve_change_lags(self):
        # Check that changing the object-level `lags` attribute alters the
        # solution span
        model = self.Model(range(10))
        results = model.solve()

        self.assertEqual(model.lags, 1)
        self.assertEqual(results[0], list(range(1, 9 + 1)))

        model.lags = 2
        results = model.solve()

        self.assertEqual(model.lags, 2)
        self.assertEqual(results[0], list(range(2, 9 + 1)))

    def test_solve_change_leads(self):
        # Check that changing the object-level `leads` attribute alters the
        # solution span
        model = self.Model(range(10))
        results = model.solve()

        self.assertEqual(model.leads, 0)
        self.assertEqual(results[0], list(range(1, 9 + 1)))

        model.leads = 1
        results = model.solve()

        self.assertEqual(model.leads, 1)
        self.assertEqual(results[0], list(range(1, 8 + 1)))

    def test_solve_change_lags_and_leads(self):
        # Check that changing the object-level `lags` and `leads` attributes
        # alters the solution span
        model = self.Model(range(10))
        results = model.solve()

        self.assertEqual(model.lags, 1)
        self.assertEqual(model.leads, 0)

        self.assertEqual(results[0], list(range(1, 9 + 1)))

        model.lags = 2
        model.leads = 1
        results = model.solve()

        self.assertEqual(model.lags, 2)
        self.assertEqual(model.leads, 1)
        self.assertEqual(results[0], list(range(2, 8 + 1)))

    def test_solve_keyword_passthrough(self):
        # Check that keyword arguments pass through the solution stack without
        # triggering any errors
        model = self.Model(range(10))
        model.solve(custom_keyword=True)

    def test_solve_t_keyword_passthrough(self):
        # Check that keyword arguments pass through the solution stack without
        # triggering any errors
        model = self.Model(range(10))

        for t, _ in model.iter_periods():
            model.solve_t(t, custom_keyword=True)

    def test_solve_t_negative(self):
        # Check support for negative values of `t` in `solve_t()`
        model = self.Model(range(1945, 2010 + 1), alpha_1=0.5)

        model.G[-1] = 20
        model.solve_t(-1)

        self.assertEqual(model.Y[-1].round(1), 40.0)
        self.assertEqual(model.YD[-1].round(1), 40.0)
        self.assertEqual(model.C[-1].round(1), 20.0)

    def test_solve_t_negative_with_offset(self):
        # Check support for negative values of `t` in `solve_t()`, with an
        # offset
        model = self.Model(range(1945, 2010 + 1), alpha_1=0.5)

        model.G[-1] = 20
        model.solve_t(-1, offset=-1)

        self.assertEqual(model.Y[-1].round(1), 40.0)
        self.assertEqual(model.YD[-1].round(1), 40.0)
        self.assertEqual(model.C[-1].round(1), 20.0)

    def test_offset_error_lag_solve(self):
        # Check for an `IndexError` if `offset` points prior to the span of the
        # current model instance
        model = self.Model(range(1945, 2010 + 1))

        with self.assertRaises(IndexError):
            # With Model *SIM*, trying to solve the second period (remember
            # there's a lag in the model) with an offset of -2 should fail
            model.solve(offset=-2)

    def test_offset_error_lag_solve_period(self):
        # Check for an `IndexError` if `offset` points prior to the span of the
        # current model instance
        model = self.Model(range(1945, 2010 + 1))

        with self.assertRaises(IndexError):
            # With Model *SIM*, trying to solve the second period (remember
            # there's a lag in the model) with an offset of -2 should fail
            model.solve_period(1946, offset=-2)

    def test_offset_error_lead_solve(self):
        # Check for an `IndexError` if `offset` points beyond the span of the
        # current model instance
        model = self.Model(range(1945, 2010 + 1))

        with self.assertRaises(IndexError):
            # With Model *SIM*, trying to solve the final period with an offset
            # of +1 should fail
            model.solve(offset=1)

    def test_offset_error_lead_solve_t(self):
        # Check for an `IndexError` if `offset` points beyond the span of the
        # current model instance
        model = self.Model(range(1945, 2010 + 1))

        with self.assertRaises(IndexError):
            # With Model *SIM*, trying to solve the final period with an offset
            # of +1 should fail
            model.solve_t(-1, offset=1)

    def test_solve_return_values(self):
        # Check that the return values from `solve()` are as expected
        model = self.Model(['A', 'B', 'C'])
        labels, indexes, solved = model.solve()

        self.assertEqual(labels, ['B', 'C'])
        self.assertEqual(indexes, [1, 2])
        self.assertEqual(solved, [True, True])

    def test_iter_solve_t(self):
        # Use `iter_periods()` and `solve_t` to solve the model
        model = self.Model(range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4)
        model.G[15:] = 20
        model.theta[15:] = 0.2

        for t, period in model.iter_periods():
            model.solve_t(t)

        # Check final period (steady-state) results
        self.assertAlmostEqual(model.C[-1], 80.0, delta=self.DELTA)
        self.assertAlmostEqual(model.YD[-1], 80.0, delta=self.DELTA)
        self.assertAlmostEqual(model.H[-1], 80.0, delta=self.DELTA)
        self.assertAlmostEqual(model.Y[-1], 100.0, delta=self.DELTA)
        self.assertAlmostEqual(model.T[-1], 20.0, delta=self.DELTA)
        self.assertAlmostEqual(model.G[-1], 20.0, delta=self.DELTA)

    def test_iter_solve_period(self):
        # Use `iter_periods()` and `solve_period` to solve the model
        model = self.Model(range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4)
        model.G[15:] = 20
        model.theta[15:] = 0.2

        for t, period in model.iter_periods():
            model.solve_period(period)

        # Check final period (steady-state) results
        self.assertAlmostEqual(model.C[-1], 80.0, delta=self.DELTA)
        self.assertAlmostEqual(model.YD[-1], 80.0, delta=self.DELTA)
        self.assertAlmostEqual(model.H[-1], 80.0, delta=self.DELTA)
        self.assertAlmostEqual(model.Y[-1], 100.0, delta=self.DELTA)
        self.assertAlmostEqual(model.T[-1], 20.0, delta=self.DELTA)
        self.assertAlmostEqual(model.G[-1], 20.0, delta=self.DELTA)

    def test_solve_add_variable(self):
        # Check that extending the model with new variables at runtime makes no
        # difference to the model solution
        base = self.Model(
            range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2
        )

        # Add further variables to a copy and check types before solving
        extended = base.copy()

        extended.add_variable('I', 20, dtype=int)
        extended.add_variable('J', 20)
        extended.add_variable('K', 20, dtype=float)
        extended.add_variable('L', False, dtype=bool)

        self.assertEqual(extended.I.dtype, int)
        self.assertEqual(extended.J.dtype, float)
        self.assertEqual(extended.K.dtype, float)
        self.assertEqual(extended.L.dtype, bool)

        self.assertEqual(base.values.shape, (9, 66))
        self.assertEqual(extended.values.shape, (13, 66))

        # Solve and check solution values match
        base.solve()
        extended.solve()

        self.assertTrue(np.allclose(base.values, extended.values[:-4, :]))

        self.assertEqual(base.values.shape, (9, 66))
        self.assertEqual(extended.values.shape, (13, 66))

        self.assertEqual(extended.I.dtype, int)
        self.assertEqual(extended.J.dtype, float)
        self.assertEqual(extended.K.dtype, float)
        self.assertEqual(extended.L.dtype, bool)

    def test_min_iter_solve(self):
        # Check that `min_iter` forces a minimum number of iterations

        # With no further arguments, the model with all parameters set to zero
        # should solve in one iteration per period
        model = self.Model(range(1945, 2010 + 1))
        model.solve()

        self.assertEqual(model.status[0], '-')
        self.assertTrue((model.status[1:] == '.').all())

        self.assertEqual(model.iterations[0], -1)
        self.assertTrue((model.iterations[1:] == 1).all())

        # The same model should still solve in one iteration but it should be
        # possible to force more iterations
        model = self.Model(range(1945, 2010 + 1))
        model.solve(min_iter=10)

        self.assertEqual(model.status[0], '-')
        self.assertTrue((model.status[1:] == '.').all())

        self.assertEqual(model.iterations[0], -1)
        self.assertTrue((model.iterations[1:] == 10).all())

    def test_min_iter_solve_t(self):
        # Check that `min_iter` forces a minimum number of iterations

        # With no further arguments, the model with all parameters set to zero
        # should solve in one iteration per period
        model = self.Model(range(1945, 2010 + 1))

        for t, _ in model.iter_periods():
            model.solve_t(t)

        self.assertEqual(model.status[0], '-')
        self.assertTrue((model.status[1:] == '.').all())

        self.assertEqual(model.iterations[0], -1)
        self.assertTrue((model.iterations[1:] == 1).all())

        # The same model should still solve in one iteration but it should be
        # possible to force more iterations
        model = self.Model(range(1945, 2010 + 1))

        for t, _ in model.iter_periods():
            model.solve_t(t, min_iter=10)

        self.assertEqual(model.status[0], '-')
        self.assertTrue((model.status[1:] == '.').all())

        self.assertEqual(model.iterations[0], -1)
        self.assertTrue((model.iterations[1:] == 10).all())

    def test_min_iter_error_solve(self):
        # Check that models raise an error if `min_iter` exceeds `max_iter`
        model = self.Model(range(1945, 2010 + 1))

        with self.assertRaises(ValueError):
            model.solve(min_iter=10, max_iter=5)

    def test_min_iter_error_solve_t(self):
        # Check that models raise an error if `min_iter` exceeds `max_iter`
        model = self.Model(range(1945, 2010 + 1))

        with self.assertRaises(ValueError):
            model.solve_t(1, min_iter=10, max_iter=5)

    def test_solve_t_before_errors(self):
        # Check that error handling applies to `solve_t_before()`
        class Model(self.Model):
            def solve_t_before(self, t: int, *args, **kwargs) -> None:
                _ = self.H[t] / 0  # Divide by zero

        model = Model(range(10))

        with self.assertRaises(fsic.exceptions.SolutionError):
            model.solve()

    def test_solve_t_after_errors(self):
        # Check that error handling applies to `solve_t_after()`
        class Model(self.Model):
            def solve_t_after(self, t: int, *args, **kwargs) -> None:
                _ = self.H[t] / 0  # Divide by zero

        model = Model(range(10))

        with self.assertRaises(fsic.exceptions.SolutionError):
            model.solve()


class TestCustomModel(unittest.TestCase):
    SCRIPT = """
s = 1 - (C / YD)  # Unless handled, this equation will generate a NaN on the first iteration
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
"""
    SYMBOLS = fsic.parse_model(SCRIPT)

    # Tolerance for absolute differences to be considered almost equal
    DELTA = 0.05

    def setUp(self):
        def custom_converter(symbol):
            lhs, _ = map(str.strip, symbol.code.split('=', maxsplit=1))
            return """\
# {}
with warnings.catch_warnings():
    warnings.simplefilter('error')

    try:
        {}

    except RuntimeWarning:
        if errors == 'raise':
            {} = np.nan
            self.status[t] = 'E'
            self.iterations[t] = iteration

            raise fsic.exceptions.SolutionError(
                'Numerical solution error after {{}} iterations(s) '
                'in period with label: {{}} (index: {{}})'
                .format(iteration, self.span[t], t))

        elif errors == 'skip':
            {} = np.nan
            return

        elif errors == 'ignore':
            {} = np.nan

        elif errors == 'replace':
            {} = 0

        else:
            raise ValueError('Invalid `errors` argument: {{}}'.format(errors))""".format(
                symbol.equation, symbol.code, lhs, lhs, lhs, lhs
            )

        self.Model = fsic.build_model(self.SYMBOLS, converter=custom_converter)

    def test_custom_solve_raise(self):
        # Check that the custom code leads to a `SolutionError` in the absence
        # of any alternative error handling
        model = self.Model(
            range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2
        )

        with self.assertRaises(fsic.exceptions.SolutionError):
            model.solve()

        # `s` should have generated a NaN
        self.assertTrue(np.isclose(model.s[0], 0))
        self.assertTrue(np.isnan(model.s[1]))
        self.assertTrue(np.allclose(model.s[2:], 0))

        # The `SolutionError` should be recorded in `status`
        self.assertEqual(model.status[0], '-')
        self.assertEqual(model.status[1], 'E')
        self.assertTrue(np.all(model.status[2:] == '-'))

        # The single pass in `iterations` should also be recorded
        self.assertEqual(model.iterations[0], -1)
        self.assertEqual(model.iterations[1], 1)
        self.assertTrue(np.all(model.iterations[2:] == -1))

        # All other values should be unchanged
        for x in ['C', 'YD', 'Y', 'T', 'H']:
            self.assertTrue(np.allclose(model[x], 0))

        self.assertTrue(np.allclose(model.alpha_1, 0.6))
        self.assertTrue(np.allclose(model.alpha_2, 0.4))
        self.assertTrue(np.allclose(model.G, 20))
        self.assertTrue(np.allclose(model.theta, 0.2))

    def test_custom_solve_skip(self):
        # Check that the custom code correctly skips on the first iteration of
        # each period
        model = self.Model(
            range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2
        )

        model.solve(errors='skip')

        # `s` should just be NaNs
        self.assertTrue(np.isclose(model.s[0], 0))
        self.assertTrue(np.all(np.isnan(model.s[1:])))

        # Solution status should indicate [S]kipped
        self.assertEqual(model.status[0], '-')
        self.assertTrue(np.all(model.status[1:] == 'S'))

        # Just one iteration per period
        self.assertEqual(model.iterations[0], -1)
        self.assertTrue(np.all(model.iterations[1:] == 1))

        # All other values should be unchanged
        for x in ['C', 'YD', 'Y', 'T', 'H']:
            self.assertTrue(np.allclose(model[x], 0))

        self.assertTrue(np.allclose(model.alpha_1, 0.6))
        self.assertTrue(np.allclose(model.alpha_2, 0.4))
        self.assertTrue(np.allclose(model.G, 20))
        self.assertTrue(np.allclose(model.theta, 0.2))

    def test_custom_solve_ignore(self):
        # Check that the custom code solves as usual with `errors='ignore'`
        model = self.Model(
            range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2
        )

        model._evaluate(1, errors='ignore')
        self.assertTrue(np.isnan(model.s[1]))

        model.solve(errors='ignore')

        # `s` should be entirely non-NaN
        self.assertFalse(np.any(np.isnan(model.s)))

        # The other model variables should have converged to their stationary
        # state values
        self.assertAlmostEqual(model.C[-1], 80, delta=self.DELTA)
        self.assertAlmostEqual(model.YD[-1], 80, delta=self.DELTA)
        self.assertAlmostEqual(model.H[-1], 80, delta=self.DELTA)
        self.assertAlmostEqual(model.Y[-1], 100, delta=self.DELTA)
        self.assertAlmostEqual(model.T[-1], 20, delta=self.DELTA)

    def test_custom_solve_replace(self):
        # Check that the custom code solves as usual with `errors='replace'`
        model = self.Model(
            range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2
        )

        model._evaluate(1, errors='replace')
        self.assertAlmostEqual(model.s[1], 0)

        model.solve(errors='replace')

        # `s` should be entirely non-NaN
        self.assertFalse(np.any(np.isnan(model.s)))

        # The other model variables should have converged to their stationary
        # state values
        self.assertAlmostEqual(model.C[-1], 80, delta=self.DELTA)
        self.assertAlmostEqual(model.YD[-1], 80, delta=self.DELTA)
        self.assertAlmostEqual(model.H[-1], 80, delta=self.DELTA)
        self.assertAlmostEqual(model.Y[-1], 100, delta=self.DELTA)
        self.assertAlmostEqual(model.T[-1], 20, delta=self.DELTA)


class TestCustomOverrides(unittest.TestCase):
    def test_evaluate_error(self):
        # Check that `solve_t()` can catch errors raised in `_evaluate()`

        class Model(fsic.BaseModel):
            def _evaluate(self, t, *args, **kwargs):
                # In base Python, this raises a `ZeroDivisionError`
                x = 1 / 0  # noqa: F841

        model = Model(range(5))

        with self.assertRaises(fsic.exceptions.SolutionError):
            model.solve()

    class RegularModel(fsic.BaseModel):
        """Standard model setup as produced by `fsic` `parse_model()` and `build_model()`."""

        ENDOGENOUS = ['G', 'C', 'YD', 'H', 'Y', 'T']
        EXOGENOUS = ['G_bar']

        PARAMETERS = ['alpha_1', 'alpha_2', 'theta']
        ERRORS = []

        NAMES = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
        CHECK = ENDOGENOUS

        LAGS = 1
        LEADS = 0

        def _evaluate(self, t: int, *, errors='raise', iteration=None, **kwargs):
            # G[t] = G_bar[t]
            self._G[t] = self._G_bar[t]

            # C[t] = alpha_1[t] * YD[t] + alpha_2[t] * H[t-1]
            self._C[t] = (
                self._alpha_1[t] * self._YD[t] + self._alpha_2[t] * self._H[t - 1]
            )

            # YD[t] = Y[t] - T[t]
            self._YD[t] = self._Y[t] - self._T[t]

            # H[t] = H[t-1] + YD[t] - C[t]
            self._H[t] = self._H[t - 1] + self._YD[t] - self._C[t]

            # Y[t] = C[t] + G[t]
            self._Y[t] = self._C[t] + self._G[t]

            # T[t] = theta[t] * Y[t]
            self._T[t] = self._theta[t] * self._Y[t]

    class BeforeModel(fsic.BaseModel):
        """Variant on `RegularModel` but with an over-riding `solve_t_before()` method."""

        ENDOGENOUS = ['G', 'C', 'YD', 'H', 'Y', 'T']
        EXOGENOUS = ['G_bar']

        PARAMETERS = ['alpha_1', 'alpha_2', 'theta']
        ERRORS = []

        NAMES = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
        CHECK = ENDOGENOUS

        LAGS = 1
        LEADS = 0

        def solve_t_before(self, t, *args, **kwargs):
            # G[t] = G_bar[t]
            self._G[t] = self._G_bar[t]

        def _evaluate(self, t, *, errors='raise', iteration=None, **kwargs):
            # C[t] = alpha_1[t] * YD[t] + alpha_2[t] * H[t-1]
            self._C[t] = (
                self._alpha_1[t] * self._YD[t] + self._alpha_2[t] * self._H[t - 1]
            )

            # YD[t] = Y[t] - T[t]
            self._YD[t] = self._Y[t] - self._T[t]

            # H[t] = H[t-1] + YD[t] - C[t]
            self._H[t] = self._H[t - 1] + self._YD[t] - self._C[t]

            # Y[t] = C[t] + G[t]
            self._Y[t] = self._C[t] + self._G[t]

            # T[t] = theta[t] * Y[t]
            self._T[t] = self._theta[t] * self._Y[t]

    class AfterModel(fsic.BaseModel):
        """Variant on `RegularModel` but with an over-riding `solve_t_after()` method."""

        ENDOGENOUS = ['G', 'C', 'YD', 'H', 'Y', 'T']
        EXOGENOUS = ['G_bar']

        PARAMETERS = ['alpha_1', 'alpha_2', 'theta']
        ERRORS = []

        NAMES = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
        CHECK = ENDOGENOUS

        LAGS = 1
        LEADS = 0

        def solve_t_after(self, t, *args, **kwargs):
            # H[t] = H[t-1] + YD[t] - C[t]
            self._H[t] = self._H[t - 1] + self._YD[t] - self._C[t]

        def _evaluate(self, t, *, errors='raise', iteration=None, **kwargs):
            # G[t] = G_bar[t]
            self._G[t] = self._G_bar[t]

            # C[t] = alpha_1[t] * YD[t] + alpha_2[t] * H[t-1]
            self._C[t] = (
                self._alpha_1[t] * self._YD[t] + self._alpha_2[t] * self._H[t - 1]
            )

            # YD[t] = Y[t] - T[t]
            self._YD[t] = self._Y[t] - self._T[t]

            # Y[t] = C[t] + G[t]
            self._Y[t] = self._C[t] + self._G[t]

            # T[t] = theta[t] * Y[t]
            self._T[t] = self._theta[t] * self._Y[t]

    class BeforeAndAfterModel(fsic.BaseModel):
        """Variant on `RegularModel` but with over-riding `solve_t_before()` and `solve_t_after()` methods."""

        ENDOGENOUS = ['G', 'C', 'YD', 'H', 'Y', 'T']
        EXOGENOUS = ['G_bar']

        PARAMETERS = ['alpha_1', 'alpha_2', 'theta']
        ERRORS = []

        NAMES = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
        CHECK = ENDOGENOUS

        LAGS = 1
        LEADS = 0

        def solve_t_before(self, t, *args, **kwargs):
            # G[t] = G_bar[t]
            self._G[t] = self._G_bar[t]

        def solve_t_after(self, t, *args, **kwargs):
            # H[t] = H[t-1] + YD[t] - C[t]
            self._H[t] = self._H[t - 1] + self._YD[t] - self._C[t]

        def _evaluate(self, t, *, errors='raise', iteration=None, **kwargs):
            # C[t] = alpha_1[t] * YD[t] + alpha_2[t] * H[t-1]
            self._C[t] = (
                self._alpha_1[t] * self._YD[t] + self._alpha_2[t] * self._H[t - 1]
            )

            # YD[t] = Y[t] - T[t]
            self._YD[t] = self._Y[t] - self._T[t]

            # Y[t] = C[t] + G[t]
            self._Y[t] = self._C[t] + self._G[t]

            # T[t] = theta[t] * Y[t]
            self._T[t] = self._theta[t] * self._Y[t]

    def test_evaluate_before(self):
        # Check that a model amended to carry out some pre-solution calculation
        # produces the same results as the regular model
        regular_model = self.RegularModel(
            range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G_bar=20, theta=0.2
        )
        regular_model.solve()

        before_model = self.BeforeModel(
            range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G_bar=20, theta=0.2
        )
        before_model.solve()

        # Quick check that results match expected results
        self.assertAlmostEqual(regular_model.Y[-1], 100.0, places=2)

        # Check model results are identical
        self.assertTrue(np.allclose(regular_model.values, before_model.values))

    def test_evaluate_after(self):
        # Check that a model amended to carry out some post-solution calculation
        # produces the same results as the regular model
        regular_model = self.RegularModel(
            range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G_bar=20, theta=0.2
        )
        regular_model.solve()

        after_model = self.AfterModel(
            range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G_bar=20, theta=0.2
        )
        after_model.solve()

        # Quick check that results match expected results
        self.assertAlmostEqual(regular_model.Y[-1], 100.0, places=2)

        # Check model results are identical
        self.assertTrue(np.allclose(regular_model.values, after_model.values))

    def test_evaluate_before_and_after(self):
        # Check that a model amended to carry out some pre- and post-solution
        # calculations produces the same results as the regular model
        regular_model = self.RegularModel(
            range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G_bar=20, theta=0.2
        )
        regular_model.solve()

        before_after_model = self.BeforeAndAfterModel(
            range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G_bar=20, theta=0.2
        )
        before_after_model.solve()

        # Quick check that results match expected results
        self.assertAlmostEqual(regular_model.Y[-1], 100.0, places=2)

        # Check model results are identical
        self.assertTrue(np.allclose(regular_model.values, before_after_model.values))


class TestParserErrors(unittest.TestCase):
    def test_invalid_index(self):
        # Check that the parser can detect an invalid index
        with self.assertRaises(fsic.exceptions.ParserError):
            # In the index, '1' and '-' are the wrong way around
            fsic.parse_model('A = A[1-]')

    def test_missing_closing_bracket(self):
        # Check that the parser can detect a missing closing bracket
        with self.assertRaises(fsic.exceptions.ParserError):
            # Missing closing bracket at the end of the string below
            fsic.parse_model('Y = C + I + G + (X - M')

    def test_misplaced_closing_bracket(self):
        # Check that the parser can detect a closing bracket without an
        # accompanying prior open bracket
        with self.assertRaises(fsic.exceptions.ParserError):
            fsic.parse_model('Y = C + I + G +)X - M')

        with self.assertRaises(fsic.exceptions.ParserError):
            fsic.parse_model('Y = C + I + G +)X - M)')

        with self.assertRaises(fsic.exceptions.ParserError):
            fsic.parse_model('Y = C + I + G +)X - M(')

    def test_extra_equals_single_equation(self):
        with self.assertRaises(fsic.exceptions.ParserError):
            fsic.parse_model('Y = C + I + G = X - M')

    def test_extra_equals_multiple_equations(self):
        with self.assertRaises(fsic.exceptions.ParserError):
            fsic.parse_model(
                """
Y = C + I + G = X - M
Z = C + I + G = X - M
"""
            )

    def test_double_definition(self):
        # Check test for endogenous variables that are set twice
        with self.assertRaises(fsic.exceptions.ParserError):
            fsic.parse_model(
                """
Y = C + I + G + X - M
Y = GVA + TSP
"""
            )

    @unittest.skipIf(
        sys.version_info[:2] < (3, 8),
        'Parser test for accidental treatment of floats as callable not supported prior to Python version 3.8',
    )
    def test_accidental_float_call(self):
        # Check that something like 'A = 0.5(B)' (missing * operator) raises a
        # `ParserError`
        with self.assertRaises(fsic.exceptions.ParserError):
            fsic.parse_model('A = 0.5(B)')

    def test_keywords_as_variables_lhs(self):
        # Check that the parser catches attempts to use reserved Python
        # keywords as endogenous variable names in a model
        for name in keyword.kwlist:
            with self.subTest(name=name):
                with self.assertRaises(fsic.exceptions.ParserError):
                    fsic.parse_model('{} = 0'.format(name))

    def test_keywords_as_variables_rhs(self):
        # Check that the parser catches attempts to use reserved Python
        # keywords as exogenous variable names in a model
        for name in keyword.kwlist:
            with self.subTest(name=name):
                with self.assertRaises(fsic.exceptions.ParserError):
                    fsic.parse_model('X = {}[-1]'.format(name))


class TestBuildErrors(unittest.TestCase):
    def test_extra_equals(self):
        symbols = fsic.parse_model('Y = C + I + G = X - M', check_syntax=False)
        with self.assertRaises(fsic.exceptions.BuildError):
            Model = fsic.build_model(symbols)  # noqa: F841


class TestSolutionErrorHandling(unittest.TestCase):
    SCRIPT = """
s = 1 - (C / Y)  # Divide-by-zero equation (generating a NaN) appears first
Y = C + G
C = {c0} + {c1} * Y
g = log(G)  # Use to test for infinity with log(0)
"""
    SYMBOLS = fsic.parse_model(SCRIPT)

    def setUp(self):
        self.Model = fsic.build_model(self.SYMBOLS)

    def test_raise_nans(self):
        # Model should halt on first period
        model = self.Model(range(10), G=20)

        with self.assertRaises(fsic.exceptions.SolutionError):
            model.solve()

        # `catch_first_error=True` rolls back non-finite values
        self.assertTrue(np.allclose(model.s, 0))

        self.assertTrue(np.all(model.status == np.array(['E'] + ['-'] * 9)))
        self.assertTrue(np.all(model.iterations == np.array([1] + [-1] * 9)))

    def test_raise_nans_old(self):  # Pre-0.8.0 behaviour
        # Model should halt on first period
        model = self.Model(range(10), G=20)

        with self.assertRaises(fsic.exceptions.SolutionError):
            model.solve(catch_first_error=False)

        self.assertTrue(np.isnan(model.s[0]))
        self.assertTrue(np.allclose(model.s[1:], 0))

        self.assertTrue(np.all(model.status == np.array(['E'] + ['-'] * 9)))
        self.assertTrue(np.all(model.iterations == np.array([1] + [-1] * 9)))

    def test_raise_prior_nans(self):
        # Model should halt if pre-existing NaNs detected
        model = self.Model(range(10), s=np.nan)

        with self.assertRaises(fsic.exceptions.SolutionError):
            model.solve()

    def test_skip_solve(self):
        model = self.Model(range(10), G=20)

        # Model should skip to next period each time
        model.solve(errors='skip')
        self.assertTrue(np.all(np.isnan(model.s)))
        self.assertTrue(np.all(model.status == 'S'))
        self.assertTrue(np.all(model.iterations == 1))

        # Re-running with 'skip' should successfully solve, though
        model.solve(errors='skip')
        self.assertTrue(np.allclose(model.s, 1))
        self.assertTrue(np.all(model.status == '.'))
        self.assertTrue(np.all(model.iterations == 2))

    def test_skip_solve_t(self):
        model = self.Model(range(10), G=20)

        # Model should skip to next period each time
        for t, _ in model.iter_periods():
            model.solve_t(t, errors='skip')

        self.assertTrue(np.all(np.isnan(model.s)))
        self.assertTrue(np.all(model.status == 'S'))
        self.assertTrue(np.all(model.iterations == 1))

        # Re-running with 'skip' should successfully solve, though
        for t, _ in model.iter_periods():
            model.solve_t(t, errors='skip')

        self.assertTrue(np.allclose(model.s, 1))
        self.assertTrue(np.all(model.status == '.'))
        self.assertTrue(np.all(model.iterations == 2))

    def test_ignore_successfully_solve(self):
        # Model should keep solving (successfully in this case)
        model = self.Model(range(10), G=20)
        model.solve(errors='ignore')

        self.assertTrue(np.allclose(model.s, 1))
        self.assertTrue(np.all(model.status == '.'))

        # Three iterations to solve:
        # 1. `s` evaluates to NaN: continue
        # 2. NaNs persist from previous iteration: continue
        # 3. Convergence check now possible (no NaNs): success
        self.assertTrue(np.all(model.iterations == 3))

    def test_ignore_successfully_solve_t(self):
        # Model should keep solving (successfully in this case)
        model = self.Model(range(10), G=20)

        for t, _ in model.iter_periods():
            model.solve_t(t, errors='ignore')

        self.assertTrue(np.allclose(model.s, 1))
        self.assertTrue(np.all(model.status == '.'))

        # Three iterations to solve:
        # 1. `s` evaluates to NaN: continue
        # 2. NaNs persist from previous iteration: continue
        # 3. Convergence check now possible (no NaNs): success
        self.assertTrue(np.all(model.iterations == 3))

    def test_ignore_unsuccessfully_solve(self):
        # Model should keep solving (unsuccessfully in this case)
        model = self.Model(range(10))
        model.solve(failures='ignore', errors='ignore')

        self.assertTrue(np.all(np.isnan(model.s)))
        self.assertTrue(np.all(model.status == 'F'))
        self.assertTrue(np.all(model.iterations == 100))

    def test_ignore_unsuccessfully_solve_t(self):
        # Model should keep solving (unsuccessfully in this case)
        model = self.Model(range(10))

        for t, _ in model.iter_periods():
            model.solve_t(t, failures='ignore', errors='ignore')

        self.assertTrue(np.all(np.isnan(model.s)))
        self.assertTrue(np.all(model.status == 'F'))
        self.assertTrue(np.all(model.iterations == 100))

    def test_ignore_prior_nans_solve(self):
        model = self.Model(range(10), s=np.nan, G=20)
        model.solve(errors='ignore')

        self.assertTrue(np.allclose(model.s, 1))
        self.assertTrue(np.all(model.status == '.'))
        self.assertTrue(np.all(model.iterations == 3))

    def test_ignore_prior_nans_solve_t(self):
        model = self.Model(range(10), s=np.nan, G=20)

        for t, _ in model.iter_periods():
            model.solve_t(t, errors='ignore')

        self.assertTrue(np.allclose(model.s, 1))
        self.assertTrue(np.all(model.status == '.'))
        self.assertTrue(np.all(model.iterations == 3))

    def test_replace_successfully_solve(self):
        # Model should replace NaNs and keep solving (successfully in this case)
        model = self.Model(range(10), G=20)
        model.solve(errors='replace')

        self.assertTrue(np.allclose(model.s, 1))
        self.assertTrue(np.all(model.status == '.'))
        self.assertTrue(np.all(model.iterations == 3))

    def test_replace_successfully_solve_t(self):
        # Model should replace NaNs and keep solving (successfully in this case)
        model = self.Model(range(10), G=20)

        for t, _ in model.iter_periods():
            model.solve_t(t, errors='replace')

        self.assertTrue(np.allclose(model.s, 1))
        self.assertTrue(np.all(model.status == '.'))
        self.assertTrue(np.all(model.iterations == 3))

    def test_replace_unsuccessfully_solve(self):
        # Model should replace NaNs and keep solving (unsuccessfully in this case)
        model = self.Model(range(10))
        model.solve(failures='ignore', errors='replace')

        self.assertTrue(np.all(np.isnan(model.s)))
        self.assertTrue(np.all(model.status == 'F'))
        self.assertTrue(np.all(model.iterations == 100))

    def test_replace_unsuccessfully_solve_t(self):
        # Model should replace NaNs and keep solving (unsuccessfully in this case)
        model = self.Model(range(10))

        for t, _ in model.iter_periods():
            model.solve_t(t, failures='ignore', errors='replace')

        self.assertTrue(np.all(np.isnan(model.s)))
        self.assertTrue(np.all(model.status == 'F'))
        self.assertTrue(np.all(model.iterations == 100))

    def test_replace_prior_nans_solve(self):
        model = self.Model(range(10), s=np.nan, G=20)
        model.solve(errors='replace')

        self.assertTrue(np.allclose(model.s, 1))
        self.assertTrue(np.all(model.status == '.'))
        self.assertTrue(np.all(model.iterations == 3))

    def test_replace_prior_nans_solve_t(self):
        model = self.Model(range(10), s=np.nan, G=20)

        for t, _ in model.iter_periods():
            model.solve_t(t, errors='replace')

        self.assertTrue(np.allclose(model.s, 1))
        self.assertTrue(np.all(model.status == '.'))
        self.assertTrue(np.all(model.iterations == 3))

    def test_raise_infinities(self):
        # Model should halt on first period because of log(0)
        model = self.Model(range(10), c0=10, C=10, Y=10)

        with self.assertRaises(fsic.exceptions.SolutionError):
            model.solve()

        # `catch_first_error=True` rolls back non-finite values
        self.assertTrue(np.allclose(model.g, 0))

        self.assertTrue(np.allclose(model.s, 0))
        self.assertTrue(np.allclose(model.C, 10))
        self.assertTrue(np.allclose(model.Y, 10))
        self.assertTrue(np.allclose(model.G, 0))
        self.assertTrue(np.allclose(model.c0, 10))
        self.assertTrue(np.allclose(model.c1, 0))

        self.assertTrue(np.all(model.status == np.array(['E'] + ['-'] * 9)))
        self.assertTrue(np.all(model.iterations == np.array([1] + [-1] * 9)))

    def test_raise_infinities_old(self):  # Pre-0.8.0 behaviour
        # Model should halt on first period because of log(0)
        model = self.Model(range(10), c0=10, C=10, Y=10)

        with self.assertRaises(fsic.exceptions.SolutionError):
            model.solve(catch_first_error=False)

        self.assertTrue(np.isinf(model.g[0]))
        self.assertTrue(np.allclose(model.g[1:], 0))

        self.assertTrue(np.allclose(model.s, 0))
        self.assertTrue(np.allclose(model.C, 10))
        self.assertTrue(np.allclose(model.Y, 10))
        self.assertTrue(np.allclose(model.G, 0))
        self.assertTrue(np.allclose(model.c0, 10))
        self.assertTrue(np.allclose(model.c1, 0))

        self.assertTrue(np.all(model.status == np.array(['E'] + ['-'] * 9)))
        self.assertTrue(np.all(model.iterations == np.array([1] + [-1] * 9)))

    def test_replace_infinities_solve(self):
        model = self.Model(range(10), c0=10, C=10, Y=10)
        model.solve(errors='replace', failures='ignore')

        self.assertTrue(np.all(np.isinf(model.g)))

        self.assertTrue(np.allclose(model.s, 0))
        self.assertTrue(np.allclose(model.C, 10))
        self.assertTrue(np.allclose(model.Y, 10))
        self.assertTrue(np.allclose(model.G, 0))
        self.assertTrue(np.allclose(model.c0, 10))
        self.assertTrue(np.allclose(model.c1, 0))

        self.assertTrue(np.all(model.status == 'F'))

    def test_replace_infinities_solve_t(self):
        model = self.Model(range(10), c0=10, C=10, Y=10)

        for t, _ in model.iter_periods():
            model.solve_t(t, errors='replace', failures='ignore')

        self.assertTrue(np.all(np.isinf(model.g)))

        self.assertTrue(np.allclose(model.s, 0))
        self.assertTrue(np.allclose(model.C, 10))
        self.assertTrue(np.allclose(model.Y, 10))
        self.assertTrue(np.allclose(model.G, 0))
        self.assertTrue(np.allclose(model.c0, 10))
        self.assertTrue(np.allclose(model.c1, 0))

        self.assertTrue(np.all(model.status == 'F'))


class TestNonConvergenceError(unittest.TestCase):
    MODEL = """
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
"""
    SYMBOLS = fsic.parse_model(MODEL)

    def setUp(self):
        self.Model = fsic.build_model(self.SYMBOLS)

    def test_nonconvergence_raise(self):
        model = self.Model(range(5), alpha_1=0.6, alpha_2=0.4, theta=0.2, G=20)

        # First period (after initial lag) should solve
        model.solve_t(1)

        # Second period (with limited number of iterations) should fail to
        # solve
        with self.assertRaises(fsic.exceptions.NonConvergenceError):
            model.solve_t(2, max_iter=5)

        self.assertTrue(np.all(model.status == np.array(['-', '.', 'F', '-', '-'])))

    def test_nonconvergence_ignore_solve(self):
        model = self.Model(range(5), alpha_1=0.6, alpha_2=0.4, theta=0.2, G=20)
        model.solve(max_iter=5, failures='ignore')
        self.assertTrue(np.all(model.status[1:] == 'F'))

    def test_nonconvergence_ignore_solve_t(self):
        model = self.Model(range(5), alpha_1=0.6, alpha_2=0.4, theta=0.2, G=20)

        for t, _ in model.iter_periods():
            model.solve_t(t, max_iter=5, failures='ignore')

        self.assertTrue(np.all(model.status[1:] == 'F'))


class TestLinkerInit(unittest.TestCase):
    SYMBOLS_NO_LAGS = fsic.parse_model('Y = C + I + G + X - M')
    SYMBOLS_WITH_LAGS = fsic.parse_model('C = {alpha_1} * YD + {alpha_2} * H[-1]')
    SYMBOLS_WITH_LEADS = fsic.parse_model('A = {a0} + {a1} * A[1] + {a2} * A[2]')

    @staticmethod
    def build_model(symbols):
        return fsic.build_model(symbols)

    def setUp(self):
        self.SubmodelNoLags = self.build_model(self.SYMBOLS_NO_LAGS)
        self.SubmodelWithLags = self.build_model(self.SYMBOLS_WITH_LAGS)
        self.SubmodelWithLeads = self.build_model(self.SYMBOLS_WITH_LEADS)

    def test_init(self):
        linker = fsic.BaseLinker(
            {
                'A': self.SubmodelNoLags(range(1990, 2005 + 1)),
                'B': self.SubmodelNoLags(range(1990, 2005 + 1)),
                'C': self.SubmodelNoLags(range(1990, 2005 + 1)),
            }
        )

        self.assertEqual(linker.LAGS, 0)
        self.assertEqual(linker.LEADS, 0)

    def test_init_different_lags(self):
        linker = fsic.BaseLinker(
            {
                'A': self.SubmodelNoLags(range(1990, 2005 + 1)),
                'B': self.SubmodelWithLags(range(1990, 2005 + 1)),
            }
        )

        self.assertEqual(linker.LAGS, 1)
        self.assertEqual(linker.LEADS, 0)

    def test_init_different_leads(self):
        linker = fsic.BaseLinker(
            {
                'A': self.SubmodelNoLags(range(1990, 2005 + 1)),
                'B': self.SubmodelWithLeads(range(1990, 2005 + 1)),
            }
        )

        self.assertEqual(linker.LAGS, 0)
        self.assertEqual(linker.LEADS, 2)

    def test_init_mixed_lags_leads(self):
        linker = fsic.BaseLinker(
            {
                'A': self.SubmodelNoLags(range(1990, 2005 + 1)),
                'B': self.SubmodelWithLags(range(1990, 2005 + 1)),
                'C': self.SubmodelWithLeads(range(1990, 2005 + 1)),
            }
        )

        self.assertEqual(linker.LAGS, 1)
        self.assertEqual(linker.LEADS, 2)

    def test_init_different_spans_error(self):
        # Check for an error if the submodel spans differ
        with self.assertRaises(fsic.exceptions.InitialisationError):
            fsic.BaseLinker(
                {
                    # Start is 1990 for 'A'
                    'A': self.SubmodelNoLags(range(1990, 2005 + 1)),
                    'B': self.SubmodelNoLags(range(1991, 2005 + 1)),
                }
            )

    def test_init_no_submodels(self):
        # Check linker initialisation if no submodels passed

        # All of the below should initialise without error, but raise a
        # `SolutionError` if attempting to solve
        linker = fsic.BaseLinker()  # No `submodels` argument
        with self.assertRaises(
            SolutionError, msg='Object `span` is empty: No periods to solve'
        ):
            linker.solve()

        linker = fsic.BaseLinker(None)  # `submodels=None`
        with self.assertRaises(
            SolutionError, msg='Object `span` is empty: No periods to solve'
        ):
            linker.solve()

        linker = fsic.BaseLinker({})  # `submodels={}'
        with self.assertRaises(
            SolutionError, msg='Object `span` is empty: No periods to solve'
        ):
            linker.solve()


class TestLinkerSolve(unittest.TestCase):
    # Tolerance for absolute differences to be considered almost equal
    DELTA = 0.05

    SYMBOLS = fsic.parse_model(
        """
Y = C + I + G + X - M
M = {mu} * Y"""
    )

    class Linker(fsic.BaseLinker):
        def evaluate_t_before(self, t, *args, **kwargs):
            self.submodels['A'].X[t] = self.submodels['B'].M[t]
            self.submodels['B'].X[t] = self.submodels['A'].M[t]

    def setUp(self):
        self.Model = fsic.build_model(self.SYMBOLS)

    def test_solve(self):
        # Check that linker behaves and solves as expected with custom methods
        linker = self.Linker(
            {
                'A': self.Model(range(1990, 2005 + 1)),
                'B': self.Model(range(1990, 2005 + 1)),
            }
        )

        # All values should start at zero
        for k, submodel in linker.submodels.items():
            with self.subTest(submodel=k):
                self.assertTrue(np.allclose(submodel.values, 0.0))

        # Add in a propensity to import and government expenditure
        for k, submodel in linker.submodels.items():
            submodel.mu = 0.2
            submodel.G = 20

        linker.solve()

        # Check results
        self.assertTrue(np.all(linker.status == '.'))

        for k, submodel in linker.submodels.items():
            with self.subTest(submodel=k):
                self.assertTrue(np.all(submodel.status == '.'))
                self.assertTrue(np.all(submodel.iterations == linker.iterations))

                # fmt: off
                self.assertTrue(np.allclose(submodel['Y'], 20.0))
                self.assertTrue(np.allclose(submodel['C'],  0.0))
                self.assertTrue(np.allclose(submodel['I'],  0.0))
                self.assertTrue(np.allclose(submodel['G'], 20.0))
                self.assertTrue(np.allclose(submodel['X'],  4.0))
                self.assertTrue(np.allclose(submodel['M'],  4.0))
                # fmt: on

    def test_solve_min_iter(self):
        # Check that linker behaves and solves as expected with custom methods
        # and a minimum of number of iterations
        linker = self.Linker(
            {
                'A': self.Model(range(1990, 2005 + 1)),
                'B': self.Model(range(1990, 2005 + 1)),
            }
        )

        # All values should start at zero
        for k, submodel in linker.submodels.items():
            with self.subTest(submodel=k):
                self.assertTrue(np.allclose(submodel.values, 0.0))

        # Add in a propensity to import and government expenditure
        for k, submodel in linker.submodels.items():
            submodel.mu = 0.2
            submodel.G = 20

        linker.solve(min_iter=50)

        # Check results
        self.assertTrue(np.all(linker.status == '.'))
        self.assertTrue(np.all(linker.iterations == 50))

        for k, submodel in linker.submodels.items():
            with self.subTest(submodel=k):
                self.assertTrue(np.all(submodel.status == '.'))
                self.assertTrue(np.all(submodel.iterations == linker.iterations))

                # fmt: off
                self.assertTrue(np.allclose(submodel['Y'], 20.0))
                self.assertTrue(np.allclose(submodel['C'],  0.0))
                self.assertTrue(np.allclose(submodel['I'],  0.0))
                self.assertTrue(np.allclose(submodel['G'], 20.0))
                self.assertTrue(np.allclose(submodel['X'],  4.0))
                self.assertTrue(np.allclose(submodel['M'],  4.0))
                # fmt: on

    def test_solve_selective(self):
        # Check that the linker can solve selected submodels
        linker = self.Linker(
            {
                'A': self.Model(range(1990, 2005 + 1), mu=0.2, G=20),
                'B': self.Model(range(1990, 2005 + 1), mu=0.2, G=20),
            }
        )

        linker.solve(submodels=['A'])

        # Check results
        for k, submodel in linker.submodels.items():
            with self.subTest(submodel=k):
                # fmt: off
                self.assertTrue(np.allclose(submodel['C'],  0.0))
                self.assertTrue(np.allclose(submodel['I'],  0.0))
                self.assertTrue(np.allclose(submodel['G'], 20.0))
                # fmt: on

        self.assertTrue(np.allclose(linker.submodels['A'].Y, 16 + 2 / 3))
        self.assertTrue(np.allclose(linker.submodels['A'].X, 0))
        self.assertTrue(np.allclose(linker.submodels['A'].M, 3 + 1 / 3))

        self.assertTrue(np.allclose(linker.submodels['B'].Y, 0))
        self.assertTrue(np.allclose(linker.submodels['B'].X, 3 + 1 / 3))
        self.assertTrue(np.allclose(linker.submodels['B'].M, 0))

    def test_solve_selective_key_error(self):
        # Check for error with nonexistent specified submodel
        linker = self.Linker(
            {
                'A': self.Model(range(1990, 2005 + 1), mu=0.2, G=20),
                'B': self.Model(range(1990, 2005 + 1), mu=0.2, G=20),
            }
        )

        with self.assertRaises(KeyError):
            linker.solve(submodels=['C'])

    def test_solve_min_iter_error(self):
        linker = self.Linker(
            {
                'A': self.Model(range(1990, 2005 + 1), mu=0.2, G=20),
                'B': self.Model(range(1990, 2005 + 1), mu=0.2, G=20),
            }
        )

        # Check for error if `min_iter` > `max_iter`
        with self.assertRaises(ValueError):
            linker.solve(min_iter=10, max_iter=5)

    def test_solve_nonconvergence_error(self):
        linker = self.Linker(
            {
                'A': self.Model(range(1990, 2005 + 1), mu=0.2, G=20),
                'B': self.Model(range(1990, 2005 + 1), mu=0.2, G=20),
            }
        )

        # Too few iterations to solve: Should raise an error
        with self.assertRaises(fsic.exceptions.NonConvergenceError):
            linker.solve(max_iter=1)

    def test_solve_nonconvergence_continue(self):
        linker = self.Linker(
            {
                'A': self.Model(range(1990, 2005 + 1), mu=0.2, G=20),
                'B': self.Model(range(1990, 2005 + 1), mu=0.2, G=20),
            }
        )

        # Too few iterations to solve but `failures='ignore'` should let the
        # method complete without error
        linker.solve(max_iter=1, failures='ignore')


class TestLinkerCopy(unittest.TestCase):
    SYMBOLS = fsic.parse_model('Y = C + I + G + X - M')

    class Linker(fsic.BaseLinker):
        ENDOGENOUS = ['X', 'M', 'NX']
        NAMES = ENDOGENOUS
        CHECK = ENDOGENOUS

        def evaluate_t_after(self, t, *args, **kwargs):
            self._X[t] = sum(submodel.X[t] for submodel in self.submodels.values())
            self._M[t] = sum(submodel.M[t] for submodel in self.submodels.values())
            self._NX[t] = self._X[t] - self._M[t]

    def setUp(self):
        self.Model = fsic.build_model(self.SYMBOLS)

    def test_copy_method(self):
        linker = self.Linker(
            {
                'A': self.Model(range(1990, 2005 + 1)),
                'B': self.Model(range(1990, 2005 + 1)),
                'C': self.Model(range(1990, 2005 + 1)),
            }
        )

        duplicate_linker = linker.copy()
        duplicate_linker.values = np.full((3, 16), -1)

        for v in duplicate_linker.submodels.values():
            v.values = np.ones((6, 16))

        self.assertEqual(linker.values.shape, (3, 16))
        self.assertTrue(np.allclose(linker.values, 0))

        for k, v in linker.submodels.items():
            with self.subTest(submodel=k):
                self.assertEqual(v.values.shape, (6, 16))
                self.assertTrue(np.allclose(v.values, 0))

        self.assertEqual(duplicate_linker.values.shape, (3, 16))
        self.assertTrue(np.allclose(duplicate_linker.values, -1))

        for k, v in duplicate_linker.submodels.items():
            with self.subTest(submodel=k):
                self.assertEqual(v.values.shape, (6, 16))
                self.assertTrue(np.allclose(v.values, 1))

    def test_copy_dunder_method(self):
        linker = self.Linker(
            {
                'A': self.Model(range(1990, 2005 + 1)),
                'B': self.Model(range(1990, 2005 + 1)),
                'C': self.Model(range(1990, 2005 + 1)),
            }
        )

        duplicate_linker = copy.copy(linker)
        duplicate_linker.values = np.full((3, 16), -1)

        for v in duplicate_linker.submodels.values():
            v.values = np.ones((6, 16))

        self.assertEqual(linker.values.shape, (3, 16))
        self.assertTrue(np.allclose(linker.values, 0))

        for k, v in linker.submodels.items():
            with self.subTest(submodel=k):
                self.assertEqual(v.values.shape, (6, 16))
                self.assertTrue(np.allclose(v.values, 0))

        self.assertEqual(duplicate_linker.values.shape, (3, 16))
        self.assertTrue(np.allclose(duplicate_linker.values, -1))

        for k, v in duplicate_linker.submodels.items():
            with self.subTest(submodel=k):
                self.assertEqual(v.values.shape, (6, 16))
                self.assertTrue(np.allclose(v.values, 1))

    def test_deepcopy_dunder_method(self):
        linker = self.Linker(
            {
                'A': self.Model(range(1990, 2005 + 1)),
                'B': self.Model(range(1990, 2005 + 1)),
                'C': self.Model(range(1990, 2005 + 1)),
            }
        )

        duplicate_linker = copy.deepcopy(linker)
        duplicate_linker.values = np.full((3, 16), -1)

        for v in duplicate_linker.submodels.values():
            v.values = np.ones((6, 16))

        self.assertEqual(linker.values.shape, (3, 16))
        self.assertTrue(np.allclose(linker.values, 0))

        for k, v in linker.submodels.items():
            with self.subTest(submodel=k):
                self.assertEqual(v.values.shape, (6, 16))
                self.assertTrue(np.allclose(v.values, 0))

        self.assertEqual(duplicate_linker.values.shape, (3, 16))
        self.assertTrue(np.allclose(duplicate_linker.values, -1))

        for k, v in duplicate_linker.submodels.items():
            with self.subTest(submodel=k):
                self.assertEqual(v.values.shape, (6, 16))
                self.assertTrue(np.allclose(v.values, 1))


class TestLinkerMisc(unittest.TestCase):
    def test_size(self):
        # Check that `size` returns the total number of array elements across
        # the linker and its constituent models
        a = fsic.BaseModel(range(10))
        a.add_variable('A', 0)

        b = a.copy()
        b.add_variable('B', 1)

        c = b.copy()
        c.add_variable('C', 2)

        linker = fsic.BaseLinker({'A': a, 'B': b, 'C': c})
        for i, a in enumerate('XYZ'):
            linker.add_variable(a, i)

        self.assertEqual(linker.size, 90)

    def test_sizes(self):
        # Check that `sizes` returns, as a dictionary, the number of array
        # elements in the linker and each of its constituent models
        a = fsic.BaseModel(range(10))
        a.add_variable('A', 0)

        b = a.copy()
        b.add_variable('B', 1)

        c = b.copy()
        c.add_variable('C', 2)

        linker = fsic.BaseLinker({'A': a, 'B': b, 'C': c})
        for i, a in enumerate('XYZ'):
            linker.add_variable(a, i)

        self.assertEqual(
            linker.sizes,
            {
                '_': 30,
                'A': 10,
                'B': 20,
                'C': 30,
            },
        )

    def test_nbytes(self):
        # Check that `nbytes` returns the total bytes consumed by the array
        # elements
        a = fsic.BaseModel(range(10))
        a.add_variable('A', 0)

        b = a.copy()
        b.add_variable('B', 1)

        c = b.copy()
        c.add_variable('C', 2)

        linker = fsic.BaseLinker({'A': a, 'B': b, 'C': c})
        for i, a in enumerate('XYZ'):
            linker.add_variable(a, i)

        self.assertEqual(
            linker.nbytes,
            (90 * 8)         # Array elements
            + (10 * 8 * 4)   # Iterations
            + (10 * 4 * 4),  # Statuses
        )  # fmt: skip

    def test_strict(self):
        # Check that `strict=True` works for linkers as it does for base
        # `VectorContainer`s
        linker = fsic.BaseLinker({'A': fsic.BaseModel(range(10))})

        linker.add_variable('X', 0)
        self.assertEqual(linker.values.shape, (1, 10))

        linker.Y = 'This should work fine'
        self.assertEqual(linker.Y, 'This should work fine')

        linker.Y = 'This should still work fine'
        self.assertEqual(linker.Y, 'This should still work fine')

        linker.strict = True
        with self.assertRaises(AttributeError):
            linker.Z = "This shouldn't work"

        self.assertEqual(linker.values.shape, (1, 10))


@unittest.skipIf(not pandas_installed, 'Requires `pandas`')
class TestPandasIndexing(unittest.TestCase):
    SCRIPT = """
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
"""
    SYMBOLS = fsic.parse_model(SCRIPT)

    def setUp(self):
        self.Model = fsic.build_model(self.SYMBOLS)

    def test_multiindex_indexing(self):
        # Indexing tests to check that `BaseModel` can use, as a `span`
        # attribute value, `pandas` `MultiIndex` objects
        model = self.Model(
            pd.MultiIndex.from_product(
                [range(2000, 2005 + 1), range(1, 3 + 1)], names=['year', 'term']
            ),
            alpha_1=0.6,
            alpha_2=0.4,
            G=20,
            theta=0.2,
        )

        expected = np.array([20.0] * 18)
        self.assertTrue(np.allclose(model.G, expected))

        model['G', : (2002, 3)] = 10
        expected[:9] = 10
        self.assertTrue(np.allclose(model.G, expected))

        model['G', :2001] = 15
        expected[:6] = 15
        self.assertTrue(np.allclose(model.G, expected))

        model['G', (2004, 3) :] = 25
        expected[-4:] = 25
        self.assertTrue(np.allclose(model.G, expected))

        model['G', 2003:] = 30
        expected[9:] = 30
        self.assertTrue(np.allclose(model.G, expected))

        model['G', (2000, 3)] = 25
        expected[2] = 25
        self.assertTrue(np.allclose(model.G, expected))

        model['G', 2001] = 0
        expected[3:6] = 0
        self.assertTrue(np.allclose(model.G, expected))

        model['G', (2003, 2) : (2005, 1)] = 5
        expected[10:-2] = 5
        self.assertTrue(np.allclose(model.G, expected))

        model['G', 2001:2003] = 20
        expected[3:12] = 20
        self.assertTrue(np.allclose(model.G, expected))

        self.assertTrue(np.allclose(
            model.G,
            np.array([15.0, 15.0, 25.0,
                      20.0, 20.0, 20.0,
                      20.0, 20.0, 20.0,
                      20.0, 20.0, 20.0,
                       5.0,  5.0,  5.0,
                       5.0, 30.0, 30.0, ])))  # fmt: skip

        self.assertTrue(np.allclose(model['G', (2000, 3)], 25.0))
        self.assertTrue(np.allclose(model['G', 2001], 20.0))

        self.assertTrue(np.allclose(model['G', : (2000, 2)], 15.0))
        self.assertTrue(
            np.allclose(model['G', :2001], [15.0, 15.0, 25.0, 20.0, 20.0, 20.0])
        )

        self.assertTrue(np.allclose(model['G', (2005, 2) :], 30.0))
        self.assertTrue(
            np.allclose(model['G', 2004:], [5.0, 5.0, 5.0, 5.0, 30.0, 30.0])
        )

        self.assertTrue(np.allclose(model['G', 2001:2003], 20.0))
        self.assertTrue(
            np.allclose(model['G', 2003:2004], [20.0, 20.0, 20.0, 5.0, 5.0, 5.0])
        )

    def test_periodindex_indexing(self):
        # Indexing tests to check that `BaseModel` can use, as a `span`
        # attribute value, `pandas` `PeriodIndex` objects
        model = self.Model(
            pd.period_range(start='1990-01-01', end='1995-12-31', freq='Q'),
            alpha_1=0.6,
            alpha_2=0.4,
            G=20,
            theta=0.2,
        )

        expected = np.array([20.0] * 24)
        self.assertTrue(np.allclose(model.G, expected))

        model['G', :'1990Q4'] = 10
        expected[:4] = 10
        self.assertTrue(np.allclose(model.G, expected))

        model['G', :'1992'] = 15
        expected[:12] = 15
        self.assertTrue(np.allclose(model.G, expected))

        model['G', '1993Q1':] = 25
        expected[12:] = 25
        self.assertTrue(np.allclose(model.G, expected))

        model['G', '1994':] = 30
        expected[16:] = 30
        self.assertTrue(np.allclose(model.G, expected))

        model['G', '1991Q1'] = 25
        expected[4] = 25
        self.assertTrue(np.allclose(model.G, expected))

        model['G', '1992'] = 0
        expected[8:12] = 0
        self.assertTrue(np.allclose(model.G, expected))

        model['G', '1992Q1':'1993Q4'] = 5
        expected[8:16] = 5
        self.assertTrue(np.allclose(model.G, expected))

        model['G', '1993':'1994'] = 20
        expected[12:20] = 20
        self.assertTrue(np.allclose(model.G, expected))

        self.assertTrue(np.allclose(
            model.G,
            np.array([15.0] * 4 +           # 1990
                     [25.0] + [15.0] * 3 +  # 1991
                     [ 5.0] * 4 +           # 1992
                     [20.0] * 4 +           # 1993
                     [20.0] * 4 +           # 1994
                     [30.0] * 4             # 1995
                     )))  # fmt: skip

        self.assertTrue(np.allclose(model['G', '1991Q1'], 25))
        self.assertTrue(np.allclose(model['G', '1990'], 15))

        self.assertTrue(np.allclose(model['G', :'1990Q4'], 15))
        self.assertTrue(np.allclose(model['G', :'1990'], 15))

        self.assertTrue(np.allclose(model['G', '1995Q1':], 30))
        self.assertTrue(np.allclose(model['G', '1995':], 30))

        self.assertTrue(np.allclose(model['G', '1993':'1994'], 20))
        self.assertTrue(np.allclose(model['G', '1993Q1':'1994Q4'], 20))

    def test_periodindex_solution(self):
        # Solution tests to check that `BaseModel` can use, as a `span`
        # attribute value, `pandas` `PeriodIndex` objects
        model = self.Model(
            pd.period_range(start='1990-01', end='2010-12', freq='Q'),
            alpha_1=0.6,
            alpha_2=0.4,
            G=20,
            theta=0.2,
        )

        model.solve(end='2010Q2')
        model.solve_t(-2)
        model.solve_period('2010Q4')

        self.assertTrue(np.isclose(model.C[-1], 80))
        self.assertTrue(np.isclose(model.YD[-1], 80))
        self.assertTrue(np.isclose(model.Y[-1], 100))
        self.assertTrue(np.isclose(model.T[-1], 20))
        self.assertTrue(np.isclose(model.H[-1], 80))

    def test_periodindex_solve_error(self):
        # Check that `solve()` raises an error if the arguments imply multiple
        # periods
        model = self.Model(pd.period_range(start='1990-01', end='2010-12', freq='Q'))

        with self.assertRaises(KeyError):
            model.solve(start='2000')

        with self.assertRaises(KeyError):
            model.solve(end='2000')

        with self.assertRaises(KeyError):
            model.solve(start='1995', end='2000')

    def test_periodindex_solve_period_error(self):
        # Check that `solve_period()` raises an error if the argument implies
        # multiple periods
        model = self.Model(pd.period_range(start='1990-01', end='2010-12', freq='Q'))

        with self.assertRaises(KeyError):
            model.solve_period('2000')


if __name__ == '__main__':
    unittest.main()
