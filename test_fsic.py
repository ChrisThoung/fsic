# -*- coding: utf-8 -*-
"""
test_fsic
=========
Test suite for FSIC.

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

import functools
import keyword
import unittest
import sys

import numpy as np

import fsic


class TestRegexes(unittest.TestCase):

    def test_equation_re(self):
        # Test that the equation regex correctly identifies individual
        # equations in a system
        script = '''
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
'''
        expected = [
            '(C =\n     {alpha_1} * YD +\n     {alpha_2} * H[-1])',
            'YD = (Y -\n      T)',
            'Y = C + G',
            'T = {theta} * Y',
            'H = (\n     H[-1] + YD - C\n)',
        ]

        self.assertEqual(fsic.equation_re.findall(script), expected)


class TestTerm(unittest.TestCase):

    def test_str(self):
        # Check that `str` representations are as expected
        self.assertEqual(str(fsic.Term('C', fsic.Type.VARIABLE, -1)), 'C[t-1]')
        self.assertEqual(str(fsic.Term('C', fsic.Type.VARIABLE,  0)), 'C[t]')
        self.assertEqual(str(fsic.Term('C', fsic.Type.VARIABLE,  1)), 'C[t+1]')


class TestParsers(unittest.TestCase):

    def test_split_equations(self):
        # Test that `split_equations()` correctly identifies individual
        # equations in a system
        script = '''
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
'''
        expected = [
            '(C =\n     {alpha_1} * YD +\n     {alpha_2} * H[-1])',
            'YD = (Y -\n      T)',
            'Y = C + G',
            'T = {theta} * Y',
            'H = (\n     H[-1] + YD - C\n)',
            'H = H[-1] + (\n    YD - C)',
            'H = (H[-1] +\n     YD) - C',
        ]

        self.assertEqual(fsic.split_equations(script), expected)

    def test_parse_terms(self):
        # Test that `parse_terms()` correctly identifies individual terms in an
        # expression
        expression = 'C = exp({alpha_1} * log(YD) + {alpha_2} * log(H[-1]) + <epsilon>)'
        expected = [
            fsic.Term(name='C', type=fsic.Type.VARIABLE, index_=0),
            fsic.Term(name='exp', type=fsic.Type.FUNCTION, index_=None),
            fsic.Term(name='alpha_1', type=fsic.Type.PARAMETER, index_=0),
            fsic.Term(name='log', type=fsic.Type.FUNCTION, index_=None),
            fsic.Term(name='YD', type=fsic.Type.VARIABLE, index_=0),
            fsic.Term(name='alpha_2', type=fsic.Type.PARAMETER, index_=0),
            fsic.Term(name='log', type=fsic.Type.FUNCTION, index_=None),
            fsic.Term(name='H', type=fsic.Type.VARIABLE, index_=-1),
            fsic.Term(name='epsilon', type=fsic.Type.ERROR, index_=0),
        ]

        self.assertEqual(fsic.parse_terms(expression), expected)

    def test_parse_equation_terms(self):
        # Test that `parse_equation_terms()` correctly identifies individual
        # terms in an equation
        equation = 'C = exp({alpha_1} * log(YD) + {alpha_2} * log(H[-1]) + <epsilon>)'
        expected = [
            fsic.Term(name='C', type=fsic.Type.ENDOGENOUS, index_=0),
            fsic.Term(name='exp', type=fsic.Type.FUNCTION, index_=None),
            fsic.Term(name='alpha_1', type=fsic.Type.PARAMETER, index_=0),
            fsic.Term(name='log', type=fsic.Type.FUNCTION, index_=None),
            fsic.Term(name='YD', type=fsic.Type.EXOGENOUS, index_=0),
            fsic.Term(name='alpha_2', type=fsic.Type.PARAMETER, index_=0),
            fsic.Term(name='log', type=fsic.Type.FUNCTION, index_=None),
            fsic.Term(name='H', type=fsic.Type.EXOGENOUS, index_=-1),
            fsic.Term(name='epsilon', type=fsic.Type.ERROR, index_=0),
        ]

        self.assertEqual(fsic.parse_equation_terms(equation), expected)

    def test_parse_equation(self):
        # Test that `parse_equation()` correctly identifies the symbols that
        # make up an equation
        equation = 'C = exp({alpha_1} * log(YD) + {alpha_2} * log(H[-1]) + <epsilon>)'
        expected = [
            fsic.Symbol(name='C', type=fsic.Type.ENDOGENOUS, lags=0, leads=0,
                        equation='C[t] = exp(alpha_1[t] * log(YD[t]) + '
                                            'alpha_2[t] * log(H[t-1]) + '
                                            'epsilon[t])',
                        code='self._C[t] = np.exp(self._alpha_1[t] * np.log(self._YD[t]) + '
                                                 'self._alpha_2[t] * np.log(self._H[t-1]) + '
                                                 'self._epsilon[t])'),
            fsic.Symbol(name='exp', type=fsic.Type.FUNCTION, lags=None, leads=None, equation=None, code=None),
            fsic.Symbol(name='alpha_1', type=fsic.Type.PARAMETER, lags=0, leads=0, equation=None, code=None),
            fsic.Symbol(name='log', type=fsic.Type.FUNCTION, lags=None, leads=None, equation=None, code=None),
            fsic.Symbol(name='YD', type=fsic.Type.EXOGENOUS, lags=0, leads=0, equation=None, code=None),
            fsic.Symbol(name='alpha_2', type=fsic.Type.PARAMETER, lags=0, leads=0, equation=None, code=None),
            fsic.Symbol(name='H', type=fsic.Type.EXOGENOUS, lags=-1, leads=0, equation=None, code=None),
            fsic.Symbol(name='epsilon', type=fsic.Type.ERROR, lags=0, leads=0, equation=None, code=None),
        ]

        self.assertEqual(fsic.parse_equation(equation), expected)

    def test_parse_equation_function_replacement(self):
        # Check that function replacement only applies to functions explicitly
        # defined in `fsic.replacement_function_names`. For example:
        #  - log -> np.log     (replaced)
        #  - np.log -> np.log  (unchanged)
        equation = 'C = log(A) + np.log(B)'
        expected = [
            fsic.Symbol(name='C', type=fsic.Type.ENDOGENOUS, lags=0, leads=0,
                        equation='C[t] = log(A[t]) + np.log(B[t])',
                        code='self._C[t] = np.log(self._A[t]) + np.log(self._B[t])'),
            fsic.Symbol(name='log', type=fsic.Type.FUNCTION, lags=None, leads=None, equation=None, code=None),
            fsic.Symbol(name='A', type=fsic.Type.EXOGENOUS, lags=0, leads=0, equation=None, code=None),
            fsic.Symbol(name='np.log', type=fsic.Type.FUNCTION, lags=None, leads=None, equation=None, code=None),
            fsic.Symbol(name='B', type=fsic.Type.EXOGENOUS, lags=0, leads=0, equation=None, code=None),
        ]

        self.assertEqual(fsic.parse_equation(equation), expected)

    def test_parse_equation_namespace_functions(self):
        # Check that functions in namespaces (below, `mean` in `np` [NumPy])
        # are left unchanged
        equation = 'Y = np.mean(X)'
        expected = [
            fsic.Symbol(name='Y', type=fsic.Type.ENDOGENOUS, lags=0, leads=0,
                        equation='Y[t] = np.mean(X[t])',
                        code='self._Y[t] = np.mean(self._X[t])'),
            fsic.Symbol(name='np.mean', type=fsic.Type.FUNCTION, lags=None, leads=None, equation=None, code=None),
            fsic.Symbol(name='X', type=fsic.Type.EXOGENOUS, lags=0, leads=0, equation=None, code=None),
        ]

        self.assertEqual(fsic.parse_equation(equation), expected)

    def test_parse_model(self):
        # Test that `parse_model()` correctly identifies the symbols that make
        # up a system of equations
        model = '''
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
'''
        expected = [
            fsic.Symbol(name='C', type=fsic.Type.ENDOGENOUS, lags=0, leads=0,
                        equation='C[t] = alpha_1[t] * YD[t] + alpha_2[t] * H[t-1]',
                        code='self._C[t] = self._alpha_1[t] * self._YD[t] + self._alpha_2[t] * self._H[t-1]'),
            fsic.Symbol(name='alpha_1', type=fsic.Type.PARAMETER, lags=0, leads=0, equation=None, code=None),
            fsic.Symbol(name='YD', type=fsic.Type.ENDOGENOUS, lags=0, leads=0,
                        equation='YD[t] = Y[t] - T[t]',
                        code='self._YD[t] = self._Y[t] - self._T[t]'),
            fsic.Symbol(name='alpha_2', type=fsic.Type.PARAMETER, lags=0, leads=0, equation=None, code=None),
            fsic.Symbol(name='H', type=fsic.Type.ENDOGENOUS, lags=-1, leads=0,
                        equation='H[t] = H[t-1] + YD[t] - C[t]',
                        code='self._H[t] = self._H[t-1] + self._YD[t] - self._C[t]'),
            fsic.Symbol(name='Y', type=fsic.Type.ENDOGENOUS, lags=0, leads=0,
                        equation='Y[t] = C[t] + G[t]',
                        code='self._Y[t] = self._C[t] + self._G[t]'),
            fsic.Symbol(name='T', type=fsic.Type.ENDOGENOUS, lags=0, leads=0,
                        equation='T[t] = theta[t] * Y[t]',
                        code='self._T[t] = self._theta[t] * self._Y[t]'),
            fsic.Symbol(name='G', type=fsic.Type.EXOGENOUS, lags=0, leads=0, equation=None, code=None),
            fsic.Symbol(name='theta', type=fsic.Type.PARAMETER, lags=0, leads=0, equation=None, code=None),
        ]

        self.assertEqual(fsic.parse_model(model), expected)

    def test_parser_no_lhs(self):
        # Check that invalid equations (missing a left-hand side expression)
        # lead to a `ParserError`
        with self.assertRaises(fsic.ParserError):
            fsic.parse_model('A + B + C')

    def test_accidental_int_call(self):
        # Check that a missing operator between an integer and a variable is
        # caught properly
        with self.assertRaises(fsic.ParserError):
            fsic.parse_model('3(A)')

    def test_accidental_float_call(self):
        # Check that a missing operator between a float and a variable is
        # caught properly
        with self.assertRaises(fsic.ParserError):
            fsic.parse_model('3.0(A)')

    def test_missing_operator_int(self):
        # Check that a missing operator between an integer and a variable is
        # caught properly
        with self.assertRaises(fsic.ParserError):
            fsic.parse_model('3A')

    def test_missing_operator_float(self):
        # Check that a missing operator between a float and a variable is
        # caught properly
        with self.assertRaises(fsic.ParserError):
            fsic.parse_model('3.0A')


class TestInit(unittest.TestCase):

    SCRIPT = '''
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
'''
    SYMBOLS = fsic.parse_model(SCRIPT)

    def setUp(self):
        self.Model = fsic.build_model(self.SYMBOLS)

    def test_init_with_arrays(self):
        model = self.Model(range(5), G=np.arange(0, 10, 2), alpha_1=[0.6] * 5)
        self.assertEqual(model.values.shape, (9, 5))
        self.assertTrue(np.allclose(
            model.values,
            np.array([
                [0.0] * 5,
                [0.0] * 5,
                [0.0] * 5,
                [0.0] * 5,
                [0.0] * 5,
                [0.0, 2.0, 4.0, 6.0, 8.0],
                [0.6] * 5,
                [0.0] * 5,
                [0.0] * 5,
            ])))

    def test_init_dimension_error(self):
        with self.assertRaises(fsic.DimensionError):
            # C is invalid because it has the wrong shape
            self.Model(range(10), C=[0, 0])


class TestInterface(unittest.TestCase):

    SCRIPT = '''
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
'''
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

        with self.assertRaises(fsic.DimensionError):
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

        with self.assertRaises(fsic.DimensionError):
            model.status = ['-', '.']  # Incompatible dimensions

    def test_iter_periods(self):
        # Check properties of the iterator returned by `iter_periods()`
        model = self.Model(range(2000, 2009 + 1))

        self.assertEqual(len(model.iter_periods()), 9)
        self.assertEqual(len(model.iter_periods(start=2005)), 5)

        self.assertEqual(list(model.iter_periods()),
                         [(i, 2000 + i) for i in range(1, 9 + 1)])


class TestModelContainerMethods(unittest.TestCase):

    SCRIPT = 'C = {alpha_1} * YD + {alpha_2} * H[-1]'
    SYMBOLS = fsic.parse_model(SCRIPT)

    def setUp(self):
        self.Model = fsic.build_model(self.SYMBOLS)

    def test_getitem_by_name(self):
        # Test variable access by name
        model = self.Model(['{}Q{}'.format(y, q)
                                 for y in range(1990, 1995 + 1)
                                 for q in range(1, 4 + 1)])
        model.YD = np.arange(len(model.span))

        self.assertTrue(np.allclose(model['YD'],
                                    np.arange(len(model.span))))

    def test_getitem_by_name_error(self):
        # Test that variable access raises a KeyError if the name isn't a model
        # variable
        model = self.Model(['{}Q{}'.format(y, q)
                                 for y in range(1990, 1995 + 1)
                                 for q in range(1, 4 + 1)])
        model.YD = np.arange(len(model.span))

        with self.assertRaises(KeyError):
            model['ABC']

    def test_getitem_by_name_and_index(self):
        # Test simultaneous variable and single period access
        model = self.Model(['{}Q{}'.format(y, q)
                                 for y in range(1990, 1995 + 1)
                                 for q in range(1, 4 + 1)])
        model.YD = np.arange(len(model.span))

        self.assertEqual(model['YD', '1991Q1'], 4.0)

    def test_getitem_by_name_and_slice_to(self):
        # Test simultaneous variable and slice access
        model = self.Model(['{}Q{}'.format(y, q)
                                 for y in range(1990, 1995 + 1)
                                 for q in range(1, 4 + 1)])
        model.YD = np.arange(len(model.span))

        self.assertTrue(np.allclose(model['YD', :'1990Q4'],
                                    np.arange(3 + 1)))

    def test_getitem_by_name_and_slice_from(self):
        # Test simultaneous variable and slice access
        model = self.Model(['{}Q{}'.format(y, q)
                                 for y in range(1990, 1995 + 1)
                                 for q in range(1, 4 + 1)])
        model.YD = np.arange(len(model.span))

        self.assertTrue(np.allclose(model['YD', '1995Q1':],
                                    np.arange(20, 23 + 1)))

    def test_getitem_by_name_and_slice_step(self):
        # Test simultaneous variable and slice access
        model = self.Model(['{}Q{}'.format(y, q)
                                 for y in range(1990, 1995 + 1)
                                 for q in range(1, 4 + 1)])
        model.YD = np.arange(len(model.span))

        self.assertTrue(np.allclose(model['YD', ::4],
                                    np.arange(0, 23 + 1, 4)))

    def test_setitem_by_name_with_number(self):
        # Test variable assignment by name, setting all elements to a single
        # value
        model = self.Model(['{}Q{}'.format(y, q)
                                 for y in range(1990, 1995 + 1)
                                 for q in range(1, 4 + 1)])
        model.YD = np.arange(len(model.span))

        model['YD'] = 10
        self.assertTrue(np.allclose(model['YD'],
                                    np.full(len(model.span), 10, dtype=float)))

    def test_setitem_by_name_with_array(self):
        # Test variable assignment by name, replacing with a new array
        model = self.Model(['{}Q{}'.format(y, q)
                                 for y in range(1990, 1995 + 1)
                                 for q in range(1, 4 + 1)])
        model.YD = np.arange(len(model.span))

        model['YD'] = np.arange(0, len(model.span) * 2, 2)
        self.assertTrue(np.allclose(model['YD'],
                                    np.arange(0, len(model.span) * 2, 2)))

    def test_setitem_by_name_and_index(self):
        # Test variable assignment by name and single period
        model = self.Model(['{}Q{}'.format(y, q)
                                 for y in range(1990, 1995 + 1)
                                 for q in range(1, 4 + 1)])
        model.YD = np.arange(len(model.span))

        model['YD', '1990Q1'] = 100

        expected = np.arange(len(model.span))
        expected[0] = 100

        self.assertTrue(np.allclose(model['YD'],
                                    expected))

    def test_setitem_by_name_and_slice_to(self):
        # Test variable assignment by name and slice
        model = self.Model(['{}Q{}'.format(y, q)
                                 for y in range(1990, 1995 + 1)
                                 for q in range(1, 4 + 1)])
        model.YD = np.arange(len(model.span))

        model['YD', :'1990Q4'] = 100

        expected = np.arange(len(model.span))
        expected[:4] = 100

        self.assertTrue(np.allclose(model['YD'],
                                    expected))

    def test_setitem_by_name_and_slice_from(self):
        # Test variable assignment by name and slice
        model = self.Model(['{}Q{}'.format(y, q)
                                 for y in range(1990, 1995 + 1)
                                 for q in range(1, 4 + 1)])
        model.YD = np.arange(len(model.span))

        model['YD', '1995Q1':] = 100

        expected = np.arange(len(model.span))
        expected[-4:] = 100

        self.assertTrue(np.allclose(model['YD'],
                                    expected))

    def test_setitem_dimension_error(self):
        # Test check for misaligned dimensions at assignment
        model = self.Model(['{}Q{}'.format(y, q)
                                 for y in range(1990, 1995 + 1)
                                 for q in range(1, 4 + 1)])
        model.YD = np.arange(len(model.span))

        with self.assertRaises(fsic.DimensionError):
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

    def _evaluate(self, t: int, **kwargs: Dict[str, Any]) -> None:
        pass'''

        symbols = fsic.parse_model('Y = C + G')
        # Delete the symbol for the endogenous variable
        del symbols[0]

        code = fsic.build_model_definition(symbols)
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

    def _evaluate(self, t: int, **kwargs: Dict[str, Any]) -> None:
        pass'''

        symbols = fsic.parse_model('')
        code = fsic.build_model_definition(symbols)
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

    def _evaluate(self, t: int, **kwargs: Dict[str, Any]) -> None:
        # Y[t] = X[t] if X[t] > Z[t] else Z[t]
        self._Y[t] = self._X[t] if self._X[t] > self._Z[t] else self._Z[t]'''

        symbols = fsic.parse_model('Y = X if X > Z else Z')
        code = fsic.build_model(symbols).CODE
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

    def _evaluate(self, t: int, **kwargs: Dict[str, Any]) -> None:
        # Y[t] = X[t]
        _ = self._X[t]
        if np.isfinite(_):
            self._Y[t] = _'''

        symbols = fsic.parse_model('Y = X')

        def converter(symbol):
            lhs, rhs = map(str.strip, symbol.code.split('=', maxsplit=1))
            return '''\
# {}
_ = {}
if np.isfinite(_):
    {} = _'''.format(symbol.equation, rhs, lhs)

        code = fsic.build_model(symbols, converter=converter).CODE
        self.assertEqual(code, expected)


class TestBuildAndSolve(unittest.TestCase):

    # Tolerance for absolute differences to be considered almost equal
    DELTA = 0.05

    def test_gl2007_sim(self):
        # Test that `build_model()` correctly constructs a working model
        # definition from a simplified version of Godley and Lavoie's (2007)
        # Model SIM, and that the model solves as expected
        model = '''
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
'''
        symbols = fsic.parse_model(model)
        SIM = fsic.build_model(symbols)

        # Initialise a new model instance, set values and solve
        sim = SIM(range(1945, 2010 + 1),
                  alpha_1=0.6, alpha_2=0.4)

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
        model = '''
C = 0.60 * Y[-1] + 0.35 * Y[-2]
I = (R +
     1.0 * (PQ[-1] - PQ[-2]) +
     1.0 * (PQ[-2] - PQ[-3]) +
     0.1 * Q[-1])
PQ = max(Q, PQ[-1])
M = -380 + 0.2 * (C + I + X)
Q = C + I + G + X - M
Y = 0.72 * Q
'''
        symbols = fsic.parse_model(model)
        AMI = fsic.build_model(symbols)

        # Initialise a new model instance, set values and solve
        ami = AMI(range(1, 24 + 1))

        ami.G = [714.9, 718.5, 722.2, 725.9, 729.5,
                 733.2, 736.9, 740.5, 744.2, 747.8,
                 751.5, 755.2, 758.8, 762.5, 766.1,
                 769.8, 773.5, 777.1, 780.8, 784.5,
                 788.1, 791.8, 795.4, 799.1]
        ami.R = [518.7, 522.5, 526.2, 530.0, 533.7,
                 537.5, 541.2, 545.0, 548.7, 552.5,
                 556.2, 560.0, 563.7, 567.5, 571.2,
                 575.0, 578.7, 582.5, 586.2, 590.0,
                 593.7, 597.5, 601.2, 605.0]
        ami.X = [303.9, 308.8, 313.8, 318.7, 323.6,
                 328.5, 333.5, 338.4, 343.3, 348.3,
                 353.2, 358.1, 363.0, 368.0, 372.9,
                 377.8, 382.8, 387.7, 392.6, 397.6,
                 402.5, 407.4, 412.3, 417.3]

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

    SCRIPT = '''
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
'''
    SYMBOLS = fsic.parse_model(SCRIPT)

    def setUp(self):
        self.Model = fsic.build_model(self.SYMBOLS)

    def test_copy(self):
        model = self.Model(range(1945, 2010 + 1),
                           alpha_1=0.6, alpha_2=0.4,
                           theta=0.2)
        model.G = 20

        duplicate_model = model.copy()

        # Values should be identical at this point
        self.assertTrue(np.allclose(model.values,
                                    duplicate_model.values))

        # The solved model should have different values to the duplicate
        model.solve()
        self.assertFalse(np.allclose(model.values,
                                     duplicate_model.values))

        # The solved duplicate should match the original again
        duplicate_model.solve()
        self.assertTrue(np.allclose(model.values,
                                    duplicate_model.values))


class TestSolve(unittest.TestCase):

    SCRIPT = '''
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
'''
    SYMBOLS = fsic.parse_model(SCRIPT)

    def setUp(self):
        self.Model = fsic.build_model(self.SYMBOLS)

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

    def test_offset_error_lag(self):
        # Check for an `IndexError` if `offset` points prior to the span of the
        # current model instance
        model = self.Model(range(1945, 2010 + 1))

        with self.assertRaises(IndexError):
            # With Model *SIM*, trying to solve the second period (remember
            # there's a lag in the model) with an offset of -2 should fail
            model.solve(offset=-2)

    def test_offset_error_lead(self):
        # Check for an `IndexError` if `offset` points beyond the span of the
        # current model instance
        model = self.Model(range(1945, 2010 + 1))

        with self.assertRaises(IndexError):
            # With Model *SIM*, trying to solve the final period with an offset
            # of +1 should fail
            model.solve(offset=1)

    def test_solve_return_values(self):
        # Check that the return values from `solve()` are as expected
        model = self.Model(['A', 'B', 'C'])
        labels, indexes, solved = model.solve()

        self.assertEqual(labels, ['B', 'C'])
        self.assertEqual(indexes, [1, 2])
        self.assertEqual(solved, [True, True])


class TestParserErrors(unittest.TestCase):

    def test_extra_equals_single_equation(self):
        with self.assertRaises(fsic.ParserError):
            fsic.parse_model('Y = C + I + G = X - M')

    def test_extra_equals_multiple_equations(self):
        with self.assertRaises(fsic.ParserError):
            fsic.parse_model('''
Y = C + I + G = X - M
Z = C + I + G = X - M
''')

    def test_double_definition(self):
        # Check test for endogenous variables that are set twice
        with self.assertRaises(fsic.ParserError):
            fsic.parse_model('''
Y = C + I + G + X - M
Y = GVA + TSP
''')

    @unittest.skipIf(sys.version_info[:2] < (3, 8),
                     'Parser test for accidental treatment of floats as callable not supported prior to Python version 3.8')
    def test_accidental_float_call(self):
        # Check that something like 'A = 0.5(B)' (missing * operator) raises a
        # `ParserError`
        with self.assertRaises(fsic.ParserError):
            fsic.parse_model('A = 0.5(B)')

    def test_keywords_as_variables_lhs(self):
        # Check that the parser catches attempts to use reserved Python
        # keywords as endogenous variable names in a model
        for name in keyword.kwlist:
            with self.subTest(name=name):
                with self.assertRaises(fsic.ParserError):
                    fsic.parse_model('{} = 0'.format(name))

    def test_keywords_as_variables_rhs(self):
        # Check that the parser catches attempts to use reserved Python
        # keywords as exogenous variable names in a model
        for name in keyword.kwlist:
            with self.subTest(name=name):
                with self.assertRaises(fsic.ParserError):
                    fsic.parse_model('X = {}[-1]'.format(name))


class TestBuildErrors(unittest.TestCase):

    def test_extra_equals(self):
        symbols = fsic.parse_model('Y = C + I + G = X - M', check_syntax=False)
        with self.assertRaises(fsic.BuildError):
            Model = fsic.build_model(symbols)


class TestSolutionErrorHandling(unittest.TestCase):

    SCRIPT = '''
s = 1 - (C / Y)  # Divide-by-zero equation (generating a NaN) appears first
Y = C + G
C = {c0} + {c1} * Y
g = log(G)  # Use to test for infinity with log(0)
'''
    SYMBOLS = fsic.parse_model(SCRIPT)

    def setUp(self):
        self.Model = fsic.build_model(self.SYMBOLS)

    def test_raise_nans(self):
        # Model should halt on first period
        model = self.Model(range(10), G=20)

        with self.assertRaises(fsic.SolutionError):
            model.solve()

        self.assertTrue(np.isnan(model.s[0]))
        self.assertTrue(np.allclose(model.s[1:], 0))

        self.assertTrue(np.all(model.status == np.array(['E'] + ['-'] * 9)))
        self.assertTrue(np.all(model.iterations == np.array([1] + [-1] * 9)))

    def test_raise_prior_nans(self):
        # Model should halt if pre-existing NaNs detected
        model = self.Model(range(10), s=np.nan)

        with self.assertRaises(fsic.SolutionError):
            model.solve()

    def test_skip(self):
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

    def test_ignore_successfully(self):
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

    def test_ignore_unsuccessfully(self):
        # Model should keep solving (unsuccessfully in this case)
        model = self.Model(range(10))
        model.solve(failures='ignore', errors='ignore')

        self.assertTrue(np.all(np.isnan(model.s)))
        self.assertTrue(np.all(model.status == 'F'))
        self.assertTrue(np.all(model.iterations == 100))

    def test_ignore_prior_nans(self):
        model = self.Model(range(10), s=np.nan, G=20)
        model.solve(errors='ignore')

        self.assertTrue(np.allclose(model.s, 1))
        self.assertTrue(np.all(model.status == '.'))
        self.assertTrue(np.all(model.iterations == 3))

    def test_replace_successfully(self):
        # Model should replace NaNs and keep solving (successfully in this case)
        model = self.Model(range(10), G=20)
        model.solve(errors='replace')

        self.assertTrue(np.allclose(model.s, 1))
        self.assertTrue(np.all(model.status == '.'))
        self.assertTrue(np.all(model.iterations == 3))

    def test_replace_unsuccessfully(self):
        # Model should replace NaNs and keep solving (unsuccessfully in this case)
        model = self.Model(range(10))
        model.solve(failures='ignore', errors='replace')

        self.assertTrue(np.all(np.isnan(model.s)))
        self.assertTrue(np.all(model.status == 'F'))
        self.assertTrue(np.all(model.iterations == 100))

    def test_replace_prior_nans(self):
        model = self.Model(range(10), s=np.nan, G=20)
        model.solve(errors='replace')

        self.assertTrue(np.allclose(model.s, 1))
        self.assertTrue(np.all(model.status == '.'))
        self.assertTrue(np.all(model.iterations == 3))

    def test_raise_infinities(self):
        # Model should halt on first period because of log(0)
        model = self.Model(range(10), c0=10, C=10, Y=10)

        with self.assertRaises(fsic.SolutionError):
            model.solve()

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

    def test_replace_infinities(self):
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


class TestNonConvergenceError(unittest.TestCase):

    MODEL = '''
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
'''
    SYMBOLS = fsic.parse_model(MODEL)

    def setUp(self):
        self.Model = fsic.build_model(self.SYMBOLS)

    def test_nonconvergence_raise(self):
        model = self.Model(range(5),
                           alpha_1=0.6, alpha_2=0.4,
                           theta=0.2, G=20)

        # First period (after initial lag) should solve
        model.solve_t(1)

        # Second period (with limited number of iterations) should fail to
        # solve
        with self.assertRaises(fsic.NonConvergenceError):
            model.solve_t(2, max_iter=5)

        self.assertTrue(np.all(model.status ==
                               np.array(['-', '.', 'F', '-', '-'])))

    def test_nonconvergence_ignore(self):
        model = self.Model(range(5),
                           alpha_1=0.6, alpha_2=0.4,
                           theta=0.2, G=20)
        model.solve(max_iter=5, failures='ignore')
        self.assertTrue(np.all(model.status[1:] == 'F'))


if __name__ == '__main__':
    unittest.main()
