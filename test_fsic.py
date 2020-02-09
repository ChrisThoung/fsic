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
import unittest

import numpy as np

import fsic


SIM = fsic.build_model(fsic.parse_model(
'''
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
'''))


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
            fsic.Term(name='C', type=fsic.Type.VARIABLE, index=0),
            fsic.Term(name='exp', type=fsic.Type.FUNCTION, index=None),
            fsic.Term(name='alpha_1', type=fsic.Type.PARAMETER, index=0),
            fsic.Term(name='log', type=fsic.Type.FUNCTION, index=None),
            fsic.Term(name='YD', type=fsic.Type.VARIABLE, index=0),
            fsic.Term(name='alpha_2', type=fsic.Type.PARAMETER, index=0),
            fsic.Term(name='log', type=fsic.Type.FUNCTION, index=None),
            fsic.Term(name='H', type=fsic.Type.VARIABLE, index=-1),
            fsic.Term(name='epsilon', type=fsic.Type.ERROR, index=0),
        ]

        self.assertEqual(fsic.parse_terms(expression), expected)

    def test_parse_equation_terms(self):
        # Test that `parse_equation_terms()` correctly identifies individual
        # terms in an equation
        equation = 'C = exp({alpha_1} * log(YD) + {alpha_2} * log(H[-1]) + <epsilon>)'
        expected = [
            fsic.Term(name='C', type=fsic.Type.ENDOGENOUS, index=0),
            fsic.Term(name='exp', type=fsic.Type.FUNCTION, index=None),
            fsic.Term(name='alpha_1', type=fsic.Type.PARAMETER, index=0),
            fsic.Term(name='log', type=fsic.Type.FUNCTION, index=None),
            fsic.Term(name='YD', type=fsic.Type.EXOGENOUS, index=0),
            fsic.Term(name='alpha_2', type=fsic.Type.PARAMETER, index=0),
            fsic.Term(name='log', type=fsic.Type.FUNCTION, index=None),
            fsic.Term(name='H', type=fsic.Type.EXOGENOUS, index=-1),
            fsic.Term(name='epsilon', type=fsic.Type.ERROR, index=0),
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


class TestInit(unittest.TestCase):

    def test_init_dimension_error(self):
        with self.assertRaises(fsic.DimensionError):
            # C is invalid because it has the wrong shape
            SIM(range(10), C=[0, 0])


class TestInterface(unittest.TestCase):

    def setUp(self):
        self.model = SIM(range(10))

    def test_dir(self):
        # Check that `dir(model)` includes the model's variable names
        for name in self.model.names + ['span', 'names', 'status', 'iterations']:
            self.assertIn(name, dir(self.model))

    def test_modify_iterations(self):
        # Check that the `iterations` attribute stays as a NumPy array
        iterations = np.full(10, -1, dtype=int)

        self.assertEqual(self.model.iterations.shape, iterations.shape)
        self.assertTrue(np.all(self.model.iterations == iterations))

        # Set all values to 1
        self.model.iterations = 1
        iterations[:] = 1

        self.assertEqual(self.model.iterations.shape, iterations.shape)
        self.assertTrue(np.all(self.model.iterations == iterations))

        # Assign a sequence
        self.model.iterations = range(0, 20, 2)
        iterations = np.arange(0, 20, 2)

        self.assertEqual(self.model.iterations.shape, iterations.shape)
        self.assertTrue(np.all(self.model.iterations == iterations))

    def test_modify_iterations_errors(self):
        # Check that `iterations` assignment errors are as expected
        with self.assertRaises(fsic.DimensionError):
            self.model.iterations = [0, 1]  # Incompatible dimensions

    def test_modify_status(self):
        # Check that the `status` attribute stays as a NumPy array
        status = np.full(10, '-')

        self.assertEqual(self.model.status.shape, status.shape)
        self.assertTrue(np.all(self.model.status == status))

        # Set all values to '.'
        self.model.status = '.'
        status[:] = '.'

        self.assertEqual(self.model.status.shape, status.shape)
        self.assertTrue(np.all(self.model.status == status))

        # Assign a sequence
        self.model.status = ['-', '.'] * 5
        status = np.array(['-', '.'] * 5)

        self.assertEqual(self.model.status.shape, status.shape)
        self.assertTrue(np.all(self.model.status == status))

    def test_modify_status_errors(self):
        # Check that `status` assignment errors are as expected
        with self.assertRaises(fsic.DimensionError):
            self.model.status = ['-', '.']  # Incompatible dimensions

class TestModelContainerMethods(unittest.TestCase):

    def setUp(self):
        # Initialise a new model object for each test
        Model = fsic.build_model(
            fsic.parse_model('C = {alpha_1} * YD + {alpha_2} * H[-1]'))
        self.model = Model(['{}Q{}'.format(y, q)
                            for y in range(1990, 1995 + 1)
                            for q in range(1, 4 + 1)])
        self.model.YD = np.arange(len(self.model.span))

    def test_getitem_by_name(self):
        # Test variable access by name
        self.assertTrue(np.allclose(self.model['YD'],
                                    np.arange(len(self.model.span))))

    def test_getitem_by_name_error(self):
        # Test that variable access raises a KeyError if the name isn't a model
        # variable
        with self.assertRaises(KeyError):
            self.model['ABC']

    def test_getitem_by_name_and_index(self):
        # Test simultaneous variable and single period access
        self.assertEqual(self.model['YD', '1991Q1'], 4.0)

    def test_getitem_by_name_and_slice_to(self):
        # Test simultaneous variable and slice access
        self.assertTrue(np.allclose(self.model['YD', :'1990Q4'],
                                    np.arange(3 + 1)))

    def test_getitem_by_name_and_slice_from(self):
        # Test simultaneous variable and slice access
        self.assertTrue(np.allclose(self.model['YD', '1995Q1':],
                                    np.arange(20, 23 + 1)))

    def test_getitem_by_name_and_slice_step(self):
        # Test simultaneous variable and slice access
        self.assertTrue(np.allclose(self.model['YD', ::4],
                                    np.arange(0, 23 + 1, 4)))

    def test_setitem_by_name_with_number(self):
        # Test variable assignment by name, setting all elements to a single
        # value
        self.model['YD'] = 10
        self.assertTrue(np.allclose(self.model['YD'],
                                    np.full(len(self.model.span), 10, dtype=float)))

    def test_setitem_by_name_with_array(self):
        # Test variable assignment by name, replacing with a new array
        self.model['YD'] = np.arange(0, len(self.model.span) * 2, 2)
        self.assertTrue(np.allclose(self.model['YD'],
                                    np.arange(0, len(self.model.span) * 2, 2)))

    def test_setitem_by_name_and_index(self):
        # Test variable assignment by name and single period
        self.model['YD', '1990Q1'] = 100

        expected = np.arange(len(self.model.span))
        expected[0] = 100

        self.assertTrue(np.allclose(self.model['YD'],
                                    expected))

    def test_setitem_by_name_and_slice_to(self):
        # Test variable assignment by name and slice
        self.model['YD', :'1990Q4'] = 100

        expected = np.arange(len(self.model.span))
        expected[:4] = 100

        self.assertTrue(np.allclose(self.model['YD'],
                                    expected))

    def test_setitem_by_name_and_slice_from(self):
        # Test variable assignment by name and slice
        self.model['YD', '1995Q1':] = 100

        expected = np.arange(len(self.model.span))
        expected[-4:] = 100

        self.assertTrue(np.allclose(self.model['YD'],
                                    expected))

    def test_setitem_dimension_error(self):
        # Test check for misaligned dimensions at assignment
        with self.assertRaises(fsic.DimensionError):
            self.model.C = [0, 0]


round_1dp = functools.partial(round, ndigits=1)

class TestBuildAndSolve(unittest.TestCase):

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
        self.assertEqual(round_1dp(sim.C[-1]), 80.0)
        self.assertEqual(round_1dp(sim.YD[-1]), 80.0)
        self.assertEqual(round_1dp(sim.H[-1]), 80.0)
        self.assertEqual(round_1dp(sim.Y[-1]), 100.0)
        self.assertEqual(round_1dp(sim.T[-1]), 20.0)
        self.assertEqual(round_1dp(sim.G[-1]), 20.0)

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
        self.assertEqual(round_1dp(ami.C[-1]), 3631.3)
        self.assertEqual(round_1dp(ami.I[-1]), 1136.2)
        self.assertEqual(round_1dp(ami.PQ[-1]), 6394.5)
        self.assertEqual(round_1dp(ami.M[-1]), 657.0)
        self.assertEqual(round_1dp(ami.Q[-1]), 5327.0)
        self.assertEqual(round_1dp(ami.Y[-1]), 3835.4)


class TestCopy(unittest.TestCase):

    def setUp(self):
        self.model = SIM(range(1945, 2010 + 1),
                         alpha_1=0.6, alpha_2=0.4,
                         theta=0.2)

    def test_copy(self):
        self.model.G = 20

        duplicate_model = self.model.copy()

        # Values should be identical at this point
        self.assertTrue(np.allclose(self.model.values,
                                    duplicate_model.values))

        # The solved model should have different values to the duplicate
        self.model.solve()
        self.assertFalse(np.allclose(self.model.values,
                                     duplicate_model.values))

        # The solved duplicate should match the original again
        duplicate_model.solve()
        self.assertTrue(np.allclose(self.model.values,
                                    duplicate_model.values))


if __name__ == '__main__':
    unittest.main()
