# -*- coding: utf-8 -*-
"""
test_fsic_fortran
=================
Test suite for Fortran-accelerated FSIC-based economic models.

Example equations/models are adapted from:

    Godley, W., Lavoie, M. (2007)
    *Monetary economics: An integrated approach to credit, money, income, production and wealth*,
    Palgrave Macmillan

Godley and Lavoie (2007) implementation also informed by Gennaro Zezza's EViews
programs:

    http://gennaro.zezza.it/software/eviews/glch03.php
"""

import glob
import importlib
import os
import subprocess
import unittest

import numpy as np

import fsic

try:
    from . import test_fsic
    imported_as_package = True
except ImportError:
    import test_fsic
    imported_as_package = False


class FortranTestWrapper:

    # Define a path specific to each test case to avoid import issues from
    # trying to redefine the same module on disk at runtime
    TEST_MODULE_NAME = None

    # Override as `True` either here or in individual subclasses to retain the
    # intermediate output (source and compiled Fortran code)
    DEBUG = False

    def clean(self):
        # Just return if `DEBUG`ging: Don't delete the source and compiled
        # Fortran code
        if self.DEBUG:
            return

        # Delete intermediate and compiled files
        files_to_delete = (glob.glob(os.path.join(os.path.split(__file__)[0], '{}*.f95').format(self.TEST_MODULE_NAME)) +
                           glob.glob(os.path.join(os.path.split(__file__)[0], '{}*.*.so').format(self.TEST_MODULE_NAME)))

        for path in files_to_delete:
            os.remove(path)

    @staticmethod
    def build_model(symbols, test_module_name):
        # Switch the working directory to the same one that contains this test
        # script
        original_working_directory = os.getcwd()
        os.chdir(os.path.split(__file__)[0])

        # Write out a file of Fortran code
        fortran_definition = fsic.fortran.build_fortran_definition(symbols)
        with open('{}.f95'.format(test_module_name), 'w') as f:
            f.write(fortran_definition)

        # Compile the code
        output = subprocess.run(['f2py',
                                 '-c', '{}.f95'.format(test_module_name),
                                 '-m', test_module_name],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # Print output on failed compile
        try:
            output.check_returncode()
        except subprocess.CalledProcessError as e:
            print(e.stdout.decode())
            raise

        # Construct the class
        PythonClass = fsic.build_model(symbols)

        class FortranClass(fsic.fortran.FortranEngine, PythonClass):
            if imported_as_package:
                ENGINE = importlib.import_module('.{}'.format(test_module_name),
                                                 os.path.split(os.path.split(__file__)[0])[1])
            else:
                ENGINE = importlib.import_module(test_module_name)

        # Switch back to the original directory
        os.chdir(original_working_directory)

        return FortranClass

    def setUp(self):
        self.clean()
        self.Model = self.build_model(self.SYMBOLS, self.TEST_MODULE_NAME)

    def tearDown(self):
        self.clean()


class TestInit(FortranTestWrapper, test_fsic.TestInit):
    TEST_MODULE_NAME = 'fsic_test_fortran_testinit'

class TestInterface(FortranTestWrapper, test_fsic.TestInterface):
    TEST_MODULE_NAME = 'fsic_test_fortran_testinterface'

class TestModelContainerMethods(FortranTestWrapper, test_fsic.TestModelContainerMethods):
    TEST_MODULE_NAME = 'fsic_test_fortran_testmodelcontainermethods'

class TestCopy(FortranTestWrapper, test_fsic.TestCopy):
    TEST_MODULE_NAME = 'fsic_test_fortran_testcopy'

class TestSolve(FortranTestWrapper, test_fsic.TestSolve):
    TEST_MODULE_NAME = 'fsic_test_fortran_testsolve'


class TestSolutionErrorHandling(FortranTestWrapper, test_fsic.TestSolutionErrorHandling):
    TEST_MODULE_NAME = 'fsic_test_fortran_testsolutionerrorhandling'

    @unittest.skip('Version 0.8.0 behaviour not yet implemented in Fortran')
    def test_raise_nans(self):
        super().test_raise_nans()

    @unittest.skip('Version 0.8.0 behaviour not yet implemented in Fortran')
    def test_raise_infinities(self):
        super().test_raise_infinities()


class TestNonConvergenceError(FortranTestWrapper, test_fsic.TestNonConvergenceError):
    TEST_MODULE_NAME = 'fsic_test_fortran_testnonconvergenceerror'

class TestLinkerCopy(FortranTestWrapper, test_fsic.TestLinkerCopy):
    TEST_MODULE_NAME = 'fsic_test_fortran_testlinkercopy'

class TestLinkerSolve(FortranTestWrapper, test_fsic.TestLinkerSolve):
    TEST_MODULE_NAME = 'fsic_test_fortran_testlinkersolve'


class TestLinkerInit(FortranTestWrapper, test_fsic.TestLinkerInit):
    TEST_MODULE_NAME = 'fsic_test_fortran_testlinkerinit'

    def setUp(self):
        self.SubmodelNoLags = self.build_model(self.SYMBOLS_NO_LAGS, 'fsic_test_fortran_testlinkerinit_no_lags')
        self.SubmodelWithLags = self.build_model(self.SYMBOLS_WITH_LAGS, 'fsic_test_fortran_testlinkerinit_with_lags')
        self.SubmodelWithLeads = self.build_model(self.SYMBOLS_WITH_LEADS, 'fsic_test_fortran_testlinkerinit_with_leads')


class TestCompile(FortranTestWrapper, unittest.TestCase):

    TEST_MODULE_NAME = 'fsic_test_fortran_testcompile'

    # Define a large number of variables of the same type (here, arbitrarily,
    # exogenous variables) to check that index numbers in the resulting Fortran
    # code are line-wrapped correctly
    SCRIPT = ('RESULT = A + B + C + D + E + F + G + H + I + J + '
                       'K + L + M + N + O + P + Q + R + S + T + '
                       'U + V + W + X + Y + Z')
    SYMBOLS = fsic.parse_model(SCRIPT)

    def test_continuation_lines(self):
        # Test line wrapping for long lines
        # No code here: Check that the code successfully compiles during
        # `setUp()`
        pass


class TestBuild(unittest.TestCase):

    def test_lags(self):
        # Check that the `lags` keyword argument imposes a lag length on the
        # final model

        # No symbols: Take specified lag length
        self.assertIn(
            'integer :: lags = 2, leads = 0',
            fsic.fortran.build_fortran_definition([],
                             lags=2))

        # Symbol with lag of 1: Take specified lag length (impose 2)
        self.assertIn(
            'integer :: lags = 2, leads = 0',
            fsic.fortran.build_fortran_definition([fsic.parser.Symbol(name='C', type=fsic.parser.Type.ENDOGENOUS, lags=-1, leads=0, equation=None, code=None)],
                             lags=2))

        # Symbol with lag of 3: Take specified lag length (impose 2)
        self.assertIn(
            'integer :: lags = 2, leads = 0',
            fsic.fortran.build_fortran_definition([fsic.parser.Symbol(name='C', type=fsic.parser.Type.ENDOGENOUS, lags=-3, leads=0, equation=None, code=None)],
                             lags=2))

    def test_leads(self):
        # Check that the `leads` keyword argument imposes a lead length on the
        # final model

        # No symbols: Take specified lead length
        self.assertIn(
            'integer :: lags = 0, leads = 2',
            fsic.fortran.build_fortran_definition([],
                             leads=2),
            2)

        # Symbol with lead of 1: Take specified lead length (impose 2)
        self.assertIn(
            'integer :: lags = 0, leads = 2',
            fsic.fortran.build_fortran_definition([fsic.parser.Symbol(name='C', type=fsic.parser.Type.ENDOGENOUS, lags=0, leads=1, equation=None, code=None)],
                             leads=2),
            2)

        # Symbol with lead of 3: Take specified lead length (impose 2)
        self.assertIn(
            'integer :: lags = 0, leads = 2',
            fsic.fortran.build_fortran_definition([fsic.parser.Symbol(name='C', type=fsic.parser.Type.ENDOGENOUS, lags=0, leads=3, equation=None, code=None)],
                             leads=2),
            2)

    def test_min_lags(self):
        # Check that the `min_lags` keyword argument ensures a minimum lag
        # length

        # No symbols: Take specified minimum lag length
        self.assertIn(
            'integer :: lags = 2, leads = 0',
            fsic.fortran.build_fortran_definition([],
                             min_lags=2))

        # Symbol with lag of 1: Take minimum lag length (impose 2)
        self.assertIn(
            'integer :: lags = 2, leads = 0',
            fsic.fortran.build_fortran_definition([fsic.parser.Symbol(name='C', type=fsic.parser.Type.ENDOGENOUS, lags=-1, leads=0, equation=None, code=None)],
                             min_lags=2))

        # Symbol with lag of 3: Ignore minimum lag length (set to 3)
        self.assertIn(
            'integer :: lags = 3, leads = 0',
            fsic.fortran.build_fortran_definition([fsic.parser.Symbol(name='C', type=fsic.parser.Type.ENDOGENOUS, lags=-3, leads=0, equation=None, code=None)],
                             min_lags=2))

    def test_min_leads(self):
        # Check that the `min_leads` keyword argument ensures a minimum lead
        # length

        # No symbols: Take specified minimum lead length
        self.assertIn(
            'integer :: lags = 0, leads = 2',
            fsic.fortran.build_fortran_definition([],
                             min_leads=2))

        # Symbol with lead of 1: Take minimum lead length (impose 2)
        self.assertIn(
            'integer :: lags = 0, leads = 2',
            fsic.fortran.build_fortran_definition([fsic.parser.Symbol(name='C', type=fsic.parser.Type.ENDOGENOUS, lags=0, leads=-1, equation=None, code=None)],
                             min_leads=2))

        # Symbol with lead of 3: Ignore minimum lead length (set to 3)
        self.assertIn(
            'integer :: lags = 0, leads = 3',
            fsic.fortran.build_fortran_definition([fsic.parser.Symbol(name='C', type=fsic.parser.Type.ENDOGENOUS, lags=0, leads=-3, equation=None, code=None)],
                             min_leads=2))


class TestBuildAndSolve(FortranTestWrapper, unittest.TestCase):

    TEST_MODULE_NAME = 'fsic_test_fortran_testbuildandsolve'

    # Definition of a stripped-down Model *SIM* from Chapter 3 of Godley and
    # Lavoie (2007)
    SCRIPT = '''
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
'''
    SYMBOLS = fsic.parse_model(SCRIPT)

    def setUp(self):
        super().setUp()

        PythonClass = fsic.build_model(self.SYMBOLS)
        FortranClass = self.Model

        # Instantiate a Python and a corresponding Fortran instance of the
        # model
        self.model_python = PythonClass(range(100), alpha_1=0.6, alpha_2=0.4)
        self.model_fortran = FortranClass(range(100), alpha_1=0.6, alpha_2=0.4)

    def test_initialisation_error(self):
        # Check that the Fortran class catches a missing (unlinked) Fortran
        # module
        with self.assertRaises(fsic.fortran.InitialisationError):
            fsic.fortran.FortranEngine(range(10))

    def test_evaluate(self):
        # Check Python- and Fortran-based evaluate functions generate the same
        # results

        # Python
        self.model_python.G = 20
        self.model_python.theta = 0.2

        for i in self.model_python.span[3:10]:
            for _ in range(100):
                self.model_python._evaluate(i)

        for _ in range(100):
            self.model_python._evaluate(-10)

        # Fortran
        self.model_fortran.G = 20
        self.model_fortran.theta = 0.2

        for i in self.model_fortran.span[3:10]:
            for _ in range(100):
                self.model_fortran._evaluate(i)

        for _ in range(100):
            self.model_fortran._evaluate(-10)

        # Comparison
        self.assertEqual(self.model_python.values.shape, self.model_fortran.values.shape)
        self.assertTrue(np.allclose(self.model_python.values, self.model_fortran.values))

    def test_evaluate_index_errors(self):
        # Check model instances correctly throw errors if period (`t`)
        # parameters are out of bounds
        with self.assertRaises(IndexError):
            self.model_python._evaluate(100)

        with self.assertRaises(IndexError):
            self.model_fortran._evaluate(100)

        with self.assertRaises(IndexError):
            self.model_python._evaluate(-100)

        with self.assertRaises(IndexError):
            self.model_fortran._evaluate(-100)

    def test_solve_t_max_iter(self):
        # Check that the Fortran code returns the correct number of iterations
        # if it reaches `max_iter`
        # (In Fortran, a loop that completes seems to leave the counter 1
        # higher than the loop limit. This isn't the case if the loop exits
        # early.)

        # Python
        self.model_python.G = 20
        self.model_python.theta = 0.2

        self.model_python.solve(max_iter=2, failures='ignore')
        self.assertTrue(np.all(self.model_python.iterations[1:] == 2))

        # Fortran
        self.model_fortran.G = 20
        self.model_fortran.theta = 0.2

        self.model_fortran.solve(max_iter=2, failures='ignore')
        self.assertTrue(np.all(self.model_fortran.iterations[1:] == 2))

    def test_solve_t_errors_ignore(self):
        # Check that the `errors='ignore'` solve option ignores NaNs properly

        # Python
        self.model_python.G = 20
        self.model_python.theta = 0.2
        self.model_python.C[2] = np.NaN

        # Solution should halt because of the pre-existing NaN
        with self.assertRaises(fsic.exceptions.SolutionError):
            self.model_python.solve()

        self.model_python.solve(errors='ignore')

        # Fortran
        self.model_fortran.G = 20
        self.model_fortran.theta = 0.2
        self.model_fortran.C[2] = np.NaN

        # Solution should halt because of the pre-existing NaN
        with self.assertRaises(fsic.exceptions.SolutionError):
            self.model_fortran.solve()

        self.model_fortran.solve(errors='ignore')

        # Comparison
        self.assertEqual(self.model_python.values.shape, self.model_fortran.values.shape)
        self.assertTrue(np.allclose(self.model_python.values, self.model_fortran.values))

    def test_solve(self):
        # Check Python- and Fortran-based solve functions generate the same
        # results

        # Python
        self.model_python.G = 20
        self.model_python.theta = 0.2

        self.model_python.solve()

        # Fortran
        self.model_fortran.G = 20
        self.model_fortran.theta = 0.2

        self.model_fortran.solve()

        # Comparison
        self.assertEqual(self.model_python.values.shape, self.model_fortran.values.shape)
        self.assertTrue(np.allclose(self.model_python.values, self.model_fortran.values))

    def test_solve_offset(self):
        # Check Python- and Fortran-based solve functions generate the same
        # results with the `offset` keyword argument

        # Python
        self.model_python.G = 20
        self.model_python.theta = 0.2

        self.model_python.solve(offset=-1)

        # Fortran
        self.model_fortran.G = 20
        self.model_fortran.theta = 0.2

        self.model_fortran.solve(offset=-1)

        # Comparison
        self.assertEqual(self.model_python.values.shape, self.model_fortran.values.shape)
        self.assertTrue(np.allclose(self.model_python.values, self.model_fortran.values))

    def test_solve_max_iter_non_convergence_error(self):
        # Check Python- and Fortran-based solve functions generate the same
        # error if the model fails to solve

        # Python
        self.model_python.G = 20
        self.model_python.theta = 0.2

        with self.assertRaises(fsic.exceptions.NonConvergenceError):
            self.model_python.solve(max_iter=5)

        # Fortran
        self.model_fortran.G = 20
        self.model_fortran.theta = 0.2

        with self.assertRaises(fsic.exceptions.NonConvergenceError):
            self.model_fortran.solve(max_iter=5)

        # Comparison
        self.assertEqual(self.model_python.values.shape, self.model_fortran.values.shape)
        self.assertTrue(np.allclose(self.model_python.values, self.model_fortran.values))


if __name__ == '__main__':
    unittest.main()
