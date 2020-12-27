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
import fsic_fortran

import test_fsic


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
        files_to_delete = (glob.glob('{}.f95'.format(self.TEST_MODULE_NAME)) +
                           glob.glob('{}.*.so'.format(self.TEST_MODULE_NAME)))

        for path in files_to_delete:
            os.remove(path)

    def setUp(self):
        self.clean()

        # Write out a file of Fortran code
        fortran_definition = fsic_fortran.build_fortran_definition(self.SYMBOLS)
        with open('{}.f95'.format(self.TEST_MODULE_NAME), 'w') as f:
            f.write(fortran_definition)

        # Compile the code
        output = subprocess.run(['f2py',
                                 '-c', '{}.f95'.format(self.TEST_MODULE_NAME),
                                 '-m', self.TEST_MODULE_NAME],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # Print output on failed compile
        try:
            output.check_returncode()
        except subprocess.CalledProcessError as e:
            print(e.stdout.decode())
            raise

        # Construct the class
        PythonClass = fsic.build_model(self.SYMBOLS)

        class FortranClass(fsic_fortran.FortranEngine, PythonClass):
            ENGINE = importlib.import_module(self.TEST_MODULE_NAME)

        self.Model = FortranClass

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

class TestNonConvergenceError(FortranTestWrapper, test_fsic.TestNonConvergenceError):
    TEST_MODULE_NAME = 'fsic_test_fortran_testnonconvergenceerror'


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
        with self.assertRaises(fsic_fortran.InitialisationError):
            fsic_fortran.FortranEngine(range(10))

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
        with self.assertRaises(fsic.SolutionError):
            self.model_python.solve()

        self.model_python.solve(errors='ignore')

        # Fortran
        self.model_fortran.G = 20
        self.model_fortran.theta = 0.2
        self.model_fortran.C[2] = np.NaN

        # Solution should halt because of the pre-existing NaN
        with self.assertRaises(fsic.SolutionError):
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

        with self.assertRaises(fsic.NonConvergenceError):
            self.model_python.solve(max_iter=5)

        # Fortran
        self.model_fortran.G = 20
        self.model_fortran.theta = 0.2

        with self.assertRaises(fsic.NonConvergenceError):
            self.model_fortran.solve(max_iter=5)

        # Comparison
        self.assertEqual(self.model_python.values.shape, self.model_fortran.values.shape)
        self.assertTrue(np.allclose(self.model_python.values, self.model_fortran.values))


if __name__ == '__main__':
    unittest.main()
