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
import os
import subprocess
import unittest

import numpy as np

import fsic
import fsic_fortran

import test_fsic


class FortranTestWrapper:

    def clean(self):
        # Delete intermediate and compiled files
        files_to_delete = (glob.glob('fsic_test_tmp.f95') +
                           glob.glob('fsic_test_tmp.*.so'))

        for path in files_to_delete:
            os.remove(path)

    def setUp(self):
        self.clean()

        # Write out a file of Fortran code
        fortran_definition = fsic_fortran.build_fortran_definition(self.SYMBOLS)
        with open('fsic_test_tmp.f95', 'w') as f:
            f.write(fortran_definition)

        # Compile the code
        output = subprocess.run(['f2py', '-c', 'fsic_test_tmp.f95', '-m', 'fsic_test_tmp'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output.check_returncode()

        # Construct the class
        PythonClass = fsic.build_model(self.SYMBOLS)

        import fsic_test_tmp
        class FortranClass(fsic_fortran.FortranEngine, PythonClass):
            ENGINE = fsic_test_tmp

        self.Model = FortranClass

    def tearDown(self):
        self.clean()


class TestInit(FortranTestWrapper, test_fsic.TestInit):
    pass

class TestInterface(FortranTestWrapper, test_fsic.TestInterface):
    pass

class TestModelContainerMethods(FortranTestWrapper, test_fsic.TestModelContainerMethods):
    pass

class TestCopy(FortranTestWrapper, test_fsic.TestCopy):
    pass

class TestSolve(FortranTestWrapper, test_fsic.TestSolve):
    pass

class TestSolutionErrorHandling(FortranTestWrapper, test_fsic.TestSolutionErrorHandling):
    pass

class TestNonConvergenceError(FortranTestWrapper, test_fsic.TestNonConvergenceError):
    pass


class TestBuildAndSolve(FortranTestWrapper, unittest.TestCase):

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
