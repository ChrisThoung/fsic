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


# Definition of a stripped-down Model *SIM* from Chapter 3 of Godley and Lavoie
# (2007)
symbols = fsic.parse_model(
'''
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
''')

SIM = fsic.build_model(symbols)


class TestBuildAndSolve(unittest.TestCase):

    def clean(self):
        # Delete intermediate and compiled files
        files_to_delete = glob.glob('fsic_tmp.f95') + glob.glob('fsic_tmp.*.so')

        for path in files_to_delete:
            os.remove(path)

    def setUp(self):
        self.clean()

        # Generate and write out a file of Fortran code
        fortran_definition = fsic_fortran.build_fortran_definition(symbols)

        with open('fsic_tmp.f95', 'w') as f:
            f.write(fortran_definition)

        # Compile the code
        subprocess.run(['f2py', '-c', 'fsic_tmp.f95', '-m', 'fsic_tmp'])

        # Create a new class to embed the compiled Fortran module
        import fsic_tmp
        class SIMFortran(fsic_fortran.FortranEngine, SIM):
            ENGINE = fsic_tmp

        # Instantiate a Python and a corresponding Fortran instance of the
        # model
        self.model_python = SIM(range(100), alpha_1=0.6, alpha_2=0.4)
        self.model_fortran = SIMFortran(range(100), alpha_1=0.6, alpha_2=0.4)

    def tearDown(self):
        self.clean()

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
