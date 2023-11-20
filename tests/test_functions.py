# -*- coding: utf-8 -*-
"""
test_functions
==============
Test suite for `fsic` `functions` module.
"""

import unittest

import numpy as np

from fsic.core import VectorContainer
from fsic.functions import diff, dlog, lag, lead


class TestFunctions(unittest.TestCase):

    def test_lag(self):
        # Check lag operations
        x = np.arange(5, dtype=float)
        self.assertTrue(np.allclose(x, np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)))

        # A lag order of 0 should leave the array unchanged
        self.assertTrue(np.allclose(x, lag(x, 0)))

        self.assertTrue(np.array_equal(lag(x),
                                       np.array([np.nan, 0.0, 1.0, 2.0, 3.0], dtype=float),
                                       equal_nan=True))
        self.assertTrue(np.array_equal(lag(x, 2),
                                       np.array([np.nan, np.nan, 0.0, 1.0, 2.0], dtype=float),
                                       equal_nan=True))

    def test_lead(self):
        # Check lead operations
        x = np.arange(5, dtype=float)
        self.assertTrue(np.allclose(x, np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)))

        # A lead of 0 should leave the array unchanged
        self.assertTrue(np.allclose(x, lead(x, 0)))

        self.assertTrue(np.array_equal(lead(x),
                                       np.array([1.0, 2.0, 3.0, 4.0, np.nan], dtype=float),
                                       equal_nan=True))
        self.assertTrue(np.array_equal(lead(x, 2),
                                       np.array([2.0, 3.0, 4.0, np.nan, np.nan], dtype=float),
                                       equal_nan=True))

    def test_diff(self):
        # Check difference operations
        x = np.arange(5, dtype=float)
        self.assertTrue(np.allclose(x, np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)))

        # A difference of 0 should leave the array unchanged
        self.assertTrue(np.allclose(x, diff(x, 0)))

        self.assertTrue(np.array_equal(diff(x),
                                       np.array([np.nan, 1.0, 1.0, 1.0, 1.0], dtype=float),
                                       equal_nan=True))
        self.assertTrue(np.array_equal(diff(x, 2),
                                       np.array([np.nan, np.nan, 2.0, 2.0, 2.0], dtype=float),
                                       equal_nan=True))

    def test_dlog(self):
        # Check difference of log operations
        x = np.arange(1, 5 + 1, dtype=float) ** 2

        self.assertTrue(np.isnan(dlog(x)[0]))
        self.assertTrue(np.allclose(dlog(x)[1:], np.array([np.log(4) - np.log(1),
                                                           np.log(9) - np.log(4),
                                                           np.log(16) - np.log(9),
                                                           np.log(25) - np.log(16)])))


class TestFunctionsEval(unittest.TestCase):

    def test_eval_lag(self):
        # Check lag operations in `eval()`
        container = VectorContainer(range(5))
        container.add_variable('x', np.arange(5, dtype=float))

        self.assertTrue(np.allclose(container.x, np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)))

        self.assertTrue(np.array_equal(container.eval('lag(x)'),
                                       np.array([np.nan, 0.0, 1.0, 2.0, 3.0], dtype=float),
                                       equal_nan=True))
        self.assertTrue(np.array_equal(container.eval('lag(x, 2)'),
                                       np.array([np.nan, np.nan, 0.0, 1.0, 2.0], dtype=float),
                                       equal_nan=True))

    def test_eval_lead(self):
        # Check lead operations in `eval()`
        container = VectorContainer(range(5))
        container.add_variable('x', np.arange(5, dtype=float))

        self.assertTrue(np.allclose(container.x, np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)))

        self.assertTrue(np.array_equal(container.eval('lead(x)'),
                                       np.array([1.0, 2.0, 3.0, 4.0, np.nan], dtype=float),
                                       equal_nan=True))
        self.assertTrue(np.array_equal(container.eval('lead(x, 2)'),
                                       np.array([2.0, 3.0, 4.0, np.nan, np.nan], dtype=float),
                                       equal_nan=True))

    def test_eval_diff(self):
        # Check difference operations in `eval()`
        container = VectorContainer(range(5))
        container.add_variable('x', np.arange(5, dtype=float))

        self.assertTrue(np.allclose(container.x, np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)))

        self.assertTrue(np.array_equal(container.eval('diff(x)'),
                                       np.array([np.nan, 1.0, 1.0, 1.0, 1.0], dtype=float),
                                       equal_nan=True))
        self.assertTrue(np.array_equal(container.eval('diff(x, 2)'),
                                       np.array([np.nan, np.nan, 2.0, 2.0, 2.0], dtype=float),
                                       equal_nan=True))


if __name__ == '__main__':
    unittest.main()
