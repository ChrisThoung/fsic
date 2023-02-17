# -*- coding: utf-8 -*-
"""
test_fsic_functions
===================
Test suite for `fsic` `functions` module.
"""

import unittest

import numpy as np

from fsic.functions import diff, lag, lead


class TestFunctions(unittest.TestCase):

    def test_lag(self):
        # Check lag operations
        x = np.arange(5, dtype=float)
        self.assertTrue(np.allclose(x, np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)))

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

        self.assertTrue(np.array_equal(diff(x),
                                       np.array([np.nan, 1.0, 1.0, 1.0, 1.0], dtype=float),
                                       equal_nan=True))
        self.assertTrue(np.array_equal(diff(x, 2),
                                       np.array([np.nan, np.nan, 2.0, 2.0, 2.0], dtype=float),
                                       equal_nan=True))


if __name__ == '__main__':
    unittest.main()
