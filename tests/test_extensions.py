# -*- coding: utf-8 -*-
"""
test_extensions
===============
Test suite for fsic extensions.
"""

import math
import unittest

import numpy as np

import fsic


pandas_installed = True

try:
    import pandas as pd
except ModuleNotFoundError:
    pandas_installed = False


class TestAliasMixin(unittest.TestCase):
    # TODO: Add support for Fortran tests

    script = '''
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
'''
    symbols = fsic.parse_model(script)
    SIM = fsic.build_model(symbols)

    class SIMAlias(fsic.extensions.AliasMixin, SIM):
        ALIASES = {
            'GDP': 'Y',
            'expenditure': 'output',
            'output': 'income',
            'income': 'Y',
            'mpc_income': 'alpha_1',
            'mpc_wealth': 'alpha_2',
            'income_tax_rate': 'theta',
        }

        PREFERRED_NAMES = ['GDP', 'alpha_1', 'alpha_2']

    def test_alias_mapping(self):
        # Check that alias mappings are resolved to the lowest-level variable
        model = self.SIMAlias(range(10), alpha_1=0.6, mpc_wealth=0.4)

        self.assertEqual(
            model.aliases,
            {'GDP': 'Y',
             'expenditure': 'Y',
             'output': 'Y',
             'income': 'Y',
             'mpc_income': 'alpha_1',
             'mpc_wealth': 'alpha_2',
             'income_tax_rate': 'theta', })

    def test_init(self):
        # Check __init__() works with aliases
        model = self.SIMAlias(range(10), alpha_1=0.6, mpc_wealth=0.4)

        self.assertTrue(np.allclose(model.mpc_income, np.array([0.6] * 10)))
        self.assertTrue(np.allclose(model.alpha_2, np.array([0.4] * 10)))

    def test_get_set(self):
        # Check that get and set methods work with aliases
        model = self.SIMAlias(range(-5, 5 + 1))

        self.assertTrue(np.allclose(model.Y, np.array([0.0] * 11)))

        model.Y = 50
        self.assertTrue(np.allclose(model.Y, np.array([50.0] * 11)))

        model['GDP'] = 100
        self.assertTrue(np.allclose(model.Y, np.array([100.0] * 11)))

        model.expenditure[1:3] = 75
        self.assertTrue(np.allclose(model.output, np.array([100.0, 75.0, 75.0] + [100.0] * 8)))

        model['output', 0] = 125
        self.assertTrue(np.allclose(model['income'], np.array([100.0, 75.0, 75.0, 100.0, 100.0, 125.0] +
                                                              [100.0] * 5)))

        self.assertTrue(math.isclose(model['income', 0], 125))

    def test_dir(self):
        # Check that aliases appear in the list of methods and attributes
        model = self.SIMAlias(range(5))

        full_list = set(dir(model))
        expected_to_be_present = set(['GDP', 'Y', 'expenditure', 'output', 'income',
                                      'mpc_income', 'alpha_1', 'mpc_wealth', 'alpha_2', 'income_tax_rate',
                                      'theta'])

        # Check that the intersection of the sets leaves the expected variable
        # names
        self.assertEqual(full_list & expected_to_be_present, expected_to_be_present)

    def test_ipython_key_completions(self):
        # Check that aliases appear in the list of IPython method and attribute
        # completions
        model = self.SIMAlias(range(5))

        full_list = set(model._ipython_key_completions_())
        expected_to_be_present = set(['GDP', 'Y', 'expenditure', 'output', 'income',
                                      'mpc_income', 'alpha_1', 'mpc_wealth', 'alpha_2', 'income_tax_rate',
                                      'theta'])

        # Check that the intersection of the sets leaves the expected variable
        # names
        self.assertEqual(full_list & expected_to_be_present, expected_to_be_present)

    @unittest.skipIf(not pandas_installed, 'Requires `pandas`')
    def test_to_dataframe_default(self):
        # Check that `to_dataframe()` returns, by default, the same result as
        # the base version
        model = self.SIMAlias(range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2)
        model.solve()

        pd.testing.assert_frame_equal(model.to_dataframe(),
                                      super(self.SIMAlias, model).to_dataframe())

    @unittest.skipIf(not pandas_installed, 'Requires `pandas`')
    def test_to_dataframe_aliases(self):
        # Check that `to_dataframe()` correctly resolves and renames aliases
        model = self.SIMAlias(range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2)
        model.solve()

        result = model.to_dataframe(use_aliases=True)

        default_dataframe = super(self.SIMAlias, model).to_dataframe()
        self.assertNotEqual(list(result.columns), list(default_dataframe))
        pd.testing.assert_frame_equal(result,
                                      default_dataframe.set_axis(
                                          result.columns,
                                          axis='columns'))

        self.assertEqual(list(result.columns),
                         ['C', 'YD', 'H', 'GDP', 'T',
                          'G', 'alpha_1', 'alpha_2', 'income_tax_rate', 'status',
                          'iterations'])


if __name__ == '__main__':
    unittest.main()
