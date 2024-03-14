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

    script = """
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
"""
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
            {
                'GDP': 'Y',
                'expenditure': 'Y',
                'output': 'Y',
                'income': 'Y',
                'mpc_income': 'alpha_1',
                'mpc_wealth': 'alpha_2',
                'income_tax_rate': 'theta',
            },
        )

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
        self.assertTrue(
            np.allclose(model.output, np.array([100.0, 75.0, 75.0] + [100.0] * 8))
        )

        model['output', 0] = 125
        self.assertTrue(
            np.allclose(
                model['income'],
                np.array([100.0, 75.0, 75.0, 100.0, 100.0, 125.0] + [100.0] * 5),
            )
        )

        self.assertTrue(math.isclose(model['income', 0], 125))

    def test_dir(self):
        # Check that aliases appear in the list of methods and attributes
        model = self.SIMAlias(range(5))

        full_list = set(dir(model))
        expected_to_be_present = set(
            [
                'GDP',
                'Y',
                'expenditure',
                'output',
                'income',
                'mpc_income',
                'alpha_1',
                'mpc_wealth',
                'alpha_2',
                'income_tax_rate',
                'theta',
            ]
        )

        # Check that the intersection of the sets leaves the expected variable
        # names
        self.assertEqual(full_list & expected_to_be_present, expected_to_be_present)

    def test_ipython_key_completions(self):
        # Check that aliases appear in the list of IPython method and attribute
        # completions
        model = self.SIMAlias(range(5))

        full_list = set(model._ipython_key_completions_())
        expected_to_be_present = set(
            [
                'GDP',
                'Y',
                'expenditure',
                'output',
                'income',
                'mpc_income',
                'alpha_1',
                'mpc_wealth',
                'alpha_2',
                'income_tax_rate',
                'theta',
            ]
        )

        # Check that the intersection of the sets leaves the expected variable
        # names
        self.assertEqual(full_list & expected_to_be_present, expected_to_be_present)

    @unittest.skipIf(not pandas_installed, 'Requires `pandas`')
    def test_to_dataframe_default(self):
        # Check that `to_dataframe()` returns, by default, the same result as
        # the base version
        model = self.SIMAlias(
            range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2
        )
        model.solve()

        pd.testing.assert_frame_equal(
            model.to_dataframe(), super(self.SIMAlias, model).to_dataframe()
        )

    @unittest.skipIf(not pandas_installed, 'Requires `pandas`')
    def test_to_dataframe_aliases(self):
        # Check that `to_dataframe()` correctly resolves and renames aliases
        model = self.SIMAlias(
            range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2
        )
        model.solve()

        result = model.to_dataframe(use_aliases=True)

        default_dataframe = super(self.SIMAlias, model).to_dataframe()
        self.assertNotEqual(list(result.columns), list(default_dataframe))
        pd.testing.assert_frame_equal(
            result, default_dataframe.set_axis(result.columns, axis='columns')
        )

        self.assertEqual(
            list(result.columns),
            [
                'C',
                'YD',
                'H',
                'GDP',
                'T',
                'G',
                'alpha_1',
                'alpha_2',
                'income_tax_rate',
                'status',
                'iterations',
            ],
        )


@unittest.skipIf(not pandas_installed, 'Requires `pandas`')
class TestPandasFeaturesMixin(unittest.TestCase):
    # TODO: Rework for Fortran compatibility
    SCRIPT = """\
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
    """
    SYMBOLS = fsic.parse_model(SCRIPT)
    SIM = fsic.build_model(SYMBOLS)

    class Model(fsic.extensions.model.PandasIndexFeaturesMixin, SIM):
        pass

    def test_reindex(self):
        # Test `pandas`-based version of `reindex()`

        from pandas.testing import assert_frame_equal

        def test(model):
            model.solve()

            results = model.to_dataframe()

            # Check standard in-sample reindexing
            expected = results.reindex(index=range(1, 9))
            model = model.reindex(range(1, 9))

            assert_frame_equal(model.to_dataframe(), expected)

            # Check standard out-of-sample reindexing
            expected = expected.reindex(index=range(5, 15 + 1))
            expected.loc[9:, 'status'] = '-'
            expected.loc[9:, 'iterations'] = -1
            expected['iterations'] = expected['iterations'].astype(int)

            model = model.reindex(range(5, 15 + 1))
            assert_frame_equal(model.to_dataframe(), expected)

            # Check implementation of `pandas` 'ffill' to future out-of-sample
            # periods
            expected = expected.reindex(index=range(5, 15 + 1), method='ffill')
            expected.loc[11:, 'status'] = '-'
            expected.loc[11:, 'iterations'] = -1
            expected['iterations'] = expected['iterations'].astype(int)

            model = model.reindex(range(5, 15 + 1), method='ffill')
            assert_frame_equal(model.to_dataframe(), expected)

            # Check implementation of `pandas` 'bfill' to past out-of-sample
            # periods
            expected = expected.reindex(index=range(-10, 0 + 1), method='bfill')
            expected.loc[:0, 'status'] = '-'
            expected.loc[:0, 'iterations'] = -1
            expected['iterations'] = expected['iterations'].astype(int)

            model = model.reindex(range(-10, 0 + 1), method='bfill')
            assert_frame_equal(model.to_dataframe(), expected)

            # Test selective filling by different `pandas` methods
            expected = expected.reindex(range(-12, 2 + 1), method='nearest')
            expected.loc[:-11, :'G'] = 0.0
            expected.loc[1:, :'T'] = 0.0

            model = model.reindex(
                range(-12, 2 + 1),
                fill_value=0.0,
                nearest_=model.PARAMETERS,
                ffill_='G',
            )

            assert_frame_equal(model.to_dataframe(), expected)

        # Test alternative `strict` values
        test(self.Model(range(-5, 10 + 1), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2))
        test(
            self.Model(
                range(-5, 10 + 1),
                alpha_1=0.6,
                alpha_2=0.4,
                G=20,
                theta=0.2,
                strict=True,
            )
        )
        test(
            self.Model(
                range(-5, 10 + 1),
                alpha_1=0.6,
                alpha_2=0.4,
                G=20,
                theta=0.2,
                strict=False,
            )
        )

    def test_reindex_strict(self):
        # Check for errors (or not) with different values of `strict`

        # Default (`strict=False`) should raise no errors
        self.Model(
            range(-5, 10 + 1), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2
        ).reindex(range(-2, 2 + 1), A=2)  # A isn't actually a variable in the model

        # Explicit `strict=False` should also raise no errors
        self.Model(
            range(-5, 10 + 1), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2, strict=False
        ).reindex(range(-2, 2 + 1), A=2)

        # Whereas `strict=True` should raise an error
        with self.assertRaises(KeyError):
            self.Model(
                range(-5, 10 + 1),
                alpha_1=0.6,
                alpha_2=0.4,
                G=20,
                theta=0.2,
                strict=True,
            ).reindex(range(-2, 2 + 1), A=2)


class TestTracerMixin(unittest.TestCase):
    SCRIPT = """\
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
    """
    SYMBOLS = fsic.parse_model(SCRIPT)
    SIM = fsic.build_model(SYMBOLS)

    class Model(fsic.extensions.model.TracerMixin, SIM):
        pass

    @unittest.skipIf(not pandas_installed, 'Requires `pandas`')
    def test_solve_trace_all(self):
        # Test `solve()` method with default trace settings

        from pandas.testing import assert_frame_equal

        # Standard model (to compare final results)
        model_without_trace = self.SIM(
            range(10), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2
        )
        model_without_trace.solve()

        # Model with trace
        model_with_trace = self.Model(
            range(10), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2
        )

        # Check all `Trace` objects are empty
        for period in model_with_trace.span:
            with self.subTest(period=period):
                self.assertTrue(model_with_trace['trace', period].is_empty())

        # Solve model
        model_with_trace.solve(trace=True)

        # Check solved `Trace` objects are non-empty
        for _, period in model_with_trace.iter_periods():
            with self.subTest(period=period):
                self.assertFalse(model_with_trace['trace', period].is_empty())

        # Check that the model results are the same, with and without the trace
        assert_frame_equal(
            model_with_trace.to_dataframe(), model_without_trace.to_dataframe()
        )

        # Check the `Trace` results
        results = model_with_trace['trace', 5].to_dataframe()

        self.assertTrue(np.allclose(results['alpha_1'], 0.6))
        self.assertTrue(np.allclose(results['alpha_2'], 0.4))

        self.assertTrue(np.allclose(results['G'], 20.0))
        self.assertTrue(np.allclose(results['theta'], 0.2))

        self.assertTrue(
            np.allclose(
                results.loc[['start', 'before', 0], ['C', 'YD', 'H', 'Y', 'T']], 0.0
            )
        )
        self.assertTrue(
            np.allclose(
                results.loc[1, ['C', 'YD', 'H', 'Y', 'T']],
                [15.59609257, 0, 23.39413886, 35.59609257, 7.119218515],
            )
        )
        self.assertTrue(
            np.allclose(
                results.loc[74, ['C', 'YD', 'H', 'Y', 'T']],
                [48.45402418, 54.76321934, 45.2994266, 68.45402418, 13.69080484],
            )
        )
        self.assertTrue(
            np.allclose(
                results.loc['end', ['C', 'YD', 'H', 'Y', 'T']],
                [48.45402418, 54.76321934, 45.2994266, 68.45402418, 13.69080484],
            )
        )

    @unittest.skipIf(not pandas_installed, 'Requires `pandas`')
    def test_solve_trace_off(self):
        # Test `solve()` method with trace off

        from pandas.testing import assert_frame_equal

        # Standard model (to compare final results)
        model_without_trace = self.SIM(
            range(10), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2
        )
        model_without_trace.solve()

        # Model with trace
        model_with_trace = self.Model(
            range(10), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2
        )

        # Check all `Trace` objects are empty
        for period in model_with_trace.span:
            with self.subTest(period=period):
                self.assertTrue(model_with_trace['trace', period].is_empty())

        # Solve model
        model_with_trace.solve()  # Default is trace off

        # Check all `Trace` objects are still empty
        for period in model_with_trace.span:
            with self.subTest(period=period):
                self.assertTrue(model_with_trace['trace', period].is_empty())

        # Check that the model results are the same, with and without the trace
        assert_frame_equal(
            model_with_trace.to_dataframe(), model_without_trace.to_dataframe()
        )


if __name__ == '__main__':
    unittest.main()
