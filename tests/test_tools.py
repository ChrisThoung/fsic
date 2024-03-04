# -*- coding: utf-8 -*-
"""
test_tools
==========
Test suite for supporting fsic tools.
"""

import unittest

import fsic


pandas_installed = True

try:
    import pandas as pd
except ModuleNotFoundError:
    pandas_installed = False


@unittest.skipIf(not pandas_installed, 'Requires `pandas`')
class TestPandasFunctions(unittest.TestCase):
    SYMBOLS = fsic.parse_model(
        'Y = C + float(G)'
    )  # `float` has no material effect: Added just to have a function in the example
    MODEL = fsic.build_model(SYMBOLS)

    def test_symbols_to_dataframe(self):
        result = fsic.tools.symbols_to_dataframe(self.SYMBOLS)
        expected = pd.DataFrame(
            {
                'name': ['Y', 'C', 'float', 'G'],
                'type': [
                    fsic.parser.Type.ENDOGENOUS,
                    fsic.parser.Type.EXOGENOUS,
                    fsic.parser.Type.FUNCTION,
                    fsic.parser.Type.EXOGENOUS,
                ],
                'lags': [0, 0, None, 0],
                'leads': [0, 0, None, 0],
                'equation': ['Y[t] = C[t] + float(G[t])', None, None, None],
                'code': [
                    'self._Y[t] = self._C[t] + float(self._G[t])',
                    None,
                    None,
                    None,
                ],
            }
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_dataframe_to_symbols(self):
        # Check by way of a roundtrip: symbols -> DataFrame -> symbols
        result = fsic.tools.symbols_to_dataframe(self.SYMBOLS)
        expected = pd.DataFrame(
            {
                'name': ['Y', 'C', 'float', 'G'],
                'type': [
                    fsic.parser.Type.ENDOGENOUS,
                    fsic.parser.Type.EXOGENOUS,
                    fsic.parser.Type.FUNCTION,
                    fsic.parser.Type.EXOGENOUS,
                ],
                'lags': [0, 0, None, 0],
                'leads': [0, 0, None, 0],
                'equation': ['Y[t] = C[t] + float(G[t])', None, None, None],
                'code': [
                    'self._Y[t] = self._C[t] + float(self._G[t])',
                    None,
                    None,
                    None,
                ],
            }
        )

        # Initial (i.e. pre-)check only
        pd.testing.assert_frame_equal(result, expected)

        # Check by value
        self.assertEqual(fsic.tools.dataframe_to_symbols(result), self.SYMBOLS)
        self.assertEqual(fsic.tools.dataframe_to_symbols(expected), self.SYMBOLS)

        # Check by string representation
        for before, after in zip(self.SYMBOLS, fsic.tools.dataframe_to_symbols(result)):
            with self.subTest(symbol=before):
                self.assertEqual(str(before), str(after))
                self.assertEqual(repr(before), repr(after))

    def test_model_to_dataframe(self):
        model = self.MODEL(range(5))

        # Check list of names is unchanged
        self.assertEqual(model.names, model.NAMES)

        result = fsic.tools.model_to_dataframe(model)
        expected = pd.DataFrame(
            {'Y': 0.0, 'C': 0.0, 'G': 0.0, 'status': '-', 'iterations': -1},
            index=range(5),
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_model_to_dataframe_no_status(self):
        model = self.MODEL(range(5))

        # Check list of names is unchanged
        self.assertEqual(model.names, model.NAMES)

        expected = fsic.tools.model_to_dataframe(model).drop('status', axis='columns')
        result = fsic.tools.model_to_dataframe(model, status=False)

        pd.testing.assert_frame_equal(result, expected)

    def test_model_to_dataframe_no_iterations(self):
        model = self.MODEL(range(5))

        # Check list of names is unchanged
        self.assertEqual(model.names, model.NAMES)

        expected = fsic.tools.model_to_dataframe(model).drop(
            'iterations', axis='columns'
        )
        result = fsic.tools.model_to_dataframe(model, iterations=False)

        pd.testing.assert_frame_equal(result, expected)

    def test_model_to_dataframe_no_status_or_iterations(self):
        model = self.MODEL(range(5))

        # Check list of names is unchanged
        self.assertEqual(model.names, model.NAMES)

        expected = fsic.tools.model_to_dataframe(model).drop(
            ['status', 'iterations'], axis='columns'
        )
        result = fsic.tools.model_to_dataframe(model, status=False, iterations=False)

        pd.testing.assert_frame_equal(result, expected)

    def test_model_to_dataframe_additional_variables(self):
        # Check that extending the model with extra variables carries through
        # to the results DataFrame (including preserving variable types)
        model = self.MODEL(range(5))

        # Check list of names is unchanged
        self.assertEqual(model.names, model.NAMES)

        model.add_variable('I', 0, dtype=int)
        model.add_variable('J', 0)
        model.add_variable('K', 0, dtype=float)
        model.add_variable('L', False, dtype=bool)

        # Check list of names is now changed
        self.assertEqual(model.names, model.NAMES + ['I', 'J', 'K', 'L'])

        result = fsic.tools.model_to_dataframe(model)
        expected = pd.DataFrame({
            'Y': 0.0,
            'C': 0.0,
            'G': 0.0,
            'I': 0,      # int
            'J': 0.0,    # float
            'K': 0.0,    # float (forced)
            'L': False,  # bool
            'status': '-',
            'iterations': -1
        }, index=range(5))  # fmt: skip

        pd.testing.assert_frame_equal(result, expected)

    def test_model_to_dataframe_internal_variables(self):
        # Check that `model_to_dataframe()` selectively returns internal
        # variables (denoted by a leading underscore)
        model = self.MODEL(range(5))

        # Check list of names is unchanged
        self.assertEqual(model.names, model.NAMES)

        expected = pd.DataFrame(
            {'Y': 0.0, 'C': 0.0, 'G': 0.0, 'status': '-', 'iterations': -1},
            index=range(5),
        )

        pd.testing.assert_frame_equal(fsic.tools.model_to_dataframe(model), expected)

        # Add an internal variable
        model.add_variable('_A', -1, dtype=int)

        # By default, exclude internal variables from the DataFrame
        pd.testing.assert_frame_equal(fsic.tools.model_to_dataframe(model), expected)

        # Use `include_internal=True` to also return internal variables
        expected_internal = expected.copy()
        expected_internal.insert(3, '_A', -1)

        pd.testing.assert_frame_equal(
            fsic.tools.model_to_dataframe(model, include_internal=True),
            expected_internal,
        )

    def test_model_to_dataframe_core(self):
        model = self.MODEL(range(5))

        # Check list of names is unchanged
        self.assertEqual(model.names, model.NAMES)

        result = model.to_dataframe()
        expected = pd.DataFrame(
            {'Y': 0.0, 'C': 0.0, 'G': 0.0, 'status': '-', 'iterations': -1},
            index=range(5),
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_model_to_dataframe_core_no_status(self):
        model = self.MODEL(range(5))

        # Check list of names is unchanged
        self.assertEqual(model.names, model.NAMES)

        result = model.to_dataframe(status=False)
        expected = pd.DataFrame(
            {'Y': 0.0, 'C': 0.0, 'G': 0.0, 'iterations': -1}, index=range(5)
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_model_to_dataframe_core_no_iterations(self):
        model = self.MODEL(range(5))

        # Check list of names is unchanged
        self.assertEqual(model.names, model.NAMES)

        result = model.to_dataframe(iterations=False)
        expected = pd.DataFrame(
            {
                'Y': 0.0,
                'C': 0.0,
                'G': 0.0,
                'status': '-',
            },
            index=range(5),
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_model_to_dataframe_core_no_status_or_iterations(self):
        model = self.MODEL(range(5))

        # Check list of names is unchanged
        self.assertEqual(model.names, model.NAMES)

        result = model.to_dataframe(status=False, iterations=False)
        expected = pd.DataFrame(
            {
                'Y': 0.0,
                'C': 0.0,
                'G': 0.0,
            },
            index=range(5),
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_model_to_dataframe_additional_variables_core(self):
        # Check that extending the model with extra variables carries through
        # to the results DataFrame (including preserving variable types)
        model = self.MODEL(range(5))

        # Check list of names is unchanged
        self.assertEqual(model.names, model.NAMES)

        model.add_variable('I', 0, dtype=int)
        model.add_variable('J', 0)
        model.add_variable('K', 0, dtype=float)
        model.add_variable('L', False, dtype=bool)

        # Check list of names is now changed
        self.assertEqual(model.names, model.NAMES + ['I', 'J', 'K', 'L'])

        result = model.to_dataframe()
        expected = pd.DataFrame({
            'Y': 0.0,
            'C': 0.0,
            'G': 0.0,
            'I': 0,      # int
            'J': 0.0,    # float
            'K': 0.0,    # float (forced)
            'L': False,  # bool
            'status': '-',
            'iterations': -1
        }, index=range(5))  # fmt: skip

        pd.testing.assert_frame_equal(result, expected)

    def test_linker_to_dataframe(self):
        Submodel = fsic.build_model(fsic.parse_model('Y = C + I + G + X - M'))

        model = fsic.BaseLinker(
            {
                'A': Submodel(range(1990, 2005 + 1)),
                'B': Submodel(range(1990, 2005 + 1)),
                'C': Submodel(range(1990, 2005 + 1)),
            },
            name='test',
        )
        model.add_variable('D', 0.0)

        linker_results = fsic.tools.model_to_dataframe(model)

        pd.testing.assert_frame_equal(
            linker_results,
            pd.DataFrame(
                {
                    'D': 0.0,
                    'status': '-',
                    'iterations': -1,
                },
                index=range(1990, 2005 + 1),
            ),
        )

    def test_linker_to_dataframe_core(self):
        Submodel = fsic.build_model(fsic.parse_model('Y = C + I + G + X - M'))

        model = fsic.BaseLinker(
            {
                'A': Submodel(range(1990, 2005 + 1)),
                'B': Submodel(range(1990, 2005 + 1)),
                'C': Submodel(range(1990, 2005 + 1)),
            },
            name='test',
        )
        model.add_variable('D', 0.0)

        linker_results = model.to_dataframe()

        pd.testing.assert_frame_equal(
            linker_results,
            pd.DataFrame(
                {
                    'D': 0.0,
                    'status': '-',
                    'iterations': -1,
                },
                index=range(1990, 2005 + 1),
            ),
        )

    def test_linker_to_dataframes(self):
        Submodel = fsic.build_model(fsic.parse_model('Y = C + I + G + X - M'))

        model = fsic.BaseLinker(
            {
                'A': Submodel(range(1990, 2005 + 1)),
                'B': Submodel(range(1990, 2005 + 1)),
                'C': Submodel(range(1990, 2005 + 1)),
            },
            name='test',
        )
        model.add_variable('D', 0.0)

        results = fsic.tools.linker_to_dataframes(model)

        pd.testing.assert_frame_equal(
            results['test'],
            pd.DataFrame(
                {
                    'D': 0.0,
                    'status': '-',
                    'iterations': -1,
                },
                index=range(1990, 2005 + 1),
            ),
        )

        expected = pd.DataFrame({x: 0.0 for x in 'YCIGXM'}, index=range(1990, 2005 + 1))
        expected['status'] = '-'
        expected['iterations'] = -1

        for name, submodel in model.submodels.items():
            with self.subTest(submodel=name):
                pd.testing.assert_frame_equal(results[name], expected)

    def test_linker_to_dataframes_core(self):
        Submodel = fsic.build_model(fsic.parse_model('Y = C + I + G + X - M'))

        model = fsic.BaseLinker(
            {
                'A': Submodel(range(1990, 2005 + 1)),
                'B': Submodel(range(1990, 2005 + 1)),
                'C': Submodel(range(1990, 2005 + 1)),
            },
            name='test',
        )
        model.add_variable('D', 0.0)

        results = model.to_dataframes()

        pd.testing.assert_frame_equal(
            results['test'],
            pd.DataFrame(
                {
                    'D': 0.0,
                    'status': '-',
                    'iterations': -1,
                },
                index=range(1990, 2005 + 1),
            ),
        )

        expected = pd.DataFrame({x: 0.0 for x in 'YCIGXM'}, index=range(1990, 2005 + 1))
        expected['status'] = '-'
        expected['iterations'] = -1

        for name, submodel in model.submodels.items():
            with self.subTest(submodel=name):
                pd.testing.assert_frame_equal(results[name], expected)


networkx_installed = True

try:
    import networkx as nx
except ModuleNotFoundError:
    networkx_installed = False


@unittest.skipIf(not networkx_installed, 'Requires `networkx`')
class TestNetworkXFunctions(unittest.TestCase):
    SYMBOLS = fsic.parse_model('Y = C + G')

    def test_symbols_to_graph(self):
        result = fsic.tools.symbols_to_graph(self.SYMBOLS)

        expected = nx.DiGraph()
        expected.add_nodes_from(['Y[t]'], equation='Y[t] = C[t] + G[t]')
        expected.add_edge('C[t]', 'Y[t]')
        expected.add_edge('G[t]', 'Y[t]')

        self.assertEqual(result.nodes, expected.nodes)
        self.assertEqual(result.edges, expected.edges)


sympy_installed = True

try:
    import sympy
except ModuleNotFoundError:
    sympy_installed = False


@unittest.skipIf(not sympy_installed, 'Requires `SymPy`')
class TestSympyFunctions(unittest.TestCase):
    def test_symbols_to_sympy(self):
        result = fsic.tools.symbols_to_sympy(fsic.parse_model('Y = C + G'))
        expected = {
            sympy.Symbol('Y'): sympy.Eq(sympy.Symbol('Y'), sympy.sympify('C + G')),
        }

        self.assertEqual(result, expected)

    def test_symbols_to_sympy_singleton(self):
        # Test (forced) conversion of 'S' to a SymPy Symbol (rather than a
        # Singleton)
        result = fsic.tools.symbols_to_sympy(fsic.parse_model('S = YD - C'))
        expected = {
            sympy.Symbol('S'): sympy.Eq(sympy.Symbol('S'), sympy.sympify('YD - C')),
        }

        self.assertEqual(result, expected)

    def test_symbols_to_sympy_singleton_and_imaginary(self):
        # Also test (forced) conversion of 'I' to a SymPy Symbol (rather
        # than an imaginary number)
        result = fsic.tools.symbols_to_sympy(fsic.parse_model('I = S'))
        expected = {
            sympy.Symbol('I'): sympy.Eq(sympy.Symbol('I'), sympy.Symbol('S')),
        }

        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
