# -*- coding: utf-8 -*-
"""
test_fsictools
==============
Test suite for supporting FSIC tools.
"""

import unittest
import warnings

import fsic
import fsictools


pandas_installed = True

try:
    import pandas as pd
except ModuleNotFoundError:
    pandas_installed = False


@unittest.skipIf(not pandas_installed, 'Requires `pandas`')
class TestPandasFunctions(unittest.TestCase):

    SYMBOLS = fsic.parse_model('Y = C + G')
    MODEL = fsic.build_model(SYMBOLS)

    def test_symbols_to_dataframe(self):
        result = fsictools.symbols_to_dataframe(self.SYMBOLS)
        expected = pd.DataFrame({
            'name': ['Y', 'C', 'G'],
            'type': [fsic.Type.ENDOGENOUS, fsic.Type.EXOGENOUS, fsic.Type.EXOGENOUS],
            'lags': 0,
            'leads': 0,
            'equation': ['Y[t] = C[t] + G[t]', None, None],
            'code': ['self._Y[t] = self._C[t] + self._G[t]', None, None],
        })

        pd.testing.assert_frame_equal(result, expected)

    def test_model_to_dataframe(self):
        model = self.MODEL(range(5))

        # Check list of names is unchanged
        self.assertEqual(model.names, model.NAMES)

        result = fsictools.model_to_dataframe(model)
        expected = pd.DataFrame({
            'Y': 0.0,
            'C': 0.0,
            'G': 0.0,
            'status': '-',
            'iterations': -1
        }, index=range(5))

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

        result = fsictools.model_to_dataframe(model)
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
        }, index=range(5))

        pd.testing.assert_frame_equal(result, expected)

    def test_linker_to_dataframes(self):
        Submodel = fsic.build_model(fsic.parse_model('Y = C + I + G + X - M'))

        model = fsic.BaseLinker({
            'A': Submodel(range(1990, 2005 + 1)),
            'B': Submodel(range(1990, 2005 + 1)),
            'C': Submodel(range(1990, 2005 + 1)),
        })

        results = fsictools.linker_to_dataframes(model)

        pd.testing.assert_frame_equal(results['_'],
                                      pd.DataFrame({'status': '-',
                                                    'iterations': -1, },
                                                   index=range(1990, 2005 + 1)))

        expected = pd.DataFrame({x: 0.0 for x in 'YCIGXM'},
                                index=range(1990, 2005 + 1))
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
        result = fsictools.symbols_to_graph(self.SYMBOLS)

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
        result = fsictools.symbols_to_sympy(fsic.parse_model('Y = C + G'))
        expected = {
            sympy.Symbol('Y'): sympy.Eq(sympy.Symbol('Y'), sympy.sympify('C + G')),
        }

        self.assertEqual(result, expected)

    def test_symbols_to_sympy_singleton(self):
        # Test (forced) conversion of 'S' to a SymPy Symbol (rather than a
        # Singleton)
        result = fsictools.symbols_to_sympy(fsic.parse_model('S = YD - C'))
        expected = {
            sympy.Symbol('S'): sympy.Eq(sympy.Symbol('S'), sympy.sympify('YD - C')),
        }

        self.assertEqual(result, expected)

    def test_symbols_to_sympy_singleton_and_imaginary(self):
        # Also test (forced) conversion of 'I' to a SymPy Symbol (rather
        # than an imaginary number)
        result = fsictools.symbols_to_sympy(fsic.parse_model('I = S'))
        expected = {
            sympy.Symbol('I'): sympy.Eq(sympy.Symbol('I'), sympy.Symbol('S')),
        }

        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
