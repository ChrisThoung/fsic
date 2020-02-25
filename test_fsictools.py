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


found_pandas = True

try:
    import pandas as pd
except ModuleNotFoundError:
    found_pandas = False
    warnings.warn('`pandas` not found: Skipping `pandas`-based tests')


if found_pandas:

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
            result = fsictools.model_to_dataframe(self.MODEL(range(5)))
            expected = pd.DataFrame({
                'Y': 0.0,
                'C': 0.0,
                'G': 0.0,
                'status': '-',
                'iterations': -1
            }, index=range(5))

            pd.testing.assert_frame_equal(result, expected)


found_networkx = True

try:
    import networkx as nx
except ModuleNotFoundError:
    found_networkx = False
    warnings.warn('`networkx` not found: Skipping `networkx`-based tests')


if found_networkx:

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


found_sympy = True

try:
    import sympy
except ModuleNotFoundError:
    found_sympy = False
    warnings.warn('`sympy` not found: Skipping `sympy`-based tests')


if found_sympy:

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
