# -*- coding: utf-8 -*-
"""
test_parser
===========
Test suite for fsic model/equation parser.
"""

import unittest

import numpy as np

import fsic


class TestParserPeriodIndexing(unittest.TestCase):

    def test_period_index_verbatim(self):
        # Check that the user can specify integer periods by enclosing them in
        # backticks
        test_input = 'Cb = C[`2000`]'

        # Check symbols are generated correctly
        expected = [
            fsic.parser.Symbol(name='Cb',
                               type=fsic.parser.Type.ENDOGENOUS,
                               lags=0,
                               leads=0,
                               equation="Cb[t] = C[2000]", code="self._Cb[t] = self['C', 2000]"),
            fsic.parser.Symbol(name='C',
                               type=fsic.parser.Type.EXOGENOUS,
                               lags=0,
                               leads=0,
                               equation=None,
                               code=None)
        ]

        symbols = fsic.parse_model(test_input)
        self.assertEqual(symbols, expected)

        # Create a model class definition
        Model = fsic.build_model(symbols)

        # Solve the model
        model = Model(range(2000, 2005 + 1))

        model['C', 2000] = 10
        model.solve(start=2001)

        # Check results are as expected
        self.assertTrue(np.allclose(
            model.Cb,
            np.array([0.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        ))
        self.assertTrue(np.allclose(
            model.C,
            np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ))

    def test_period_index_verbatim_str_single_quote(self):
        # Check that the user can specify periods by enclosing them in
        # backticks (single quote string variant)
        test_input = "Cb = C[`'2000'`]"

        # Check symbols are generated correctly
        expected = [
            fsic.parser.Symbol(name='Cb',
                               type=fsic.parser.Type.ENDOGENOUS,
                               lags=0,
                               leads=0,
                               equation="Cb[t] = C['2000']", code="self._Cb[t] = self['C', '2000']"),
            fsic.parser.Symbol(name='C',
                               type=fsic.parser.Type.EXOGENOUS,
                               lags=0,
                               leads=0,
                               equation=None,
                               code=None)
        ]

        symbols = fsic.parse_model(test_input)
        self.assertEqual(symbols, expected)

        # Create a model class definition
        Model = fsic.build_model(symbols)

        # Solve the model
        model = Model(list(map(str, range(2000, 2005 + 1))))

        model['C', '2000'] = 10
        model.solve(start='2001')

        # Check results are as expected
        self.assertTrue(np.allclose(
            model.Cb,
            np.array([0.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        ))
        self.assertTrue(np.allclose(
            model.C,
            np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ))

    def test_period_index_verbatim_str_double_quote(self):
        # Check that the user can specify periods by enclosing them in
        # backticks (double quote string variant)
        test_input = 'Cb = C[`"2000"`]'

        # Check symbols are generated correctly
        expected = [
            fsic.parser.Symbol(name='Cb',
                               type=fsic.parser.Type.ENDOGENOUS,
                               lags=0,
                               leads=0,
                               # TODO: Standardise on either single or double quotes?
                               equation='Cb[t] = C["2000"]', code="self._Cb[t] = self['C', \"2000\"]"),
            fsic.parser.Symbol(name='C',
                               type=fsic.parser.Type.EXOGENOUS,
                               lags=0,
                               leads=0,
                               equation=None,
                               code=None)
        ]

        symbols = fsic.parse_model(test_input)
        self.assertEqual(symbols, expected)

        # Create a model class definition
        Model = fsic.build_model(symbols)

        # Solve the model
        model = Model(list(map(str, range(2000, 2005 + 1))))

        model['C', '2000'] = 10
        model.solve(start='2001')

        # Check results are as expected
        self.assertTrue(np.allclose(
            model.Cb,
            np.array([0.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        ))
        self.assertTrue(np.allclose(
            model.C,
            np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ))


if __name__ == '__main__':
    unittest.main()
