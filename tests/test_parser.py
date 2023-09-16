# -*- coding: utf-8 -*-
"""
test_parser
===========
Test suite for fsic model/equation parser.
"""

import unittest

import fsic


class TestParserPeriodIndexing(unittest.TestCase):

    def test_period_index_verbatim(self):
        # Check that the user can specify integer periods by enclosing them in
        # backticks
        test_input = 'Cb = C[`2000`]'
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


if __name__ == '__main__':
    unittest.main()
