# -*- coding: utf-8 -*-
"""
pandas_indexing
===============
Example of how to use a `pandas` `PeriodIndex` for the span of a model. This
might be useful when dealing with models of higher-than-annual frequency.

The example model in this script is a simplified five-equation version of
Godley and Lavoie’s (2007) Model *SIM*.

Reference:

    Godley, W., Lavoie, M. (2007),
    *Monetary economics: an integrated approach to
    credit, money, income, production and wealth*,
    Palgrave Macmillan
"""

import pandas as pd

import fsic


# Define the example model
script = """
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
"""

# Parse the script and generate a class definition
SIM = fsic.build_model(fsic.parse_model(script))


if __name__ == '__main__':
    # At instantiation, use a `PeriodIndex` as the first argument (here,
    # generated by `period_range()`), in place of a typical sequence variable
    model = SIM(
        pd.period_range(
            start='1990Q1', end='2000Q4', freq='Q'
        ),  # Quarterly, 1990Q1-2000Q4
        alpha_1=0.6,
        alpha_2=0.4,
    )

    # Period-level indexing works as usual
    model['G', '1990Q2':] = 20
    model['theta', '1990Q2':] = 0.2

    # With a `PeriodIndex` we can also get/set entire years of values
    model['G', '1995'] = 25
    model['G', '1996':] = 30
    model['G', '1998':'1999'] = 35

    # Model solution works as usual
    model.solve()

    # Note that, as a design decision, arguments to the `solve_()` methods must
    # still specify the exact period (e.g. '2000Q1') rather than a year
    # (e.g. '2000')
    # Uncomment the line below to see this

    # model.solve_period('2000')  # Raises a `KeyError`
