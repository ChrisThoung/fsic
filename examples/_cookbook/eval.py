# -*- coding: utf-8 -*-
"""
eval
====
Example to show how to use `eval()` as an alternative to statements that
operate on the `BaseModel` (or `BaseLinker`) objects themselves. This works in
a similar way to `eval()` in pandas.

**Note that `eval()` uses the Python `eval()` function**.

The example model in this script is a simplified five-equation version of
Godley and Lavoie’s (2007) Model *SIM*.

Reference:

    Godley, W., Lavoie, M. (2007),
    *Monetary economics: an integrated approach to
    credit, money, income, production and wealth*,
    Palgrave Macmillan
"""

import math

import numpy as np

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
symbols = fsic.parse_model(script)
SIM = fsic.build_model(symbols)


if __name__ == '__main__':
    # Setup and solve the model as usual --------------------------------------
    model = SIM(range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4)
    model.G = 20
    model.theta = 0.2

    model.solve()

    # Calculate the stationary state flow of aggregate income -----------------

    # As in Section 3.5 ('Steady-state solutions') of Godley and Lavoie (2007),
    # at its stationary state, output in Model *SIM* is given by Y* = G / theta

    # We could calculate this from the model attributes (which is fine for a
    # simple expression like this)
    print('Y* (Python statements):')
    print(model.G / model.theta)

    # Or we could pass an expression to `eval()` to perform the same. By
    # dropping the `model.` part, this may be more convenient, especially for
    # longer expressions
    print('\nY* (`eval()`):')
    print(model.eval('G / theta'))

    # Two ways of calculating income as a proportion of the steady state
    print('\nY/Y* (Python statements):')
    print((model.Y / (model.G / model.theta)).round(2))

    print('\nY/Y* (`eval()`):')
    print(model.eval('Y / (G / theta)').round(2))

    # Indexing ----------------------------------------------------------------

    # Variable indexing works as usual
    print(
        '\nFirst five years of C/Y (can ignore the divide-by-zero in the first year):'
    )
    print(model.eval('(C/Y)[:5]').round(2))

    print("\nFirst five years of C/Y (use `warnings_='ignore'` to suppress warnings):")

    # The `warnings_` argument can take anything valid for
    # `warnings.simplefilter()` e.g. 'always' to convert warnings to errors:
    # https://docs.python.org/3/library/warnings.html#describing-warning-filters
    print(model.eval('(C/Y)[:5]', warnings_='ignore').round(2))

    print('\nLast five years of Y/Y*:')
    print(model.eval('Y[-5:] / (G / theta)[-5:]').round(2))

    # Use backticks to access elements by period label
    print('\nY / Y* over 1960-70 (`eval()`):')
    print(model.eval('(Y / (G / theta))[`1960`:`1970`]').round(2))

    # The above compares to the usual Python approach below
    print('\nY / Y* over 1960-70 (Python statements):')
    print(
        (
            model['Y', 1960:1970] / (model['G', 1960:1970] / model['theta', 1960:1970])
        ).round(2)
    )

    # Other operations --------------------------------------------------------

    # Pass a dictionary of functions and/or modules to make other operations
    # available to `eval()`
    print('\nOther operations:')
    print(
        model.eval(
            'round(log(C[1]), 2), np.log(C[1:4]).round(2)',
            locals={'log': math.log, 'np': np},
        )
    )
