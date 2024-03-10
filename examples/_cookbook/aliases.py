# -*- coding: utf-8 -*-
"""
aliases
=======
Example to show how to use aliases to bind further names to model
variables. See the Notes section of the `AliasMixin` docstring for more
information.

The example model in this script is a simplified five-equation version of
Godley and Lavoie’s (2007) Model *SIM*.

Reference:

    Godley, W., Lavoie, M. (2007),
    *Monetary economics: an integrated approach to
    credit, money, income, production and wealth*,
    Palgrave Macmillan
"""

import numpy as np

from fsic.extensions import AliasMixin
import fsic


# Setup a model class definition as usual
script = """
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
"""
symbols = fsic.parse_model(script)
SIM = fsic.build_model(symbols)


# Create a version of the class with aliases
class SIMAlias(AliasMixin, SIM):
    ALIASES = {
        # Add 'GDP' as an alternative name for 'Y'
        'GDP': 'Y',
        # Also add other names for output (and note how aliases can be chained
        # to point to other aliases)
        'expenditure': 'output',
        'output': 'income',
        'income': 'Y',
        # Other aliases
        'mpc_income': 'alpha_1',
        'mpc_wealth': 'alpha_2',
        'income_tax_rate': 'theta',
    }

    # Set a list of preferred names for tabling, preferring 'GDP' to 'Y' but
    # retaining the original 'alpha_1' and 'alpha_2' labels. All others resolve
    # to their aliases in the order Python handles the contents of the
    # `ALIASES` dictionary above (which depends on the Python version).
    PREFERRED_NAMES = ['GDP', 'alpha_1', 'alpha_2']


if __name__ == '__main__':
    # Run Model *SIM*  as usual -----------------------------------------------
    model = SIM(range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4)

    model.G = 20
    model.theta = 0.2

    model.solve()

    # Run the same model with aliases -----------------------------------------
    # Aliases are completely interchangeable with the original variable names
    model_with_aliases = SIMAlias(
        range(1945, 2010 + 1),
        alpha_1=0.6,     # Parameter as listed in Godley and Lavoie (2007)
        mpc_wealth=0.4,  # Alias for `alpha_2`
    )  # fmt: skip

    model_with_aliases.G = 20  # Original variable names work as usual
    model_with_aliases.income_tax_rate = 0.2  # Aliases also work

    model_with_aliases.solve()

    # Check (show) results match ----------------------------------------------
    assert np.allclose(model.values, model_with_aliases.values)

    # Print results as DataFrames, with `use_aliases=True` to use the aliases
    print(model_with_aliases.to_dataframe().round(2))
    print(model_with_aliases.to_dataframe(use_aliases=True).round(2))
