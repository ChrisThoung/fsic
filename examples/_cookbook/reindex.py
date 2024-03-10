# -*- coding: utf-8 -*-
"""
reindex
=======
Examples of how to use the `BaseModel.reindex()` method to change the span of a
model object.

The example model in this script is a simplified five-equation version of
Godley and Lavoie’s (2007) Model *SIM*.

Reference:

    Godley, W., Lavoie, M. (2007),
    *Monetary economics: an integrated approach to
    credit, money, income, production and wealth*,
    Palgrave Macmillan
"""

import fsic


# Define the example model
script = """
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
"""

# Parse the script and create a class definition
SIM = fsic.build_model(fsic.parse_model(script))


if __name__ == '__main__':
    # Create a model object over 1945-60, set its inputs and solve
    model = SIM(range(1945, 1960 + 1), alpha_1=0.6, alpha_2=0.4)

    model.G = 20
    model.theta = 0.2

    model.solve()
    print(model.to_dataframe().round(2))

    # Use `reindex()` to create a copy with a different span
    # By default, variables are filled for new periods (NaN for floats, 0 for
    # ints, '' for strs and False for bools)
    # The `status` and `iterations` attributes are filled with '-' and -1,
    # respectively (but this can be over-ridden)
    copied_with_nans = model.reindex(range(1955, 1970 + 1))
    print(copied_with_nans.to_dataframe().round(2))

    # Use `fill_value` to change the default for new periods
    copied_with_blanket_value = model.reindex(range(1950, 1965 + 1), fill_value=0)
    print(copied_with_blanket_value.to_dataframe().round(2))

    # Use variable names as keywords to set individual fill values
    copied_and_filled = model.reindex(
        range(1960, 1975 + 1), fill_value=0, alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2
    )
    print(copied_and_filled.to_dataframe().round(2))

    copied_and_filled.solve()
    print(copied_and_filled.to_dataframe().round(2))
