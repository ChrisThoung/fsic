# -*- coding: utf-8 -*-
"""
polars_support
==============
Examples demonstrating Polars compatibility.

The example model in this script is a simplified five-equation version of
Godley and Lavoie’s (2007) Model *SIM*.

Reference:

    Godley, W., Lavoie, M. (2007),
    *Monetary economics: an integrated approach to
    credit, money, income, production and wealth*,
    Palgrave Macmillan
"""

import polars as pl

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
    # Using the `from_dataframe()` class method -------------------------------

    # Initialise a model with the usual parameters and assumptions
    initial_values = SIM(
        range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2
    )

    # Save the model in this state to disk (giving the index a name before saving)
    initial_values.to_dataframe().rename_axis(index='time').to_csv('sim_inputs.csv')

    # Read the CSV file back in using Polars
    input_data = pl.read_csv('sim_inputs.csv')

    # Polars DataFrames have no concept of an index which, by default, `fsic`
    # uses to set a model's span
    # In these cases, set the index explicitly with the `index_col` argument
    model = SIM.from_dataframe(input_data, index_col='time')

    # Solve the model and check its outputs
    model.solve()
    print(model.to_dataframe())
