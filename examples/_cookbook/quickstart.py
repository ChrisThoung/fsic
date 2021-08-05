# -*- coding: utf-8 -*-
"""
quickstart
==========
Copy of the quickstart code from the README file, implementing Model *SIM* (the
SIMplest model with government money) from Chapter 3 of Godley and Lavoie
(2007).

Reference:

    Godley, W., Lavoie, M. (2007),
    *Monetary economics: an integrated approach to
    credit, money, income, production and wealth*,
    Palgrave Macmillan
"""

import fsic


script = '''
# Keynesian/Kaleckian quantity adjustment equalises demand and supply
C_s = C_d  # Household final consumption expenditure
G_s = G_d  # Government expenditure
T_s = T_d  # Taxes
N_s = N_d  # Labour

# Disposable income
YD = (W * N_s) - T_s

# Taxes
T_d = {theta} * W * N_s

# Household final consumption expenditure
C_d = {alpha_1} * YD + {alpha_2} * H_h[-1]

# Money: Government liability
H_s = H_s[-1] + G_d - T_d

# Money: Household asset
H_h = H_h[-1] + YD - C_d

# National income
Y = C_s + G_s

# Labour demand
N_d = Y / W
'''

# Parse `script` to identify the constituent symbols (endogenous/exogenous
# variables, parameters and equations)
symbols = fsic.parse_model(script)

# Embed the economic logic in a new Python class
SIM = fsic.build_model(symbols)

# Initialise a new model instance over 1945-2010
model = SIM(range(1945, 2010 + 1))

# Set parameters and input values
model.alpha_1 = 0.6  # Propensity to consume out of current disposable income
model.alpha_2 = 0.4  # Propensity to consume out of lagged wealth

model['W'] = 1       # Wages (alternative variable access by name rather than attribute)

# Exogenous government expenditure beginning in the second period
model.G_d[1:] = 20  # Regular list/NumPy-like indexing by position

# Income tax rate of 20% beginning in the second period
model['theta', 1946:] = 0.2  # `pandas`-like indexing by label: variable and period

# Solve the model
# (`max_iter` increases the maximum number of iterations, to ensure
# convergence)
model.solve(max_iter=350)


# `model.status` lists the solution state of each period as one of:
#
# * `-` : still to be solved (no attempt made)
# * `.` : solved successfully
# * `F` : failed to solve
#
# The solved model's values are in `model.values`, which is a 2D NumPy array in
# which each item of the:
#
# * first axis ('rows') is a *variable*, with corresponding labels in `model.names`
# * second axis ('columns') is a *period*, with corresponding labels in
#   `model.span`
#
# If you've installed [`pandas`](https://pandas.pydata.org/), you can convert the
# contents of the model to a DataFrame for inspection, using
# `model_to_dataframe()`, from the `fsic.tools` module:

import fsic.tools

results = fsic.tools.model_to_dataframe(model)

print(results.round(2))
