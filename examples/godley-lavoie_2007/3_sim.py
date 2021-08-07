# -*- coding: utf-8 -*-
"""
3_sim
=====
FSIC implementation of Model *SIM* (the SIMplest model with government money)
from Chapter 3 of Godley and Lavoie (2007).

While FSIC only requires NumPy, this example also uses:

* `pandas`, to generate a DataFrame of final results using `fsic.tools`
* `matplotlib`, to replicate Figure 3.2 ('Disposable income and consumption
  starting from scratch') of Godley and Lavoie (2007)

Outputs:

1. Prints a complete set of results to the terminal, as a table (`pandas`
   DataFrame)
2. Replicates Figure 3.2 of Godley and Lavoie (2007), saving the chart to
   'figure-3.2.png'

Reference:

    Godley, W., Lavoie, M. (2007),
    *Monetary economics: an integrated approach to
    credit, money, income, production and wealth*,
    Palgrave Macmillan
"""

import matplotlib.pyplot as plt
import pandas as pd

import fsic


# Inline comments give the corresponding equation numbers from Godley and
# Lavoie (2007) - for reference only; FSIC ignores these.
# 'A' suffix indicates a slight amendment to be compatible with the FSIC
# parser.
script = '''
# Keynesian/Kaleckian quantity adjustment equalises demand and supply
C_s = C_d                                   # 3.1
G_s = G_d                                   # 3.2
T_s = T_d                                   # 3.3
N_s = N_d                                   # 3.4

# Disposable income
YD = W * N_s - T_s                          # 3.5

# Taxes
T_d = {theta} * W * N_s                     # 3.6

# Household final consumption expenditure
C_d = {alpha_1} * YD + {alpha_2} * H_h[-1]  # 3.7

# Money: Government liability
H_s = H_s[-1] + G_d - T_d                   # 3.8A

# Money: Household asset
H_h = H_h[-1] + YD - C_d                    # 3.9A

# National income
Y = C_s + G_s                               # 3.10

# Labour demand
N_d = Y / W                                 # 3.11
'''

symbols = fsic.parse_model(script)
SIM = fsic.build_model(symbols)


if __name__ == '__main__':
    # Create a new model object and set values
    model = SIM(range(1957, 2001 + 1))

    model.alpha_1 = 0.6  # Propensity to consume out of current disposable income
    model.alpha_2 = 0.4  # Propensity to consume out of previous period's wealth

    model.W = 1  # Wages

    # Models implement container methods to index values by variable name
    # (e.g. 'G_d') and period (e.g. 1960 onwards)
    model['G_d', 1960:] = 20  # Government expenditure

    # Alternatively, access variables as attributes and index as you would any
    # other Python sequence
    model.theta[3:] = 0.2  # Income tax rate

    # Solve the model with an increased maximum number of iterations, to ensure
    # convergence
    model.solve(max_iter=350)

    # Store results to a DataFrame and print to the screen
    results = fsic.tools.model_to_dataframe(model)
    print(results.round(1))

    # Replicate Figure 3.2 of Godley and Lavoie (2007): Disposable income and
    # consumption starting from scratch
    plt.figure()

    results['YD'].plot(color='#33C3F0', label='Disposable income (YD)')
    results['C_d'].plot(color='#FF4F2E', label='Consumption (C)', linestyle='--')

    # Horizontal line for stationary-state levels of YD and C_d
    plt.plot(results.index, [80] * len(results.index), color='k', linewidth=1)

    plt.xlim(min(results.index), max(results.index))
    plt.xticks(results.index[::4], results.index[::4])

    plt.ylim(0, 90)
    plt.yticks([0, 15, 30, 45, 60, 75, 80], [0, 15, 30, 45, 60, 75, 80])

    plt.legend()

    plt.savefig('figure-3.2.png')
