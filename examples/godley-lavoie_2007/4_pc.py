# -*- coding: utf-8 -*-
"""
4_pc
====
FSIC implementation of Model *PC*, a model of government money with portfolio
choice, from Chapter 4 of Godley and Lavoie (2007).

Godley and Lavoie (2007) analyse Model *PC* beginning from an initial
stationary state. This script first finds that stationary state, matching the
starting values in Zezza's (2006) EViews scripts. The script then analyses a
step-change in the interest rate, as in Godley and Lavoie (2007).

While FSIC only requires NumPy, this example also uses:

* `pandas`, to generate a DataFrame of results using `fsictools`
* `matplotlib`, to replicate, from Godley and Lavoie (2007):
    * Figure 4.3: Evolution of the shares of bills and money balances in the
      portfolio of households, following an increase of 100 points in the rate
      of interest on bills
    * Figure 4.4: Evolution of disposable income and household consumption
      following an increase of 100 points in the rate of interest on bills

Outputs:

1. Replicates Figures 4.3 and 4.4 of Godley and Lavoie (2007), saving the chart
   to 'figures-4.3,4.4.png'

References:

    Godley, W., Lavoie, M. (2007),
    *Monetary economics: an integrated approach to
    credit, money, income, production and wealth*,
    Palgrave Macmillan

    Zezza, G. (2006),
    'EViews macros for building models in *Wynne Godley and Marc Lavoie*
    Monetary Economics: an integrated approach to
    credit, money, income, production and wealth',
    http://gennaro.zezza.it/software/eviews/glch04.php
"""

import matplotlib.pyplot as plt
import pandas as pd

import fsic
import fsictools


# Inline comments give the corresponding equation numbers from Godley and
# Lavoie (2007) - for reference only; FSIC ignores comments, just as Python
# does.
# 'A' suffix indicates a slight amendment to be compatible with the FSIC
# parser.
script = '''
Y = C + G                                                       # 4.1
YD = Y - T + r[-1] * Bh[-1]                                     # 4.2
T = {theta} * (Y + r[-1] * Bh[-1])                              # 4.3
V = V[-1] + YD - C                                              # 4.4
C = {alpha_1} * YD + {alpha_2} * V[-1]                          # 4.5
Hh = V - Bh                                                     # 4.6
Bh = V * ({lambda_0} + {lambda_1} * r - {lambda_2} * (YD / V))  # 4.7A
Bs = Bs[-1] + (G + r[-1] * Bs[-1]) - (T + r[-1] * Bcb[-1])      # 4.8A
Hs = Hs[-1] + Bcb - Bcb[-1]                                     # 4.9A
Bcb = Bs - Bh                                                   # 4.10
r = r_bar                                                       # 4.11
'''

symbols = fsic.parse_model(script)
PC = fsic.build_model(symbols)


if __name__ == '__main__':
    # 1. Find the stationary state of the model from an initial set of
    #    parameter values (from Zezza, 2006)
    starting_from_zero = PC(range(100),  # Enough periods to reach the stationary state
                            alpha_1=0.6, alpha_2=0.4,
                            lambda_0=0.635, lambda_1=5, lambda_2=0.01)

    starting_from_zero.G = 20
    starting_from_zero.theta = 0.2

    starting_from_zero.r_bar = 0.025
    starting_from_zero.r[0] = starting_from_zero.r[0]

    # Solve the model:
    #  - increase the maximum number of iterations for convergence
    #  - Equation 4.7A has a division operation: starting from zero, this may
    #    initially generate a NaN in solution - ignore this because it should
    #    be overwritten in a later iteration
    starting_from_zero.solve(max_iter=200, errors='ignore')

    # Take the results from the last period as the stationary state
    stationary_state = dict(zip(starting_from_zero.names,
                                starting_from_zero.values[:, -1]))


    # 2. Starting from that stationary state, simulate an increase in the
    #    interest rate
    interest_rate_scenario = PC(range(1945, 2010 + 1),
                                **stationary_state)

    # Increase the interest rate by one percentage point beginning in 1960
    interest_rate_scenario['r_bar', 1960:] += 0.01

    # Solve the model with the default solution options (needs fewer iterations
    # from the stationary state and no need to catch NaNs)
    interest_rate_scenario.solve(max_iter=200)


    # 3. Reproduce Figures 4.3 and 4.4 of Godley and Lavoie (2007)
    #    Code copied from:
    #        https://www.christhoung.com/2018/07/08/fsic-gl2007-pc/
    results = fsictools.model_to_dataframe(interest_rate_scenario)

    # Calculate bills and money holdings as a share of household wealth
    results['Sb'] = results.eval('Bh / V')
    results['Sh'] = results.eval('Hh / V')

    # Select same timespan as the original charts
    results_to_plot = results.loc[1950:2000, :]

    # Set up plot areas
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.suptitle('Effects of a step-change in the interest rate from 2.5% to 3.5% in 1960')

    # Recreate Figure 4.3 (shares of bills and money in households' portfolios)
    ax1.set_title("Shares of financial assets in households' portfolios")

    ax1.plot(results_to_plot.index, results_to_plot['Sb'] * 100,
             label='Bills',
             color='#33C3F0')
    ax1.plot(results_to_plot.index, results_to_plot['Sh'] * 100,
             label='Money',
             color='#FF4F2E', linestyle='--')

    ax1.set_xlim(min(results_to_plot.index), max(results_to_plot.index))
    ax1.set_ylabel('%')
    ax1.legend()

    # Recreate Figure 4.4 (disposable income and household consumption)
    ax2.set_title('Disposable income and household consumption')

    ax2.plot(results_to_plot.index, results_to_plot['YD'].values,
             label='Disposable income',
             color='#33C3F0')
    ax2.plot(results_to_plot.index, results_to_plot['C'].values,
             label='Consumption',
             color='#FF4F2E', linestyle='--')

    ax2.set_xlim(min(results_to_plot.index), max(results_to_plot.index))
    ax2.set_ylim(86, 91)
    ax2.legend(loc='lower right')

    plt.savefig('figures-4.3,4.4.png')
