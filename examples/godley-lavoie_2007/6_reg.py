# -*- coding: utf-8 -*-
"""
6_reg
=====
FSIC implementation of Model *REG*, a model of a two-region economy with a
single government, fiscal and monetary system, and currency, from Chapter 6 of
Godley and Lavoie (2007). This model disaggregates Model *PC* (see '4_pc.py')
into two regions: 'North' and 'South'. Parameter values come from Zezza (2006).

Godley and Lavoie (2007) analyse Model *REG* beginning from an initial
stationary state. This script first finds that stationary state, matching (more
or less) the starting values in Zezza's (2006) EViews script.

This example also shows how to use the `offset` keyword argument in the solve
methods to copy over values from another period before solving. This can
substantially improve solution times.

While FSIC only requires NumPy, this example also uses:

* `pandas`, to generate a DataFrame of results using `fsic.tools`
* `matplotlib`, to replicate, from Godley and Lavoie (2007), Figures 6.1, 6.2,
  6.3, 6.4, 6.5, 6.6 and 6.7. These figures consist of four pairs of charts in
  a (4 x 2) grid that shows the results of four experiments with Model *REG*:
    1. an increase in the propensity to import by the South
    2. an increase in government expenditure in the South
    3. an increase in the propensity to save of Southern households
    4. a decrease in Southern households' liquidity preference
  Each pair of charts consists of:
    1. a comparison of financial balances (household net acquisition of
       financial assets; government budget, trade) relative to the original
       stationary state baseline
    2. evolution of North and South GDP in the experiment (scenario)
  (Godley and Lavoie, 2007, don't report a GDP chart for the fourth
  experiment.)

Outputs:

1. Replicates Figures 6.1, 6.2, 6.3, 6.4, 6.5, 6.6 and 6.7 of Godley and Lavoie
   (2007), saving the charts to 'figures-6.1t6.7.png'

References:

    Godley, W., Lavoie, M. (2007),
    *Monetary economics: an integrated approach to
    credit, money, income, production and wealth*,
    Palgrave Macmillan

    Zezza, G. (2006),
    'EViews macros for building models in *Wynne Godley and Marc Lavoie*
    Monetary Economics: an integrated approach to
    credit, money, income, production and wealth',
    http://gennaro.zezza.it/software/eviews/glch06.php
"""

import matplotlib.pyplot as plt

from pandas import DataFrame
import pandas as pd

import fsic
import fsic.tools


# Inline comments give the corresponding equation numbers from Godley and
# Lavoie (2007) - for reference only; FSIC ignores comments, just as Python
# does.
# 'A' suffix indicates a slight amendment to be compatible with the FSIC
# parser.
script = '''
Y_N = C_N + G_N + X_N - IM_N                                                  # 6.1
Y_S = C_S + G_S + X_S - IM_S                                                  # 6.2

IM_N = {mu_N} * Y_N                                                           # 6.3
IM_S = {mu_S} * Y_S                                                           # 6.4

X_N = IM_S                                                                    # 6.5
X_S = IM_N                                                                    # 6.6

YD_N = Y_N - T_N + r[-1] * Bh_N[-1]                                           # 6.7
YD_S = Y_S - T_S + r[-1] * Bh_S[-1]                                           # 6.8

T_N = {theta} * (Y_N + r[-1] * Bh_N[-1])                                      # 6.9
T_S = {theta} * (Y_S + r[-1] * Bh_S[-1])                                      # 6.10

V_N = V_N[-1] + (YD_N - C_N)                                                  # 6.11
V_S = V_S[-1] + (YD_S - C_S)                                                  # 6.12

C_N = {alpha_1_N} * YD_N + {alpha_2_N} * V_N[-1]                              # 6.13
C_S = {alpha_1_S} * YD_S + {alpha_2_S} * V_S[-1]                              # 6.14

Hh_N = V_N - Bh_N                                                             # 6.15
Hh_S = V_S - Bh_S                                                             # 6.16

Bh_N = V_N * ({lambda_0_N} + {lambda_1_N} * r - {lambda_2_N} * (YD_N / V_N))  # 6.17A
Bh_S = V_S * ({lambda_0_S} + {lambda_1_S} * r - {lambda_2_S} * (YD_S / V_S))  # 6.18A

T = T_N + T_S                                                                 # 6.19
G = G_N + G_S                                                                 # 6.20
Bh = Bh_N + Bh_S                                                              # 6.21
Hh = Hh_N + Hh_S                                                              # 6.22

Bs = Bs[-1] + (G + r[-1] * Bs[-1]) - (T + r[-1] * Bcb[-1])                    # 6.23A
Hs = Hs[-1] + (Bcb - Bcb[-1])                                                 # 6.24A
Bcb = Bs - Bh                                                                 # 6.25

r = r_bar                                                                     # 6.26
'''

symbols = fsic.parse_model(script)
REG = fsic.build_model(symbols)


def make_model_results(model: fsic.BaseModel) -> DataFrame:
    """Return the model results, with supplementary variables, as a `pandas` DataFrame."""
    results = fsic.tools.model_to_dataframe(model)[model.names]

    # Take first difference of household wealth to construct a flow measure
    results['D(V_N)'] = results['V_N'].diff()
    results['D(V_S)'] = results['V_S'].diff()

    results['GovtBal_N'] = results.eval('T_N - G_N') - results['r'].shift() * results['Bh_N'].shift()
    results['GovtBal_S'] = results.eval('T_S - G_S') - results['r'].shift() * results['Bh_S'].shift()

    results['NX_N'] = results.eval('X_N - IM_N')
    results['NX_S'] = results.eval('X_S - IM_S')

    return results

def make_scenario_charts(financial_balances_plot: 'AxesSubplot', gdp_plot: 'AxesSubplot', scenario_results: DataFrame, baseline_results: DataFrame) -> None:
    """Create plots (Southern financial balances and both regions' GDP)."""
    # Calculate difference from baseline
    difference_from_baseline = scenario_results - baseline_results

    # Financial balances plot
    financial_balances_plot.plot(difference_from_baseline.index, [0] * len(difference_from_baseline.index),
                                 color='k', linewidth=0.75)

    financial_balances_plot.plot(difference_from_baseline.index, difference_from_baseline['D(V_S)'],
                                 label='Change in household wealth of the South region', color='#33C3F0', linestyle='-')
    financial_balances_plot.plot(difference_from_baseline.index, difference_from_baseline['GovtBal_S'],
                                 label='Government balance with the South region', color='#FF4F2E', linestyle=':')
    financial_balances_plot.plot(difference_from_baseline.index, difference_from_baseline['NX_S'],
                                 label='Trade balance of the South region', color='#77C3AF', linestyle='--')

    financial_balances_plot.set_xlim(min(difference_from_baseline.index), max(difference_from_baseline.index))

    # GDP plot
    gdp_plot.plot(scenario_results.index, [scenario_results['Y_N'].iloc[0]] * len(difference_from_baseline.index),
                  color='k', linewidth=0.75)

    gdp_plot.plot(scenario_results.index, scenario_results['Y_N'],
                  label='North region GDP', color='#33C3F0', linestyle='-')
    gdp_plot.plot(scenario_results.index, scenario_results['Y_S'],
                  label='South region GDP', color='#FF4F2E', linestyle='--')

    gdp_plot.set_xlim(min(scenario_results.index), max(scenario_results.index))


if __name__ == '__main__':
    # 1. Find the stationary state of the model from an initial set of
    #    parameter values (from Zezza, 2006)
    starting_from_zero = REG(
        range(500),  # Enough periods to reach the stationary state
        alpha_1_N=0.6, alpha_2_N=0.4, lambda_0_N=0.635, lambda_1_N=5, lambda_2_N=0.01, mu_N=0.18781,
        alpha_1_S=0.7, alpha_2_S=0.3, lambda_0_S=0.670, lambda_1_S=6, lambda_2_S=0.07, mu_S=0.18781)

    # Fiscal policy
    starting_from_zero.G_N = starting_from_zero.G_S = 20
    starting_from_zero.theta = 0.2

    # Monetary policy
    starting_from_zero.r_bar = 0.025
    starting_from_zero.r[0] = starting_from_zero.r_bar[0]

    # Solve the model:
    #  - increase the maximum number of iterations for convergence
    #  - copying the values from the previous period (with `offset=-1`) before
    #    solution improves solution times as the model approaches its
    #    stationary state
    #  - Equations 6.17A and 6.18A have division operations: starting from
    #    zero, this may initially generate a NaN in solution - ignore this
    #    because it should be overwritten in a later iteration
    starting_from_zero.solve(max_iter=2000, offset=-1, errors='ignore')

    # Take the results from the last period as the stationary state
    stationary_state = dict(zip(starting_from_zero.names,
                                starting_from_zero.values[:, -1]))

    # Copy the stationary state to a new baseline model instance
    baseline = REG(range(1945, 2010 + 1), **stationary_state)
    baseline.solve(offset=-1)

    baseline_results = make_model_results(baseline).loc[1950:2000, :]


    # 2. Experiments with Model *REG*
    #    (Input values come from Zezza, 2006)

    # 2.1 An increase in the propensity to import of the South
    #     (from Section 6.5.1 of Godley and Lavoie, 2007)
    import_propensity_scenario = baseline.copy()
    import_propensity_scenario['mu_S', 1960:] = 0.20781
    import_propensity_scenario.solve(max_iter=3500, offset=-1)

    imports_results = make_model_results(import_propensity_scenario).loc[1950:2000, :]

    # 2.2 An increase in the government expenditures of the South
    #     (from Section 6.5.2 of Godley and Lavoie, 2007)
    government_expenditure_scenario = baseline.copy()
    government_expenditure_scenario['G_S', 1960:] = 25
    government_expenditure_scenario.solve(max_iter=2000, offset=-1)

    government_results = make_model_results(government_expenditure_scenario).loc[1950:2000, :]

    # 2.3 An increase in the propensity to save of the Southern households
    #     (from Section 6.5.3 of Godley and Lavoie, 2007)
    consumption_propensity_scenario = baseline.copy()
    consumption_propensity_scenario['alpha_1_S', 1960:] = 0.6
    consumption_propensity_scenario.solve(max_iter=2000, offset=-1)

    consumption_results = make_model_results(consumption_propensity_scenario).loc[1950:2000, :]

    # 2.4 A change in the liquidity preference of the Southern households
    #     (from Section 6.5.4 of Godley and Lavoie, 2007)
    liquidity_preference_scenario = baseline.copy()
    liquidity_preference_scenario['lambda_0_S', 1960:] = 0.75
    liquidity_preference_scenario.solve(max_iter=2000, offset=-1)

    liquidity_results = make_model_results(liquidity_preference_scenario).loc[1950:2000, :]

    # 3. Replicate Figures 6.1, 6.2, 6.3, 6.4, 6.5, 6.6 and 6.7 of Godley and
    #    Lavoie (2007)

    # Set up plot area
    _, axes = plt.subplots(4, 2, figsize=(12, 20))
    plt.suptitle('Experiments with Model $\it{REG}$')

    # Create individual plots
    make_scenario_charts(*axes[0], imports_results, baseline_results)
    make_scenario_charts(*axes[1], government_results, baseline_results)
    make_scenario_charts(*axes[2], consumption_results, baseline_results)
    make_scenario_charts(*axes[3], liquidity_results, baseline_results)

    # Add plot labels
    axes[0, 0].set_title('Evolution of financial balances in the South region')
    axes[0, 1].set_title('Evolution of GDP in the North and South regions')

    # Use y-axis labels to name the individual experiments
    axes[0, 0].set_ylabel('Increase in propensity to import of the South')
    axes[1, 0].set_ylabel('Increase in government expenditure in the South')
    axes[2, 0].set_ylabel('Increase in propensity to save of Southern households')
    axes[3, 0].set_ylabel("Decrease in Southern households' liquidity preference")

    # Add legends to the bottom of the chart
    axes[-1, 0].legend(loc='upper left', bbox_to_anchor=(0.0, -0.1))
    axes[-1, 1].legend(loc='upper left', bbox_to_anchor=(0.0, -0.1))

    plt.savefig('figures-6.1t6.7.png')
