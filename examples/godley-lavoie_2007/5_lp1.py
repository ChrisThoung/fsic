# -*- coding: utf-8 -*-
"""
5_lp1
=====
fsic implementation of Model *LP1*, a model of long-term bonds, capital gains
and liquidity preference, from Chapter 5 of Godley and Lavoie (2007). Parameter
values come from Zezza (2006).

Godley and Lavoie (2007) analyse Model *LP1* beginning from an initial
stationary state. This script first finds that stationary state, matching (more
or less) the starting values in Zezza's (2006) EViews script. The script then
analyses the impact of an increase in both short- and long-term interest rates,
as in Godley and Lavoie (2007).

This example also shows how to use conditional expressions in a model
definition to avoid generating NaNs (from divide-by-zero operations). This is
new in fsic version 0.5.0.dev and an alternative to handling NaNs using the
`errors` keyword argument in `solve()`.

While fsic only requires NumPy, this example also uses:

* `pandas`, to generate a DataFrame of results using `fsic.tools`
* `matplotlib`, to replicate, from Godley and Lavoie (2007):
    * Figure 5.2: Evolution of the wealth to disposable income ratio, following
                  an increase in both the short-term and long-term interest
                  rates
    * Figure 5.3: Evolution of household consumption and disposable income,
                  following an increase in both the short-term and long-term
                  interest rates
    * Figure 5.4: Evolution of the bonds to wealth ratio and the bills to
                  wealth ratio, following an increase from 3% to 4% in the
                  short-term interest rate, while the long-term interest rate
                  moves from 5% to 6.67%

Outputs:

1. Replicates Figures 5.2, 5.3 and 5.4 of Godley and Lavoie (2007), saving the
   charts to 'figures-5.2,5.3,5.4.png'

References:

    Godley, W., Lavoie, M. (2007),
    *Monetary economics: an integrated approach to
    credit, money, income, production and wealth*,
    Palgrave Macmillan

    Zezza, G. (2006),
    'EViews macros for building models in *Wynne Godley and Marc Lavoie*
    Monetary Economics: an integrated approach to
    credit, money, income, production and wealth',
    http://gennaro.zezza.it/software/eviews/glch05.php
"""

import matplotlib.pyplot as plt

import fsic


# Inline comments give the corresponding equation numbers from Godley and
# Lavoie (2007) - for reference only; fsic ignores comments, just as Python
# does.
# 'A' suffix indicates a slight amendment to be compatible with the fsic
# parser.
script = '''
Y = C + G                                                                    # 5.1
YDr = Y - T + r_b[-1] * Bh[-1] + BLh[-1]                                     # 5.2
T = {theta} * (Y + r_b[-1] * Bh[-1] + BLh[-1])                               # 5.3
V = V[-1] + (YDr - C) + CG                                                   # 5.4
CG = (p_bL - p_bL[-1]) * BLh[-1]                                             # 5.5A
C = {alpha_1} * YDr_e + {alpha_2} * V[-1]                                    # 5.6
V_e = V[-1] + (YDr_e - C) + CG                                               # 5.7
Hh = V   - Bh - (p_bL * BLh)                                                 # 5.8
Hd = V_e - Bd - (p_bL * BLd)                                                 # 5.9

Bd = V_e * ({lambda_20} +                                                    # 5.10A
            {lambda_22} * r_b +
            {lambda_23} * ERr_bL +
            # Note the conditional expression to guard against NaNs
            {lambda_24} * (YDr_e / V_e if V_e > 0 else 0))

BLd = V_e / p_bL * ({lambda_30} +                                            # 5.11
                    {lambda_32} * r_b +
                    {lambda_33} * ERr_bL +
                    # Note the conditional expression to guard against NaNs
                    {lambda_34} * (YDr_e / V_e if V_e > 0 else 0))

Bh = Bd                                                                      # 5.12
BLh = BLd                                                                    # 5.13
Bs = (Bs[-1] +                                                               # 5.14A
      (G + r_b[-1] * Bs[-1] + BLs[-1]) -
      (T + r_b[-1] * Bcb[-1]) -
      ((BLs - BLs[-1]) * p_bL))
Hs = Hs[-1] + (Bcb - Bcb[-1])                                                # 5.15A
Bcb = Bs - Bh                                                                # 5.16
BLs = BLh                                                                    # 5.17
ERr_bL = r_bL + {chi} * (p_bL_e - p_bL) / p_bL                               # 5.18
r_bL = 1 / p_bL                                                              # 5.19
p_bL_e = p_bL                                                                # 5.20
CG_e = {chi} * (p_bL_e - p_bL) * BLh                                         # 5.21
YDr_e = YDr[-1]                                                              # 5.22
r_b = r_b_bar                                                                # 5.23
p_bL = p_bL_bar                                                              # 5.24
'''

symbols = fsic.parse_model(script)
LP1 = fsic.build_model(symbols)


if __name__ == '__main__':
    # 1. Find the stationary state of the model from an initial set of
    #    parameter values (from Zezza, 2006)
    #    Note that Zezza (2006) uses entirely positive values for the lambda
    #    parameters, setting the signs in the equation specifications. Below,
    #    the equations (and signs) follow Godley and Lavoie (2007), reversing
    #    the signs on the parameter values below as needed
    starting_from_zero = LP1(range(100),  # Enough periods to reach the stationary state
                             alpha_1=0.8, alpha_2=0.2, chi=0.1,
                             lambda_20=0.44196, lambda_22=1.1, lambda_23=-1,  lambda_24=-0.03,
                             lambda_30=0.3997,  lambda_32=-1,  lambda_33=1.1, lambda_34=-0.03)

    # Fiscal policy
    starting_from_zero.G = 20
    starting_from_zero.theta = 0.1938

    # Monetary policy
    starting_from_zero.r_b_bar = 0.03
    starting_from_zero.p_bL_bar = 20

    # Copy values for first period (not solved)
    starting_from_zero.r_b[0] = starting_from_zero.r_b_bar[0]
    starting_from_zero.p_bL[0] = starting_from_zero.p_bL_bar[0]

    # Solve and take the results from the last period as the stationary state
    starting_from_zero.solve()

    stationary_state = dict(zip(starting_from_zero.names,
                                starting_from_zero.values[:, -1]))


    # 2. Starting from that stationary state, simulate an increase in the
    #    short- and long-term interest rates
    interest_rate_scenario = LP1(range(1945, 2010 + 1), **stationary_state)

    interest_rate_scenario['r_b_bar', 1960:] = 0.04
    interest_rate_scenario['p_bL_bar', 1960:] = 15

    interest_rate_scenario.solve()


    # 3. Reproduce Figures 5.2, 5.3 and 5.4 of Godley and Lavoie (2007)
    results = fsic.tools.model_to_dataframe(interest_rate_scenario)

    # Construct ratios for graphing
    results['V:YD'] = results.eval('V / YDr')
    results['Bh:V'] = results.eval('Bh / V')
    results['BLh:V'] = results.eval('p_bL * BLh / V')

    # Select same timespan as the original charts
    results_to_plot = results.loc[1950:2000, :]

    # Set up plot areas
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    plt.suptitle('Effects of an increase in the short- and long-term interest rates')

    # Recreate Figure 5.2 (wealth-to-disposable-income ratio)
    ax1.set_title('Evolution of the wealth-to-disposable-income ratio')

    ax1.plot(results_to_plot.index, results_to_plot['V:YD'].values,
             label='Wealth-to-disposable-income ratio',
             color='#33C3F0')

    ax1.set_xlim(min(results_to_plot.index), max(results_to_plot.index))
    ax1.legend()

    # Recreate Figure 5.3 (household consumption and disposable income)
    ax2.set_title('Evolution of household consumption and disposable income')

    ax2.plot(results_to_plot.index, results_to_plot['YDr'].values,
             label='Disposable income',
             color='#33C3F0')
    ax2.plot(results_to_plot.index, results_to_plot['C'].values,
             label='Consumption',
             color='#FF4F2E', linestyle='--')

    ax2.set_xlim(min(results_to_plot.index), max(results_to_plot.index))
    ax2.legend()

    # Recreate Figure 5.4 (bonds-to-wealth and bills-to-wealth ratios)
    ax3.set_title('Evolution of the bonds-to-wealth and bills-to-wealth ratios')

    ax3.plot(results_to_plot.index, results_to_plot['Bh:V'] * 100,
             label='Bills-to-wealth ratio',
             color='#33C3F0')
    ax3.plot(results_to_plot.index, results_to_plot['BLh:V'] * 100,
             label='Bonds-to-wealth ratio',
             color='#FF4F2E', linestyle='--')

    ax3.set_xlim(min(results_to_plot.index), max(results_to_plot.index))
    ax3.set_ylabel('%')
    ax3.legend()

    plt.savefig('figures-5.2,5.3,5.4.png')
