# -*- coding: utf-8 -*-
"""
7_bmw
=====
fsic implementation of Model *BMW*, the simplest *bank-money-world*
model. Parameter values come from Zezza (2006).

Godley and Lavoie (2007) analyse Model *BMW* beginning from an initial
stationary state. This script first finds that stationary state, matching (more
or less) the starting values in Zezza's (2006) EViews script.

While fsic only requires NumPy, this example also uses:

* `matplotlib`, to replicate, from Godley and Lavoie (2007), Figures 7.1, 7.2,
  7.3, 7.4 and 7.5

Outputs:

1. Replicates Figures 7.1, 7.2, 7.3, 7.4 and 7.5 of Godley and Lavoie (2007),
   saving the charts to 'figures-7.1t7.5.png'
    * This file also includes versions of Figures 7.4 and 7.5 for an increase
      in autonomous consumption; and 7.2 for an increase in the propensity to
      save.

References:

    Godley, W., Lavoie, M. (2007),
    *Monetary economics: an integrated approach to
    credit, money, income, production and wealth*,
    Palgrave Macmillan

    Zezza, G. (2006),
    'EViews macros for building models in *Wynne Godley and Marc Lavoie*
    Monetary Economics: an integrated approach to
    credit, money, income, production and wealth',
    http://gennaro.zezza.it/software/eviews/glch07.php
"""

import matplotlib.pyplot as plt
import numpy as np

import fsic


# Inline comments give the corresponding equation numbers from Godley and
# Lavoie (2007) - for reference only; fsic ignores comments, just as Python
# does.
# 'A' suffix indicates a slight amendment to be compatible with the fsic
# parser.
# 'B' suffix indicates a code change for ease of solution. This applies to
# Equation 7.15 only.
script = '''
C_s = C_d                                               # 7.1
I_s = I_d                                               # 7.2
N_s = N_d                                               # 7.3
L_s = L_s[-1] + (L_d - L_d[-1])                         # 7.4A
Y = C_s + I_s                                           # 7.5
WB_d = Y - r_l[-1] * L_d[-1] - AF                       # 7.6
AF = {delta} * K[-1]                                    # 7.7
L_d = L_d[-1] + I_d - AF                                # 7.8A

# Godley and Lavoie (2007) has 'M_{d-1}' as the final term. Looks to be a typo:
# it should be 'M_{h-1}' (which also matches Zezza [2006]).
YD = WB_s + r_m[-1] * M_h[-1]                           # 7.9

M_h = M_h[-1] + YD - C_d                                # 7.10A
M_s = M_s[-1] + (L_s - L_s[-1])                         # 7.11A
r_m = r_l                                               # 7.12
WB_s = W * N_s                                          # 7.13
N_d = Y / pr                                            # 7.14

# Equation below only divides by `N_d` if positive, to avoid a NaN propagating
# through the solution (NB `fsic.fortran` can't handle this)
W = WB_d / (N_d if N_d > 0 else 1)                      # 7.15B

C_d = {alpha_0} + {alpha_1} * YD + {alpha_2} * M_h[-1]  # 7.16
K = K[-1] + (I_d - DA)                                  # 7.17
DA = {delta} * K[-1]                                    # 7.18
K_T = {kappa} * Y[-1]                                   # 7.19
I_d = {gamma} * (K_T - K[-1]) + DA                      # 7.20
r_l = r_l_bar                                           # 7.21
'''

symbols = fsic.parse_model(script)
BMW = fsic.build_model(symbols)


if __name__ == '__main__':
    # 1. Find the stationary state of the model from an initial set of
    #    parameter values (from Zezza, 2006)
    starting_from_zero = BMW(
        range(100),  # Enough periods to reach the stationary state
        alpha_0=25, alpha_1=0.75, alpha_2=0.1,
        delta=0.1, gamma=0.15, kappa=1,
        pr=1)

    # Set the interest rate
    starting_from_zero.r_l_bar = 0.04
    starting_from_zero.r_l[0] = starting_from_zero.r_m[0] = starting_from_zero.r_l_bar[0]

    starting_from_zero.solve(max_iter=400)

    # Take the results from the last period to be the stationary state
    stationary_state = dict(zip(starting_from_zero.names,
                                starting_from_zero.values[:, -1]))

    # Run a baseline with these values
    baseline = BMW(range(1945, 2010 + 1), **stationary_state)
    baseline.solve(max_iter=250)

    # 2. An increase in autonomous consumption expenditures
    consumption_scenario = baseline.copy()
    consumption_scenario['alpha_0', 1960:] = 28
    consumption_scenario.solve(start=1960, max_iter=400)

    # 3. An increase in the propensity to save out of disposable income
    saving_scenario = baseline.copy()
    saving_scenario['alpha_1', 1960:] = 0.74
    saving_scenario.solve(start=1960, max_iter=400)

    # -------------------------------------------------------------------------
    # Reproduce the figures from Godley and Lavoie (2007)
    _, axes = plt.subplots(2, 4, figsize=(24, 12))
    plt.suptitle('Model $\it{BMW}$: A simple model with private bank money')

    axes[0, 0].set_ylabel('Increase in autonomous consumption expenditures')
    axes[1, 0].set_ylabel('Increase in the propensity to save')

    # Figure 7.1: Evolution of household disposable income and consumption,
    #             following an increase in autonomous consumption expenditures
    axes[0, 0].plot(consumption_scenario.span, consumption_scenario.YD,
                    label='Household disposable income', color='#33C3F0', linestyle='-')
    axes[0, 0].plot(consumption_scenario.span, consumption_scenario.C_d,
                    label='Consumption', color='#FF4F2E', linestyle='--')

    axes[0, 0].set_xlim(1950, 2000)
    axes[0, 0].legend()
    axes[0, 0].set_title('Figure 7.1: Household disposable income and consumption')

    # Figure 7.2: Evolution of gross investment and disposable investment,
    #             following an increase in autonomous consumption expenditures
    axes[0, 1].plot(consumption_scenario.span, consumption_scenario.I_d,
                    label='Investment', color='#4563F2', linestyle='-')
    axes[0, 1].plot(consumption_scenario.span, consumption_scenario.DA,
                    label='Depreciation', color='#33C3F0', linestyle='--')

    axes[0, 1].set_xlim(1950, 2000)
    axes[0, 1].legend()
    axes[0, 1].set_title('Figure 7.2: Investment and depreciation')

    # Figure 7.4a: Evolution of the output to capital ratio (Y/K−1)
    axes[0, 2].plot(consumption_scenario.span,
                    consumption_scenario.Y / np.hstack([np.nan, consumption_scenario.K[:-1]]),
                    label='Output:capital ratio', color='#77C3AF', linestyle='-')

    axes[0, 2].set_xlim(1950, 2000)
    axes[0, 2].legend()
    axes[0, 2].set_title('Figure 7.4a: Output:capital ratio')

    # Figure 7.5a: Evolution of the real wage rate (W)
    axes[0, 3].plot(consumption_scenario.span, consumption_scenario.W,
                    label='Real wage rate', color='#FF992E', linestyle='-')

    axes[0, 3].set_xlim(1950, 2000)
    axes[0, 3].legend()
    axes[0, 3].set_title('Figure 7.5a: Real wage rate')

    # Figure 7.3: Evolution of household disposable income and consumption,
    #             following an increase in the propensity to save out of
    #             disposable income
    axes[1, 0].plot(saving_scenario.span, saving_scenario.YD,
                    label='Household disposable income', color='#33C3F0', linestyle='-')
    axes[1, 0].plot(saving_scenario.span, saving_scenario.C_d,
                    label='Consumption', color='#FF4F2E', linestyle='--')

    axes[1, 0].set_xlim(1950, 2000)
    axes[1, 0].set_title('Figure 7.3: Household disposable income and consumption')

    # Figure 7.2a: Evolution of gross investment and disposable investment
    axes[1, 1].plot(saving_scenario.span, saving_scenario.I_d,
                    label='Investment', color='#4563F2', linestyle='-')
    axes[1, 1].plot(saving_scenario.span, saving_scenario.DA,
                    label='Depreciation', color='#33C3F0', linestyle='--')

    axes[1, 1].set_xlim(1950, 2000)
    axes[1, 1].set_title('Figure 7.2a: Investment and depreciation')

    # Figure 7.4: Evolution of the output to capital ratio (Y/K−1), following
    #             an increase in the propensity to save out of disposable
    #             income
    axes[1, 2].plot(saving_scenario.span,
                    saving_scenario.Y / np.hstack([np.nan, saving_scenario.K[:-1]]),
                    label='Output:capital ratio', color='#77C3AF', linestyle='-')

    axes[1, 2].set_xlim(1950, 2000)
    axes[1, 2].set_title('Figure 7.4: Output:capital ratio')

    # Figure 7.5: Evolution of the real wage rate (W), following an increase in
    #             the propensity to save out of disposable income
    axes[1, 3].plot(saving_scenario.span, saving_scenario.W,
                    label='Real wage rate', color='#FF992E', linestyle='-')

    axes[1, 3].set_xlim(1950, 2000)
    axes[1, 3].set_title('Figure 7.5: Real wage rate')

    plt.savefig('figures-7.1t7.5.png')
