# -*- coding: utf-8 -*-
"""
fsic implementation of Model *DIS*, representing disequilibrium (of a kind) in
the goods market, from Chapter 9 of Godley and Lavoie (2007). Parameter values
come from Zezza (2006).

While fsic only requires NumPy, this example also uses:

* `matplotlib`, to replicate, from Godley and Lavoie (2007), Figures 9.1, 9.2
  and 9.3

Outputs:

1. Replicates Figures 9.1, 9.2 and 9.3 of Godley and Lavoie (2007), saving the
   charts to 'figures-9.1t9.3.png'

References:

    Godley, W., Lavoie, M. (2007),
    *Monetary economics: an integrated approach to
    credit, money, income, production and wealth*,
    Palgrave Macmillan

    Zezza, G. (2006),
    'EViews macros for building models in *Wynne Godley and Marc Lavoie*
    Monetary Economics: an integrated approach to
    credit, money, income, production and wealth',
    https://gennaro.zezza.it/software/eviews/glch09.php
"""

import matplotlib.pyplot as plt

import fsic


# Variable definitions, taken from Godley and Lavoie (2007) ('Notations Used in
# the Book')
descriptions = {
    'c':       'Consumption goods demand by households, in real terms',
    'C':       'Consumption goods supply by firms, in nominal terms',

    'F':       'Sum of bank and firm profits',
    'F_b':     'Target profits of banks',

    'IN':      'Realized stock of inventories, at current unit costs',
    # Note trailing underscore here, to avoid collision with Python keyword `in`
    'in_':     'Realized stock of inventories, in real terms',

    'in_e':    'Short-run target level (expected level) of inventories, in real terms',
    'in_T':    'Long-run target level of inventories, in real terms',
    'L_d':     'Loans demanded by firms from banks',
    'L_s':     'Loans supplied by banks to firms',
    'M_h':     'Money deposits actually held by households',
    'm_h':     'Real money balances held by households',
    'M_s':     'Money supplied by the banks',
    'N':       'Demand for labour',
    'NHUC':    'Normal historic unit cost',
    'p':       'Price level',
    'pr':      'Labour productivity, or trend labour productivity',
    'r_l':     'Rate of interest on bank loans',
    'r_l_bar': 'Rate of interest on bank loans (exogenous)',
    'r_m':     'Rate of interest on deposits',
    'S':       'Sales in nominal terms',
    's':       'Realized real sales (in widgets)',
    's_e':     'Expected real sales',
    'UC':      'Unit cost of production',
    'W':       'Nominal wage rate',
    'WB':      'The wage bill, in nominal terms',
    'Y':       'National income, in nominal terms',
    'y':       'Real output',
    'YD':      'Disposable income of households',
    'yd_hs':   'Haig–Simons realised real disposable income',
    'yd_hs_e': 'Haig–Simons expected real disposable income',

    'add':     'Spread of loan rate over the deposit rate',

    'alpha_0': 'Autonomous consumption',
    'alpha_1': 'Propensity to consume out of regular income',
    'alpha_2': 'Propensity to consume out of past wealth',
    'beta':    'Reaction parameter related to expectations',
    'gamma':   'Partial adjustment function that applies to inventories and fixed capital',
    'epsilon': 'Another reaction parameter related to expectations',
    'phi':     'Costing margin in pricing',
    'sigma_T': 'Target (current) inventories to sales ratio',
}

# Inline comments give the corresponding equation numbers from Godley and
# Lavoie (2007) - for reference only; fsic ignores comments, just as Python
# does.
# 'A' suffix indicates a slight amendment to be compatible with the fsic
# parser. This applies to Equation 9.22
# 'B' suffix indicates a code change for ease of solution. This applies to
# Equations 9.7, 9.9 and 9.25
script = '''
y = s_e + (in_e - in_[-1])                                        # 9.1
in_T = {sigma_T} * s_e                                            # 9.2
in_e = in_[-1] + {gamma} * (in_T - in_[-1])                       # 9.3
in_ = in_[-1] + (y - s)                                           # 9.4
s_e = {beta} * s[-1] + (1 - {beta}) * s_e[-1]                     # 9.5
s = c                                                             # 9.6
N = y / (pr if pr > 0 else 1)                                     # 9.7B
WB = N * W                                                        # 9.8
UC = WB / (y if y > 0 else 1)                                     # 9.9B
IN = in_ * UC                                                     # 9.10
S = p * s                                                         # 9.11
p = (1 + {phi}) * NHUC                                            # 9.12
NHUC = (1 - {sigma_T}) * UC + {sigma_T} * (1 + r_l[-1]) * UC[-1]  # 9.13
F = S - WB + (IN - IN[-1]) - r_l[-1] * IN[-1]                     # 9.14
L_d = IN                                                          # 9.15
L_s = L_d                                                         # 9.16
M_s = L_s                                                         # 9.17
r_l = r_l_bar                                                     # 9.18
r_m = r_l - add                                                   # 9.19

# Note typo in original: L[-1] instead of L_d[-1]
F_b = (r_l[-1] * L_d[-1]) - (r_m[-1] * M_h[-1])                   # 9.20

YD = WB + F + F_b + r_m[-1] * M_h[-1]                             # 9.21
M_h = M_h[-1] + (YD - C)                                          # 9.22A
yd_hs = c + (m_h - m_h[-1])                                       # 9.23
C = c * p                                                         # 9.24
m_h = M_h / (p if p > 0 else 1)                                   # 9.25B
c = {alpha_0} + {alpha_1} * yd_hs_e + {alpha_2} * m_h[-1]         # 9.26
yd_hs_e = {epsilon} * yd_hs[-1] + (1 - {epsilon}) * yd_hs_e[-1]   # 9.27

# Auxiliary equation
Y = (s * p) + (in_ - in_[-1]) * UC
'''

symbols = fsic.parse_model(script)

class DIS(fsic.build_model(symbols)):
    __slots__ = descriptions


if __name__ == '__main__':
    # Set up the steady state baseline ----------------------------------------
    baseline = DIS(range(1945, 2010 + 1),
                   alpha_0=15, alpha_1=0.8, alpha_2=0.1,
                   beta=0.75, epsilon=0.75, gamma=0.25, phi=0.25, sigma_T=0.15)

    baseline.add = 0.02
    baseline.pr = 1
    baseline.r_l_bar = 0.04
    baseline.W = 0.86

    baseline.r_l[0] = baseline.r_l_bar[0]
    baseline.r_m[0] = baseline.r_l[0] - baseline.add[0]

    # Calculate steady state values for the first period
    # Note the use of `eval()` below for both convenience and readability
    baseline.UC[0] = baseline.eval('W / pr')[0]
    baseline.NHUC[0] = baseline.eval('(1 + sigma_T * r_l_bar) * UC')[0]
    baseline.p[0] = baseline.eval('(1 + phi[0]) * NHUC')[0]

    # Without `eval()`, the statement below would be:
    # baseline.yd_hs[0] = baseline.alpha_0[0] / (1 - baseline.alpha_1[0] - baseline.alpha_2[0] * baseline.sigma_T[0] * baseline.UC[0] / baseline.p[0])
    baseline.yd_hs[0] = baseline.eval('alpha_0 / (1 - alpha_1 - alpha_2 * sigma_T * UC / p[0])')[0]
    baseline.s_e[0] = baseline.s[0] = baseline.c[0] = baseline.yd_hs_e[0] = baseline.yd_hs[0]

    # Set starting values for stocks
    baseline.in_e[0] = baseline.in_[0] = baseline.eval('sigma_T * s')[0]
    baseline.M_s[0] = baseline.M_h[0] = baseline.L_s[0] = baseline.L_d[0] = baseline.IN[0] = baseline.eval('in_ * UC')[0]
    baseline.m_h[0] = baseline.M_h[0] / baseline.p[0]

    # Solve the baseline
    baseline.solve()

    # Scenario: The impact of an increase in the mark-up ----------------------
    markup_scenario = baseline.copy()
    markup_scenario['phi', 1960:] = 0.3
    markup_scenario.solve(start=1960)

    # Scenario: The impact of a higher target inventories to sales ratio ------
    inventories_scenario = baseline.copy()
    inventories_scenario['sigma_T', 1960:] = 0.25
    inventories_scenario.solve(start=1960)

    # Reproduce the figures from Godley and Lavoie (2007) ---------------------
    _, axes = plt.subplots(1, 3, figsize=(24, 8))
    plt.suptitle(r'Model $\it{DIS}$: Disequilibrium (of a kind) in the goods market')

    # Figure 9.1: Evolution of (Haig–Simons) real disposable income and of real
    #             consumption, following a one-shot increase in the costing
    #             margin
    axes[0].plot(range(1955, 2000 + 1), markup_scenario['yd_hs', 1955:2000],
                 label='Haig-Simons real disposable income', color='#33C3F0', linestyle='-')
    axes[0].plot(range(1955, 2000 + 1), markup_scenario['c', 1955:2000],
                 label='Real consumption', color='#FF4F2E', linestyle='--')

    axes[0].set_xlim(1955, 2000)
    axes[0].legend()
    axes[0].set_title(r'Figure 9.1: Income and consumption effects of a one-shot\nincrease in the costing margin, $\phi$')

    # Figure 9.2: Evolution of (Haig–Simons) real disposable income and of real
    #             consumption, following an increase in the target inventories
    #             to sales ratio
    axes[1].plot(range(1955, 2000 + 1), inventories_scenario['yd_hs', 1955:2000],
                 label='Haig-Simons real disposable income', color='#33C3F0', linestyle='-')
    axes[1].plot(range(1955, 2000 + 1), inventories_scenario['c', 1955:2000],
                 label='Real consumption', color='#FF4F2E', linestyle='--')

    axes[1].set_xlim(1955, 2000)
    axes[1].legend()
    axes[1].set_title(r'Figure 9.2: Income and consumption effects of an\nincrease in the target inventories-to-sales ratio, $\sigma^T$')

    # Figure 9.3: Evolution of the desired increase in physical inventories and
    #             of the change in realized inventories, following an increase
    #             in the target inventories to sales ratio
    # Note the use of `eval()` in the two statements below to avoid either
    # having to calculate indexes manually or using `pandas`
    axes[2].plot(range(1955, 2000 + 1), inventories_scenario.eval('diff(in_)[`1955`:`2000`]'),
                 label='Change in realised inventories', color='#4563F2', linestyle='-')
    axes[2].plot(range(1955, 2000 + 1), inventories_scenario.eval('diff(in_e)[`1955`:`2000`]'),
                 label='Desired increase in physical inventories', color='#33C3F0', linestyle='--')

    axes[2].set_xlim(1955, 2000)
    axes[2].legend()
    axes[2].set_title(r'Figure 9.3: Inventory effects of an increase\nin the target inventories-to-sales ratio, $\sigma^T$')

    plt.savefig('figures-9.1t9.3.png')
