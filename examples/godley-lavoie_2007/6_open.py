# -*- coding: utf-8 -*-
"""
6_open
======
FSIC implementation of Model *OPEN*, a model of a two-country economy, from
Chapter 6 of Godley and Lavoie (2007). This model disaggregates Model *PC* (see
'4_pc.py') into two countries: 'North' and 'South'. Parameter values come from
Zezza (2006).

Whereas Godley and Lavoie (2007) define the model as a single set of equations,
the implementation below builds on the observation that the two countries have
identical structures, differing only in their parameter values, and are then
connected by a small number of further equations. This leads to an
implementation consisting of:

* two instances of the same country model to represent North and South, built
  in the same way as in examples for earlier chapters of Godley and Lavoie
  (2007)
* a linker object that nests/stores the two country models and implements the
  linking equations - this is the object the user interacts with

The linker is derived from the `fsic` `BaseLinker` class and shares similar
features and interfaces with the `fsic` `BaseModel` class. However, it's up to
the user to write the necessary code.

While FSIC only requires NumPy, this example also uses:

* `pandas`, to generate DataFrames of results using
  `fsictools.linker_to_dataframes()`
* `matplotlib`, to replicate, from Godley and Lavoie (2007), Figures 6.8, 6.9,
  6.10 and 6.11

Outputs:

1. Replicates Figures 6.8, 6.9, 6.10 and 6.11 of Godley and Lavoie (2007),
   saving the charts to 'figures-6.8t6.11.png'

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

import fsic
import fsictools


# -----------------------------------------------------------------------------
# Define the (common) structure of the individual country models
#
# Inline comments give the corresponding equation numbers from Godley and
# Lavoie (2007) - for reference only; FSIC ignores comments, just as Python
# does.
#
# 'A' suffix indicates a slight amendment to be compatible with the FSIC
# parser.
#
# Note that not all the equations appear in the script directly below. The
# others are global and connect the country models. See the later linker for
# these equations.
script = '''
# Equation order may matter when solving by the Gauss-Seidel method. In
# practice, putting the consumption function first avoids the solution
# exploding.
C = {alpha_1} * YD + {alpha_2} * V[-1]                      # 6.O.13 / 6.O.14

Y = C + G + X - IM                                          # 6.O.1 / 6.O.2
IM = {mu} * Y                                               # 6.O.3 / 6.O.4

YD = Y - T + r[-1] * Bh[-1]                                 # 6.O.7 / 6.O.8
T = {theta} * (Y + r[-1] * Bh[-1])                          # 6.O.9 / 6.O.10
V = V[-1] + YD - C                                          # 6.O.11 / 6.O.12

Hh = V - Bh                                                 # 6.O.15 / 6.O.16
Bh = V * ({lambda_0} + {lambda_1} * r) - {lambda_2} * YD    # 6.O.17A / 6.O.18A
Bs = Bs[-1] + (G + r[-1] * Bs[-1]) - (T + r[-1] * Bcb[-1])  # 6.O.19A / 6.O.20A
Bcb = Bs - Bh                                               # 6.O.21 / 6.O.22

# Note the underscore in 'or_' to avoid collisions with the Python `or`
# keyword. (Any potential collisions raise an Exception.)
or_ = or_[-1] + ((Hs - Hs[-1]) - (Bcb - Bcb[-1])) / p_g     # 6.O.23A / 6.O.24A

Hs = Hh                                                     # 6.O.25 / 6.O.26
r = r_bar                                                   # 6.O.30 / 6.O.31
'''

symbols = fsic.parse_model(script)
Country = fsic.build_model(symbols)


# Define the linker that connects the country models --------------------------
class OPEN(fsic.BaseLinker):
    """Model *OPEN*: A two-country economy."""

    # Core variables (not specific to any individual submodel) work the same
    # way as models derived from the `fsic` `BaseModel` class
    ENDOGENOUS = ['xr']
    EXOGENOUS = ['xr_bar', 'p_g_bar']

    NAMES = ENDOGENOUS + EXOGENOUS
    CHECK = ENDOGENOUS

    # Not strictly needed but using the `__slots__` attribute like this
    # provides a bit of documentation at runtime
    __slots__ = {
        'xr':      'Exchange rate',
        'xr_bar':  'Fixed exchange rate',
        'p_g_bar': 'Price of gold',
    }

    def solve_t_before(self, t, *args, **kwargs):
        """Evaluate equations that only need to solve once per period, prior to iterative solution."""
        self.xr[t] = self.xr_bar[t] # 6.O.29: Set exchange rate

        # Set price of gold in domestic currencies
        self.submodels['North'].p_g[t] = self.p_g_bar[t] / self.xr[t]  # 6.O.27A
        self.submodels['South'].p_g[t] = self.p_g_bar[t] * self.xr[t]  # 6.O.28A

    def evaluate_t_before(self, t, *args, **kwargs):
        """Evaluate equations that should solve before the individual country models in each iteration."""
        # Set Country A's exports to equal Country B's imports in the relevant
        # domestic currency (and vice versa)
        self.submodels['North'].X[t] = self.submodels['South'].IM[t] / self.xr[t]  # 6.O.5
        self.submodels['South'].X[t] = self.submodels['North'].IM[t] * self.xr[t]  # 6.O.6


if __name__ == '__main__':
    # Set up baseline ---------------------------------------------------------
    # In contrast to earlier chapter examples, have the model begin at its
    # stationary state, rather than try to solve for it

    # Stationary state stock values (common to both countries) from Zezza
    # (2006):
    # http://gennaro.zezza.it/software/eviews/v6/gl06open.prg
    country_stationary_state = {
        'Bcb': 11.622,
        'Bh':  64.865,
        'Bs':  76.486,
        'or_': 10,
        'V':   86.487,
        'Hh':  86.487 - 64.865,
        'Hs':  86.487 - 64.865,
    }

    baseline = OPEN({
        'North': Country(range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, lambda_0=0.635, lambda_1=5, lambda_2=0.01, mu=0.18781, **country_stationary_state),
        'South': Country(range(1945, 2010 + 1), alpha_1=0.7, alpha_2=0.3, lambda_0=0.670, lambda_1=6, lambda_2=0.07, mu=0.18781, **country_stationary_state),
    })

    # Set global exchange rate and price of gold
    baseline.xr_bar = 1
    baseline.xr[0] = baseline.xr_bar[0]

    baseline.p_g_bar = 1

    # Set country-level fiscal and monetary policies
    baseline.submodels['North'].G = 20
    baseline.submodels['North'].theta = 0.2
    baseline.submodels['North'].r_bar = 0.025
    baseline.submodels['North'].r = baseline.submodels['North'].r_bar[0]

    baseline.submodels['South'].G = 20
    baseline.submodels['South'].theta = 0.2
    baseline.submodels['South'].r_bar = 0.025
    baseline.submodels['South'].r = baseline.submodels['South'].r_bar[0]

    # Solve
    baseline.solve(max_iter=250)

    # -------------------------------------------------------------------------
    # 6.8 Rejecting the Mundell–Fleming approach and adopting the compensation
    #     approach

    # 6.8.1 Ever-falling gold reserves
    #       An increase in the South propensity to import
    import_propensity_scenario = baseline.copy()
    import_propensity_scenario.submodels['South']['mu', 1960:] = 0.20781
    import_propensity_scenario.solve(start=1960, max_iter=250)  # No need to solve pre-1960 again

    # 6.8.4 A further example of the compensation thesis: increasing the
    #       propensity to save
    saving_propensity_scenario = baseline.copy()
    saving_propensity_scenario.submodels['South']['alpha_2', 1960:] = 0.6
    saving_propensity_scenario.solve(start=1960, max_iter=250)  # No need to solve pre-1960 again