# -*- coding: utf-8 -*-
"""
define_simple
=============
Example fsic implementation of DEFINE-SIMPLE, from Dafermos and Nikolaidi
(2021), based on the:

* original R code: https://github.com/DEFINE-model/SIMPLE
* model description: https://define-model.org/define-simple/

This script runs three simulations:

1. Baseline
2. Green investment scenario:
    * more favourable credit conditions for green technologies, with lower
      interest rates for green relative to conventional technologies
    * a higher autonomous propensity to invest in green technologies
3. Degrowth scenario: lower propensities to consume and invest, leading to
   lower global growth

Outputs:

* three CSV files of model results, one per run: 'baseline.csv',
  'green-investment.csv' and 'degrowth.csv'
* 'results.png', a single file of charts, reproducing those from the original R
  script

As well as fsic (which depends in turn on NumPy), this script also requires:

* `pandas`, to generate DataFrames of results from the model runs
* `matplotlib`, to recreate the charts from the original R implementation

Reference:

    Dafermos, Y., Nikolaidi, M. (2021)
    'DEFINE-SIMPLE'
    https://define-model.org/define-simple/
    https://github.com/DEFINE-model/SIMPLE
"""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from fsic import BaseModel


class DEFINE_Simple(BaseModel):
    """fsic implementation of DEFINE-SIMPLE, from Dafermos and Nikolaidi (2021).

    Reference
    ---------
    Dafermos, Y., Nikolaidi, M. (2021)
    'DEFINE-SIMPLE'
    https://define-model.org/define-simple/
    https://github.com/DEFINE-model/SIMPLE
    """

    ENDOGENOUS: List[str] = [
        'Y_D', 'CO', 'D',
        'Y',
        'TP', 'DP', 'RP', 'BP',
        'I', 'IC', 'IG', 'beta',
        'K', 'KC', 'KG', 'r',
        'L', 'LC', 'LG',
        'EMIS_IN', 'CI',
        'Y_star', 'u', 'g_Y', 'lev',
        'D_red',
    ]

    EXOGENOUS: List[str] = ['CI_max', 'CI_min', 'v',]

    PARAMETERS: List[str] = [
        'alpha_0', 'alpha_1',
        'beta_0', 'beta_1',
        'c_1', 'c_2',
        'ci_1', 'ci_2',
        'int_C', 'int_D', 'int_G',
        's_F', 's_W',
    ]

    ERRORS: List[str] = []

    NAMES: List[str] = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
    CHECK: List[str] = ENDOGENOUS

    LAGS: int = 1
    LEADS: int = 0

    __slots__: Dict[str, str] = {
        'BP': 'Profits of banks ($tn) [19]',
        'CI': 'Carbon intensity (GtCO2 / $tn) [22]',
        'CI_max': 'Maximum potential value of carbon intensity (GtCO2 / $tn)',
        'CI_min': 'Minimum potential value of carbon intensity (GtCO2 / $tn)',
        'CO': 'Consumption expenditures ($tn) [2]',
        'D': 'Deposits ($tn) [3]',
        'D_red': 'Deposits ($tn) [20] - redundant equation',
        'DP': 'Distributed profits ($tn) [7]',
        'EMIS_IN': 'Industrial CO2 emissions (GtCO2) [21]',
        'I': 'Investment ($tn) [8]',
        'IC': 'Conventional investment ($tn) [12]',
        'IG': 'Green investment ($tn) [10]',
        'K': 'Capital stock ($tn) [15]',
        'KC': 'Conventional capital stock ($tn) [14]',
        'KG': 'Green capital stock ($tn) [13]',
        'L': 'Loans ($tn) [18]',
        'LC': 'Conventional loans ($tn)',
        'LG': 'Green loans ($tn)',
        'RP': 'Retained profits ($tn) [6]',
        'TP': 'Total profits of firms ($tn) [5]',
        'Y': 'Output ($tn) [4]',
        'Y_D': 'Disposable income of households ($tn) [1]',
        'Y_star': 'Potential output ($tn) [23]',

        'alpha_0': 'Base level of total investment (as a ratio to the total capital stock)',
        'alpha_1': 'Investment parameter on rate of profit',

        'beta': 'Share of green investment in total investment ([0, 1]) [11]',
        'beta_0': 'Autonomous share of green investment in total investment',
        'beta_1': 'Adjustment for the interest rate differential between green and conventional loans',

        'c_1': 'Marginal propensity to consume out of disposable income',
        'c_2': 'Marginal propensity to consume out of deposits (wealth)',

        'ci_1': 'Capital carbon intensity parameter 1: Denominator coefficient on exponential term',
        'ci_2': 'Capital carbon intensity parameter 2: Coefficient in exponential calculation',

        'g_Y': 'Growth rate of output [25]',

        'int_C': 'Interest rate on conventional loans',
        'int_D': 'Interest rate on deposits',
        'int_G': 'Interest rate on green loans',

        'lev': 'Leverage ratio [26]',

        'r': 'Rate of profit [9]',

        's_F': 'Retention rate of firms ([0, 1])',
        's_W': 'Wage share of output ([0, 1])',

        'u': 'Capacity utilisation [24]',
        'v': 'Capital productivity',
    }

    def _evaluate(self, t: int, *, errors: str = 'raise', catch_first_error: bool = True, iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
        """Evaluate the system of equations for the period at integer position `t` in the model's `span`.

        Parameters
        ----------
        t : int
            Position in `span` of the period to solve
        errors : str
            User-specified treatment on encountering numerical solution
            errors. Note that it is up to the user's over-riding code to decide
            how to handle this.
        catch_first_error : bool
            If `True` (default) and `errors='raise'`, raise an exception
            (`SolutionError`) on the first numerical error/warning during
            solution. This identifies the problem statement in the stack trace
            without modifying any values at this point.
            If `False`, only check for errors after completing an iteration,
            raising an exception (`SolutionError`) after the fact. This allows
            numerical errors (NaNs, Infs) to propagate through the solution
            before raising the exception.
        iteration : int
            The current iteration count. This is not guaranteed to take a
            non-`None` value if the user has over-ridden the default calling
            `solve_t()` method. Note that it is up to the user's over-riding
            code to decide how to handle this.
        kwargs :
            Further keyword arguments for solution
        """
        # Any equation numbers with an 'a' suffix mark modifications from the
        # original equation/code

        # Households ----------------------------------------------------------

        # 1: Disposable income of households
        #    Y_D[t] = s_W[t] * Y[t] + DP[t] + BP[t] + int_D[t] * D[t-1]
        # Disposable income, Y_D, is the sum of:
        #  - the wage share, s_W, of output, Y
        #  - distributed profits, DP
        #  - bank profits, BP
        #  - interest, int_D, earned on (previous-period) deposits, D
        self._Y_D[t] = (self._s_W[t] * self._Y[t] +
                        self._DP[t] +
                        self._BP[t] +
                        self._int_D[t] * self._D[t-1])

        # 2a: Consumption expenditure
        #     CO[t] = (c_1[t] + c_1_change[t]) * Y_D[t-1] + c_2[t] * D[t-1]
        # Household consumption depends on:
        #  - lagged disposable income, Y_D (MPC: c_1)
        #  - lagged deposits/wealth, D (MPC: c_2)
        # NB In the original implementation, the first pair of terms, which
        #    determine consumption out of lagged disposable income, includes an
        #    additive adjustment on c_1: c_1_change (as above). This is
        #    excluded below and c_1 should be adjusted directly in any
        #    scenarios.
        self._CO[t] = (self._c_1[t] * self._Y_D[t-1] +
                       self._c_2[t] * self._D[t-1])

        # 3: Deposits
        #    D[t] = D[t-1] + Y_D[t] - CO[t]
        # Deposits, D, cumulate household saving (Y_D - CO)
        self._D[t] = self._D[t-1] + self._Y_D[t] - self._CO[t]

        # Firms ---------------------------------------------------------------

        # 4: Output
        #    Y[t] = CO[t] + I[t]
        # Output, Y, is the sum of consumption, CO, and investment, I
        self._Y[t] = self._CO[t] + self._I[t]

        # 5: Total profits of firms
        #    TP[t] = Y[t] - s_W[t] * Y[t] - int_C[t] * LC[t-1] - int_G[t] * LG[t-1]
        # Total profits are calculated as output, Y, less:
        #  - labour costs: labour's share of output, s_W * Y
        #  - interest paid on conventional loans: int_C * LC
        #  - interest paid on green loans: int_G * LG
        self._TP[t] = (self._Y[t] -
                       (self._s_W[t] * self._Y[t]) -
                       (self._int_C[t] * self._LC[t-1]) -
                       (self._int_G[t] * self._LG[t-1]))

        # 6: Retained profits
        #    RP[t] = s_F[t] * TP[t]
        # Retained profits, RP, are calculated as a proportion, s_F, (firms'
        # retention rate) of total profits, TP
        self._RP[t] = self._s_F[t] * self._TP[t]

        # 7: Distributed profits
        #    DP[t] = TP[t] - RP[t]
        # Distributed profits, DP, are the remainder of total profits, TP,
        # after subtracting retained profits, RP (as above)
        self._DP[t] = self._TP[t] - self._RP[t]

        # 8a: Investment
        #     I[t] = (alpha_0[t] + alpha_0_change[t] + alpha_1[t] * r[t-1]) * K[t-1]
        # Investment, I, is a proportion of the lagged capital stock, K, with
        # components:
        #  - alpha_0: a fixed/constant proportion
        #  - alpha_1: a coefficient applied to the (lagged) rate of profit, r
        # NB In the original implementation, there are two alpha_0 terms: the
        #    main term and an additive adjustment, alpha_0_change (as
        #    above). This is excluded below and alpha_0 should be adjusted
        #    directly in any scenarios.
        self._I[t] = (self._alpha_0[t] + self._alpha_1[t] * self._r[t-1]) * self._K[t-1]

        # 9: Rate of profit
        #    r[t] = TP[t] / K[t]
        # Ratio of total profits to capital stock
        self._r[t] = self._TP[t] / self._K[t]

        # 10: Green investment
        #     IG[t] = beta[t] * I[t]
        # Green investment is a proportion, beta, of total investment
        self._IG[t] = self._beta[t] * self._I[t]

        # 11a: Share of green investment in total investment
        #      beta[t] = beta_0[t] - beta_1[t] * (int_G[t] - int_C[t]) + beta_0_change[t]
        # The share of green investment, beta, is a base share of investment,
        # beta_0, adjusted for the difference in interest rates between green
        # and conventional investment e.g. if the interest rate on green
        # investment exceeds that on conventional investment, the share of
        # green investment is depressed
        # NB In the original implementation, there are two beta_0 terms: the
        #    main term and an additive adjustment, beta_0_change (as
        #    above). This is excluded below and beta_0 should be adjusted
        #    directly in any scenarios.
        self._beta[t] = (self._beta_0[t] -
                         self._beta_1[t] * (self._int_G[t] - self._int_C[t]))

        # 12: Conventional investment
        #     IC[t] = I[t] - IG[t]
        # The difference between total and green investment
        self._IC[t] = self._I[t] - self._IG[t]

        # 13: Green capital stock
        #     KG[t] = KG[t-1] + IG[t]
        self._KG[t] = self._KG[t-1] + self._IG[t]

        # 14: Conventional capital stock
        #     KC[t] = KC[t-1] + IC[t]
        self._KC[t] = self._KC[t-1] + self._IC[t]

        # 15: Capital stock
        #     K[t] = KC[t] + KG[t]
        # Sum of conventional and green capital stocks
        self._K[t] = self._KC[t] + self._KG[t]

        # 16: Green loans
        #     LG[t] = LG[t-1] + IG[t] - beta[t] * RP[t]
        # The stock of green loans, LG, increases by the amount of green
        # investment, IG, less anything paid out of retained profits, beta *
        # RP. The split of retained profit channelled to green investments
        # follows the split of investment i.e. according to beta.
        self._LG[t] = self._LG[t-1] + (self._IG[t] - self._beta[t] * self._RP[t])

        # 17a: Conventional loans
        #      LC[t] = LC[t-1] + IC[t] + IG[t] - RP[t] - (LG[t] - LG[t-1])
        # After rearranging, this equation reads more like Equation 16 above,
        # with the stock of conventional loans, LC, increasing by investment,
        # IC, less anything paid out of retained profits, (1 - beta) * RP.
        self._LC[t] = self._LC[t-1] + (self._IC[t] - (1 - self._beta[t]) * self._RP[t])

        # 18: Total loans
        #     L[t] = LC[t] + LG[t]
        # Sum of conventional and green loans
        self._L[t] = self._LC[t] + self._LG[t]

        # Banks ---------------------------------------------------------------

        # 19: Profits of banks
        #     BP[t] = int_C[t] * LC[t-1] + int_G[t] * LG[t-1] - int_D[t] * D[t-1]
        # Banks' profits are the interest received on loans (conventional and
        # green) less interest paid on deposits
        self._BP[t] = (self._int_C[t] * self._LC[t-1] +
                       self._int_G[t] * self._LG[t-1] -
                       self._int_D[t] * self._D[t-1])

        # 20: Deposits (redundant equation)
        #     D_red[t] = L[t]
        # By the accounting of the model, D_red should equal D (deposits
        # calculated from household saving)
        self._D_red[t] = self._L[t]

        # Emissions -----------------------------------------------------------

        # 21: Industrial CO2 emissions
        #     EMIS_IN[t] = CI[t] * Y[t]
        # Calculated as the emissions intensity of output multiplied by output
        self._EMIS_IN[t] = self._CI[t] * self._Y[t]

        # 22a: Carbon intensity
        #      CI[t] = CI_max[t] -
        #              ((CI_max[t] - CI_min[t]) / (1 + ci_1[t] * exp(-ci_2[t] * (KG[t-1] / KC[t-1]))))
        # This equation calculates carbon intensity as:
        #  - maximum possible carbon intensity (per unit of output)
        #  - minus a term that increases with the degree of green capital
        # such that carbon intensity falls with a greener capital stock.
        #
        # At the limits:
        #  - all capital is conventional: KG/KC = 0 such that CI = CI_max - (CI_max - CI_min) / (1 + ci_1)
        #  - all capital is green: KG/KC = inf such that CI = CI_min

        # Calculate exponential term
        if np.isclose(self._KC[t-1], 0):
            # All capital is green
            ratio_green_conventional_capital = np.inf
            exp_term = -np.inf
        else:
            # At least some capital is conventional
            ratio_green_conventional_capital = self._KG[t-1] / self._KC[t-1]
            exp_term = -self._ci_2[t] * ratio_green_conventional_capital

        # Insert into main equation
        self._CI[t] = (
            self._CI_max[t] -
            ((self._CI_max[t] - self._CI_min[t]) / (1 + self._ci_1[t] * np.exp(exp_term)))
        )

        # Auxiliary equations -------------------------------------------------

        # 23: Potential output
        #     Y_star[t] = v[t] * K[t]
        # Capital stock, K, is endogenous while capital productivity, v, is
        # exogenous and must be set before solution
        self._Y_star[t] = self._v[t] * self._K[t]

        # 24: Capacity utilisation
        #     u[t] = Y[t] / Y_star[t]
        self._u[t] = self._Y[t] / self._Y_star[t]

        # 25a: Growth rate of output
        #      g_Y[t] = (Y[t] - Y[t-1]) / Y[t-1]
        # The below is mathematically identical to the above
        self._g_Y[t] = (self._Y[t] / self._Y[t-1]) - 1

        # 26: Leverage ratio
        #     lev[t] = L[t] / K[t]
        # Ratio of loans to capital stock
        self._lev[t] = self._L[t] / self._K[t]


# Baseline parameters come from the original R script, with numerical values
# transferred to the dictionary below
BASELINE_PARAMETERS = {
    'CI_max': 0.6,
    'CI_min': 0.05,

    'alpha_0': 0.02443275,
    'alpha_1': 0.1,
    'beta_0': 0.033942356,
    'beta_1': 1,
    'c_1': 0.937339324,
    'c_2': 0.0498,
    'ci_1': 2.451037,
    'ci_2': 3.579244,
    'int_C': 0.08,
    'int_D': 0.025,
    'int_G': 0.08,
    's_F': 0.550832042,
    's_W': 0.54,
    'v': 0.163094338,
}

# Starting (first-period) values come from the original R script, with
# numerical values transferred to the dictionary below
STARTING_VALUES = {
    'Y_D': 67.5202699513256,
    'CO': 65.3068,
    'D': 78.54002,
    'Y': 85.93,
    'TP': 33.4216759961127,
    'DP': 15.0119459474383,
    'RP': 18.4097300486743,
    'BP': 4.19796025267249,
    'I': 20.6232,
    'IC': 19.9232,
    'IG': 0.7,
    'beta': 0.0339423561813879,
    'K': 731.76802758917,
    'KC': 706.930096554586,
    'KG': 24.8379310345833,
    'r': 0.0456725010331776,
    'L': 78.54002,
    'LC': 75.8741866666666,
    'LG': 2.66583333333333,
    'EMIS_IN': 36.6,
    'CI': 0.425928080996159,
    'Y_star': 119.347222222222,
    'u': 0.72,
    'g_Y': 0.029,
    'lev': 0.107329122124605,
    'D_red': 78.54002,
}


if __name__ == '__main__':
    # Baseline ----------------------------------------------------------------
    baseline = DEFINE_Simple(range(2018, 2100 + 1), strict=True,
                             **BASELINE_PARAMETERS)

    # Insert values for the first period (which can't be solved, because of the
    # lags in the model)
    for k, v in STARTING_VALUES.items():
        baseline[k][0] = v

    # Solve with `offset=-1`, to copy the previous period's endogenous values
    # before solving (otherwise, the default order of the equations generates
    # NaNs from divide-by-zeroes)
    baseline.solve(offset=-1)

    # Green investment scenario -----------------------------------------------
    # Lower interest rates on green investment, raise interest rates on
    # conventional investment and increase autonomous investment in green
    # technologies
    green_investment_scenario = baseline.copy()
    green_investment_scenario['int_G', 2022:] = 0.04
    green_investment_scenario['int_C', 2022:] = 0.12
    green_investment_scenario['beta_0', 2022:] += 0.3

    green_investment_scenario.solve()

    # Degrowth scenario -------------------------------------------------------
    # Lower propensity to consume out of disposable income and lower autonomous
    # investment
    degrowth_scenario = baseline.copy()
    degrowth_scenario['c_1', 2022:] -= 0.01
    degrowth_scenario['alpha_0', 2022:] -= 0.01

    degrowth_scenario.solve()

    # Write results to CSV files (requires pandas) ----------------------------
    baseline.to_dataframe().to_csv('baseline.csv')
    green_investment_scenario.to_dataframe().to_csv('green-investment.csv')
    degrowth_scenario.to_dataframe().to_csv('degrowth.csv')

    # Create charts (requires matplotlib) -------------------------------------
    styles = {
        'baseline': {'label': 'Baseline',         'color': '#FF4F2E', 'linestyle': '-' },
        'green':    {'label': 'Green investment', 'color': '#77C3AF', 'linestyle': '--'},
        'degrowth': {'label': 'Degrowth',         'color': '#33C3F0', 'linestyle': '--'},
    }

    _, axes = plt.subplots(1, 5, figsize=(30, 6))
    plt.suptitle('DEFINE-SIMPLE: Results')

    # Figure 1: Annual output growth
    axes[0].plot(baseline.span, [0] * len(baseline.span),
                 label='_nolegend_', linewidth=0.5, color='black')

    axes[0].plot(baseline.span, baseline.g_Y * 100, **styles['baseline'])
    axes[0].plot(green_investment_scenario.span, green_investment_scenario.g_Y * 100, **styles['green'])
    axes[0].plot(degrowth_scenario.span, degrowth_scenario.g_Y * 100, **styles['degrowth'])

    axes[0].set_title('Figure 1: Annual output growth')
    axes[0].set_xlim(2018, 2100)
    axes[0].set_ylabel('Output growth (% pa)')

    # Figure 2: Leverage
    axes[1].plot(baseline.span, baseline.lev * 100, **styles['baseline'])
    axes[1].plot(green_investment_scenario.span, green_investment_scenario.lev * 100, **styles['green'])
    axes[1].plot(degrowth_scenario.span, degrowth_scenario.lev * 100, **styles['degrowth'])

    axes[1].set_title('Figure 2: Leverage (ratio of loans to capital)')
    axes[1].set_xlim(2018, 2100)
    axes[1].set_ylim(0, None)
    axes[1].set_ylabel("Firms' leverage (%)")

    # Figure 3: Proportion of green investment
    axes[2].plot(baseline.span, baseline.beta * 100, **styles['baseline'])
    axes[2].plot(green_investment_scenario.span, green_investment_scenario.beta * 100, **styles['green'])
    axes[2].plot(degrowth_scenario.span, degrowth_scenario.beta * 100, **styles['degrowth'])

    axes[2].set_title('Figure 3: Proportion of green investment')
    axes[2].set_xlim(2018, 2100)
    axes[2].set_ylim(0, None)
    axes[2].set_ylabel('Share of green investment in total investment (%)')

    # Figure 4: Annual CO2 emissions
    axes[3].plot(baseline.span, baseline.EMIS_IN, **styles['baseline'])
    axes[3].plot(green_investment_scenario.span, green_investment_scenario.EMIS_IN, **styles['green'])
    axes[3].plot(degrowth_scenario.span, degrowth_scenario.EMIS_IN, **styles['degrowth'])

    axes[3].set_title('Figure 4: Annual global $CO_2$ emissions')
    axes[3].set_xlim(2018, 2100)
    axes[3].set_ylim(0, None)
    axes[3].set_ylabel('$CO_2$ emissions (Gt$CO_2$)')

    # Figure 5: CO2 intensity
    axes[4].plot(baseline.span, baseline.CI, **styles['baseline'])
    axes[4].plot(green_investment_scenario.span, green_investment_scenario.CI, **styles['green'])
    axes[4].plot(degrowth_scenario.span, degrowth_scenario.CI, **styles['degrowth'])

    axes[4].set_title('Figure 5: $CO_2$ intensity')
    axes[4].set_xlim(2018, 2100)
    axes[4].set_ylim(0, None)
    axes[4].set_ylabel('$CO_2$ intensity (Gt$CO_2$ / \$tn)')

    axes[4].legend(loc='lower right')

    # Save to disk
    plt.savefig('results.png')
