# -*- coding: utf-8 -*-
"""
estimate_equations
==================
Estimate the equations of Klein Model I using different estimation techniques:
 - ordinary least squares (OLS)
 - two-stage least squares (2SLS)
 - limited information maximum likelihood (LIML)
 - three-stage least squares (3SLS)

Download the data from Greene (2012) and run 'process_data.py' (to create
'data.csv') before running this script.

This script uses:

* [`pandas`](https://pandas.pydata.org/) to read in the data and construct the
  final sets of parameters for solution
* Kevin Sheppard's [`linearmodels`](https://github.com/bashtage/linearmodels)
  package for estimation

References:

    Giles, D. E. A. (2012)
    'Estimating and simulating an SEM',
    *Econometrics Beat: Dave Giles' blog*, 19/05/2012
    https://davegiles.blogspot.com/2012/05/estimating-simulating-sem.html

    Greene, W. H. (2012)
    *Econometric analysis*,
    7th edition,
    Pearson
    Datasets available from:
    http://people.stern.nyu.edu/wgreene/Text/econometricanalysis.htm
    (see Data Sets > 'Table F10.3: Klein's Model I')

    Klein, L. R. (1950)
    *Economic fluctuations in the United States, 1921-1941*,
    *Cowles Commission for Research in Economics*, **11**,
    New York: John Wiley & Sons / London: Chapman & Hall
    https://cowles.yale.edu/cfm-11
"""

from numbers import Number
from typing import Any, Dict, Mapping
import warnings

from linearmodels import IV2SLS, IVLIML, IV3SLS

from pandas import DataFrame
import pandas as pd

from fsic.functions import lag  # noqa: F401


def prepend_keys(parameters: Mapping[str, Any], prefix: str) -> Dict[str, Number]:
    """Prepend `prefix` to the keys in `parameters`."""
    return {'{}{}'.format(prefix, k): v for k, v in parameters.items()}


def pack_parameters(*args: Mapping[str, Any]) -> Dict[str, Any]:
    """Consolidate the arguments (sets of parameters) into a single dictionary of parameters."""
    # Mapping of names:
    #  - from econometric estimation in this script (keys)
    #  - to model parameters in the solution script (values)
    replacements = {
        'consumption_Intercept':   'alpha_0',
        'consumption_P':           'alpha_1',
        'consumption_lag(P)':      'alpha_2',
        'consumption_W':           'alpha_3',

        'investment_Intercept':    'beta_0',
        'investment_P':            'beta_1',
        'investment_lag(P)':       'beta_2',
        'investment_lag(K)':       'beta_3',

        'private_wages_Intercept': 'gamma_0',
        'private_wages_X':         'gamma_1',
        'private_wages_lag(X)':    'gamma_2',
        'private_wages_time':      'gamma_3',
    }  # fmt: skip

    # Rename the parameters and store to a single dictionary
    parameters = {}

    for group in args:
        parameters.update({replacements[k]: v for k, v in group.items()})

    return parameters


if __name__ == '__main__':
    data = pd.read_csv('data.csv', index_col=0)
    parameters = {}

    # OLS (equivalent to IV2SLS with no instruments) --------------------------
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')

        consumption = IV2SLS.from_formula('C ~ P + lag(P) + W', data=data).fit()
        investment = IV2SLS.from_formula('I ~ P + lag(P) + lag(K)', data=data).fit()
        private_wages = IV2SLS.from_formula('Wp ~ X + lag(X) + time', data=data).fit()

    parameters['OLS'] = pack_parameters(
        prepend_keys(consumption.params, 'consumption_'),
        prepend_keys(investment.params, 'investment_'),
        prepend_keys(private_wages.params, 'private_wages_'),
    )

    # 2SLS --------------------------------------------------------------------
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')

        consumption = IV2SLS.from_formula(
            'C ~ 1 + lag(P) + [P + W ~ Wg + lag(K) + lag(X) + time + G + T]',
            data=data,
        ).fit()

        investment = IV2SLS.from_formula(
            'I ~ 1 + lag(P) + lag(K) + [P ~ Wg + lag(X) + time + G + T]',
            data=data,
        ).fit()

        private_wages = IV2SLS.from_formula(
            'Wp ~ 1 + lag(X) + time + [X ~ lag(P) + Wg + lag(K) + G + T]',
            data=data,
        ).fit()

    parameters['2SLS'] = pack_parameters(
        prepend_keys(consumption.params, 'consumption_'),
        prepend_keys(investment.params, 'investment_'),
        prepend_keys(private_wages.params, 'private_wages_'),
    )

    # LIML --------------------------------------------------------------------
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')

        consumption = IVLIML.from_formula(
            'C ~ 1 + lag(P) + [P + W ~ Wg + lag(K) + lag(X) + time + G + T]',
            data=data,
        ).fit()

        investment = IVLIML.from_formula(
            'I ~ 1 + lag(P) + lag(K) + [P ~ Wg + lag(X) + time + G + T]',
            data=data,
        ).fit()

        private_wages = IVLIML.from_formula(
            'Wp ~ 1 + lag(X) + time + [X ~ lag(P) + Wg + lag(K) + G + T]',
            data=data,
        ).fit()

    parameters['LIML'] = pack_parameters(
        prepend_keys(consumption.params, 'consumption_'),
        prepend_keys(investment.params, 'investment_'),
        prepend_keys(private_wages.params, 'private_wages_'),
    )

    # 3SLS --------------------------------------------------------------------
    equations = {
        'consumption':   'C ~ 1 + lag(P) + [P + W ~ Wg + lag(K) + lag(X) + time + G + T]',
        'investment':    'I ~ 1 + lag(P) + lag(K) + [P ~ Wg + lag(X) + time + G + T]',
        'private_wages': 'Wp ~ 1 + lag(X) + time + [X ~ lag(P) + Wg + lag(K) + G + T]',
    }  # fmt: skip

    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')

        system = IV3SLS.from_formula(equations, data=data).fit()

    parameters['3SLS'] = pack_parameters(system.params)

    # Write results to a CSV file ---------------------------------------------
    results = DataFrame(parameters)
    results.to_csv('parameters.csv')
