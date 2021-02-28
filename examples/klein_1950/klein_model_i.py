# -*- coding: utf-8 -*-
"""
klein_model_i
=============
Example FSIC implementation of Klein Model I, from Klein (1950).

Run 'estimate_equations.py' first to generate 'parameters.csv', a CSV file of
parameter estimates for the equations.

While FSIC only requires NumPy, this example also uses:

* `pandas`, to read the input data and, using `fsictools`, generate DataFrames
  of model results
* `matplotlib` to create plots of the endogenous variables

Reference:

    Klein, L. R. (1950)
    *Economic fluctuations in the United States, 1921-1941*,
    *Cowles Commission for Research in Economics*, **11**,
    New York: John Wiley & Sons / London: Chapman & Hall
    https://cowles.yale.edu/cfm-11
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pandas import DataFrame
import pandas as pd

import fsic
import fsictools


# Variable names and descriptions, adapted from Klein (1950)
descriptions = {
    'C': 'Consumption ($1934bn)',
    'I': 'Net investment ($1934bn)',
    'G': 'Exogenous investment ($1934bn)',
    'Y': 'Net national income ($1934bn)',
    'T': 'Business taxes ($1934bn)',
    'K': 'End-of-year stock of capital ($1934bn)',                     # K_{-1}, i.e. lagged, in Klein (1950)
    'P': 'Profits ($1934bn)',                                          # Π in Klein (1950)
    'Wp': 'Labor income originating in private employment ($1934bn)',  # W1 in Klein (1950)
    'Wg': 'Labor income originating in government ($1934bn)',          # W2 in Klein (1950)
    'time': 'Time trend (1931=0)',                                     # (t - 1931) in Klein (1950)
}

# Define the model's equations and parse to a set of symbols
script = '''
C  = {alpha_0} + {alpha_1} * P + {alpha_2} * P[-1] + {alpha_3} * (Wp + Wg) + <C_r>
I  = {beta_0}  + {beta_1}  * P + {beta_2}  * P[-1] + {beta_3}  * K[-1]     + <I_r>

Wp = ({gamma_0} +
      {gamma_1} * (Y     + T     - Wg    ) +
      {gamma_2} * (Y[-1] + T[-1] - Wg[-1]) +
      {gamma_3} * time +
      <Wp_r>)

Y = (C + I + G) - T
P = Y - (Wp + Wg)
K = K[-1] + I
'''
symbols = fsic.parse_model(script)

# Subclass the output from `fsic.build_model()`, to bind the variable
# descriptions (use `help()` on either the class or a class instance to see
# these descriptions in the data descriptors)
class KleinModelI(fsic.build_model(symbols)):
    __slots__ = descriptions


if __name__ == '__main__':
    # Setup -------------------------------------------------------------------
    # Read data
    data = pd.read_csv('data.csv', index_col=0)
    data = data.ffill()  # Remove the NaN at the end of 'K' (capital stock)

    # Read parameters
    parameters = pd.read_csv('parameters.csv', index_col=0)

    # Instantiate the model with data, but no parameters
    base = KleinModelI.from_dataframe(data)

    # Dictionary to store the results from the various runs
    results = {
        # `base` holds the original (i.e. actual) data
        'Actual': fsictools.model_to_dataframe(base)
    }


    # Solve for each set of parameters ----------------------------------------
    for technique, estimates in parameters.iteritems():
        # Copy the base and insert the parameters
        model = base.copy()
        model.replace(**estimates)

        # Solve from 1921 onwards, not 1920, to avoid problem of lagged NaNs
        model.solve(start=1921)

        results[technique] = fsictools.model_to_dataframe(model)


    # Create plots of the endogenous variables --------------------------------
    _, axes = plt.subplots(3, 2, figsize=(12, 15))

    plt.suptitle('Klein Model I: Comparison of actual and simulated results')

    for ax, name in zip(axes.flatten(), sorted(KleinModelI.ENDOGENOUS)):
        # Extract results (across multiple runs) for the current variable as a DataFrame
        plot_data = DataFrame({k: df[name] for k, df in results.items()})

        ax.plot(plot_data['Actual'].index, plot_data['Actual'], color='#FF4F2E', linestyle='-')
        ax.plot(plot_data['OLS'].index,    plot_data['OLS'],    color='#77C3AF', linestyle='--')
        ax.plot(plot_data['2SLS'].index,   plot_data['2SLS'],   color='#33C3F0', linestyle='--')
        ax.plot(plot_data['3SLS'].index,   plot_data['3SLS'],   color='#4563F2', linestyle='--')

        description = KleinModelI.__slots__[name].split('(')[0].strip()
        ax.set_title('{}: {}'.format(name, description))

        # Set axis limits and labels
        ax.set_xlim(1920, 1941)
        ax.set_xticks(range(1920, 1941, 5))

        ax.set_ylabel('$1934bn')

    # Add legend
    axes[-1, -1].legend(handles=[
        Line2D([], [], color='#FF4F2E', linestyle='-',  label='Actual'),
        Line2D([], [], color='#77C3AF', linestyle='--', label='OLS'),
        Line2D([], [], color='#33C3F0', linestyle='--', label='2SLS'),
        Line2D([], [], color='#4563F2', linestyle='--', label='3SLS'),
    ], loc='lower right', bbox_to_anchor=(1.015, -0.425))

    plt.savefig('results.png')
