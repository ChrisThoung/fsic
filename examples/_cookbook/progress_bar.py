# -*- coding: utf-8 -*-
"""
progress_bar
============
Example of how to add a progress bar using
[`tqdm`](https://github.com/tqdm/tqdm) to report the number of periods that
have been and are still to be solved.

For this, the `fsic` `BaseModel` class implements an `iter_periods()`
method. This method returns a `fsic` `PeriodIter` object. `PeriodIter` is an
iterator over (index, label) pairs and has a defined length (supporting
`len()`). The length corresponds to the number of periods (index-label pairs)
in the sequence to allow the slightly tidier:

    for t, period in tqdm(model.iter_periods()):
        ...

over:

    for t, period in tqdm(list(model.iter_periods())):
        ...

This example shows two possible implementations:

1. Call `iter_periods()` directly each time, wrapping the iterator with `tqdm`
2. Over-ride `iter_periods()` at the class level, again using `tqdm` but with
   `ProgressBarMixin` from `fsic.extensions`

The example model in this script is a simplified five-equation version of
Godley and Lavoie’s (2007) Model *SIM*.

References:

    Godley, W., Lavoie, M. (2007),
    *Monetary economics: an integrated approach to
    credit, money, income, production and wealth*,
    Palgrave Macmillan
"""

import time

from tqdm import tqdm

from fsic.extensions import ProgressBarMixin
import fsic


# Define the example model
script = '''
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
'''

# Parse the script and generate a class definition
SIM = fsic.build_model(fsic.parse_model(script))


if __name__ == '__main__':
    # Initialise a model instance (which will be copied in each example)
    base = SIM(range(1945, 2010 + 1),
               alpha_1=0.6, alpha_2=0.4,
               G=20, theta=0.2)

    # -------------------------------------------------------------------------
    # 1. Call `iter_periods()` directly each time, wrapping the iterator with
    #    `tqdm` (two ways shown below)

    # 1a. Wrap `iter_periods()`
    print('1a. Wrap `iter_periods()`:')

    model = base.copy()
    for t, period in tqdm(model.iter_periods()):
        model.solve_t(t)

    # 1b. Use a context manager and update the period each step
    print('\n1b. Use a context manager and update the period each step:')

    model = base.copy()
    period_iter = model.iter_periods()

    with tqdm(period_iter) as pbar:
        for t, period in period_iter:
            # Update the progress bar with the current period
            pbar.set_description(str(period))

            # Add a delay to make the example clearer (not needed for regular
            # use)
            time.sleep(0.03)

            model.solve_t(t)

            # Advance the progress bar on solution
            pbar.update(1)

        # Set the final status
        pbar.set_description('DONE')


    # -------------------------------------------------------------------------
    # 2. Over-ride `iter_periods()` at the class level using a mixin

    # Apply the extension to a new model class
    class SIM_ProgressBar(ProgressBarMixin, SIM):
        pass

    print('\n2. Over-ride `iter_periods()` at the class level:')

    model = SIM_ProgressBar(range(1945, 2010 + 1),
                            alpha_1=0.6, alpha_2=0.4,
                            G=20, theta=0.2)
    model.solve(show_progress=True)
