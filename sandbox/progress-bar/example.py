# -*- coding: utf-8 -*-
"""
example
=======
Experiments extending [`tqdm`](https://github.com/tqdm/tqdm)-based progress
bars, to eventually augment `ProgressBarMixin`(?).

This script runs various examples of the augmented `ProgressBarMixin`.

See the notes in 'progress_bar.py' for discussion of the implementation.

The example model in this script is a simplified five-equation version of
Godley and Lavoie’s (2007) Model *SIM*.

Reference:

    Godley, W., Lavoie, M. (2007),
    *Monetary economics: an integrated approach to
    credit, money, income, production and wealth*,
    Palgrave Macmillan
"""

import time

from pandas.testing import assert_frame_equal
import pandas as pd

from tqdm import tqdm

import fsic

from progress_bar import ProgressBarMixin


# Set up an example model
SCRIPT = """
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
"""
SYMBOLS = fsic.parse_model(SCRIPT)

SIM = fsic.build_model(SYMBOLS)


# Wrap the model class with a progress bar
class SIM_ProgressBar(ProgressBarMixin, SIM):
    def solve_t(self, *args, **kwargs):
        time.sleep(0.05)  # Add a delay for readability
        return super().solve_t(*args, **kwargs)


if __name__ == '__main__':
    # Solve for the core model with no extensions -----------------------------
    baseline = SIM(range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2)
    baseline.solve()
    expected = baseline.to_dataframe()

    # Solve the model with different progress bar settings --------------------

    # Set up a copy using the wrapped version of the model class
    model = SIM_ProgressBar(
        range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2
    )

    # Solve the progress bar version with the progress bar turned off (the
    # default)
    test = model.copy()
    test.solve()
    assert_frame_equal(test.to_dataframe(), expected)

    # Solve the progress bar version with the default progress bar
    test = model.copy()
    test.solve(progress_bar=True)
    assert_frame_equal(test.to_dataframe(), expected)

    # Solve the progress bar version with a custom progress bar (here, just the
    # default `tqdm`)
    test = model.copy()
    test.solve(progress_bar=tqdm)
    assert_frame_equal(test.to_dataframe(), expected)

    # Solve the progress bar version with a preset progress bar (equal to the
    # default in this case)
    test = model.copy()
    test.solve(progress_bar=ProgressBarMixin.TqdmFsicStandard)
    assert_frame_equal(test.to_dataframe(), expected)

    # Solve the model with a pandas PeriodIndex to check compatibility --------
    model = SIM_ProgressBar(
        pd.period_range(start='1990Q1', end='2005Q4', freq='Q'),
        alpha_1=0.6,
        alpha_2=0.4,
        G=20,
        theta=0.2,
    )

    test = model.copy()
    test.solve(progress_bar=True)
