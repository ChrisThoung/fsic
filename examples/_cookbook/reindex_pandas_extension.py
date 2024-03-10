# -*- coding: utf-8 -*-
"""
reindex_pandas_extension
========================
Examples of how to use the more advanced `PandasIndexFeaturesMixin` extension
to `reindex` model objects to have different spans.

The example model in this script is a simplified five-equation version of
Godley and Lavoie’s (2007) Model *SIM*.

Reference:

    Godley, W., Lavoie, M. (2007),
    *Monetary economics: an integrated approach to
    credit, money, income, production and wealth*,
    Palgrave Macmillan
"""

import pandas as pd

import fsic


# Define the example model
SCRIPT = """
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
"""

# Parse the script and create a class definition
SIM = fsic.build_model(fsic.parse_model(SCRIPT))


# Extend the model with more advanced `pandas` indexing features
class SIM_Extended(fsic.extensions.model.PandasIndexFeaturesMixin, SIM):
    pass


if __name__ == '__main__':
    # Setup -------------------------------------------------------------------
    print('\nSETUP')

    # Create a model object over 1945Q1-1946Q1, set its inputs and solve
    model = SIM(
        pd.period_range(start='1945Q1', end='1946Q1', freq='Q'),
        alpha_1=0.6,
        alpha_2=0.4,
        G=20,
        theta=0.2,
    )

    model.solve()

    print(' - Original:')
    print(model.to_dataframe().round(1))

    # Do the same with the extended version to show that the results are the
    # same
    model_extended = SIM_Extended(
        pd.period_range(start='1945Q1', end='1946Q1', freq='Q'),
        alpha_1=0.6,
        alpha_2=0.4,
        G=20,
        theta=0.2,
    )

    model_extended.solve()

    print('\n - Extended:')
    print(model_extended.to_dataframe().round(1))

    # Standard arguments ------------------------------------------------------
    print('\nSTANDARD ARGUMENTS')

    # The core `reindex()` method is no-frills, only offering the following
    # arguments:
    #  - fill_value: value to use when adding new periods (default: NaN)
    #  - model variables as keywords, to set individual fill values
    #    (over-riding the above)
    #
    # And... that's about it.

    print(' - Original:')
    print(
        model.reindex(
            pd.period_range(start='1945Q3', end='1946Q3', freq='Q'),
            fill_value=0.0,
            G=25,
        )
        .to_dataframe()
        .round(1)
    )

    print('\n - Extended:')
    print(
        model_extended.reindex(
            pd.period_range(start='1945Q3', end='1946Q3', freq='Q'),
            fill_value=0.0,
            G=25,
        )
        .to_dataframe()
        .round(1)
    )

    # Using `pandas` ----------------------------------------------------------
    print('\nPANDAS FEATURES')

    # As above, the `pandas`-augmented method in `PandasIndexFeaturesMixin`
    # replicates the behaviour of the core `reindex()` method. But it also adds
    # further keyword arguments to access more sophisticated features, relying
    # on the `pandas` `Series.reindex()` method:
    # https://pandas.pydata.org/docs/reference/api/pandas.Series.reindex.html
    #
    # Further keyword arguments are as follows (matching the `pandas`
    # arguments):
    #  - method: Approach for filling missing values on reindexing
    #            (default: `None`)
    #  - copy: Whether to return a copy of the new index (default: `True`)
    #  - limit: Maximum number of consecutive elements to fill
    #  - tolerance: Maximum distance between original and new labels
    #
    # Of the above, `method` is the most important

    # Use the standard `pandas` `method` argument to...

    # ...back fill all variables...
    print(' - Backfill (all):')
    print(
        model_extended.reindex(
            pd.period_range(start='1944Q3', end='1946Q3', freq='Q'), method='bfill'
        )
        .to_dataframe()
        .round(1)
    )

    # ...forward fill all variables...
    print('\n - Forward fill (all):')
    print(
        model_extended.reindex(
            pd.period_range(start='1944Q3', end='1946Q3', freq='Q'), method='ffill'
        )
        .to_dataframe()
        .round(1)
    )

    # ...apply nearest-neighbour filling
    print('\n - Nearest (all):')
    print(
        model_extended.reindex(
            pd.period_range(start='1944Q3', end='1946Q3', freq='Q'), method='nearest'
        )
        .to_dataframe()
        .round(1)
    )

    # As well as the arguments from the `pandas` `reindex()` method, the
    # extension provides variable-by-variable controls over the filling method
    # with the following arguments available:
    #  - back fill: `backfill_`, `bfill_`
    #  - forward fill: `pad_`, `ffill_`
    #  - nearest: `nearest+`
    #
    # Note the underscores in the names, to help avoid potential name
    # collisions.
    #
    # Arguments can either be variable names (e.g. 'G') or a sequence of
    # variable names (['G', 'alpha_1' etc])
    #
    # This supports statements like the examples below.

    # Use nearest neighbour on parameters only
    print('\n - Nearest (parameters only) and forward fill (G only):')
    print(
        model_extended.reindex(
            pd.period_range(start='1944Q3', end='1946Q3', freq='Q'),
            ffill_='G',
            nearest_=model_extended.PARAMETERS,
        )
        .to_dataframe()
        .round(1)
    )

    # Use zero as the default fill value, forward fill exogenous values and,
    # for parameters, use nearest values
    model_extended_reindexed = model_extended.reindex(
        pd.period_range(start='1944Q3', end='1946Q3', freq='Q'),
        fill_value=0.0,
        ffill_=model.EXOGENOUS,
        nearest_=model.PARAMETERS,
    )

    print('\n - Nearest (parameters only), forward fill (G only) and zeroes elsewhere:')
    print(model_extended_reindexed.to_dataframe().round(1))

    # Solve model
    model_extended_reindexed.solve(start='1946Q2', offset=-1)

    print('\n - Solve the model:')
    print(model_extended_reindexed.to_dataframe().round(1))
