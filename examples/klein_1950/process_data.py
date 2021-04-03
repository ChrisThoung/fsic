# -*- coding: utf-8 -*-
"""
process_data
============
Create a processed/amended dataset for estimation and solution.

Before running this script, download the data from Greene (2012) - 'Table
F10.3: Klein's Model I' (under 'Data Sets') - and save to this folder as
'TableF10-3.csv'.
http://people.stern.nyu.edu/wgreene/Text/econometricanalysis.htm
"""

import pandas as pd


if __name__ == '__main__':
    data = pd.read_csv('TableF10-3.csv', index_col=0)

    # Clean up column names
    data.columns = data.columns.str.strip()
    data = data.rename(columns={'WP': 'Wp', 'WG': 'Wg'})

    # Extend index to cover 1919
    data = data.reindex(range(1919, 1941 + 1))

    # Re-align K (capital stock)
    data['K'] = data.pop('K1').shift(-1)

    # Construct variable for total wages: W = Wp + Wg
    data['W'] = data.eval('Wp + Wg')

    # Add time trend: 1931=0
    data['time'] = data.index - 1931

    data.to_csv('data.csv')
