# -*- coding: utf-8 -*-
"""
ami
===
Example FSIC implementation of Almon's (2017) AMI (Accelerator-Multiplier
Interaction) model of an imaginary economy, as set out in Chapter 1 ('What is
an economic model and why make one?').

While FSIC only requires NumPy, this example also uses:

* `pandas`, to read the input data and, using `fsic.tools`, generate a DataFrame
  of final results
* `matplotlib` to replicate, more or less, the accompanying chart in Almon
  (2017)

Reference:

    Almon, C. (2017)
    *The craft of economic modeling*,
    Third, enlarged edition, *Inforum*
    http://www.inforum.umd.edu/papers/TheCraft.html
"""

import matplotlib.pyplot as plt
import pandas as pd

import fsic


script = '''
C = 0.60 * Y[-1] + 0.35 * Y[-2]
I = (R +
     1.0 * (PQ[-1] - PQ[-2]) +
     1.0 * (PQ[-2] - PQ[-3]) +
     0.1 * ( Q[-1] -  Q[-2]))
PQ = max(Q, PQ[-1])
M = -380 + 0.2 * (C + I + X)
Q = C + I + G + X - M
Y = 0.72 * Q
'''

symbols = fsic.parse_model(script)
AMI = fsic.build_model(symbols)


if __name__ == '__main__':
    # Read the input data
    data = pd.read_csv('data.csv', index_col=0)

    # Instantiate a new model object
    # The `from_dataframe()` class method automatically extracts data
    # (contents) and metadata (index) from a DataFrame, as a more convenient
    # alternative to:
    #   model = AMI(data.index.tolist(), **dict(data.iteritems()))
    model = AMI.from_dataframe(data)

    # Solve with `errors='ignore'`, to handle missing values (NaNs) in the
    # input data
    model.solve(errors='ignore')

    # Store results to a DataFrame and print to the screen
    results = fsic.tools.model_to_dataframe(model)
    print(results.round(1))

    # Plot output (Q) as in Figure 1.1 of Almon (2017), saving to disk
    plt.figure()

    results['Q'].plot(color='#FF4F2E', label='Output (Q)')
    plt.legend()

    plt.savefig('output.png')
