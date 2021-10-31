# fsic

[![Build Status](https://github.com/ChrisThoung/fsic/actions/workflows/python-package.yml/badge.svg)](https://github.com/ChrisThoung/fsic/actions/workflows/python-package.yml)

Tools for macroeconomic modelling in Python.


## How to...

* install the package, in any of the following ways:
    * `$ python setup.py install`
    * `$ pip install .`
    * `$ make install`
* install dependencies, using the requirements files in the
  [requirements](requirements) folder:
    * minimum requirements (for completeness only: handled automatically by
      installing the package): `pip install -r requirements/minimum.txt`
    * optional requirements (to use all feature of `fsic.tools`): `pip install
      -r requirements/optional.txt`
    * development requirements (to run the test suite as recommended e.g. as in
      the [makefile](makefile)): `pip install -r requirements/development.txt`
    * dependencies to run the examples: `pip install -r
      requirements/examples.txt`
* see more examples of how to use FSIC: view the contents of the
  [examples](examples/) folder
    * the [makefile](makefile) can run all the examples in sequence: `$ make
      examples`
* run the test suite, in any of the following ways:
    * run any of the individual 'test_*.py' files in the [tests](tests/) folder
      e.g. `python tests/test_fsic.py`
    * with [`unittest`](https://docs.python.org/3/library/unittest.html)
      e.g. `python -m unittest discover .`
    * with a `unittest`-compatible test framework like
      [`pytest`](https://docs.pytest.org/en/stable/) e.g. `pytest`
    * with the [makefile](makefile): `$ make test`


## Quickstart

To specify and solve Model *SIM* from Chapter 3 of Godley and Lavoie (2007)
(also available as a file of Python code in
[examples/_cookbook/quickstart.py](examples/_cookbook/quickstart.py)):

```python
import fsic

script = '''
# Keynesian/Kaleckian quantity adjustment equalises demand and supply
C_s = C_d  # Household final consumption expenditure
G_s = G_d  # Government expenditure
T_s = T_d  # Taxes
N_s = N_d  # Labour

# Disposable income
YD = (W * N_s) - T_s

# Taxes
T_d = {theta} * W * N_s

# Household final consumption expenditure
C_d = {alpha_1} * YD + {alpha_2} * H_h[-1]

# Money: Government liability
H_s = H_s[-1] + G_d - T_d

# Money: Household asset
H_h = H_h[-1] + YD - C_d

# National income
Y = C_s + G_s

# Labour demand
N_d = Y / W
'''

# Parse `script` to identify the constituent symbols (endogenous/exogenous
# variables, parameters and equations)
symbols = fsic.parse_model(script)

# Embed the economic logic in a new Python class
SIM = fsic.build_model(symbols)

# Initialise a new model instance over 1945-2010
model = SIM(range(1945, 2010 + 1))

# Set parameters and input values
model.alpha_1 = 0.6  # Propensity to consume out of current disposable income
model.alpha_2 = 0.4  # Propensity to consume out of lagged wealth

model['W'] = 1       # Wages (alternative variable access by name rather than attribute)

# Exogenous government expenditure beginning in the second period
model.G_d[1:] = 20  # Regular list/NumPy-like indexing by position

# Income tax rate of 20% beginning in the second period
model['theta', 1946:] = 0.2  # `pandas`-like indexing by label: variable and period

# Solve the model
# (`max_iter` increases the maximum number of iterations, to ensure
# convergence)
model.solve(max_iter=350)
```

`model.status` lists the solution state of each period as one of:

* `-` : still to be solved (no attempt made)
* `.` : solved successfully
* `F` : failed to solve

The solved model's values are in `model.values`, which is a 2D NumPy array in
which each item of the:

* first axis ('rows') is a *variable*, with corresponding labels in `model.names`
* second axis ('columns') is a *period*, with corresponding labels in
  `model.span`

If you've installed [`pandas`](https://pandas.pydata.org/), you can convert the
contents of the model to a DataFrame for inspection, using the object's
`to_dataframe()` method (which wraps `fsic.tools.model_to_dataframe()`).

```python
results = model.to_dataframe()

print(results.round(2))
```

|      |   C_s |   C_d |  G_s |   T_s |   T_d |    N_s |    N_d | ... |  G_d |   W | theta | alpha_1 | alpha_2 | status | iterations |
| ---- | ----- | ----- | ---- | ----- | ----- | ------ | ------ | --- | ---- | --- | ----- | ------- | ------- | ------ | ---------- |
| 1945 |  0.00 |  0.00 |  0.0 |  0.00 |  0.00 |   0.00 |   0.00 | ... |  0.0 | 1.0 |   0.0 |     0.6 |     0.4 |      - |         -1 |
| 1946 | 18.46 | 18.46 | 20.0 |  7.69 |  7.69 |  38.46 |  38.46 | ... | 20.0 | 1.0 |   0.2 |     0.6 |     0.4 |      . |        338 |
| 1947 | 27.93 | 27.93 | 20.0 |  9.59 |  9.59 |  47.93 |  47.93 | ... | 20.0 | 1.0 |   0.2 |     0.6 |     0.4 |      . |        338 |
| 1948 | 35.94 | 35.94 | 20.0 | 11.19 | 11.19 |  55.94 |  55.94 | ... | 20.0 | 1.0 |   0.2 |     0.6 |     0.4 |      . |        338 |
| 1949 | 42.72 | 42.72 | 20.0 | 12.54 | 12.54 |  62.72 |  62.72 | ... | 20.0 | 1.0 |   0.2 |     0.6 |     0.4 |      . |        338 |
| ...  |   ... |   ... |  ... |   ... |   ... |    ... |    ... | ... |  ... | ... |   ... |     ... |     ... |    ... |        ... |
| 2006 | 80.00 | 80.00 | 20.0 | 20.00 | 20.00 | 100.00 | 100.00 | ... | 20.0 | 1.0 |   0.2 |     0.6 |     0.4 |      . |        348 |
| 2007 | 80.00 | 80.00 | 20.0 | 20.00 | 20.00 | 100.00 | 100.00 | ... | 20.0 | 1.0 |   0.2 |     0.6 |     0.4 |      . |        348 |
| 2008 | 80.00 | 80.00 | 20.0 | 20.00 | 20.00 | 100.00 | 100.00 | ... | 20.0 | 1.0 |   0.2 |     0.6 |     0.4 |      . |        348 |
| 2009 | 80.00 | 80.00 | 20.0 | 20.00 | 20.00 | 100.00 | 100.00 | ... | 20.0 | 1.0 |   0.2 |     0.6 |     0.4 |      . |        348 |
| 2010 | 80.00 | 80.00 | 20.0 | 20.00 | 20.00 | 100.00 | 100.00 | ... | 20.0 | 1.0 |   0.2 |     0.6 |     0.4 |      . |        348 |

For further examples, see the [examples](examples/) folder or my
[post](https://www.christhoung.com/2018/07/08/fsic-gl2007-pc/) and accompanying
[notebook](https://github.com/ChrisThoung/website/tree/master/code/2018-07-08_fsic_pc)
about Model *PC* from Chapter 4 of Godley and Lavoie (2007).


## References

Godley, W., Lavoie, M. (2007),
*Monetary economics: an integrated approach to
credit, money, income, production and wealth*,
Palgrave Macmillan
