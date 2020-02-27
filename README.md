# fsic

[![Build
Status](https://travis-ci.org/ChrisThoung/fsic.svg?branch=master)](https://travis-ci.org/ChrisThoung/fsic)

Tools for macroeconomic modelling in Python.


## How to...

* install the Python modules (`fsic` and `fsictools`): run either:  
  `$ python setup.py install`  
  or: `$ pip install .`
* see more examples of how to use FSIC: view the contents of the 'examples'
  folder
* run the test suite: run `python test_fsics.py` or use a `unittest`-compatible
  test framework (e.g. `unittest`, `pytest`, `nose`/`nose2`; other test
  frameworks are available) - these are the same tests that [Travis
  CI](https://travis-ci.org/ChrisThoung/fsic) runs
    * note that there's a second set of tests, 'test_fsictools.py', for the
      optional supporting `fsictools` module, which has further dependencies to
      run in full: [pandas](https://pandas.pydata.org/),
      [NetworkX](https://networkx.github.io/) and
      [SymPy](https://www.sympy.org/en/index.html) (none of these are needed to
      run a FSIC model)


## Quickstart

To specify and solve Model *SIM* from Chapter 3 of Godley and Lavoie (2007):

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

model.W = 1          # Wages

model.G_d[1:] = 20   # Exogenous government expenditure beginning in the second period
model.theta = 0.2    # Income tax rate

# Solve the model
# (`max_iter` increases the maximum number of iterations, to ensure
# convergence)
model.solve(max_iter=200)
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
contents of the model to a DataFrame for inspection, using
`model_to_dataframe()`, from the `fsictools` module:

```python
import fsictools

results = fsictools.model_to_dataframe(model)
```

For a more detailed example, see my
[post](https://www.christhoung.com/2018/07/08/fsic-gl2007-pc/) and accompanying
[notebook](https://github.com/ChrisThoung/website/tree/master/code/2018-07-08_fsic_pc)
about Model *PC* from Chapter 4 of Godley and Lavoie (2007).


## References

Godley, W., Lavoie, M. (2007),
*Monetary economics: an integrated approach to
credit, money, income, production and wealth*,
Palgrave Macmillan
