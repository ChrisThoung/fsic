# Klein (1950) *Economic fluctuations in the United States, 1921-1941*

Example FSIC implementation of Klein Model I, from Klein (1950).

This example shows how to:

* specify an economic model
* estimate parameters for the model's equations using Kevin Sheppard's
  [`linearmodels`](https://github.com/bashtage/linearmodels) package
* run the model with the estimated parameters
* use `fsictools` to convert the model results to a DataFrame for inspection


## Usage

1. Run ['estimate_equations.py'](estimate_equations.py) to generate a CSV file
   of parameter estimates, 'parameters.csv', using different estimation
   techniques.
2. Run ['klein_model_i.py'](klein_model_i.py) with *this folder* as the current
   working directory.


## Notes

The notation varies slightly from the original in Klein (1950). See the
`descriptions` variable in ['klein_model_i.py'](klein_model_i.py) for details.


## References

Giles, D. E. A. (2012)
'Estimating and simulating an SEM',
*Econometrics Beat: Dave Giles' blog*, 19/05/2012  
[https://davegiles.blogspot.com/2012/05/estimating-simulating-sem.html](https://davegiles.blogspot.com/2012/05/estimating-simulating-sem.html)

Klein, L. R. (1950)
*Economic fluctuations in the United States, 1921-1941*,
*Cowles Commission for Research in Economics*, **11**,
New York: John Wiley & Sons / London: Chapman & Hall  
[https://cowles.yale.edu/cfm-11](https://cowles.yale.edu/cfm-11)
