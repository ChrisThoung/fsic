# Klein (1950) *Economic fluctuations in the United States, 1921-1941*

Example fsic implementation of Klein Model I, from Klein (1950).

This example shows how to:

* specify an economic model
* estimate parameters for the model's equations using Kevin Sheppard's
  [`linearmodels`](https://github.com/bashtage/linearmodels) package
* run the model with the estimated parameters
* use `fsic.tools` to convert the model results to a DataFrame for inspection


## Usage

With *this folder* as the current working directory:

1. Download the data from Greene (2012) - 'Table F10.3: Klein's Model I' (under
   'Data Sets'), saving to this folder as 'TableF10-3.csv':  
   [http://people.stern.nyu.edu/wgreene/Text/econometricanalysis.htm](http://people.stern.nyu.edu/wgreene/Text/econometricanalysis.htm)
2. Run ['process_data.py'](process_data.py) to generate 'data.csv', a slightly
   amended version of the original data.
3. Run ['estimate_equations.py'](estimate_equations.py) to generate a CSV file
   of parameter estimates, 'parameters.csv', using different estimation
   techniques.
4. Run ['klein_model_i.py'](klein_model_i.py). This solves the model using the
   various sets of parameter estimates and generates 'results.png', which plots
   the results against the actual data.

Alternatively, use the accompanying [makefile](makefile) to run the above
steps.


## Notes

Broadly, the notation follows Greene (2012) which varies slightly from the
original in Klein (1950). See the `descriptions` variable in
['klein_model_i.py'](klein_model_i.py) for details of the variables.


## References

Giles, D. E. A. (2012)
'Estimating and simulating an SEM',
*Econometrics Beat: Dave Giles' blog*, 19/05/2012  
[https://davegiles.blogspot.com/2012/05/estimating-simulating-sem.html](https://davegiles.blogspot.com/2012/05/estimating-simulating-sem.html)

Greene, W. H. (2012)
*Econometric analysis*,
7th edition,
Pearson  
Datasets available from:
[http://people.stern.nyu.edu/wgreene/Text/econometricanalysis.htm](http://people.stern.nyu.edu/wgreene/Text/econometricanalysis.htm)  
(see Data Sets > 'Table F10.3: Klein's Model I')

Klein, L. R. (1950)
*Economic fluctuations in the United States, 1921-1941*,
*Cowles Commission for Research in Economics*, **11**,
New York: John Wiley & Sons / London: Chapman & Hall  
[https://cowles.yale.edu/cfm-11](https://cowles.yale.edu/cfm-11)
