# Almon (2017) *The craft of economic modeling*

Example FSIC implementation of Almon's (2017) AMI (Accelerator-Multiplier
Interaction) model of an imaginary economy, as set out in Chapter 1 ('What is
an economic model and why make one?').

This example shows how to:

* read data from a CSV file to a `pandas` DataFrame and load the data into a
  model object
* run the model with error-handling set to 'ignore', to skip over NaNs (from
  the input data) with the expectation that the model will eventually overwrite
  them with numerical values
* use `fsictools` to convert the model results to a DataFrame for inspection
  and plotting
* reproduce (more or less) Figure 1.1 of Almon (2017) ('Q in simple model')
  using `matplotlib`, saving the plot to disk as 'output.png'


## Usage

1. Run 'ami.py' with *this folder* as the current working directory.

Then:

* Inspect the model results in the console output.
* View the plot in 'output.png'.


## Notes

The equations (in 'ami.py') come from Chapter 1 (Pages 21-22) of Almon (2017).

The input data (in 'data.csv') also come from Chapter 1 (Page 25) of Almon
(2017) and were converted from the source PDF file to a (machine-readable) CSV
file.


## References

Almon, C. (2017)
*The craft of economic modeling*, Third, enlarged edition,
licensed under [Creative Commons Attribution-NonCommercial 4.0 International
(BY-NC)](https://creativecommons.org/licenses/by-nc/4.0/legalcode)  
[http://www.inforum.umd.edu/papers/TheCraft.html](http://www.inforum.umd.edu/papers/TheCraft.html)
