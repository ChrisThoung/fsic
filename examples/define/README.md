# DEFINE: Dynamic Ecosystem-FINance-Economy

Example fsic implementation of DEFINE-SIMPLE, from Dafermos and Nikolaidi
(2021), based on the original [R code](https://github.com/DEFINE-model/SIMPLE)
and [model description](https://define-model.org/define-simple/).

This example shows how to:

* specify the model
* run the model to produce results for a baseline and two scenarios (a green
  investment scenario and a degrowth scenario)
* print the results to CSV files, one per run
* reproduce the charts from the original implementation, as a single file


## Usage

With *this folder* as the current working directory:

1. Run [define_simple.py](define_simple.py) to generate results for:
    * the **baseline**
    * a **green investment scenario**, with more favourable credit conditions
      for green technologies (lower interest rates relative to conventional
      technologies) and a higher autonomous propensity to invest in green
      technologies
    * a **degrowth scenario**, with lower propensities to consume and invest,
      lowering global growth


## Notes

Differences between the equations in this implementation and the original
concern:

* some rearrangement of individual equations while leaving the results
  unchanged
* removal of separate 'change' variables to apply to certain model parameters,
  with a preference instead for copying the model object and modifying the
  parameters directly


## References

Dafermos, Y., Nikolaidi, M. (2021),
'DEFINE-SIMPLE'  
[https://define-model.org/define-simple/](https://define-model.org/define-simple/)  
[https://github.com/DEFINE-model/SIMPLE](https://github.com/DEFINE-model/SIMPLE)
