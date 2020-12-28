# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added

- New `PeriodIter` class as the return value from
  `BaseModel.iter_periods()`. This returns (index, label) pairs as before but
  also implements a `__len__()` magic method to give the number of periods to
  loop over (the length of the iterator).

### Changed

- Added blank line between equations in Python-generated model code.

### Fixed

- Fixed Fortran code generator to text wrap variable index numbers over
  multiple lines as needed, for when models grow to large(ish) numbers of
  variables.


## [0.6.3.dev] - 2020-12-26

### Added

- Support for custom code generators to convert `Symbol` objects to model
  Python code.
- Further checks and messages for errors in equations.
- Added type hints to Python model template.
- Support for numerical error handling in Fortran implementation, mirroring
  existing Python implementation.
- Added `__contains__()` magic method to `BaseModel` class, to test that a
  variable, say, `G` is defined in a model instance e.g. `G in model`.

### Changed

- `build_model_definition()` now adds the normalised equation as a comment
  before each line of model code
- Parser test to trap accidental treatment of floats as callable
  (e.g. 'A = 0.5(B)') now only runs for Python 3.8 and above. (Doesn't look
  trivial to handle in earlier versions of Python.)
- Refactored some FSIC Python unit tests to make them easier to reuse as FSIC
  Fortran tests.

### Fixed

- Corrected parser handling of functions to:
   - properly account for functions in namespaces, leaving them unchanged
     e.g. to prevent `np.mean` -> `np[t].mean`
   - only replace function names on an exact basis
     e.g. to prevent `np.log` -> `np.np.log`
- Fixed handling of reverse indexing in Fortran code.


## [0.6.2.dev] - 2020-06-27

### Added

- Extend handling of NaNs to also cover infinities e.g. from log(0) operations.
- Add separate method, `iter_periods()` to loop through model solution periods.
- Improve error catching and messages for parser errors.


## [0.6.1.dev] - 2020-05-16

Fixes to the Fortran functions to more closely mimic the behaviour of the
original Python code. Ideally, the two codebases would behave identically with
no need for the user to know which engine is running under the bonnet.

### Fixed

- Fix iteration counter in Fortran code. When a loop completes in Fortran, the
  final value of the iteration counter is 1 higher than the limit on the
  loop. Subtract 1 to correct for this.
- Further work to improve consistency of behaviour between the Python and
  Fortran implementations including: treatment of `errors` argument in the
  solve methods.


## [0.6.0.dev] - 2020-05-14

Add support to generate and link in compilable Fortran code. API (especially)
subject to change.

### Added

- Add new module, 'fsic_fortran.py', to generate and link in compilable Fortran
  code, to speed up model solution. Includes a function to generate Fortran code from a
  set of model symbols (as you normally would for a Python-based model) and a
  new subclass to handle the compiled routines.
- Add accompanying test suite for `fsic_fortran` module,
  'test_fsic_fortran.py'.
- Add setter for `values` property in `VectorContainer` and `BaseModel`
  classes, to completely overwrite their contents.
- Add new exception, `InitialisationError` to catch implementation problems at
  model instantiation (as distinct from problems with the initial model
  inputs). Only added to the new `FortranEngine` class for now (to check for a
  linked module).
- Add new model attribute ,`engine`, which is assigned at instantiation, to
  signal the *expected* solution method/implementation.


## [0.5.2.dev] - 2020-04-24

### Added

- Add return values to solve methods to indicate solution success or failure.

### Changed

- Refactored container behaviour from `BaseModel` into a separate class,
  `VectorContainer`, in readiness for future model linker development.


## [0.5.1.dev] - 2020-04-04

### Added

- Error trapping for invalid `offset` arguments to model solve methods.
- Tests to confirm explicit support for negative indexing in `solve_t()`.


## [0.5.0.dev] - 2020-03-29

### Added

- Add support for conditional expressions.

### Changed

- Updated error message for parser failures.


## [0.4.1.dev] - 2020-03-24

### Added

- Added support for incomplete model specifications i.e. no equations
  (endogenous variables); or no symbols at all.


## [0.4.0.dev] - 2020-03-15

### Added

- Add option to catch non-convergence in solution (now throws this error by
  default)
- Initial set of unit tests for `fsictools` module.
- New folder of example model implementations.

### Fixed

- Added some (tentative) workarounds to `fsictools.symbols_to_sympy()` to
  prevent conversion to imaginary numbers ('I') and singletons ('S').

### Changed

- Code generator now removes trailing space after opening brackets; and leading
  space before closing brackets.


## [0.3.1.dev] - 2020-02-19

Minor features and improvements.

### Added

- Add option to copy initial values for solution from another period with new
  `offset` keyword argument to `BaseModel.solve_t()`. This argument is also
  passed through by the other `solve_` methods.
- New `symbols_to_graph()` utility function (in `fsictools`) to generate a
  NetworkX directed graph from a list of symbols.


## [0.3.0.dev] - 2020-02-16

Improvements to the robustness of the parser and model builder, as well as to
handle numerical errors (NaNs) in solution.

### Added

- Add options to handle numerical errors (NaNs) in solution.
- Optional syntax checking (default on) in model parser.
- New `ParserError` exception to signal failures to process a model definition
  string.
- New `BuildError` exception to signal failures to generate a model class
  definition.
- New `symbols_to_sympy()` utility function (in `fsictools`) to examine a
  system of equations using `SymPy`.

### Removed

- Removed built-in symbol re-ordering from `build_model*()` functions. The user
  must now re-order the symbols themselves.


## [0.2.1.dev] - 2020-02-02

Minor changes to improve usability.

### Added

- Support in model objects (derived from `BaseModel`) for tab completion of
  variable names when indexing e.g. `model['`.
- New `fsictools` function, `symbols_to_dataframe()`, to inspect a model
  definition more easily (requires [`pandas`](https://pandas.pydata.org/)).

### Changed

- Renamed `fsictools.to_dataframe()` to `fsictools.model_to_dataframe()`.


## [0.2.0.dev] - 2019-12-03

As presented in this
[post](https://www.christhoung.com/2019/07/27/fsic-update/) and its
accompanying
[notebook](https://github.com/ChrisThoung/website/tree/master/code/2019-07-27_fsic_update).

New 'fsictools.py' module implements `to_dataframe()`, as detailed in the above
post and copied over from this later
[post](https://www.christhoung.com/2019/12/03/sympy-sim/) and its accompanying
[notebook](https://github.com/ChrisThoung/website/tree/master/code/2019-12-03_sympy_sim).

### Added

- New 'fsictools.py' module for supporting operations not directly related to
  model specification and solution.
- Dynamic `__dir__()` magic method to `BaseModel` class to list model
  variables.
- Support in `BaseModel` class for `copy()` and `deepcopy()` methods.
- Support for keyword arguments to be passed through the stack of solution
  methods.

### Changed

- Structure of `BaseModel` variable lists altered for easier customisation with
  manual edits.
- Internal structure of `BaseModel` `iterations` and `status` attributes
  changed to conform to that of economic model variables.


## [0.1.0.dev] - 2018-07-08

As presented in this
[post](https://www.christhoung.com/2018/07/08/fsic-gl2007-pc/) and its
accompanying
[notebook](https://github.com/ChrisThoung/website/tree/master/code/2018-07-08_fsic_pc).

### Added

- New 'fsic.py' module and accompanying test suite, 'test_fsic.py', as well as
  a setup script for installation, 'setup.py'.
