# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added

- Add return values to solve methods to indicate solution success or failure.


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
