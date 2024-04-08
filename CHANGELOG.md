# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added

- Added (lowercase) `endogenous` and `check` as instance-level attributes in
  `BaseModel` and `BaseLinker`, initially copied on instantiation from the
  class-level versions.
- Added `reindex()` methods to `VectorContainer` and `BaseModel`.
- Added `to_dataframe()` method to `VectorContainer`.
- Added support for empty linkers i.e. linker instances with no submodels.
- Added new `functions` module for operations on NumPy arrays.
- Added support for `functions` as standard in `eval()`.
- Added support for named periods in `parser` model definitions.
- Added exception chaining in `parser`.
- Improved error messages when `strict=True`, to suggest alternative variables
  (similar to improved error messages in Python 3.10 onwards).
- Added fallback indexing method, `_locate_period_in_span_fallback()`, if
  `span`-specific methods fail. This extends `_VALID_INDEX_METHODS` to handle
  callable objects. As a side-benefit, this also adds support for NumPy arrays
  as `span` attributes.
- Added `try...except` to catch indexing errors in
  `VectorContainer._locate_period_in_span()`, to then raise as `KeyError`s.
- Added check (and error) in `iter_periods()` if attempting to solve an
  instance with an empty span i.e. if there are no periods to solve.
- Added new keyword argument `include_internal` to control
  `model_to_dataframe()` handling of internal model variables with names
  prefixed with an underscore.
- Added new `PandasIndexFeaturesMixin` extension for more sophisticated `span`
  handling.
- Added new `TracerMixin` extension for `BaseModel`-derived classes, to track
  iteration-by-iteration variable results.

### Changed

- Reorganised `core` module into a subpackage of modules:
    - `containers`
    - `interfaces`
    - `models`
    - `linkers`
- Switched from dated 'setup.py' configuration to more modern pyproject.toml.
- Switched to uppercase constant names.
- Removed `NotImplementedError`s from over-rideable methods.
- Changed treatment of attribute addition, to use new `add_attribute()` method
  in `VectorContainer`.
- Changed treatment of lags and leads to derive instance-level attributes from
  class-level defaults in both `BaseModel` and `BaseLinker`.

### Deprecated
### Removed
### Fixed

- Corrected error in `ModelInterface.add_variable()` that added variable names
  as a list of character strings ('ABC' -> ['A', 'B', 'C']) rather than a
  string in a list ('ABC' -> ['ABC']).
- Added code in `tool.dataframe_to_symbols()` to ensure the correct dtypes.

### Security


## [0.8.0.dev] - 2023-01-22

**Version 0.8.0 is the first release of fsic as a Python package, rather than a
  set of individual Python modules.**

### Added

- Added support in the parser for user-written Python code (enclosed in
  backticks) to be inserted unmodified (verbatim).
- Added parser support for user-specified lag and lead lengths, and for
  user-specified minimum lag and lead lengths.
- Added check to parser for incomplete pairs of opening and closing braces.
- Added support for Symbols of endogenous variables but no accompanying
  equation e.g. because of a separate Symbol of verbatim code.
- Added error handling for `solve_t_before()` and `solve_t_after()`.
- Added new `strict` attribute (and `__init__()` keyword argument) to
  `VectorContainer` (and, in turn, `BaseModel` and `BaseLinker`) to optionally
  guard against adding non-variable attributes. If `strict=True`:
    - at instantiation, attempts to add unrecognised/unlisted variables
      (i.e. any not in the object's `NAMES` attribute) lead to an
      `InitialisationError`
    - only `add_variable()` can expand the object
- Added support for single value replacement in `VectorContainer` and
  `ModelInterface` (and, in turn, `BaseModel`).
- Added initial version of `eval()` method to `VectorContainer` (and, in turn,
  `BaseModel` and `BaseLinker`).
- Added new `size` property to `VectorContainer` and `ModelInterface` (and, in
  turn, `BaseModel`), as well as `BaseLinker`, to report the number of elements
  in the objects' arrays.
- Added new `sizes` property to `BaseLinker` to report the number of elements
  in each of: the linker and its constituent submodels.
- Added new `nbytes` property to `VectorContainer` (and, in turn, `BaseModel`)
  and `BaseLinker` to report the total bytes consumed by the elements in the
  objects' arrays.
- Added `to_dataframe``(s)``()` methods to `BaseModel` and `BaseLinker`,
  calling the corresponding functions in `fsic.tools`.
- Added new `dataframe_to_symbols()` function to `fsic.tools`, reversing
  `symbols_to_dataframe()` and permitting a roundtrip.
- Added `status` and `iterations` keyword arguments to `model_to_dataframe()`
  and `linker_to_dataframes()`, to optionally exclude non-data model contents.
- Added support to `VectorContainer` (and, in turn, `BaseModel` and
  `BaseLinker`) for `pandas` index objects as `span` attribute values.
- Added new `extensions` sub-package to provide additional (and optional)
  functionality, currently consisting of:
    - `AliasMixin` (in `fsic.extensions` / `fsic.extensions.common`): support
      for multiple/alternative names for model variables
    - `ProgressBarMixin` (in `fsic.extensions` / `fsic.extensions.common`):
      transferring the `tqdm`-based example into the main `fsic` package

### Changed

- Reorganised source files into a Python package (`fsic`):
    - Split original `fsic` module into:
        - `core`
        - `parser`
        - `exceptions`
    - `fsic_fortran` is now `fsic.fortran`
    - `fsictools` is now `fsic.tools`
- Changed handling of numerical solution errors when
  `errors='raise'`. Previously (before version 0.8.0), any numerical solution
  error would lead to an exception after evaluating all equations during a
  single iteration. This propagated NaNs/Infs through the solution (for
  inspection) but gave no indication as to which equation caused the original
  problem. The default behaviour now is to catch the first error and raise an
  exception at that point, to identify the problem statement. This involves a
  new keyword argument in the `solve_()` methods: `catch_first_error`. By
  default, `catch_first_error` is `True`, leading to the new
  behaviour. Pre-0.8.0 behaviour can be recovered with
  `catch_first_error=False`.
- Implemented exception chaining in `BaseModel.solve_t()`.
- Changed solution status handling from strings to an enumeration.


## [0.7.1.dev] - 2021-08-04

Various additions and changes to improve consistency and checks (both parser
and solution).

**Versions 0.7.x are the last releases to take the form of a set of individual
  Python modules. Version 0.8.0 onwards implements fsic as a Python package.**

### Added

- Implemented over-rideable `solve_t_before()` and `solve_t_after()` methods in
  `BaseModel`, following a similar pattern to those in `BaseLinker`.
- Added further keyword arguments to pass from `solve_t()` into `_evaluate()`:
  `errors` (from the user) and `iteration` (from the solution loop).
- Added check in `split_equations_iter()` for unnecessary leading whitespace in
  equations, to raise a more helpful error message.
- Added further checks to `parse_equation()` for consistency with
  `parse_model()`.
- Added further parser checks and messages to catch unmatched brackets and
  invalid index expressions.
- Python implementation of `BaseModel.solve_t()` now catches evaluation errors
  and raises a `SolutionError`.

### Changed

- Changed convergence checks to take the absolute differences between
  iterations (rather than the squared differences).
- In `build_model()` and `build_model_definition()`, `converter` is now a
  keyword-only argument.

### Fixed

- Corrected keyword passthrough from `BaseModel.solve()` by making sure
  `iter_periods()` can absorb the arguments, even if not needed.


## [0.7.0.dev] - 2021-04-03

New `BaseLinker` class to solve multiple `BaseModel`(-derived) instances as a
linked multi-entity model e.g. for multiple countries or regions connected by
trade.

Various other changes and fixes.

**Versions 0.7.x are the last releases to take the form of a set of individual
  Python modules. Version 0.8.0 onwards implements fsic as a Python package.**

### Added

- Solution:
   - New `min_iter` keyword argument in solution methods to force a minimum
     number of iterations before testing for convergence.
- New `BaseLinker` class to nest and solve multiple `BaseModel`-derived
  instances.
   - New `linker_to_dataframes()` function in `fsictools` to extract results as
     a set of DataFrames.
- Additions to `BaseModel`:
   - New `from_dataframe()` class method to instantiate a model from a `pandas`
     DataFrame.
   - Further keyword arguments in `solve()` (i.e. `**kwargs`) are now passed on
     to both `iter_periods()` and `solve_t()` to support custom keywords in the
     user's own code. (See
     [examples/_cookbook/progress_bar.py](examples/_cookbook/progress_bar.py)
     for an example.)
   - Over-rode `VectorContainer.add_variable()` to properly extend a model
     instance's store while supporting variables of non-default
     `dtype`s. Updated `fsictools.model_to_dataframe()` to preserve those
     `dtype`s.
- Additions to `ModelInterface`, feeding through to both `BaseModel` and
  `BaseLinker`:
   - `ModelInterface` now stores the default `dtype` as an object attribute.
- Additions to `VectorContainer`, feeding through to both `BaseModel` and
  `BaseLinker`:
   - New `replace_values()` method to update object contents *en masse*, with a
     similar interface to `__init__()`.
- Additions to parser:
   - `build_model_definition()` now optionally produces code without type
     hints.
- `fsic_fortran`:
   - Implemented Fortran equivalent of `solve()` method (including Python
     wrapper) for further speed gains by bypassing Python-Fortran data transfer
     each period (as happens with calls to `solve_t()`).
   - `solve_t()` now catches NaNs and infinities prior to solution (if
     desired).
   - Solution methods now support `offset` argument.

### Changed

- Refactored various parts of `BaseModel` to be able to share more code with
  the new `BaseLinker` class.
- `PeriodIter` no longer consumes its contents on iteration i.e. it is now
  reusable.
- Minor: `solve()` method now initialises `solved` as a list of `None`s to help
  with debugging. Previously, the list was initialised with `False`.

### Fixed

- Corrected instantiation of `BaseModel` objects to initialise variables from a
  copy of `BaseModel.NAMES`. This ensures that adding variables to an object
  only modifies that specific object, rather than the underlying class.
- Corrected handling of `copy.deepcopy()` in `fsic` `BaseModel`
  class. Implemented similar (and correct) behaviour in new `BaseLinker` class.


## [0.6.4.dev] - 2021-01-03

### Added

- New `PeriodIter` class as the return value from
  `BaseModel.iter_periods()`. This returns (index, label) pairs as before but
  also implements a `__len__()` magic method to give the number of periods to
  loop over (the length of the iterator).
- Explicit checks for use of reserved Python keywords as variable
  names. Previously, these were silently dropped but now lead to
  `ParserError`s.

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
- Refactored some fsic Python unit tests to make them easier to reuse as fsic
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
