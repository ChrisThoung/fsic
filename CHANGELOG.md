# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added

- New `fsictools` function, `symbols_to_dataframe()`, to inspect a model
  definition more easily (requires [`pandas`](https://pandas.pydata.org/))

### Changed

- Renamed `fsictools.to_dataframe()` to `fsictools.model_to_dataframe()`


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
