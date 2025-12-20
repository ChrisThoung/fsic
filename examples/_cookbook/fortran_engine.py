# -*- coding: utf-8 -*-
"""
fortran_engine
==============
Example of how to generate and link Fortran code for faster model solution,
comparing performance gains from making the substitution at different points.

The example model in this script is a simplified five-equation version of
Godley and Lavoie’s (2007) Model *SIM*.

**Requires NumPy with a working F2PY setup.**

The solution of a model can be divided into:

1. The outer solution loop, over a sequence of periods e.g. 2018, 2019 etc.
   This is handled by the `solve()` method.
2. For each period, an inner iteration loop, which repeatedly evaluates the
   system of equations, continuing until, hopefully, the solution converges.
   This is handled by the `solve_t()` method.
3. The evaluation step itself, which represents one pass through the system.
   This is embedded in the `_evaluate()` method, which must be defined for each
   model implementation.

In the core implementation, all three are written in Python, operating on NumPy
arrays. The `fsic.fortran` module provides tools to generate Fortran
equivalents. The Fortran implementations are then wrapped in Python methods
with identical interfaces to the original Python code. This way, the user
doesn't need to make any changes to how they call the code while benefitting
from any speed gains.

Performance gains depend on whether the faster solution outweighs the
additional overhead of transferring data from Python to Fortran and back
again. In practice, calls to `_evaluate()` take *longer* (are slower) using the
Fortran version because of the time needed to move data around. Using the
Fortran version of `solve_t()` does confer speed gains because of the more
limited data transfer (in and out just once per period). The Fortran version of
`solve()` is faster still because data transfer only happens at the start and
end of the process, regardless of the number of periods.

References:

    Godley, W., Lavoie, M. (2007),
    *Monetary economics: an integrated approach to
    credit, money, income, production and wealth*,
    Palgrave Macmillan
"""

import functools
import importlib
import statistics
import subprocess
import timeit
from typing import List, Tuple
import warnings

import fsic


# If available, use the old (deprecated) `numpy.distutils` module to compile
# the Fortran code (pre-Python 3.12)
# If not available, use the newer backend (which requires Meson)
numpy_distutils_available = True

try:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        import numpy.distutils  # noqa: F401
except ModuleNotFoundError:
    numpy_distutils_available = False


# -----------------------------------------------------------------------------
# Generate model symbols and the Python class as usual

script = """
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
"""

symbols = fsic.parse_model(script)
SIM = fsic.build_model(symbols)


# -----------------------------------------------------------------------------
# Use the model symbols to generate equivalent Fortran code, writing to disk
fortran_code = fsic.fortran.build_fortran_definition(symbols)

with open('sim_fortran.f95', 'w') as f:
    f.write(fortran_code)

# Compile the Fortran code
# TODO: Look more closely at how best to resolve Meson versus old-style builds
if numpy_distutils_available:
    # Pre-Python 3.12 (numpy.distutils)
    compiler_options = ['f2py', '-c', 'sim_fortran.f95', '-m', 'sim_fortran']
    output = subprocess.run(
        compiler_options, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output.check_returncode()

else:
    try:
        compiler_options = ['f2py', '-c', 'sim_fortran.f95', '-m', 'sim_fortran']
        output = subprocess.run(
            compiler_options, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output.check_returncode()

    except subprocess.CalledProcessError:
        # Python 3.12 onwards (Meson): May need to remove pre-existing files
        # with '--clean'
        compiler_options = [
            'f2py',
            '-c',
            'sim_fortran.f95',
            '--clean',
            '-m',
            'sim_fortran',
        ]
        output = subprocess.run(
            compiler_options, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output.check_returncode()

# Define a new class that combines:
#  - the original Python SIM class (from above), including its underlying
#    functionality
#  - the `FortranEngine` class, to handle calls to the compiled code
#  - the compiled code itself, as an importable module
#
# The interface of this class is identical to the base Python one.


class SIMFortran(fsic.fortran.FortranEngine, SIM):
    ENGINE = importlib.import_module('sim_fortran')


# -----------------------------------------------------------------------------
# The canonical implementation above uses the Fortran version of `solve()` (if
# called). This is the fastest version because it bypasses Python at the levels
# of `solve_t()` and `_evaluate()`, directly calling their Fortran equivalents,
# instead.
#
# The two classes below are just examples to show how to force calls to the
# Python-wrapped Fortran code for `solve_t()` and `_evaluate()`.
#
# **This is only to show how performance varies by substituting Fortran at
#   different points.**


class SIMFortran_SolveT(fsic.fortran.FortranEngine, SIM):
    """Use pure Python `solve()` to call Fortran version of `solve_t()`."""

    ENGINE = importlib.import_module('sim_fortran')

    def solve(self, *args, **kwargs):
        return SIM.solve(self, *args, **kwargs)


class SIMFortran_Evaluate(fsic.fortran.FortranEngine, SIM):
    """Use pure Python `solve()` and `solve_t()` to call Fortran version of `_evaluate()`."""

    ENGINE = importlib.import_module('sim_fortran')

    def solve(self, *args, **kwargs):
        return SIM.solve(self, *args, **kwargs)

    def solve_t(self, *args, **kwargs):
        return SIM.solve_t(self, *args, **kwargs)


if __name__ == '__main__':

    def run(class_definition):
        """Instantiate and solve `class_definition`."""
        model = class_definition(
            range(1945, 2010 + 1), alpha_1=0.6, alpha_2=0.4, G=20, theta=0.2
        )
        model.solve()

    # Compare the performance of the various implementations
    repeat = 15
    number = 3

    performance_test = functools.partial(timeit.repeat, repeat=repeat, number=number)

    def run_and_summarise(times: List[float]) -> Tuple[float]:
        """Return the mean and standard deviation of the `times`."""
        return statistics.mean(times), statistics.stdev(times)

    # fmt: off
    python_times           = run_and_summarise(performance_test('run(SIM)',                 globals=globals()))
    fortran_times          = run_and_summarise(performance_test('run(SIMFortran)',          globals=globals()))
    fortran_solve_t_times  = run_and_summarise(performance_test('run(SIMFortran_SolveT)',   globals=globals()))
    fortran_evaluate_times = run_and_summarise(performance_test('run(SIMFortran_Evaluate)', globals=globals()))

    print('Means, [standard deviations] and (speed gains relative to Python) of solution times:')
    print(' - Python:                       {:.4f}s [{:.4f}s] (x{:.1f})'.format(*python_times,           python_times[0] / python_times[0]))
    print(' - Fortran (complete):           {:.4f}s [{:.4f}s] (x{:.1f})'.format(*fortran_times,          python_times[0] / fortran_times[0]))
    print(' - Fortran (`solve_t()` only):   {:.4f}s [{:.4f}s] (x{:.1f})'.format(*fortran_solve_t_times,  python_times[0] / fortran_solve_t_times[0]))
    print(' - Fortran (`_evaluate()` only): {:.4f}s [{:.4f}s] (x{:.1f})'.format(*fortran_evaluate_times, python_times[0] / fortran_evaluate_times[0]))
    # fmt: on
