# -*- coding: utf-8 -*-
"""
fortran_engine
==============
Example of how to generate and link Fortran code for faster model solution.

**Requires NumPy with a working F2PY setup.**

The example model in this script is a simplified five-equation version of
Godley and Lavoie’s (2007) Model *SIM*.

References:

    Godley, W., Lavoie, M. (2007),
    *Monetary economics: an integrated approach to
    credit, money, income, production and wealth*,
    Palgrave Macmillan
"""

import importlib
import statistics
import subprocess
import timeit

import fsic
import fsic_fortran


# -----------------------------------------------------------------------------
# Generate model symbols and the Python class as usual

script = '''
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
'''

symbols = fsic.parse_model(script)
SIM = fsic.build_model(symbols)


# -----------------------------------------------------------------------------
# Use the model symbols to generate equivalent Fortran code, writing to disk
fortran_code = fsic_fortran.build_fortran_definition(symbols)

with open('sim_fortran.f95', 'w') as f:
    f.write(fortran_code)

# Compile the Fortran code
output = subprocess.run(['f2py', '-c', 'sim_fortran.f95', '-m', 'sim_fortran'],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output.check_returncode()

# Define a new class that combines:
#  - the original Python SIM class (from above), including its underlying
#    functionality
#  - the `FortranEngine` class, to handle calls to the compiled code
#  - the compiled code itself, as an importable module
#
# The interface of this class is identical to the base Python one.

class SIMFortran(fsic_fortran.FortranEngine, SIM):
    ENGINE = importlib.import_module('sim_fortran')


if __name__ == '__main__':
    def run(class_definition):
        """Instantiate and solve `class_definition`."""
        model = class_definition(range(1945, 2010 + 1),
                                 alpha_1=0.6, alpha_2=0.4,
                                 G=20, theta=0.2)
        model.solve()

    # Compare the performance of the two implementations
    repeat = 15
    number = 3

    python_times  = timeit.repeat('run(SIM)',        repeat=repeat, number=number, globals=globals())
    fortran_times = timeit.repeat('run(SIMFortran)', repeat=repeat, number=number, globals=globals())

    print('Mean and standard deviation of results:')
    print(' - Python:  {:.4f}s ({:.4f}s)'.format(statistics.mean(python_times), statistics.stdev(python_times)))
    print(' - Fortran: {:.4f}s ({:.4f}s)'.format(statistics.mean(fortran_times), statistics.stdev(fortran_times)))
