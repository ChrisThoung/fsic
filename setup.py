# -*- coding: utf-8 -*-
"""
Adapted from:
https://packaging.python.org/tutorials/packaging-projects/
"""

import setuptools
from fsic import __version__


with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name='fsic',
    version=__version__,
    author='Chris Thoung',
    author_email='chris.thoung@gmail.com',
    description='Tools for macroeconomic modelling in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ChrisThoung/fsic',
    py_modules=['fsic', 'fsictools', 'fsic_fortran'],
    python_requires='>=3.6',

    install_requires=[
        'numpy',
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
    platforms=['Any'],
)
