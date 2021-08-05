# -*- coding: utf-8 -*-
"""
Adapted from:
https://packaging.python.org/tutorials/packaging-projects/
"""

import os
import re
import setuptools


# Get version number without having to `import` the `fsic` module (and
# attempting to import NumPy before it gets installed). Idea from:
# https://packaging.python.org/guides/single-sourcing-package-version/
def get_version():
    with open(os.path.join('fsic', '__init__.py')) as f:
        for line in f:
            if line.startswith('__version__'):
                return re.split(r'''["']''', line)[1]


with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name='fsic',
    version=get_version(),
    author='Chris Thoung',
    author_email='chris.thoung@gmail.com',
    description='Tools for macroeconomic modelling in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ChrisThoung/fsic',
    packages=setuptools.find_packages(include=['fsic']),
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
