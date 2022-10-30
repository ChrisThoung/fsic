# -*- coding: utf-8 -*-
"""
Adapted from:
https://packaging.python.org/tutorials/packaging-projects/
"""

import itertools
import glob
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


# Assemble requirements files into sets of optional dependencies
extra_requirements = {}

for path in glob.glob(os.path.join('requirements', '*.txt')):
    name = os.path.splitext(os.path.split(path)[1])[0]
    extra_requirements[name] = list(map(str.strip, open(path)))

extra_requirements['all'] = list(set(
    itertools.chain(*extra_requirements.values())))


setuptools.setup(
    name='fsic',
    version=get_version(),
    author='Chris Thoung',
    author_email='chris.thoung@gmail.com',
    description='Tools for macroeconomic modelling in Python',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ChrisThoung/fsic',
    packages=setuptools.find_packages(include=['fsic']),
    python_requires='>=3.6',

    install_requires=extra_requirements['minimal'],
    extras_require=extra_requirements,

    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
    ],
    platforms=['Any'],
)
