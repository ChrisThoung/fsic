# -*- coding: utf-8 -*-
"""
fsic
====
Tools for macroeconomic modelling in Python.
"""

__version__ = '0.8.0.dev'


import copy
import enum
import itertools
import keyword
import re
import textwrap
from typing import Any, Callable, Dict, Hashable, Iterator, List, Match, NamedTuple, Optional, Sequence, Tuple, Union
import warnings

import numpy as np

from .exceptions import BuildError, DimensionError, DuplicateNameError, EvalError, InitialisationError, NonConvergenceError, ParserError, SolutionError, SymbolError
