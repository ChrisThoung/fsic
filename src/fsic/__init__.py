# -*- coding: utf-8 -*-
"""
Tools for macroeconomic modelling in Python.
"""

__version__ = '0.8.0.dev'

from fsic import core, exceptions, extensions, fortran, functions, parser, tools  # noqa: F401

from .core import BaseLinker, BaseModel  # noqa: F401
from .parser import build_model, build_model_definition, parse_model  # noqa: F401
