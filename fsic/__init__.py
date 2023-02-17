# -*- coding: utf-8 -*-
"""
Tools for macroeconomic modelling in Python.
"""

__version__ = '0.8.0.dev'

from fsic import core
from fsic import exceptions
from fsic import extensions
from fsic import fortran
from fsic import functions
from fsic import parser
from fsic import tools

from .core import BaseModel, BaseLinker
from .parser import parse_model, build_model, build_model_definition
