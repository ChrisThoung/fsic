# -*- coding: utf-8 -*-
"""
fsic
====
Tools for macroeconomic modelling in Python.
"""

__version__ = '0.8.0.dev'

from .core import BaseModel, BaseLinker
from .parser import parse_model, build_model

import fsic.core as core
import fsic.exceptions as exceptions
import fsic.fortran as fortran
import fsic.parser as parser
import fsic.tools as tools
