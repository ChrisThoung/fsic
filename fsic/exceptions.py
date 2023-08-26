# -*- coding: utf-8 -*-
"""
Custom exceptions.
"""


class FSICError(Exception):
    pass


class BuildError(FSICError):
    pass


class DimensionError(FSICError):
    pass


class DuplicateNameError(FSICError):
    pass


class InitialisationError(FSICError):
    pass


class NonConvergenceError(FSICError):
    pass


class ParserError(FSICError):
    pass


class SolutionError(FSICError):
    pass


class SymbolError(FSICError):
    pass


class FortranEngineError(FSICError):
    pass
