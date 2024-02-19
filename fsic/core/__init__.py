# -*- coding: utf-8 -*-
"""
Core `fsic` classes for defining and solving economic models. Economic models
are implemented as derived classes of those in this subpackage, inheriting the
necessary attributes and methods that make up the API.

Base classes:

* `VectorContainer`, for data handling
* `BaseModel`, for a single economic model
* `BaseLinker`, to store and solve multiple model instances i.e. as a
  multi-region/entity model
"""

from .containers import VectorContainer  # noqa: F401
from .models import BaseModel  # noqa: F401
from .linkers import BaseLinker  # noqa: F401
