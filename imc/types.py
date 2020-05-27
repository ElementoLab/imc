#!/usr/bin/env python

"""
Types used in the library as defined here.

See https://docs.python.org/3/library/typing.html
for more information.
"""

from __future__ import annotations
import os
from typing import Union, TypeVar
import pathlib

import matplotlib  # type: ignore
import pandas  # type: ignore
import numpy  # type: ignore


class Path(pathlib.Path):
    """
    A pathlib.Path child class that allows concatenation with strings using the
    addition operator.

    In addition, it implements the ``startswith`` and ``endswith`` methods
    just like in the base :obj:`str` type.
    """

    _flavour = (
        pathlib._windows_flavour  # pylint: disable=W0212
        if os.name == "nt"
        else pathlib._posix_flavour  # pylint: disable=W0212
    )

    def __add__(self, string: str) -> "Path":
        return Path(str(self) + string)

    def startswith(self, string: str) -> bool:
        return str(self).startswith(string)

    def endswith(self, string: str) -> bool:
        return str(self).endswith(string)

    def replace_(self, patt: str, repl: str) -> "Path":
        return Path(str(self).replace(patt, repl))


GenericType = TypeVar("GenericType")

# type aliasing (done with Union to distinguish from other declared variables)
Array = Union[numpy.ndarray]
Series = Union[pandas.Series]
MultiIndexSeries = Union[pandas.Series]
DataFrame = Union[pandas.DataFrame]

Figure = Union[matplotlib.figure.Figure]
Axis = Union[matplotlib.axis.Axis]
Patch = Union[matplotlib.patches.Patch]
ColorMap = Union[matplotlib.colors.LinearSegmentedColormap]
