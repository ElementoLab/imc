from __future__ import annotations
import os
from typing import Union, TypeVar
import pathlib

import matplotlib
import pandas
import numpy


class Path(pathlib.Path):
    """
    A pathlib.Path child that allows concatenation with strings using the
    addition operator
    """

    _flavour = pathlib._windows_flavour if os.name == "nt" else pathlib._posix_flavour

    def __add__(self, string: str) -> "Path":
        return Path(str(self) + string)

    def startswith(self, string: str) -> bool:
        return str(self).startswith(string)

    def endswith(self, string: str) -> bool:
        return str(self).endswith(string)

    def replace_(self, patt: str, repl: str) -> Path:
        return Path(str(self).replace(patt, repl))


GenericType = TypeVar("GenericType")

# type aliasing (done with Union to distinguish from other declared variables)
Axis = Union[matplotlib.axis.Axis]
Figure = Union[matplotlib.figure.Figure]
Patch = Union[matplotlib.patches.Patch]
Array = Union[numpy.ndarray]
DataFrame = Union[pandas.DataFrame]
Series = Union[pandas.Series]
MultiIndexSeries = Union[pandas.Series]
ColorMap = Union[matplotlib.colors.LinearSegmentedColormap]
