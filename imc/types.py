"""
Specific types or type aliases used in the library.
"""

from __future__ import annotations
import os
import typing as tp
import pathlib
import argparse

import matplotlib
import pandas
import numpy
from anndata import AnnData as _AnnData


class Path(pathlib.Path):
    """
    A pathlib.Path child class that allows concatenation with strings
    by overloading the addition operator.

    In addition, it implements the ``startswith`` and ``endswith`` methods
    just like in the base :obj:`str` type.

    The ``replace_`` implementation is meant to be an implementation closer
    to the :obj:`str` type.

    Iterating over a directory with ``iterdir`` that does not exists
    will return an empty iterator instead of throwing an error.

    Creating a directory with ``mkdir`` allows existing directory and
    creates parents by default.
    """

    _flavour = (
        pathlib._windows_flavour  # type: ignore[attr-defined]  # pylint: disable=W0212
        if os.name == "nt"
        else pathlib._posix_flavour  # type: ignore[attr-defined]  # pylint: disable=W0212
    )

    def __add__(self, string: str) -> Path:
        return Path(str(self) + string)

    def startswith(self, string: str) -> bool:
        return str(self).startswith(string)

    def endswith(self, string: str) -> bool:
        return str(self).endswith(string)

    def replace_(self, patt: str, repl: str) -> Path:
        return Path(str(self).replace(patt, repl))

    def iterdir(self) -> tp.Generator:
        if self.exists():
            yield from [Path(x) for x in pathlib.Path(str(self)).iterdir()]
        yield from []

    def mkdir(self, mode=0o777, parents: bool = True, exist_ok: bool = True) -> Path:
        super().mkdir(mode=mode, parents=parents, exist_ok=exist_ok)
        return self


GenericType = tp.TypeVar("GenericType")

# type aliasing (done with Union to distinguish from other declared variables)


# Args = Union[argparse.Namespace]
class Args(argparse.Namespace, tp.Mapping[str, tp.Any]):
    pass


# Series = Union[pandas.Series]
class Series(pandas.Series, tp.Mapping[tp.Any, tp.Any]):
    pass


Array = tp.Union[numpy.ndarray]

MultiIndexSeries = tp.Union[pandas.Series]
DataFrame = tp.Union[pandas.DataFrame]
AnnData = tp.Union[_AnnData]

Figure = tp.Union[matplotlib.figure.Figure]
Axis = tp.Union[matplotlib.axis.Axis]
Patch = tp.Union[matplotlib.patches.Patch]
ColorMap = tp.Union[matplotlib.colors.LinearSegmentedColormap]
