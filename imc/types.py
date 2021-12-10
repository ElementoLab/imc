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


__all__ = [
    "Path",
    "GenericType",
    "Args",
    "Array",
    "MultiIndexSeries",
    "Series",
    "DataFrame",
    "AnnData",
    "Figure",
    "Axis",
    "Patch",
    "ColorMap",
]


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

    def unlink(self, missing_ok: bool = True) -> Path:
        super().unlink(missing_ok=missing_ok)
        return self

    def mkdir(self, mode=0o777, parents: bool = True, exist_ok: bool = True) -> Path:
        super().mkdir(mode=mode, parents=parents, exist_ok=exist_ok)
        return self

    def glob(self, pattern: str) -> tp.Generator:
        # to support ** with symlinks: https://bugs.python.org/issue33428
        from glob import glob

        if "**" in pattern:
            sep = "/" if self.is_dir() else ""
            yield from map(
                Path,
                glob(self.as_posix() + sep + pattern, recursive=True),
            )
        else:
            yield from super().glob(pattern)


GenericType = tp.TypeVar("GenericType")

# type aliasing (done with Union to distinguish from other declared variables)


# # Args = Union[argparse.Namespace]
# class Args(argparse.Namespace, tp.Mapping[str, tp.Any]):
#     pass


# # Series = Union[pandas.Series]
# class Series(pandas.Series, tp.Mapping[tp.Any, tp.Any]):
#     pass


Args = tp.Union[argparse.Namespace, tp.Mapping[str, tp.Any]]

Array = numpy.ndarray

MultiIndexSeries = pandas.Series
Series = pandas.Series
DataFrame = pandas.DataFrame
AnnData = _AnnData

Figure = matplotlib.figure.Figure
Axis = matplotlib.axis.Axis
Patch = matplotlib.patches.Patch
ColorMap = matplotlib.colors.LinearSegmentedColormap
