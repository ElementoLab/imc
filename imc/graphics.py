#!/usr/bin/env python

from typing import Dict, Tuple, List, Union, Optional, Callable  # , Literal
import warnings
from functools import wraps


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# from skimage.exposure import equalize_hist as eq

from imc.types import DataFrame, Series, Array, Figure, Axis, Patch, ColorMap
from imc.utils import minmax_scale

DEFAULT_PIXEL_UNIT_NAME = r"$\mu$m"


SEQUENCIAL_CMAPS = [
    "Purples",
    "Greens",
    "Oranges",
    "Greys",
    "Reds",
    "Blues",
    "YlOrBr",
    "YlOrRd",
    "OrRd",
    "PuRd",
    "RdPu",
    "BuPu",
    "GnBu",
    "PuBu",
    "YlGnBu",
    "PuBuGn",
    "BuGn",
    "YlGn",
    "binary",
    "gist_yarg",
    "gist_gray",
    "gray",
    "bone",
    "pink",
    "spring",
    "summer",
    "autumn",
    "winter",
    "cool",
    "Wistia",
    "hot",
    "afmhot",
    "gist_heat",
    "copper",
]


def to_color_series(x: Series, cmap: Optional[str] = "Greens") -> Series:
    """Map a numeric pandas series to a series of RBG values."""
    return Series(plt.get_cmap(cmap)(minmax_scale(x)).tolist(), index=x.index, name=x.name)


def to_color_dataframe(
    x: Union[Series, DataFrame], cmaps: Optional[Union[str, List[str]]] = None, offset: int = 0,
) -> DataFrame:
    """Map a numeric pandas DataFrame to RGB values."""
    if isinstance(x, pd.Series):
        x = x.to_frame()
    if cmaps is None:
        # the offset is in order to get different colors for rows and columns by default
        cmaps = [plt.get_cmap(cmap) for cmap in SEQUENCIAL_CMAPS[offset:]]
    if isinstance(cmaps, str):
        cmaps = [cmaps]
    return pd.concat([to_color_series(x[col], cmap) for col, cmap in zip(x, cmaps)], axis=1)


def _add_extra_colorbars_to_clustermap(
    grid: sns.matrix.ClusterGrid,
    datas: Union[Series, DataFrame],
    cmaps: Optional[Union[str, List[str]]] = None,
    # location: Union[Literal["col"], Literal["row"]] = "row",
    location: str = "row",
) -> None:
    """Add either a row or column colorbar to a seaborn Grid."""

    def add(data: Series, cmap: str, bbox: List[List[int]], orientation: str) -> None:
        ax = grid.fig.add_axes(matplotlib.transforms.Bbox(bbox))
        norm = matplotlib.colors.Normalize(vmin=data.min(), vmax=data.max())
        cb1 = matplotlib.colorbar.ColorbarBase(
            ax, cmap=plt.get_cmap(cmap), norm=norm, orientation=orientation, label=data.name
        )

    offset = 1 if location == "row" else 0

    if isinstance(datas, pd.Series):
        datas = datas.to_frame()
    if cmaps is None:
        cmaps = SEQUENCIAL_CMAPS[offset:]
    if isinstance(cmaps, str):
        cmaps = [cmaps]

    # get position to add new axis in existing figure
    # # get_position() returns ((x0, y0), (x1, y1))
    heat = grid.ax_heatmap.get_position()
    cbar_spacing = 0.05
    cbar_size = 0.025
    if location == "col":
        orientation = "vertical"
        dend = grid.ax_col_dendrogram.get_position()
        y0 = dend.y0
        y1 = dend.y1
        for i, (data, cmap) in enumerate(zip(datas, cmaps)):
            if i == 0:
                x0 = heat.x1
                x1 = heat.x1 + cbar_size
            else:
                x0 += cbar_size + cbar_spacing
                x1 += cbar_size + cbar_spacing
            add(datas[data], cmap, [[x0, y0], [x1, y1]], orientation)
    else:
        orientation = "horizontal"
        dend = grid.ax_row_dendrogram.get_position()
        x0 = dend.x0
        x1 = dend.x1
        for i, (data, cmap) in enumerate(zip(datas, cmaps)):
            if i == 0:
                y0 = dend.y0 - cbar_size
                y1 = dend.y0
            else:
                y0 -= cbar_size + cbar_spacing
                y1 -= cbar_size + cbar_spacing
            add(datas[data], cmap, [[x0, y0], [x1, y1]], orientation)


def _add_colorbars(
    grid: sns.matrix.ClusterGrid,
    rows: DataFrame = None,
    cols: DataFrame = None,
    row_cmaps: Optional[List[str]] = None,
    col_cmaps: Optional[List[str]] = None,
) -> None:
    """Add row and column colorbars to a seaborn Grid."""
    if rows is not None:
        _add_extra_colorbars_to_clustermap(grid, rows, location="row", cmaps=row_cmaps)
    if cols is not None:
        _add_extra_colorbars_to_clustermap(grid, cols, location="col", cmaps=col_cmaps)


def colorbar_decorator(f: Callable) -> Callable:
    """
    Decorate seaborn.clustermap in order to have numeric values passed to the
    ``row_colors`` and ``col_colors`` arguments translated into row and column
    annotations and in addition colorbars for the restpective values.
    """
    # TODO: edit original seaborn.clustermap docstring to document {row,col}_colors_cmaps arguments.
    @wraps(f)
    def clustermap(*args, **kwargs):
        cmaps = {"row": None, "col": None}
        # capture "row_cmaps" and "col_cmaps" out of the kwargs
        for arg in ["row", "col"]:
            if arg + "_colors_cmaps" in kwargs:
                cmaps[arg] = kwargs[arg + "_colors_cmaps"]
                del kwargs[arg + "_colors_cmaps"]
        # get dataframe with colors and respective colormaps for rows and cols
        # instead of the original numerical values
        _kwargs = dict(rows=None, cols=None)
        for arg in ["row", "col"]:
            if arg + "_colors" in kwargs:
                if isinstance(kwargs[arg + "_colors"], (pd.DataFrame, pd.Series)):
                    _kwargs[arg + "s"] = kwargs[arg + "_colors"]
                    kwargs[arg + "_colors"] = to_color_dataframe(
                        x=kwargs[arg + "_colors"], cmaps=cmaps[arg], offset=1 if arg == "row" else 0
                    )
        grid = f(*args, **kwargs)
        _add_colorbars(grid, **_kwargs, row_cmaps=cmaps["row"], col_cmaps=cmaps["col"])
        return grid

    return clustermap


def add_scale(
    _ax: Optional[Axis] = None, width: int = 100, unit: str = DEFAULT_PIXEL_UNIT_NAME
) -> None:
    """
    Add a scale bar to a figure.
    Should be called after plotting (usually with matplotlib.pyplot.imshow).
    """
    # these values were optimized with a 1000 x 1000 reference figure
    if _ax is None:
        _ax = plt.gca()
    height = 1 / 40
    text_separation = 40 / 1000
    __h = sum(_ax.get_ylim()) + 1
    height = height * __h
    xposition = 3
    yposition = -height - 3
    text_separation = text_separation * height
    _ax.add_patch(
        mpatches.Rectangle((xposition, yposition), width, height, color="black", clip_on=False)
    )
    _ax.text(
        xposition + width + 10,
        yposition + height + text_separation,
        s=f"{width}{unit}",
        color="black",
        ha="left",
        fontsize=4,
    )


def add_legend(patches: List[Patch], ax: Optional[Axis] = None, **kwargs) -> None:
    """Add a legend to an existing axis."""
    if ax is None:
        ax = plt.gca()
    _patches = np.asarray(patches)
    _patches = _patches[
        pd.Series([(p.get_facecolor(), p.get_label()) for p in _patches])
        .drop_duplicates()
        .index.tolist()
    ]

    defaults = dict(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    defaults.update(kwargs)
    ax.legend(handles=_patches.tolist(), **defaults)


def saturize(arr: Array) -> Array:
    """Saturize an image by channel, by minmax scalling each."""
    if np.argmin(arr.shape) == 0:
        for i in range(arr.shape[0]):
            arr[i, :, :] = minmax_scale(arr[i, :, :])
    elif np.argmin(arr.shape) == 2:
        for i in range(arr.shape[2]):
            arr[:, :, i] = minmax_scale(arr[:, :, i])
    return arr


def to_merged_colors(
    arr: Array, to_colors: Optional[List[str]] = None, return_colors: bool = False
) -> Union[Array, Tuple[Array, List[Tuple[float, float, float]]]]:
    """
    Assumes [0, 1] float array.
    to is a tuple of 3 colors.
    """
    defaults = [
        "red",
        "green",
        "blue",
        "orange",
        "purple",
        "brown",
        "pink",
        "olive",
        "cyan",
        "gray",
    ]
    # defaults = list(matplotlib.colors.TABLEAU_COLORS.values())
    n_channels = arr.shape[0]
    if to_colors is None:
        target_colors = [matplotlib.colors.to_rgb(col) for col in defaults[:n_channels]]
    elif isinstance(to_colors, list):
        assert len(to_colors) == n_channels
        target_colors = [matplotlib.colors.to_rgb(col) for col in to_colors]

    if arr.min() >= 0 and arr.max() <= 1:
        arr *= 256  # done in int space to avoid float underflow
    res = np.zeros(arr.shape[1:] + (3,))
    for i in range(n_channels):
        for j in range(3):
            res[:, :, j] = res[:, :, j] + arr[i] * target_colors[i][j]
    return saturize(res) if not return_colors else (saturize(res), target_colors)


def get_rgb_cmaps() -> Tuple[ColorMap, ColorMap, ColorMap]:
    r = np.linspace(0, 1, 100).reshape((-1, 1))
    r = [matplotlib.colors.LinearSegmentedColormap.from_list("", p * r) for p in np.eye(3)]
    return tuple(r)  # type: ignore


def get_dark_cmaps(n: int = 3, from_palette: str = "colorblind") -> List[ColorMap]:
    r = np.linspace(0, 1, 100).reshape((-1, 1))
    if n > len(sns.color_palette(from_palette)):
        warnings.warn("Chosen palette has less than the requested number of colors. " "Will reuse!")
    return [
        matplotlib.colors.LinearSegmentedColormap.from_list("", np.array(p) * r)
        for p in sns.color_palette(from_palette, n)
    ]


def get_transparent_cmaps(n: int = 3, from_palette: Optional[str] = "colorblind") -> List[ColorMap]:
    __r = np.linspace(0, 1, 100)
    if n > len(sns.color_palette(from_palette)):
        warnings.warn("Chosen palette has less than the requested number of colors. " "Will reuse!")
    return [
        matplotlib.colors.LinearSegmentedColormap.from_list("", [p + (c,) for c in __r])
        for p in sns.color_palette(from_palette, n)
    ]


# TODO: see if function can be sped up e.g. with Numba
def cell_labels_to_mask(mask: Array, labels: Union[Series, Dict]) -> Array:
    """Replaces integers in `mask` with values from the mapping in `labels`."""
    res = np.zeros(mask.shape, dtype=int)
    for k, v in labels.items():
        res[mask == k] = v
    return res


def numbers_to_rgb_colors(
    mask: Array, from_palette: str = "tab20", remove_zero: bool = True
) -> Array:
    """Colors each integer in the 2D `mask` array with a unique color by
    expanding the array to 3 dimensions."""

    ident = np.sort(np.unique(mask))
    if remove_zero:
        ident = ident[ident != 0]
    n_colors = len(ident)

    if n_colors > len(sns.color_palette(from_palette)):
        warnings.warn("Chosen palette has less than the requested number of colors." "Will reuse!")

    colors = Series(sns.color_palette(from_palette, ident.max())).reindex(ident - 1)
    res = np.zeros((mask.shape) + (3,))
    for c, i in zip(colors, ident):
        x, y = np.nonzero(np.isin(mask, i))
        res[x, y, :] = c
    return res


def get_grid_dims(dims: int, nstart: Optional[int] = None) -> Tuple[int, int]:
    """
    Given a number of `dims` subplots, choose optimal x/y dimentions of plotting
    grid maximizing in order to be as square as posible and if not with more
    columns than rows.
    """
    if nstart is None:
        n = min(dims, 1 + int(np.ceil(np.sqrt(dims))))
    else:
        n = nstart
    if (n * n) == dims:
        m = n
    else:
        a = Series(n * np.arange(1, n + 1)) / dims
        m = a[a >= 1].index[0] + 1
    assert n * m >= dims

    if n * m % dims > 1:
        try:
            n, m = get_grid_dims(dims=dims, nstart=n - 1)
        except IndexError:
            pass
    return n, m


def share_axes_by(axes: Axis, by: str) -> None:
    """
    Share given axes after figure creation.
    Useful when not all subplots of a figure should be shared.
    """
    if by == "row":
        for row in axes:
            for axs in row[1:]:
                row[0].get_shared_x_axes().join(row[0], axs)
                row[0].get_shared_y_axes().join(row[0], axs)
                axs.set_yticklabels([])
    elif by == "col":
        for col in axes:
            for axs in col[:-1]:
                col[-1].get_shared_x_axes().join(col[-1], axs)
                col[-1].get_shared_y_axes().join(col[-1], axs)
                axs.set_xticklabels([])
    elif by == "both":
        # attach all axes to upper left one
        master = axes[0, 0]
        for axs in axes.flatten():
            master.get_shared_x_axes().join(master, axs)
            master.get_shared_y_axes().join(master, axs)
        # remove both ticks from axs away from left down border
        for axs in axes[:-1, 1:].flatten():
            axs.set_xticklabels([])
            axs.set_yticklabels([])
        # remove xticks from first columns except last
        for axs in axes[:-1, 0]:
            axs.set_xticklabels([])
        # remove yticks from last row except first
        for axs in axes[-1, 1:]:
            axs.set_yticklabels([])


def plot_single_channel(
    arr: Array, axis: Optional[Axis] = None, cmap: Optional[ColorMap] = None
) -> Union[Figure, Axis]:
    """Plot a single image channel either in a new figure or in an existing axis"""
    if axis is None:
        fig, axs = plt.subplots(1, 1, figsize=(6 * 1, 6 * 1), sharex=True, sharey=True)
    axs.imshow(arr, cmap=cmap, interpolation="bilinear", rasterized=True)
    axs.axis("off")
    return fig if axis is None else axs


def plot_overlayied_channels(
    arr: Array,
    channel_labels: List[str],
    axis: Optional[Axis] = None,
    palette: Optional[str] = None,
) -> Union[Figure, Axis]:
    if axis is None:
        fig, ax = plt.subplots(1, 1, figsize=(6 * 1, 6 * 1), sharex=True, sharey=True)
    cmaps = get_transparent_cmaps(arr.shape[0], from_palette=palette)
    patches = list()
    for i, (m, c) in enumerate(zip(channel_labels, cmaps)):
        x = arr[i].squeeze()
        ax.imshow(x, cmap=c, label=m, interpolation="bilinear", rasterized=True, alpha=0.9)
        ax.axis("off")
        patches.append(mpatches.Patch(color=c(256), label=m))
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    return fig if axis is None else ax


def rasterize_scanpy(fig: Figure) -> None:
    # TODO: avoid rasterization of matplotlib.offsetbox.VPacker at least
    import warnings

    with warnings.catch_warnings(record=False) as w:
        warnings.simplefilter("always")
        clss = (matplotlib.text.Text, matplotlib.axis.XAxis, matplotlib.axis.YAxis)
        for axs in fig.axes:
            for __c in axs.get_children():
                if not isinstance(__c, clss):
                    if not __c.get_children():
                        __c.set_rasterized(True)
                    for _cc in __c.get_children():
                        if not isinstance(__c, clss):
                            _cc.set_rasterized(True)
