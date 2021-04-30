"""
Plotting functions and utilities to handle images.
"""
from typing import (
    Dict,
    Tuple,
    List,
    Union,
    Optional,
    Callable,
    Any,
    overload,
    Literal,
    Collection,
)
import warnings
from functools import wraps
import colorsys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from skimage.exposure import equalize_hist as eq

from imc.types import DataFrame, Series, Array, Figure, Axis, Patch, ColorMap
from imc.utils import minmax_scale

DEFAULT_PIXEL_UNIT_NAME = r"$\mu$m"

DEFAULT_CHANNEL_COLORS = plt.get_cmap("tab10")(np.linspace(0, 1, 10))


def close_plots(func) -> Callable:
    """
    Decorator to close all plots on function exit.
    """

    @wraps(func)
    def close(*args, **kwargs) -> None:
        func(*args, **kwargs)
        plt.close("all")

    return close


def add_scale(
    _ax: Optional[Axis] = None,
    width: int = 100,
    unit: str = DEFAULT_PIXEL_UNIT_NAME,
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
        mpatches.Rectangle(
            (xposition, yposition), width, height, color="black", clip_on=False
        )
    )
    _ax.text(
        xposition + width + 10,
        yposition + height + text_separation,
        s=f"{width}{unit}",
        color="black",
        ha="left",
        fontsize=4,
    )


def add_minmax(minmax: Tuple[float, float], _ax: Optional[Axis] = None) -> None:
    """
    Add an annotation of the min and max values of the array.
    """
    # these values were optimized with a 1000 x 1000 reference figure
    if _ax is None:
        _ax = plt.gca()
    _ax.text(
        _ax.get_xlim()[1],
        -3,
        s=f"Range: {minmax[0]:.2f} -> {minmax[1]:.2f}",
        color="black",
        ha="right",
        fontsize=4,
    )


def add_legend(
    patches: List[Patch], ax: Optional[Axis] = None, **kwargs
) -> None:
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
    else:
        raise ValueError("Do not understand order of array axis.")
    return arr


def merge_channels(
    arr: Array,
    target_colors: Optional[List[Tuple[float, float, float]]] = None,
    return_colors: bool = False,
) -> Union[Array, Tuple[Array, List[Tuple[float, float, float]]]]:
    """
    Assumes [0, 1] float array.
    to is a tuple of 3 colors.
    """
    # defaults = list(matplotlib.colors.TABLEAU_COLORS.values())
    n_channels = arr.shape[0]
    if target_colors is None:
        target_colors = [
            matplotlib.colors.to_rgb(col)
            for col in DEFAULT_CHANNEL_COLORS[:n_channels]
        ]

    if (n_channels == 3) and target_colors is None:
        m = np.moveaxis(np.asarray([eq(x) for x in arr]), 0, -1)
        res = (m - m.min((0, 1))) / (m.max((0, 1)) - m.min((0, 1)))
        return res if not return_colors else (res, target_colors)

    elif isinstance(target_colors, (list, tuple)):
        assert len(target_colors) == n_channels
        target_colors = [matplotlib.colors.to_rgb(col) for col in target_colors]

    # work in int space to avoid float underflow
    if arr.min() >= 0 and arr.max() <= 1:
        arr *= 256
    else:
        arr = saturize(arr) * 256
    res = np.zeros(arr.shape[1:] + (3,))
    for i in range(n_channels):
        for j in range(3):
            res[:, :, j] = res[:, :, j] + arr[i] * target_colors[i][j]
    # return saturize(res) if not return_colors else (saturize(res), target_colors)
    return res if not return_colors else (res, target_colors)


def rainbow_text(
    x, y, strings, colors, orientation="horizontal", ax=None, **kwargs
):
    """
    Take a list of *strings* and *colors* and place them next to each
    other, with text strings[i] being shown in colors[i].

    Parameters
    ----------
    x, y : float
        Text position in data coordinates.
    strings : list of str
        The strings to draw.
    colors : list of color
        The colors to use.
    orientation : {'horizontal', 'vertical'}
    ax : Axes, optional
        The Axes to draw into. If None, the current axes will be used.
    **kwargs
        All other keyword arguments are passed to plt.text(), so you can
        set the font size, family, etc.

    From: https://matplotlib.org/3.2.1/gallery/text_labels_and_annotations/rainbow_text.html
    """
    from matplotlib.transforms import Affine2D

    if ax is None:
        ax = plt.gca()
    t = ax.transData
    canvas = ax.figure.canvas

    assert orientation in ["horizontal", "vertical"]
    if orientation == "vertical":
        kwargs.update(rotation=90, verticalalignment="bottom")

    for s, c in zip(strings, colors):
        text = ax.text(x, y, s + " ", color=c, transform=t, **kwargs)

        # Need to draw to update the text position.
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()

        if orientation == "horizontal":
            t = text.get_transform() + Affine2D().translate(ex.width * 0.5, 0)
        else:
            t = text.get_transform() + Affine2D().translate(0, ex.height * 0.5)


def get_rgb_cmaps() -> Tuple[ColorMap, ColorMap, ColorMap]:
    r = np.linspace(0, 1, 100).reshape((-1, 1))
    r = [
        matplotlib.colors.LinearSegmentedColormap.from_list("", p * r)
        for p in np.eye(3)
    ]
    return tuple(r)  # type: ignore


def get_dark_cmaps(
    n: int = 3, from_palette: str = "colorblind"
) -> List[ColorMap]:
    r = np.linspace(0, 1, 100).reshape((-1, 1))
    if n > len(sns.color_palette(from_palette)):
        print(
            "Chosen palette has less than the requested number of colors. "
            "Will reuse!"
        )
    return [
        matplotlib.colors.LinearSegmentedColormap.from_list("", np.array(p) * r)
        for p in sns.color_palette(from_palette, n)
    ]


def get_transparent_cmaps(
    n: int = 3, from_palette: Optional[str] = "colorblind"
) -> List[ColorMap]:
    __r = np.linspace(0, 1, 100)
    if n > len(sns.color_palette(from_palette)):
        print(
            "Chosen palette has less than the requested number of colors. "
            "Will reuse!"
        )
    return [
        matplotlib.colors.LinearSegmentedColormap.from_list(
            "", [p + (c,) for c in __r]
        )
        for p in sns.color_palette(from_palette, n)
    ]


def get_random_label_cmap(n=2 ** 16, h=(0, 1), l=(0.4, 1), s=(0.2, 0.8)):
    h, l, s = (
        np.random.uniform(*h, n),
        np.random.uniform(*l, n),
        np.random.uniform(*s, n),
    )
    cols = np.stack(
        [colorsys.hls_to_rgb(_h, _l, _s) for _h, _l, _s in zip(h, l, s)], axis=0
    )
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)


random_label_cmap = get_random_label_cmap

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
        print(
            "Chosen palette has less than the requested number of colors. "
            "Will reuse!"
        )

    colors = Series(sns.color_palette(from_palette, ident.max())).reindex(
        ident - 1
    )
    res = np.zeros((mask.shape) + (3,))
    for c, i in zip(colors, ident):
        x, y = np.nonzero(np.isin(mask, i))
        res[x, y, :] = c
    return res


@overload
def get_grid_dims(
    dims: Union[int, Collection],
    return_fig: Literal[True],
    nstart: Optional[int],
) -> Figure:
    ...


@overload
def get_grid_dims(
    dims: Union[int, Collection],
    return_fig: Literal[False],
    nstart: Optional[int],
) -> Tuple[int, int]:
    ...


def get_grid_dims(
    dims: Union[int, Collection],
    return_fig: bool = False,
    nstart: Optional[int] = None,
    **kwargs,
) -> Union[Tuple[int, int], Figure]:
    """
    Given a number of `dims` subplots, choose optimal x/y dimentions of plotting
    grid maximizing in order to be as square as posible and if not with more
    columns than rows.
    """
    if not isinstance(dims, int):
        dims = len(dims)
    if nstart is None:
        n = min(dims, 1 + int(np.ceil(np.sqrt(dims))))
    else:
        n = nstart
    if (n * n) == dims:
        m = n
    else:
        a = pd.Series(n * np.arange(1, n + 1)) / dims
        m = a[a >= 1].index[0] + 1
    assert n * m >= dims

    if n * m % dims > 1:
        try:
            n, m = get_grid_dims(dims=dims, return_fig=False, nstart=n - 1)
        except IndexError:
            pass
    if not return_fig:
        return n, m
    else:
        if "figsize" not in kwargs:
            kwargs["figsize"] = (m * 4, n * 4)
        fig, ax = plt.subplots(n, m, **kwargs)
        return fig


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
        fig, axs = plt.subplots(
            1, 1, figsize=(6 * 1, 6 * 1), sharex=True, sharey=True
        )
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
        fig, ax = plt.subplots(
            1, 1, figsize=(6 * 1, 6 * 1), sharex=True, sharey=True
        )
    cmaps = get_transparent_cmaps(arr.shape[0], from_palette=palette)
    patches = list()
    for i, (m, c) in enumerate(zip(channel_labels, cmaps)):
        x = arr[i].squeeze()
        ax.imshow(
            x,
            cmap=c,
            label=m,
            interpolation="bilinear",
            rasterized=True,
            alpha=0.9,
        )
        ax.axis("off")
        patches.append(mpatches.Patch(color=c(256), label=m))
    ax.legend(
        handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0
    )
    return fig if axis is None else ax


def rasterize_scanpy(fig: Figure) -> None:
    """
    Rasterize figure containing Scatter plots of single cells
    such as PCA and UMAP plots drawn by Scanpy.
    """
    import warnings

    with warnings.catch_warnings(record=False) as w:
        warnings.simplefilter("ignore")
        yes_class = (
            matplotlib.collections.PathCollection,
            matplotlib.collections.LineCollection,
        )
        not_clss = (
            matplotlib.text.Text,
            matplotlib.axis.XAxis,
            matplotlib.axis.YAxis,
        )
        for axs in fig.axes:
            for __c in axs.get_children():
                if not isinstance(__c, not_clss):
                    if not __c.get_children():
                        if isinstance(__c, yes_class):
                            __c.set_rasterized(True)
                    for _cc in __c.get_children():
                        if not isinstance(_cc, not_clss):
                            if isinstance(_cc, yes_class):
                                _cc.set_rasterized(True)


def add_centroids(a, ax=None, res=None, column=None, algo="umap"):
    """
    a: AnnData
    ax: matplotlib.Axes.axes
    res: resolution of clusters to label
    """
    from numpy_groupies import aggregate

    if ax is None:
        ax = plt.gca()

    if column is None:
        # try to guess the clustering key_added
        if res is None:
            try:
                lab = a.obs.columns[a.obs.columns.str.contains("cluster_")][0]
            except:
                lab = a.obs.columns[a.obs.columns.str.contains("leiden")][0]
        else:
            lab = f"cluster_{res}"
    else:
        lab = column

    # # # centroids:
    offset = 0 if algo != "diffmap" else 1
    cent = aggregate(
        a.obs[lab].cat.codes,
        a.obsm[f"X_{algo}"][:, 0 + offset : 2 + offset],
        func="mean",
        axis=0,
    )
    for i, clust in enumerate(a.obs[lab].sort_values().unique()):
        ax.text(*cent[i], s=clust)
