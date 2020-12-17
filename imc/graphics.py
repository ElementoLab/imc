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
)  # , Literal
import warnings
from functools import wraps


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


DEFAULT_CHANNEL_COLORS = [
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


def is_numeric(x: Series) -> bool:
    if x.dtype in [
        "float",
        "float32",
        "float64",
        "int",
        "int32",
        "int64",
    ] or is_datetime(x):
        return True
    if (x.dtype in ["object"]) or isinstance(x.dtype, pd.CategoricalDtype):
        return False
    raise ValueError(f"Cannot transfer data type '{x.dtype}' to color!")


def is_datetime(x: Series) -> bool:
    if "datetime" in x.dtype.name:
        return True
    return False


def to_numeric(x: Series) -> Series:
    """Encode a string or categorical series to integer type."""
    res = pd.Series(
        index=x.index, dtype=float
    )  # this will imply np.nan keeps being np.nan
    for i, v in enumerate(x.value_counts().sort_index().index):
        res.loc[x == v] = i
    return res


def get_categorical_cmap(x: Series) -> matplotlib.colors.ListedColormap:
    """Choose a colormap for a categorical series encoded as ints."""
    # TODO: allow choosing from sets of categorical cmaps.
    # additional ones could be Pastel1/2, Set2/3

    # colormaps are truncated to existing values
    n = int(x.max() + 1)
    for v in [10, 20]:
        if n < v:
            return matplotlib.colors.ListedColormap(
                colors=plt.get_cmap(f"tab{v}").colors[:n], name=f"tab{v}-{n}"
            )
    if n < 40:
        return matplotlib.colors.ListedColormap(
            colors=np.concatenate(
                [
                    plt.get_cmap("tab20c")(range(20)),
                    plt.get_cmap("tab20b")(range(20)),
                ]
            )[:n],
            name=f"tab40-{n}",
        )
    raise ValueError("Only up to 40 unique values can be plotted as color.")


def to_color_series(x: Series, cmap: Optional[str] = "Greens") -> Series:
    """
    Map a numeric pandas series to a series of RBG values.
    NaN values are white.
    """
    if is_numeric(x):
        return pd.Series(
            plt.get_cmap(cmap)(minmax_scale(x)).tolist(),
            index=x.index,
            name=x.name,
        )
    # str or categorical
    res = to_numeric(x)
    cmap = get_categorical_cmap(res)
    # float values passed to cmap must be in [0.0-1.0] range
    return pd.Series(cmap(res / res.max()).tolist(), index=x.index, name=x.name)


def to_color_dataframe(
    x: Union[Series, DataFrame],
    cmaps: Optional[Union[str, List[str]]] = None,
    offset: int = 0,
) -> DataFrame:
    """Map a numeric pandas DataFrame to RGB values."""
    if isinstance(x, pd.Series):
        x = x.to_frame()
    if cmaps is None:
        # the offset is in order to get different colors for rows and columns by default
        cmaps = [plt.get_cmap(cmap) for cmap in SEQUENCIAL_CMAPS[offset:]]
    if isinstance(cmaps, str):
        cmaps = [cmaps]
    return pd.concat(
        [to_color_series(x[col], cmap) for col, cmap in zip(x, cmaps)], axis=1
    )


def _add_extra_colorbars_to_clustermap(
    grid: sns.matrix.ClusterGrid,
    datas: Union[Series, DataFrame],
    cmaps: Optional[Union[str, List[str]]] = None,
    # location: Union[Literal["col"], Literal["row"]] = "row",
    location: str = "row",
) -> None:
    """Add either a row or column colorbar to a seaborn Grid."""

    def add(
        data: Series, cmap: str, bbox: List[List[int]], orientation: str
    ) -> None:
        ax = grid.fig.add_axes(matplotlib.transforms.Bbox(bbox))
        if is_numeric(data):
            if is_datetime(data):
                data = minmax_scale(data)
            norm = matplotlib.colors.Normalize(vmin=data.min(), vmax=data.max())
            cbar = matplotlib.colorbar.ColorbarBase(
                ax,
                cmap=plt.get_cmap(cmap),
                norm=norm,
                orientation=orientation,
                label=data.name,
            )
        else:
            res = to_numeric(data)
            # res /= res.max()
            cmap = get_categorical_cmap(res)
            # norm = matplotlib.colors.Normalize(vmin=res.min(), vmax=res.max())
            cbar = matplotlib.colorbar.ColorbarBase(
                ax, cmap=cmap, orientation=orientation, label=data.name,
            )
            cbar.set_ticks(res.drop_duplicates().sort_values() / res.max())
            cbar.set_ticklabels(data.value_counts().sort_index().index)

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
        _add_extra_colorbars_to_clustermap(
            grid, rows, location="row", cmaps=row_cmaps
        )
    if cols is not None:
        _add_extra_colorbars_to_clustermap(
            grid, cols, location="col", cmaps=col_cmaps
        )


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
                if isinstance(
                    kwargs[arg + "_colors"], (pd.DataFrame, pd.Series)
                ):
                    _kwargs[arg + "s"] = kwargs[arg + "_colors"]
                    kwargs[arg + "_colors"] = to_color_dataframe(
                        x=kwargs[arg + "_colors"],
                        cmaps=cmaps[arg],
                        offset=1 if arg == "row" else 0,
                    )
        grid = f(*args, **kwargs)
        _add_colorbars(
            grid, **_kwargs, row_cmaps=cmaps["row"], col_cmaps=cmaps["col"]
        )
        return grid

    return clustermap


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
    output_colors: Optional[List[str]] = None,
    return_colors: bool = False,
) -> Union[Array, Tuple[Array, List[Tuple[float, float, float]]]]:
    """
    Assumes [0, 1] float array.
    to is a tuple of 3 colors.
    """
    # defaults = list(matplotlib.colors.TABLEAU_COLORS.values())
    n_channels = arr.shape[0]
    if output_colors is None:
        target_colors = [
            matplotlib.colors.to_rgb(col)
            for col in DEFAULT_CHANNEL_COLORS[:n_channels]
        ]

    if (n_channels == 3) and output_colors is None:
        m = np.moveaxis(np.asarray([eq(x) for x in arr]), 0, -1)
        res = (m - m.min((0, 1))) / (m.max((0, 1)) - m.min((0, 1)))
        return res if not return_colors else (res, target_colors)

    elif isinstance(output_colors, (list, tuple)):
        assert len(output_colors) == n_channels
        target_colors = [matplotlib.colors.to_rgb(col) for col in output_colors]

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
            t = text.get_transform() + Affine2D().translate(ex.width, 0)
        else:
            t = text.get_transform() + Affine2D().translate(0, ex.height)


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
        a = pd.Series(n * np.arange(1, n + 1)) / dims
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
    # TODO: avoid rasterization of matplotlib.offsetbox.VPacker at least
    import warnings

    with warnings.catch_warnings(record=False) as w:
        warnings.simplefilter("always")
        clss = (
            matplotlib.text.Text,
            matplotlib.axis.XAxis,
            matplotlib.axis.YAxis,
        )
        for axs in fig.axes:
            for __c in axs.get_children():
                if not isinstance(__c, clss):
                    if not __c.get_children():
                        __c.set_rasterized(True)
                    for _cc in __c.get_children():
                        if not isinstance(__c, clss):
                            _cc.set_rasterized(True)


def add_transparency_to_boxenplot(ax: Axis) -> None:
    patches = (
        matplotlib.collections.PatchCollection,
        matplotlib.collections.PathCollection,
    )
    [x.set_alpha(0.25) for x in ax.get_children() if isinstance(x, patches)]


from typing import Literal


@overload
def swarmboxenplot(
    data: DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    swarm: bool = True,
    boxen: bool = True,
    ax: None = None,
    test: Literal[False] = False,
) -> Figure:
    ...


@overload
def swarmboxenplot(
    data: DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    swarm: bool = True,
    boxen: bool = True,
    ax: Axis = Axis,
    test: Literal[False] = False,
) -> None:
    ...


@overload
def swarmboxenplot(
    data: DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    swarm: bool = True,
    boxen: bool = True,
    ax: None = None,
    test: Literal[True] = True,
) -> Tuple[Figure, DataFrame]:
    ...


@overload
def swarmboxenplot(
    data: DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    swarm: bool = True,
    boxen: bool = True,
    ax: Axis = Axis,
    test: Literal[True] = True,
) -> DataFrame:
    ...


def swarmboxenplot(
    data: DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    swarm: bool = True,
    boxen: bool = True,
    ax: Optional[Axis] = None,
    test: bool = True,
    multiple_testing: Union[bool, str] = "fdr_bh",
    test_upper_threshold: float = 0.05,
    test_lower_threshold: float = 0.01,
    test_kws: Optional[Dict[str, Any]] = None,
) -> Optional[Union[Figure, DataFrame, Tuple[Figure, DataFrame]]]:
    """
    # Testing:

    data = pd.DataFrame(
        [np.random.random(20), np.random.choice(['a', 'b'], 20)],
        index=['cont', 'cat']).T.convert_dtypes()
    data.loc[data['cat'] == 'b', 'cont'] *= 5
    fig = swarmboxenplot(data=data, x='cat', y='cont')


    data = pd.DataFrame(
        [np.random.random(40), np.random.choice(['a', 'b', 'c'], 40)],
        index=['cont', 'cat']).T.convert_dtypes()
    data.loc[data['cat'] == 'b', 'cont'] *= 5
    data.loc[data['cat'] == 'c', 'cont'] -= 5
    fig = swarmboxenplot(data=data, x='cat', y='cont', test_kws=dict(parametric=True))

    """
    import pingouin as pg
    import itertools

    if test_kws is None:
        test_kws = dict()

    if ax is None:
        fig, _ax = plt.subplots(1, 1, figsize=(4, 4))
    else:
        _ax = ax
    if boxen:
        sns.boxenplot(data=data, x=x, y=y, hue=hue, ax=_ax)
    if boxen and swarm:
        add_transparency_to_boxenplot(_ax)
    if swarm:
        sns.swarmplot(data=data, x=x, y=y, hue=hue, ax=_ax)
    _ax.set_xticklabels(_ax.get_xticklabels(), rotation=90)

    if test:
        # remove NaNs
        data = data.dropna(subset=[x, y])
        # remove categories with only one element
        keep = data.groupby(x).size()[data.groupby(x).size() > 1].index
        data = data.loc[data[x].isin(keep), :]
        if data[x].dtype.name == "category":
            data[x] = data[x].cat.remove_unused_categories()
        ylim = _ax.get_ylim()
        ylength = abs(ylim[1]) + abs(ylim[0])
        stat = pd.DataFrame(
            itertools.combinations(data[x].unique(), 2), columns=["A", "B"]
        )
        stat["p-unc"] = np.nan
        try:
            stat = pg.pairwise_ttests(data=data, dv=y, between=x, **test_kws)
        except (AssertionError, ValueError) as e:
            print(str(e))
        except KeyError:
            print("Only one category with values!")
        if multiple_testing is not False:
            stat["p-cor"] = pg.multicomp(
                stat["p-unc"].values, method=multiple_testing
            )[1]
            pcol = "p-cor"
        else:
            pcol = "p-unc"
        for i, (idx, row) in enumerate(
            stat.loc[stat[pcol] <= test_upper_threshold, :].iterrows()
        ):
            symbol = "**" if row[pcol] <= test_lower_threshold else "*"
            # py = data[y].quantile(0.95) - (i * (ylength / 20))
            py = data[y].max() - (i * (ylength / 20))
            _ax.plot(
                (row["A"], row["B"]), (py, py), color="black", linewidth=1.2
            )
            _ax.text(row["B"], py, s=symbol, color="black")
        _ax.set_ylim(ylim)
        return (fig, stat) if ax is None else stat
    return fig if ax is None else None
