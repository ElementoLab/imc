"""
Plotting functions and utilities to handle images.
"""

from __future__ import annotations
import typing as tp
from functools import wraps
import colorsys
from functools import partial

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm

from skimage.exposure import equalize_hist as eq

from imc.types import Figure, Axis, Array, Series, ColorMap, Patch, AnnData, Path
from imc.utils import minmax_scale

DEFAULT_PIXEL_UNIT_NAME = r"$\mu$m"

DEFAULT_CHANNEL_COLORS = plt.get_cmap("tab10")(np.linspace(0, 1, 10))


class InteractiveViewer:
    """
    An interactive image viewer for multiplexed images.

    Parameters
    ----------
    obj: ROI | Array
        An ROI object or a numpy array

    **kwargs: dict
        Additional keyword arguments to pass to matplotlib.pyplot.imshow.
    """

    def __init__(
        self,
        obj: tp.Union[_roi.ROI, Array],
        show: bool = False,
        up_key: str = "w",
        down_key: str = "s",
        log_key: str = "l",
        **kwargs,
    ):
        plt.close("all")
        self.array = obj if isinstance(obj, np.ndarray) else obj.stack
        self.labels = (
            ([""] * len(self.array))
            if isinstance(obj, np.ndarray)
            else obj.channel_labels.tolist()
        )
        self.suptitle = "" if isinstance(obj, np.ndarray) else obj.name
        self.up_key = up_key
        self.down_key = down_key
        self.log_key = log_key
        self.kwargs = kwargs

        # internal
        self.index = 0
        self.n_channels = self.array.shape[0]
        self.transforms: tp.Set[str] = set()
        self.fig, self.ax = plt.subplots(num=self.suptitle)

        # go
        self.multi_slice_viewer()
        if show:
            plt.show(block=False)
        # plt.close(self.fig)

    def multi_slice_viewer(self) -> Figure:
        """Start the viewer process."""
        self.remove_keymap_conflicts({self.up_key, self.down_key, self.log_key})

        self.ax.imshow(self.array[self.index], **self.kwargs)
        self.img = self.ax.images[0]
        self.ax.set_title(self.index, loc="left")
        if self.labels is not None:
            self.ax.set_title(self.labels[self.index])
        if self.suptitle is not None:
            self.fig.suptitle(self.suptitle)
        self.ax.set(xlabel="X", ylabel="Y")
        # TODO: add colorbar scale

        # Add event listener
        self.fig.canvas.mpl_connect("key_press_event", partial(self.process_key))

    def remove_keymap_conflicts(self, new_keys_set: tp.Set) -> None:
        """Remove conflicts between viewer keyboard shortcuts and previously existing shortcuts."""
        for prop in plt.rcParams:
            if prop.startswith("keymap."):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def process_key(self, event) -> None:
        """Process keyboard events."""
        if event.key == self.up_key:
            self.previous_slice()
        elif event.key == self.down_key:
            self.next_slice()
        elif event.key == self.log_key:
            self.log_slice()

        self.ax.set_title(self.index, loc="left")
        if self.labels is not None:
            self.ax.set_title(self.labels[self.index])

        # Report transformations
        if self.transforms:
            trans = ", ".join(self.transforms)
            self.ax.set_xlabel("X" + f"\nTransformations: '{trans}'")
        else:
            self.ax.set_xlabel("X")

        # Draw
        self.fig.canvas.draw()

    def get_slice(self) -> Array:
        """Get a array slice for the current index with current transformations."""
        a = self.array[self.index]
        if "log" in self.transforms:
            a = np.log1p(a)
        return a

    def set_image(self) -> None:
        """Update image to current index and transformations."""
        a = self.get_slice()
        self.img.set_array(a)
        self.img.set_clim(a.min(), a.max())

    def previous_slice(self) -> None:
        """Go to the previous slice."""
        self.index = (self.index - 1) % self.n_channels
        self.set_image()

    def next_slice(self) -> None:
        """Go to the next slice."""
        self.index = (self.index + 1) % self.n_channels
        self.set_image()

    def log_slice(self) -> None:
        """Go to the previous slice."""
        if "log" not in self.transforms:
            self.transforms.add("log")
        else:
            self.transforms.remove("log")
        self.set_image()


def get_volume() -> Array:
    """Get example volumetric image."""
    from urlpath import URL
    import imageio

    base_url = URL("https://prod-images-static.radiopaedia.org/images/")
    start_n = 53734044
    length = 137

    imgs = list()
    for i in tqdm(range(length)):
        url = base_url / f"{start_n + i}/{i + 1}_gallery.jpeg"
        resp = url.get()
        c = resp.content
        imgs.append(imageio.read(c, format="jpeg").get_data(0))
    img = np.asarray(imgs)
    return img


def close_plots(func) -> tp.Callable:
    """
    Decorator to close all plots on function exit.
    """

    @wraps(func)
    def close(*args, **kwargs) -> None:
        func(*args, **kwargs)
        plt.close("all")

    return close


def add_scale(
    _ax: tp.Optional[Axis] = None,
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


def add_minmax(minmax: tp.Tuple[float, float], _ax: tp.Optional[Axis] = None) -> None:
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
    patches: tp.Sequence[Patch], ax: tp.Optional[Axis] = None, **kwargs
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
    target_colors: tp.Optional[tp.Sequence[tp.Tuple[float, float, float]]] = None,
    return_colors: bool = False,
) -> tp.Union[Array, tp.Tuple[Array, tp.Sequence[tp.Tuple[float, float, float]]]]:
    """
    Assumes [0, 1] float array.
    to is a tuple of 3 colors.
    """
    # defaults = list(matplotlib.colors.TABLEAU_COLORS.values())
    n_channels = arr.shape[0]
    if target_colors is None:
        target_colors = [
            matplotlib.colors.to_rgb(col) for col in DEFAULT_CHANNEL_COLORS[:n_channels]
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


def rainbow_text(x, y, strings, colors, orientation="horizontal", ax=None, **kwargs):
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
    ax : Axes, tp.optional
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


def get_n_colors(n: int, max_value: float = 1.0) -> Array:
    """
    With modifications from https://stackoverflow.com/a/13781114/1469535
    """
    import itertools
    from fractions import Fraction

    def zenos_dichotomy():
        """
        http://en.wikipedia.org/wiki/1/2_%2B_1/4_%2B_1/8_%2B_1/16_%2B_%C2%B7_%C2%B7_%C2%B7
        """
        for k in itertools.count():
            yield Fraction(1, 2 ** k)

    def fracs():
        """
        [Fraction(0, 1), Fraction(1, 2), Fraction(1, 4), Fraction(3, 4), Fraction(1, 8), Fraction(3, 8), Fraction(5, 8), Fraction(7, 8), Fraction(1, 16), Fraction(3, 16), ...]
        [0.0, 0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875, 0.0625, 0.1875, ...]
        """
        yield Fraction(0)
        for k in zenos_dichotomy():
            i = k.denominator  # [1,2,4,8,16,...]
            for j in range(1, i, 2):
                yield Fraction(j, i)

    # can be used for the v in hsv to map linear values 0..1 to something that looks equidistant
    # bias = lambda x: (math.sqrt(x/3)/Fraction(2,3)+Fraction(1,3))/Fraction(6,5)
    def hue_to_tones(h):
        for s in [Fraction(6, 10)]:  # optionally use range
            for v in [Fraction(8, 10), Fraction(5, 10)]:  # could use range too
                yield (h, s, v)  # use bias for v here if you use range

    def hsv_to_rgb(x):
        return colorsys.hsv_to_rgb(*map(float, x))

    flatten = itertools.chain.from_iterable

    def hsvs():
        return flatten(map(hue_to_tones, fracs()))

    def rgbs():
        return map(hsv_to_rgb, hsvs())

    return np.asarray(list(itertools.islice(rgbs(), n))) * max_value


def get_rgb_cmaps() -> tp.Tuple[ColorMap, ColorMap, ColorMap]:
    r = np.linspace(0, 1, 100).reshape((-1, 1))
    r = [
        matplotlib.colors.LinearSegmentedColormap.from_list("", p * r) for p in np.eye(3)
    ]
    return tuple(r)  # type: ignore


def get_dark_cmaps(n: int = 3, from_palette: str = "colorblind") -> tp.List[ColorMap]:
    r = np.linspace(0, 1, 100).reshape((-1, 1))
    if n > len(sns.color_palette(from_palette)):
        print(
            "Chosen palette has less than the requested number of colors. " "Will reuse!"
        )
    return [
        matplotlib.colors.LinearSegmentedColormap.from_list("", np.array(p) * r)
        for p in sns.color_palette(from_palette, n)
    ]


def get_transparent_cmaps(
    n: int = 3, from_palette: tp.Optional[str] = "colorblind"
) -> tp.List[ColorMap]:
    __r = np.linspace(0, 1, 100)
    if n > len(sns.color_palette(from_palette)):
        print(
            "Chosen palette has less than the requested number of colors. " "Will reuse!"
        )
    return [
        matplotlib.colors.LinearSegmentedColormap.from_list("", [p + (c,) for c in __r])
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
def cell_labels_to_mask(mask: Array, labels: tp.Union[Series, tp.Dict]) -> Array:
    """Replaces integers in `mask` with values from the mapping in `labels`."""
    res = np.zeros(mask.shape, dtype=int)
    for k, v in labels.items():
        res[mask == k] = v
    return res


def values_to_rgb_colors(
    mask: Array, from_palette: str = None, remove_zero: bool = True
) -> tp.Tuple[Array, tp.Dict[tp.Any, tp.Tuple[float, float, float]]]:
    """
    Colors each integer in the 2D `mask` array with a unique color by
    expanding the array to 3 dimensions.
    Also returns the mapping of mask identity to color tuple.
    """
    ident = np.sort(np.unique(mask))
    if remove_zero:
        ident = ident[ident != 0]
    n_colors = len(ident)

    if from_palette is not None:
        palette = sns.color_palette(from_palette)
    else:
        palette = list(get_n_colors(n_colors))

    if n_colors > len(palette):
        print(
            "Chosen palette has less than the requested number of colors. " "Will reuse!"
        )
        palette = sns.color_palette(from_palette, ident.max())

    colors = pd.Series(palette, index=ident)
    res = np.zeros((mask.shape) + (3,))
    for c, i in zip(colors, ident):
        x, y = np.nonzero(np.isin(mask, i))
        res[x, y, :] = c
    return res, colors.to_dict()


@tp.overload
def get_grid_dims(
    dims: tp.Union[int, tp.Collection],
    return_fig: tp.Literal[True],
    nstart: tp.Optional[int],
) -> Figure:
    ...


@tp.overload
def get_grid_dims(
    dims: tp.Union[int, tp.Collection],
    return_fig: tp.Literal[False],
    nstart: tp.Optional[int],
) -> tp.Tuple[int, int]:
    ...


def get_grid_dims(
    dims: tp.Union[int, tp.Collection],
    return_fig: bool = False,
    nstart: tp.Optional[int] = None,
    **kwargs,
) -> tp.Union[tp.Tuple[int, int], Figure]:
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
    arr: Array, axis: tp.Optional[Axis] = None, cmap: tp.Optional[ColorMap] = None
) -> tp.Union[Figure, Axis]:
    """Plot a single image channel either in a new figure or in an existing axis"""
    if axis is None:
        fig, axs = plt.subplots(1, 1, figsize=(6 * 1, 6 * 1), sharex=True, sharey=True)
    axs.imshow(arr, cmap=cmap, interpolation="bilinear", rasterized=True)
    axs.axis("off")
    return fig if axis is None else axs


def plot_overlayied_channels(
    arr: Array,
    channel_labels: tp.Sequence[str],
    axis: tp.Optional[Axis] = None,
    palette: tp.Optional[str] = None,
) -> tp.Union[Figure, Axis]:
    if axis is None:
        fig, ax = plt.subplots(1, 1, figsize=(6 * 1, 6 * 1), sharex=True, sharey=True)
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
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
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


def add_centroids(
    a: AnnData,
    ax: tp.Union[tp.Sequence[Axis], Axis] = None,
    res: float = None,
    column: str = None,
    algo: str = "umap",
):
    """
    a: AnnData
    ax: matplotlib.Axes.axes
    res: resolution of clusters to label
    column: Column to be used. Has precedence over `res`.
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


def legend_without_duplicate_labels(ax: Axis, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    ax.legend(*zip(*unique), **kwargs)


import imc.data_models.roi as _roi
