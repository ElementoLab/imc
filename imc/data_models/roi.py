#! /usr/bin/env python

"""
A class to model a imaging mass cytometry acquired region of interest (ROI).
"""

import re
from typing import (
    Dict,
    Tuple,
    List,
    Sequence,
    Optional,
    Union,
    Any,
    overload,
)  # , cast

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import scipy  # type: ignore
import scipy.ndimage as ndi  # type: ignore

import matplotlib.pyplot as plt  # type: ignore
import matplotlib.patches as mpatches  # type: ignore
import seaborn as sns  # type: ignore

import networkx as nx  # type: ignore
from skimage.exposure import equalize_hist as eq  # type: ignore
from skimage.segmentation import clear_border  # type: ignore

from imc.types import Path, Figure, Axis, Patch, Array, DataFrame, Series

# from imc import LOGGER
from imc.operations import quantify_cell_intensity, quantify_cell_morphology
from imc.utils import read_image_from_file, sorted_nicely, minmax_scale
from imc.graphics import (
    add_scale as _add_scale,
    add_minmax as _add_minmax,
    get_grid_dims,
    get_transparent_cmaps,
    add_legend as _add_legend,
    cell_labels_to_mask,
    numbers_to_rgb_colors,
    merge_channels,
    rainbow_text,
)

from imc.exceptions import (
    cast,
    AttributeNotSetError,
)  # TODO: replace with typing.cast


FIG_KWS = dict(dpi=300, bbox_inches="tight")

# processed directory structure
SUBFOLDERS_PER_SAMPLE = True
DEFAULT_ROI_NAME = "roi"
ROI_STACKS_DIR = Path("tiffs")
ROI_MASKS_DIR = Path("tiffs")
ROI_UNCERTAINTY_DIR = Path("uncertainty")
ROI_SINGLE_CELL_DIR = Path("single_cell")


class ROI:
    """
    A class to model a region of interest in an IMC experiment.
    """

    # roi_name: str
    # roi_number: int
    # name: str  # roi_name shorthand
    # sample: IMCSample  # from parent

    # paths

    # data attributes
    # channel_labels: Series
    # channel_number: int

    # stack: Array
    # features: Array
    # probabilities: Array
    # uncertainty: Array
    # nuclei_mask: Array
    # cell_mask: Array

    # clusters: Series  # Index: 'obj_id'

    # shape: Tuple[int, int]

    file_types = [
        # "stack",
        # "features",
        "probabilities",
        "uncertainty",
        "nuclei_mask",
        "cell_mask",
    ]

    def __init__(
        self,
        name: str = DEFAULT_ROI_NAME,
        roi_number: Optional[int] = None,
        channel_labels: Optional[Union[Path, Series]] = None,
        root_dir: Optional[Path] = None,
        stacks_dir: Optional[
            Path
        ] = ROI_STACKS_DIR,  # TODO: make these relative to the root_dir
        masks_dir: Optional[Path] = ROI_MASKS_DIR,
        single_cell_dir: Optional[Path] = ROI_SINGLE_CELL_DIR,
        sample: Optional["IMCSample"] = None,
        **kwargs,
    ):
        # attributes
        self.name = name
        self.roi_name = name
        self.roi_number = roi_number
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.stacks_dir = stacks_dir
        self.masks_dir = masks_dir
        self.single_cell_dir = single_cell_dir
        self.channel_labels_file: Optional[Path] = (
            Path(channel_labels)
            if isinstance(channel_labels, (str, Path))
            else None
        )
        # TODO: make sure channel labels conform to internal specification: "Label(Metal\d+)"
        self._channel_labels: Optional[Series] = (
            pd.read_csv(channel_labels, index_col=0, squeeze=True)
            if isinstance(channel_labels, (str, Path))
            else channel_labels
        )
        self._channel_include = None
        self._channel_exclude = None
        # obj connections
        self.sample = sample
        self.prj: Optional["Project"] = None
        # data
        self._stack: Optional[Array] = None
        self._shape: Optional[Tuple] = None
        self._area: Optional[int] = None
        self._channel_number: Optional[int] = None
        self._probabilities: Optional[Array] = None
        self._nuclei_mask: Optional[Array] = None
        self._cell_mask_o: Optional[Array] = None
        self._cell_mask: Optional[Array] = None
        self._adjacency_graph = None
        self._clusters: Optional[Series] = None

        # Add kwargs as attributes
        self.__dict__.update(kwargs)

    def __repr__(self):
        return (
            "Region"
            + (f" {self.roi_number}" if self.roi_number is not None else "")
            + (
                f" of sample '{self.sample.name}'"
                if self.sample is not None
                else ""
            )
        )

    @property
    def channel_labels(self) -> Series:
        """Return a Series with a string for each channel in the ROIs stack."""
        if self._channel_labels is not None:
            return self._channel_labels
        sample = cast(self.sample)

        # read channel labels specifically for ROI
        channel_labels_file = Path(
            self.channel_labels_file
            or sample.root_dir / self.stacks_dir / (self.name + "_full.csv")
        )
        if not channel_labels_file.exists():
            msg = (
                "`channel_labels` was not given upon initialization "
                f"and '{channel_labels_file}' could not be found!"
            )
            raise FileNotFoundError(msg)
        # self._channel_labels = pd.read_csv(channel_labels_file, header=None, squeeze=True)
        preview = pd.read_csv(channel_labels_file, header=None, squeeze=True)
        if isinstance(preview, pd.Series):
            order = preview.to_frame(name="ChannelName").set_index(
                "ChannelName"
            )
            # read reference
            ref: DataFrame = cast(sample.panel_metadata)
            ref = ref.loc[
                ref["AcquisitionID"].isin(
                    [self.roi_number, str(self.roi_number)]
                )
            ]
            self._channel_labels = (
                order.join(ref.reset_index().set_index("ChannelName"))["index"]
                .reset_index(drop=True)
                .rename("channel")
            )
        else:
            preview = preview.dropna().set_index(0).squeeze().rename("channel")
            preview.index = preview.index.astype(int)
            self._channel_labels = preview
        return self._channel_labels

    @property
    def channel_exclude(self) -> Series:
        if self._channel_exclude is not None:
            return self._channel_exclude
        if self.channel_labels is not None:
            self._channel_exclude = pd.Series(index=self.channel_labels).fillna(
                False
            )
            return self._channel_exclude

    def set_channel_exclude(self, values: Union[List[str], Series]):
        self._channel_exclude = self.channel_labels.isin(values).set_axis(
            self.channel_labels
        )

    @property
    def channel_names(self) -> Series:
        return self.channel_labels.str.extract(r"^(.*)\(")[0]

    @property
    def channel_metals(self) -> Series:
        return self.channel_labels.str.extract(r"^.*\((.*)\)$")[0]

    @property
    def stack(self) -> Array:
        """An ndarray representing the image channel stack."""
        if self._stack is not None:
            return self._stack

        # read from file and return without storing as attribute
        mtx: Array = self.read_input(
            "stack", permissive=False, set_attribute=False
        )
        self._shape = mtx.shape
        self._channel_number = mtx.shape[0]
        return mtx

    @property
    def xstack(self):
        import xarray

        return xarray.DataArray(
            self.stack,
            name=self.name,
            dims=["channel", "X", "Y"],
            coords={"channel": self.channel_labels.values},
        )
        # .to_netcdf(file_name + ".nc", engine='h5netcdf')

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the image stack."""
        if self._shape is not None:
            return self._shape
        try:
            self._shape = (np.nan,) + self.mask.shape
        except AttributeNotSetError:
            try:
                self._shape = self.stack.shape
            except AttributeNotSetError:
                raise AttributeNotSetError(
                    "ROI does not have either stack or mask!"
                )
        return self._shape

    @property
    def area(self) -> int:
        """An array with unique integers for each cell."""
        if self._area is not None:
            return self._area
        self._area = self._get_area()
        return self._area

    @property
    def channel_number(self) -> int:
        """The number of channels in the image stack."""
        if self._channel_number is not None:
            return self._channel_number
        self._channel_number = int(self.stack.shape[0])
        return self._channel_number

    @property
    def probabilities(self) -> Array:
        if self._probabilities is not None:
            return self._probabilities
        res: Array = self.read_input("probabilities", set_attribute=False)
        return res[:3, :, :]  # return only first 3 labels

    @property
    def nuclei_mask(self) -> Array:
        """An array with unique integers for each cell."""
        if self._nuclei_mask is not None:
            return self._nuclei_mask
        return clear_border(self.read_input("nuclei_mask", set_attribute=False))

    @property
    def cell_mask_o(self) -> Array:
        """An array with unique integers for each cell."""
        if self._cell_mask_o is not None:
            return self._cell_mask_o
        self._cell_mask_o = self.read_input("cell_mask", set_attribute=False)
        return self._cell_mask_o

    @property
    def cell_mask(self) -> Array:
        """An array with unique integers for each cell."""
        if self._cell_mask is not None:
            return self._cell_mask
        self._cell_mask = clear_border(self.cell_mask_o)
        return self._cell_mask

    @property
    def mask(self) -> Array:
        return self.cell_mask

    @property
    def clusters(self):
        if self._clusters is not None:
            return self._clusters
        try:
            self.set_clusters()
        except KeyError:
            return None
        return self._clusters

    @property
    def adjacency_graph(self) -> nx.Graph:
        if self._adjacency_graph is not None:
            return self._adjacency_graph
        try:
            self._adjacency_graph = nx.readwrite.read_gpickle(
                self._get_input_filename("adjacency_graph")
            )
        except FileNotFoundError:
            return None
        return self._adjacency_graph

    def _get_area(self) -> int:
        """Get area of ROI"""
        return np.multiply(*self.shape[1:])  # type: ignore[no-any-return]

    def _get_input_filename(self, input_type: str) -> Path:
        """Get path to file with data for ROI.

        Available `input_type` values are:
            - "stack":
            - "features": Features extracted by ilastik (usually not available by default)
            - "probabilities": 3 color probability intensities predicted by ilastik
            - "uncertainty": TIFF file with uncertainty of pixel classification
            - "nuclei_mask": TIFF file with mask for nuclei
            - "cell_mask": TIFF file with mask for cells
            - "cell_type_assignments": CSV file with cell type assignemts for each cell
        """
        sample = cast(self.sample)
        to_read = {
            "stack": (self.stacks_dir, "_full.tiff"),
            # "features": (self.stacks_dir, "_ilastik_s2_Features.h5"),
            "probabilities": (self.stacks_dir, "_Probabilities.tiff"),
            # "uncertainty": (ROI_UNCERTAINTY_DIR, "_ilastik_s2_Probabilities_uncertainty.tiff"),
            # "cell_mask": (self.masks_dir, "_ilastik_s2_Probabilities_mask.tiff"),
            "cell_mask": (self.masks_dir, "_full_mask.tiff"),
            "nuclei_mask": (self.masks_dir, "_full_nucmask.tiff"),
            "cell_type_assignments": (
                self.single_cell_dir,
                ".cell_type_assignment_against_reference.csv",
            ),
            "adjacency_graph": (
                self.single_cell_dir,
                ".neighbor_graph.gpickle",
            ),
        }
        dir_, suffix = to_read[input_type]
        return cast(sample.root_dir) / cast(dir_) / (self.name + suffix)

    def get(self, attr):
        try:
            return self.__getattribute(attr)
        except AttributeError:
            return None

    def read_input(
        self,
        key: str,
        permissive: bool = False,
        set_attribute: bool = True,
        overwrite: bool = False,
        parameters: Optional[Dict] = None,
    ) -> Optional[Array]:
        """Reads in all sample-wise inputs:
            - raw stack
            - extracted features
            - probabilities
            - uncertainty
            - segmentation mask.
        If `permissive` is :obj:`True`, skips non-existing inputs."""
        if parameters is None:
            parameters = {}
        if set_attribute and not overwrite and hasattr(self, key):
            return None
        try:
            value = read_image_from_file(
                self._get_input_filename(key), **parameters
            )
        except FileNotFoundError:
            if permissive:
                return None
            raise
        if set_attribute:
            # TODO: fix assignment to @property
            setattr(self, key, value)
            return None
        if key in ["cell_mask", "nuclei_mask"]:
            self._shape = (np.nan,) + value.shape
        elif key == "stack":
            self._shape = value.shape
        return value

    def read_all_inputs(
        self,
        only_these_keys: Optional[List[str]] = None,
        permissive: bool = False,
        set_attribute: bool = True,
        overwrite: bool = False,
        parameters: Optional[Union[Dict, List]] = None,
    ) -> Optional[Dict[str, Array]]:
        """Reads in all sample-wise inputs:
            - raw stack
            - extracted features
            - probabilities
            - uncertainty
            - segmentation mask.
        If `permissive` is :obj:`True`, skips non-existing inputs."""
        only_these_keys = cast(only_these_keys or self.file_types)

        if not set_attribute and len(only_these_keys) != 1:
            msg = "With `set_attribute` False, only one in `only_these_keys` can be seleted."
            raise ValueError(msg)

        if parameters is None:
            parameters = [{}] * len(only_these_keys)
        elif isinstance(parameters, dict):
            parameters = [parameters] * len(only_these_keys)
        elif isinstance(parameters, list):
            if len(parameters) != len(only_these_keys):
                raise ValueError(
                    "Length of parameter list must match number of inputs to be read."
                )

        res = dict()
        for ftype, params in zip(only_these_keys, parameters):
            res[ftype] = self.read_input(
                ftype,
                permissive=permissive,
                set_attribute=set_attribute,
                overwrite=overwrite,
                parameters=params,
            )
        return res if not set_attribute else None

    # def _get_channels_from_string(self, string: str) -> Series:
    #     match = self.channel_labels.str.contains(re.escape((string)))
    #     if match.any():
    #         return match
    #     msg = f"Could not find out channel '{string}' in `sample.channel_labels`."
    #     # # LOGGER.error(msg)
    #     raise ValueError(msg)

    def _get_channel(
        self,
        channel: Union[int, str],
        red_func: str = "mean",
        log: bool = False,
        equalize: bool = False,
        minmax: bool = False,
        dont_warn: bool = False,
    ) -> Tuple[str, Array, Tuple[float, float]]:
        """
        Get a 2D signal array from a channel name or number.
        If the channel name matches more than one channel, return reduction of channels.

        Parameters
        ----------
        channel : {Union[int, str]}
            An integer index of `channel_labels` or a string for its value.
            If the string does not match exactly the value, the channel that
            contains it would be retrieved.
            If more than one channel matches, then those are retrieved and the
            data is reduced by `red_func`.
            If the special values "mean" or "sum" are given, the respective
            reduction of all channels is retrieved.
            The above respects the boolean array attribute `channel_exclude`.
        red_func : {str}, optional
            An array function name to reduce the data in case more than one channel is matched.
        log : {bool}
            Whether to log-transform the channel. Default is `False`.
            If multiple, will be applied per channel.
        equalize : {bool}
            Whether to equalize the histogram of the channel. Default is `False`.
            If multiple, will be applied per channel.
        equalize : {bool}
            Whether to minmax-scale the channel. Default is `False`.
        dont_warn: {bool}
            Whether to not warn the user if multiple channels are found.
            Default is `False`.

        Returns
        -------
        Tuple[str, Array, Tuple[float, float]]
            A string describing the channel and the respective array,
            the actuall array, and a tuple with the min-max values of the array prior
            to any transformation if applicable.

        Raises
        ------
        ValueError
            If `channel` cannot be found in `sample.channel_labels`.
        """
        import numba as nb

        def reduce_channels(
            stack: Array, red_func: str, ex: Union[List[bool], Series] = None
        ):
            ex = [False] * stack.shape[0] if ex is None else ex
            m = np.asarray([g(f(x)) for i, x in enumerate(stack) if not ex[i]])
            return getattr(m, red_func)(axis=0)

        @nb.vectorize(target="cpu")
        def _idenv(x):
            return x

        excluded_message = (
            "Requested channel '{0}' only matches '{1}'"
            " but this is flagged as excluded in `roi.channel_exclude`."
            " Proceeding anyway with that channel."
        )

        stack = self.stack
        f = np.log1p if log else _idenv
        g = eq if equalize else _idenv
        h = minmax_scale if minmax else _idenv

        if isinstance(channel, int):
            label = self.channel_labels.iloc[channel]
            arr = stack[channel]
        elif isinstance(channel, str):
            if channel in ["sum", "mean"]:
                arr = reduce_channels(stack, channel, self.channel_exclude)
                label = f"Channel {channel}"
                f = g = _idenv
            elif sum(self.channel_labels.values == channel) == 1:
                label = channel
                if channel in excluded:
                    print(excluded_message.format(channel, label))
                arr = stack[self.channel_labels == channel]
            else:
                match = self.channel_labels.str.contains(re.escape((channel)))
                if match.any():
                    label = ", ".join(self.channel_labels[match])
                    if match.sum() == 1:
                        if self.channel_labels[match].squeeze() in excluded:
                            print(excluded_message.format(channel, label))
                        arr = stack[match]
                    else:
                        ex = (~match) | self.channel_exclude.values
                        label = ", ".join(self.channel_labels[~ex])
                        if ex.sum() == 1:
                            print(excluded_message.format(channel, label))
                            arr = stack[match]
                        else:
                            if not dont_warn:
                                msg = f"Could not find out channel '{channel}' in `roi.channel_labels` but could find '{label}'."
                                print(msg)
                            if ex.all():
                                print(excluded_message.format(channel, label))
                                ex = ~match
                            arr = reduce_channels(stack, "mean", ex)
                            f = g = _idenv
                else:
                    msg = f"Could not find out channel '{channel}' in `sample.channel_labels`."
                    # # LOGGER.error(msg)
                    raise ValueError(msg)
        vminmax = arr.min(), arr.max()
        arr = h(g(f((arr))))
        return label, arr, vminmax

    def _get_channels(
        self, channels: List[Union[int, str]], **kwargs
    ) -> Tuple[str, Array, Array]:
        """
        Convinience function to get signal from various channels.
        """
        labels = list()
        arrays = list()
        minmaxes = list()
        for channel in channels:
            lab, arr, minmax = self._get_channel(channel, **kwargs)
            labels.append(lab.replace(", ", "-"))
            arrays.append(arr.squeeze())
            minmaxes.append(minmax)
        return ", ".join(labels), np.asarray(arrays), np.asarray(minmaxes)

    def get_mean_all_channels(self) -> Array:
        """Get an array with mean of all channels"""
        return eq(self.stack.mean(axis=0))

    def plot_channel(
        self,
        channel: Union[int, str],
        ax: Optional[Axis] = None,
        equalize: bool = True,
        log: bool = True,
        add_scale: bool = True,
        add_range: bool = True,
        **kwargs,
    ) -> Axis:
        """
        Plot a single channel.

        Supports indexing of channels either by name or integer.
        Special strings for :class:`numpy.ndarray` functions can be passed to
        reduce values across channels (first axis). Pass e.g. 'mean' or 'sum'.

        Keyword arguments are passed to :func:`~matplotlib.pyplot.imshow`
        """
        _ax = ax
        channel, p, _minmax = self._get_channel(
            channel, log=log, equalize=equalize
        )

        if _ax is None:
            _, _ax = plt.subplots(1, 1, figsize=(4, 4))
        _ax.imshow(p.squeeze(), rasterized=True, **kwargs)
        if add_scale:
            _add_scale(_ax)
        if add_range:
            _add_minmax(_minmax, _ax)
        _ax.axis("off")
        _ax.set_title(f"{self.name}\n{channel}")
        return _ax

    def plot_channels(
        self,
        channels: Optional[List[str]] = None,
        merged: bool = False,
        axes: List[Axis] = None,
        equalize: bool = None,
        log: bool = True,
        minmax: bool = True,
        add_scale: bool = True,
        add_range: bool = True,
        share_axes: bool = True,
        **kwargs,
    ) -> Optional[Figure]:
        """If axes is given it must be length channels"""
        # TODO: optimize this by avoiding reading stack for every channel
        if channels is None:
            channels = self.channel_labels.index

        if axes is None:
            n, m = (1, 1) if merged else get_grid_dims(len(channels))
            fig, _axes = plt.subplots(
                n,
                m,
                figsize=(m * 4, n * 4),
                squeeze=False,
                sharex=share_axes,
                sharey=share_axes,
            )
            fig.suptitle(f"{self.sample}\n{self}")
            _axes = _axes.flatten()
        else:
            _axes = axes

        # i = 0  # in case merged or len(channels) is 0
        if merged:
            if equalize is None:
                equalize = True

            names, arr, minmaxes = self._get_channels(
                list(channels), log=log, equalize=equalize, minmax=minmax
            )
            arr2, colors = merge_channels(arr, return_colors=True, **kwargs)
            x, y, _ = arr2.shape
            _axes[0].imshow(arr2 / arr2.max())
            x = x * 0.05
            y = y * 0.05
            bbox = dict(
                boxstyle="round",
                ec=(0.3, 0.3, 0.3, 0.5),
                fc=(0.0, 0.0, 0.0, 0.5),
            )
            rainbow_text(
                x,
                y,
                names.split(","),
                colors,
                ax=_axes[0],
                fontsize=3,
                bbox=bbox,
            )
            if add_scale:
                _add_scale(_axes[0])
            # TODO: add minmaxes, perhaps to the channel labels?
            # if add_range:
            #     _add_minmax(minmax, _ax)
        else:
            if equalize is None:
                equalize = True
            for i, channel in enumerate(channels):
                self.plot_channel(
                    channel,
                    ax=_axes[i],
                    equalize=equalize,
                    log=log,
                    add_scale=add_scale,
                    add_range=add_range,
                    **kwargs,
                )
        for _ax in _axes:  # [i + 1 :]
            _ax.axis("off")
        return fig if axes is None else None

    def stack_to_probabilities(self):
        pass

    def get_cell_type_assignments(self) -> Series:
        sample = cast(self.sample)
        # read if not set
        cell_type_assignments = getattr(
            sample,
            "cell_type_assignments",
            cast(
                sample.read_all_inputs(
                    only_these_keys=["cell_type_assignments"],
                    set_attribute=False,
                )
            )["cell_type_assignments"],
        )
        if "roi" in cell_type_assignments:
            cell_type_assignments = cell_type_assignments.query(
                f"roi == {self.roi_number}"
            )
        return cell_type_assignments["cluster"]

    @overload
    def plot_cell_type(self, cluster, ax: None) -> Figure:
        ...

    @overload
    def plot_cell_type(self, cluster, ax: Axis) -> None:
        ...

    def plot_cell_type(
        self,
        cluster,
        ax: Optional[Axis] = None,
        cmap: Optional[str] = "gray",
        add_scale: bool = True,
    ) -> Figure:
        if ax is None:
            fig, _ax = plt.subplots(1, 1, figsize=(4, 4))
        else:
            _ax = ax
        m = self.clusters == cluster
        if m.sum() == 0:
            raise ValueError(f"Cound not find cluster '{cluster}'.")
        _ax.imshow(cell_labels_to_mask(self.cell_mask, m), cmap=cmap)
        _ax.set_title(cluster)
        _ax.axis("off")

        if add_scale:
            _add_scale(_ax)
        return fig if ax is None else None

    def plot_cell_types(
        self,
        cell_type_assignments: Series = None,
        cell_type_combinations: Optional[
            Union[str, List[Tuple[str, str]]]
        ] = None,
        ax: Union[Axis, List[Axis]] = None,
        palette: Optional[str] = "tab20",
        add_scale: bool = True,
        add_legend: bool = True,
    ) -> Union[Figure, List[Patch]]:
        """
        If ax is given it must match number of `cell_type_combinations`.
        """
        if cell_type_assignments is not None:
            clusters = cell_type_assignments
        else:
            clusters = self.clusters
        if isinstance(clusters.index, pd.MultiIndex):
            clusters = clusters.loc[self.name]
        clusters.index = clusters.index.astype(int)

        if clusters.dtype.name in ["object", "category"]:
            # Replace the strings with a number
            labels = pd.Series(sorted_nicely(np.unique(clusters.values)))
            ns = labels.str.extract(r"(\d+) - .*")[0]
            # If the clusters contain a numeric prefix, use that
            # that will help having consistent color across ROIs
            if not ns.isnull().any():
                ns = ns.astype(int)
                # ns.index = ns.values
                clusters = clusters.replace(dict(zip(labels, ns)))
                # clusters -= clusters.min()
            else:
                ns = labels.index.to_series()
                clusters = clusters.replace(
                    dict(zip(labels, np.arange(len(labels))))
                )
        else:
            labels = sorted(np.unique(clusters.values))
            ns = pd.Series(range(len(labels)))

        # simply plot all cell types jointly
        # TODO: fix use of cell_type_combinations
        if cell_type_combinations in [None, "all"]:
            combs = [tuple(sorted(clusters.unique()))]
        else:
            combs = cell_type_combinations  # type: ignore

        n = len(combs)
        if ax is None:
            m = 1
            fig, axes = plt.subplots(
                n,
                m,
                figsize=(3 * m, 3 * n),
                sharex="col",
                sharey="col",
                squeeze=False,
            )
        else:
            axes = ax
            if isinstance(ax, np.ndarray) and len(ax) != n:
                raise ValueError(
                    f"Given axes must be of length of cell_type_combinations ({n})."
                )

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])[:, np.newaxis]

        bckgd_cmap = get_transparent_cmaps(1, "binary")[0]
        patches = list()
        for i, _ in enumerate(combs):
            axes[i, 0].set_title(self.name)
            # plot channel mean for texture/context
            axes[i, 0].imshow(
                self.get_mean_all_channels() * 0.1, cmap=bckgd_cmap
            )
            # plot each of the cell types with different colors
            res = cell_labels_to_mask(self.cell_mask, clusters)
            rgb = numbers_to_rgb_colors(res, from_palette=cast(palette))
            axes[i, 0].imshow(rgb)
            colors = (
                pd.Series(sns.color_palette(palette, max(ns.values + 1)))
                .reindex(ns - ns.min())
                .values
            )
            patches += [
                mpatches.Patch(color=colors[j], label=l)
                for j, l in enumerate(labels)
            ]
            axes[i, 0].axis("off")
            if add_scale:
                _add_scale(axes[i, 0])

        if add_legend:
            _add_legend(patches, axes[-1, 0])
        return fig if ax is None else patches

    def get_distinct_marker_sets(
        self, n_groups: int = 4, group_size: int = 4, save_plot: bool = False
    ) -> Tuple[DataFrame, Dict[int, List[str]]]:
        """Use cross-channel correlation to pick `n` clusters of distinct channels to overlay"""
        xcorr = pd.DataFrame(
            np.corrcoef(self.stack.reshape((self.channel_number, -1))),
            index=self.channel_labels,
            columns=self.channel_labels,
        )
        np.fill_diagonal(xcorr.values, 0)

        grid = sns.clustermap(
            xcorr,
            cmap="RdBu_r",
            center=0,
            metric="correlation",
            cbar_kws=dict(label="Pearson correlation"),
        )
        grid.ax_col_dendrogram.set_title("Pairwise channel correlation")
        if save_plot:
            grid.savefig(
                cast(self.root_dir) / "channel_pairwise_correlation.svg",
                **FIG_KWS,
            )

        c = pd.Series(
            scipy.cluster.hierarchy.fcluster(
                grid.dendrogram_col.linkage, n_groups, criterion="maxclust"
            ),
            index=xcorr.index,
        )

        marker_sets: Dict[int, List[str]] = dict()
        for _sp in range(1, n_groups + 1):
            marker_sets[_sp] = list()
            for i in np.random.choice(np.unique(c), group_size, replace=True):
                marker_sets[_sp].append(
                    np.random.choice(c[c == i].index, 1, replace=True)[0]
                )
        return (xcorr, marker_sets)

    def plot_overlayied_channels_subplots(self, n_groups: int) -> Figure:
        """
        Plot all channels of ROI in `n_groups` combinations, where each combination
        has as little overlap as possible.
        """
        stack = self.stack

        _, marker_sets = self.get_distinct_marker_sets(
            n_groups=n_groups,
            group_size=int(np.floor(self.channel_number / n_groups)),
        )

        n, m = get_grid_dims(n_groups)
        fig, axis = plt.subplots(
            n,
            m,
            figsize=(6 * m, 6 * n),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axis = axis.flatten()
        for i, (marker_set, mrks) in enumerate(marker_sets.items()):
            patches = list()
            cmaps = get_transparent_cmaps(len(mrks))
            for _, (_l, c) in enumerate(zip(mrks, cmaps)):
                x = stack[self.channel_labels == _l, :, :].squeeze()
                v = x.mean() + x.std() * 2
                axis[i].imshow(
                    x,
                    cmap=c,
                    vmin=0,
                    vmax=v,
                    label=_l,
                    interpolation="bilinear",
                    rasterized=True,
                )
                axis[i].axis("off")
                patches.append(mpatches.Patch(color=c(256), label=m))
            axis[i].legend(
                handles=patches,
                bbox_to_anchor=(1.05, 1),
                loc=2,
                borderaxespad=0.0,
                title=marker_set,
            )
        return fig

    def plot_probabilities_and_segmentation(
        self, axes: Optional[Sequence[Axis]] = None, add_scale: bool = True
    ) -> Optional[Figure]:
        """
        Visualize channel mean, DNA channel, segmentation probabilities
        and the segmented nuclei and cells.

        If `axes` is given it must have length 5
        """
        probabilities = self.probabilities
        if probabilities.shape != self.cell_mask.shape:
            probabilities = ndi.zoom(self.probabilities, (1, 0.5, 0.5))
        probabilities = np.moveaxis(probabilities, 0, -1)
        probabilities = probabilities / probabilities.max()

        dna_label, dna, minmax = self._get_channel("DNA", dont_warn=True)

        nuclei = self._get_input_filename("nuclei_mask").exists()
        ncols = 5 if nuclei else 4

        if axes is None:
            fig, _axes = plt.subplots(
                1,
                ncols,
                figsize=(ncols * 4, 4),
                gridspec_kw=dict(wspace=0.05),
                sharex=True,
                sharey=True,
            )
        else:
            _axes = axes
        _axes[0].set_ylabel(self.name)
        _axes[0].set_title("Channel mean")
        _axes[0].imshow(self.get_mean_all_channels())
        _axes[1].set_title(dna_label)
        _axes[1].imshow(eq(dna))
        _axes[2].set_title("Probabilities")
        _axes[2].imshow(probabilities)
        i = 0
        if nuclei:
            _axes[3 + i].set_title("Nuclei")
            _axes[3 + i].imshow(self.nuclei_mask > 0, cmap="binary")
            i += 1
        _axes[3 + i].set_title("Cells")
        _axes[3 + i].imshow(self.cell_mask > 0, cmap="binary")
        if add_scale:
            _add_scale(_axes[3 + i])
        # To plot jointly
        # _axes[5].imshow(probabilities)
        # _axes[5].contour(self.cell_mask, cmap="Blues")
        # _axes[5].contour(self.nuclei_mask, cmap="Reds")
        for _ax in _axes:
            _ax.axis("off")

        return fig if axes is None else None

    def quantify_cell_intensity(
        self,
        channel_include: List[str] = None,
        channel_exclude: List[str] = None,
        **kwargs,
    ) -> DataFrame:
        """Quantify intensity of each cell in each channel."""
        if channel_include is not None:
            kwargs["channel_include"] = self.channel_labels.str.contains(
                channel_include
            ).values
        if channel_exclude is not None:
            kwargs["channel_exclude"] = self.channel_labels.str.contains(
                channel_exclude
            ).values
        return quantify_cell_intensity(
            self._get_input_filename("stack"),
            self._get_input_filename("cell_mask"),
            **kwargs,
        ).rename(columns=self.channel_labels)

    def quantify_cell_morphology(self, **kwargs) -> DataFrame:
        """QUantify shape attributes of each cell."""
        return quantify_cell_morphology(
            self._get_input_filename("cell_mask"), **kwargs
        )

    def set_clusters(self, clusters: Optional[Series] = None) -> None:
        if clusters is None:
            self.sample.set_clusters(clusters=clusters, rois=[self])
        else:
            assert not isinstance(clusters.index, pd.MultiIndex)
            assert clusters.index.name == "obj_id"
            assert clusters.name == "cluster"
            self._clusters = clusters

    def cells_per_area_unit(self) -> float:
        """Get cell density in ROI."""
        cells = np.unique(self.mask) - 1
        return len(cells) / self.area
