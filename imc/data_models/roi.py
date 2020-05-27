#! /usr/bin/env python

"""
A class to model a imaging mass cytometry acquired region of interest (ROI).
"""

from __future__ import annotations  # fix the type annotatiton of not yet undefined classes
import re

from typing import Dict, Tuple, List, Sequence, Optional, Union, Any  # , cast

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
from imc.operations import quantify_cells, measure_cell_attributes
from imc.utils import read_image_from_file, sorted_nicely, minmax_scale
from imc.graphics import (
    add_scale as _add_scale,
    get_grid_dims,
    get_transparent_cmaps,
    add_legend,
    cell_labels_to_mask,
    numbers_to_rgb_colors,
)
from imc.exceptions import cast  # TODO: replace with typing.cast


FIG_KWS = dict(dpi=300, bbox_inches="tight")

# processed directory structure
SUBFOLDERS_PER_SAMPLE = True
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
        name: str,
        roi_number: int,
        channel_labels: Optional[Union[Path, Series]] = None,
        root_dir: Optional[Path] = None,
        stacks_dir: Optional[Path] = ROI_STACKS_DIR,
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
            Path(channel_labels) if isinstance(channel_labels, (str, Path)) else None
        )
        # TODO: make sure channel labels conform to internal specification: "Label(Metal\d+)"
        self._channel_labels: Optional[Series] = (
            pd.read_csv(channel_labels, index_col=0, squeeze=True)
            if isinstance(channel_labels, (str, Path))
            else channel_labels
        )
        # obj connections
        self.sample = sample
        self.prj: Optional["Project"] = None
        # data
        self._stack: Optional[Array] = None
        self._shape: Optional[Tuple] = None
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
        return f"Region '{self.roi_number}'" + (
            f" of sample '{self.sample.name}'" if self.sample is not None else ""
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
            order = preview.to_frame(name="ChannelName").set_index("ChannelName")
            # read reference
            ref: DataFrame = cast(sample.panel_metadata)
            ref = ref.loc[ref["AcquisitionID"].isin([self.roi_number, str(self.roi_number)])]
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
    def stack(self) -> Array:
        """An ndarray representing the image channel stack."""
        if self._stack is not None:
            return self._stack

        # read from file and return without storing as attribute
        mtx: Array = self.read_input("stack", permissive=False, set_attribute=False)
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
    def shape(self) -> Tuple[Any, ...]:
        """The shape of the image stack."""
        if self._shape is not None:
            return self._shape
        self._shape = self.stack.shape
        self._channel_number = self._shape[0]
        return self._shape

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
        self.set_clusters()
        return self._clusters

    @property
    def adjacency_graph(self) -> nx.Graph:
        if self._adjacency_graph is not None:
            return self._adjacency_graph
        self._adjacency_graph = nx.readwrite.read_gpickle(
            self._get_input_filename("adjacency_graph")
        )
        return self._adjacency_graph

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
            "adjacency_graph": (self.single_cell_dir, ".neighbor_graph.gpickle"),
        }
        dir_, suffix = to_read[input_type]
        return cast(sample.root_dir) / cast(dir_) / (self.name + suffix)

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
            value = read_image_from_file(self._get_input_filename(key), **parameters)
        except FileNotFoundError:
            if permissive:
                return None
            raise
        if set_attribute:
            # TODO: fix assignment to @property
            setattr(self, key, value)
            return None
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
                raise ValueError("Length of parameter list must match number of inputs to be read.")

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

    def plot_channels(
        self,
        channels: Optional[Sequence[str]] = None,
        axes: List[Axis] = None,
        share_axes: bool = True,
        **kwargs,
    ) -> Optional[Figure]:
        """If axes is given it must be length channels"""
        # TODO: optimize this by avoiding reading stack for every channel
        channels = cast(channels or self.channel_labels.index)

        if axes is None:
            m, n = get_grid_dims(len(channels))
            fig, _axes = plt.subplots(
                n, m, figsize=(m * 4, n * 4), squeeze=False, sharex=share_axes, sharey=share_axes,
            )
            fig.suptitle(f"{self.sample}\n{self}")
            _axes = _axes.flatten()
        else:
            _axes = axes
        i = 0  # in case channels is length 0
        for i, channel in enumerate(channels):
            self.plot_channel(channel, ax=_axes[i], **kwargs)
        for _ax in _axes[i + 1 :]:
            _ax.axis("off")
        return fig if axes is None else None

    def _get_channel(
        self,
        channel: Union[int, str],
        red_func: str = "sum",
        equalize: bool = False,
        dont_warn: bool = False,
    ) -> Tuple[str, Array]:
        """
        Get a 2D signal array from a channel name or number.

        Parameters
        ----------
        channel : {Union[int, str]}
            An integer index of `channel_labels` or a string for its value.
            If the string does not match exactly the value, the channel that
            contains it would be retrieved.
            If more than one channel matches, then both are retrieved and the
            data is reduced by `red_func`.
            If the special values "mean" or "sum" are given, the respective
            reduction of all channels is retrieved.
        red_func : {str}, optional
            A function to reduce the data in case more than one channel is matched.
        equalize : {bool}, optional
            Whether to equalize the histogram of the channel. Default is `False`.

        Returns
        -------
        Tuple[str, Array]
            A string describing the channel and the respective array.

        Raises
        ------
        ValueError
            If `channel` cannot be found in `sample.channel_labels`.
        """
        stack = self.stack

        if isinstance(channel, int):
            label = self.channel_labels.iloc[channel]
            arr = stack[channel]
        elif isinstance(channel, str):
            if channel in ["sum", "mean"]:
                m = np.empty_like(stack)
                for i in range(m.shape[0]):
                    m[i] = stack[i] - stack[i].mean()
                label = f"Channel {channel}"
                arr = getattr(m, channel)(axis=0)
            elif sum(self.channel_labels.values == channel) == 1:
                label = channel
                arr = stack[self.channel_labels == channel]
            else:
                match = self.channel_labels.str.contains(re.escape((channel)))
                if any(match):
                    names = ", ".join(self.channel_labels[match])
                    if match.sum() == 1:
                        label = names
                        arr = stack[match]
                    else:
                        if not dont_warn:
                            print(
                                f"Could not find out channel '{channel}' in "
                                "`sample.channel_labels` "
                                f"but could find '{names}'. Returning sum of those."
                            )
                        order = match.reset_index(drop=True)[match.values].index
                        m = np.empty((match.sum(),) + stack.shape[1:])
                        j = 0
                        for i in order:
                            m[j] = stack[i] - stack[i].mean()
                            j += 1
                        label = names
                        arr = getattr(m, red_func)(axis=0)
                else:
                    msg = f"Could not find out channel '{channel}' " "in `sample.channel_labels`."
                    # # LOGGER.error(msg)
                    raise ValueError(msg)
        if equalize:
            arr = minmax_scale(eq(arr))
        return label, arr

    def _get_channels(self, channels: List[Union[int, str]], **kwargs) -> Tuple[str, Array]:
        """
        Convinience function to get signal from various channels.
        """
        labels = list()
        arrays = list()
        for channel in channels:
            lab, arr = self._get_channel(channel, **kwargs)
            labels.append(lab.replace(", ", "-"))
            arrays.append(arr.squeeze())
        return (", ".join(labels), np.asarray(arrays))

    def get_mean_all_channels(self) -> Array:
        """Get an array with mean of all channels"""
        return eq(self.stack.mean(axis=0))

    def plot_channel(
        self,
        channel: Union[int, str],
        ax: Optional[Axis] = None,
        equalize: bool = True,
        log: bool = False,
        add_scale: bool = True,
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
        channel, p = self._get_channel(channel, equalize=equalize)
        if log:
            p += abs(p.min())
            p = np.log1p(p)

        if _ax is None:
            _, _ax = plt.subplots(1, 1, figsize=(4, 4))
        _ax.imshow(p.squeeze(), rasterized=True, **kwargs)
        if add_scale:
            _add_scale(_ax)
        _ax.axis("off")
        _ax.set_title(f"{self.name}\n{channel}")
        # TODO: add scale bar
        return _ax

    def get_cell_type_assignments(self) -> Series:
        sample = cast(self.sample)
        # read if not set
        cell_type_assignments = getattr(
            sample,
            "cell_type_assignments",
            cast(
                sample.read_all_inputs(
                    only_these_keys=["cell_type_assignments"], set_attribute=False
                )
            )["cell_type_assignments"],
        )
        if "roi" in cell_type_assignments:
            cell_type_assignments = cell_type_assignments.query(f"roi == {self.roi_number}")
        return cell_type_assignments["cluster"]

    def plot_cell_types(
        self,
        cell_type_assignments: Series = None,
        cell_type_combinations: Optional[Union[str, List[Tuple[str, str]]]] = None,
        ax: Union[Axis, List[Axis]] = None,
        palette: Optional[str] = "tab20",
        add_scale: bool = True,
    ) -> Union[Figure, List[Patch]]:
        """
        If ax is given it must match number of `cell_type_combinations`.
        """
        # TODO: add scale to one of the axes
        if cell_type_assignments is not None:
            clusters = cell_type_assignments
        else:
            clusters = self.clusters
        if isinstance(clusters.index, pd.MultiIndex):
            clusters = clusters.loc[self.name]
        clusters.index = clusters.index.astype(int)

        if clusters.dtype == "object":
            # Replace the strings with a number
            labels = pd.Series(sorted_nicely(np.unique(clusters.values)))
            ns = labels.str.extract(r"(\d+) - .*")[0]
            # If the clusters contain a numeric prefix, use that
            # that will help having consistent color across ROIs
            if not ns.isnull().any():
                ns = ns.astype(int)
                clusters = clusters.replace(dict(zip(labels, ns)))
            else:
                ns = labels.index.to_series()
                clusters = clusters.replace(dict(zip(labels, np.arange(len(labels)))))
        else:
            labels = sorted(np.unique(clusters.values))
            ns = pd.Series(range(len(labels)))

        # simply plot all cell types jointly
        if cell_type_combinations in [None, "all"]:
            combs = [tuple(sorted(clusters.unique()))]
        else:
            combs = cell_type_combinations  # type: ignore

        n = len(combs)
        if ax is None:
            m = 1
            fig, axes = plt.subplots(
                n, m, figsize=(3 * m, 3 * n), sharex="col", sharey="col", squeeze=False
            )
        else:
            axes = ax
            if isinstance(ax, np.ndarray) and len(ax) != n:
                raise ValueError(f"Given axes must be of length of cell_type_combinations ({n}).")

        bckgd_cmap = get_transparent_cmaps(1, "binary")[0]
        patches = list()
        # TODO: fix dimentionality of axes call
        for i, _ in enumerate(combs):
            axes[i, 0].set_title(self.name)
            # plot channel mean for texture/context
            axes[i, 0].imshow(self.get_mean_all_channels() * 0.1, cmap=bckgd_cmap)
            # plot each of the cell types with different colors
            res = cell_labels_to_mask(self.cell_mask, clusters)
            rgb = numbers_to_rgb_colors(res, from_palette=cast(palette))
            axes[i, 0].imshow(rgb)
            colors = pd.Series(sns.color_palette(palette, max(ns.values))).reindex(ns.values - 1)
            patches += [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
            axes[i, 0].axis("off")
            if add_scale:
                _add_scale(axes[i, 0])

        if ax is None:
            add_legend(patches, axes[-1, 0])
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
            grid.savefig(cast(self.root_dir) / "channel_pairwise_correlation.svg", **FIG_KWS)

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
                marker_sets[_sp].append(np.random.choice(c[c == i].index, 1, replace=True)[0])
        return (xcorr, marker_sets)

    def plot_overlayied_channels_subplots(self, n_groups: int) -> Figure:
        """
        Plot all channels of ROI in `n_groups` combinations, where each combination
        has as little overlap as possible.
        """
        stack = self.stack

        _, marker_sets = self.get_distinct_marker_sets(
            n_groups=n_groups, group_size=int(np.floor(self.channel_number / n_groups))
        )

        n, m = get_grid_dims(n_groups)
        fig, axis = plt.subplots(
            n, m, figsize=(6 * m, 6 * n), sharex=True, sharey=True, squeeze=False,
        )
        axis = axis.flatten()
        for i, (marker_set, mrks) in enumerate(marker_sets.items()):
            patches = list()
            cmaps = get_transparent_cmaps(len(mrks))
            for _, (_l, c) in enumerate(zip(mrks, cmaps)):
                x = stack[self.channel_labels == _l, :, :].squeeze()
                v = x.mean() + x.std() * 2
                axis[i].imshow(
                    x, cmap=c, vmin=0, vmax=v, label=_l, interpolation="bilinear", rasterized=True,
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
        self, axes: Optional[Sequence[Axis]] = None
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

        dna_label, dna = self._get_channel("DNA", dont_warn=True)

        if axes is None:
            fig, _axes = plt.subplots(
                1, 5, figsize=(5 * 4, 4), gridspec_kw=dict(wspace=0.05), sharex=True, sharey=True
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
        _axes[3].set_title("Nuclei")
        _axes[3].imshow(self.nuclei_mask > 0, cmap="binary")
        _axes[4].set_title("Cells")
        _axes[4].imshow(self.cell_mask > 0, cmap="binary")
        # To plot jointly
        # _axes[5].imshow(probabilities)
        # _axes[5].contour(self.cell_mask, cmap="Blues")
        # _axes[5].contour(self.nuclei_mask, cmap="Reds")
        for _ax in _axes:
            _ax.axis("off")

        return fig if axes is None else None

    def quantify_cell_intensity(self, **kwargs) -> DataFrame:
        """Quantify intensity of each cell in each channel."""
        return quantify_cells(
            self._get_input_filename("stack"), self._get_input_filename("cell_mask"), **kwargs
        ).rename(columns=self.channel_labels)

    def quantify_cell_morphology(self, **kwargs) -> DataFrame:
        """QUantify shape attributes of each cell."""
        return measure_cell_attributes(self._get_input_filename("cell_mask"), **kwargs)

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
        area = np.multiply(*self.shape[1:])

        return len(cells) / area
