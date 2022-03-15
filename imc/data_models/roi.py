#! /usr/bin/env python

"""
A class to model a imaging mass cytometry acquired region of interest (ROI).
"""

from __future__ import annotations
import re
import typing as tp
from functools import partial

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import scipy  # type: ignore
import scipy.ndimage as ndi  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.patches as mpatches  # type: ignore
import seaborn as sns  # type: ignore
import networkx as nx  # type: ignore
from skimage.filters import gaussian  # type: ignore
from skimage.exposure import equalize_hist as eq  # type: ignore
from skimage.segmentation import clear_border  # type: ignore

from imc.types import Path, Figure, Axis, Patch, Array, DataFrame, Series
import imc.data_models.sample as _sample
import imc.data_models.project as _project
from imc.utils import read_image_from_file, minmax_scale
from imc.graphics import (
    add_scale as _add_scale,
    add_minmax as _add_minmax,
    get_grid_dims,
    get_transparent_cmaps,
    add_legend as _add_legend,
    cell_labels_to_mask,
    values_to_rgb_colors,
    merge_channels,
    rainbow_text,
    get_random_label_cmap,
)

# TODO: replace exceptions.cast with typing.cast
from imc.exceptions import cast, AttributeNotSetError


FIG_KWS = dict(dpi=300, bbox_inches="tight")

# processed directory structure
SUBFOLDERS_PER_SAMPLE = True
DEFAULT_ROI_NAME = "roi"
DEFAULT_MASK_LAYER = "cell"
ROI_STACKS_DIR = Path("tiffs")
ROI_MASKS_DIR = Path("tiffs")
ROI_UNCERTAINTY_DIR = Path("uncertainty")
ROI_SINGLE_CELL_DIR = Path("single_cell")
MEMBRANE_MASK_AREA = 1
EXTRACELLULAR_MASK_AREA = 5


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

    # shape: tp.Tuple[int, int]

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
        roi_number: tp.Optional[int] = None,
        channel_labels: tp.Optional[tp.Union[Path, Series]] = None,
        root_dir: tp.Optional[Path] = None,
        stacks_dir: tp.Optional[Path] = None,  # TODO: make these relative to the root_dir
        masks_dir: tp.Optional[Path] = None,
        single_cell_dir: tp.Optional[Path] = None,
        sample: tp.Optional[_sample.IMCSample] = None,
        default_mask_layer: str = DEFAULT_MASK_LAYER,
        **kwargs,
    ):
        # attributes
        self.name = name
        self.roi_name = name
        self.roi_number = roi_number
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.stacks_dir = (
            Path(stacks_dir)
            if stacks_dir is not None
            else self.root_dir.parent / ROI_STACKS_DIR
            if self.root_dir is not None
            else None
        )
        self.masks_dir = (
            Path(masks_dir)
            if masks_dir is not None
            else self.root_dir.parent / ROI_MASKS_DIR
            if self.root_dir is not None
            else None
        )
        self.single_cell_dir = (
            Path(single_cell_dir)
            if single_cell_dir is not None
            else self.root_dir.parent / ROI_SINGLE_CELL_DIR
            if self.root_dir is not None
            else None
        )
        self.channel_labels_file: tp.Optional[Path] = (
            Path(channel_labels) if isinstance(channel_labels, (str, Path)) else None
        )
        self.mask_layer = default_mask_layer
        # TODO: make sure channel labels conform to internal specification: "Label(Metal\d+)"
        self._channel_labels: tp.Optional[Series] = (
            pd.read_csv(channel_labels, index_col=0, squeeze=True)
            if isinstance(channel_labels, (str, Path))
            else channel_labels
        )
        self._channel_include = None
        self._channel_exclude = None
        # obj connections
        self.sample = sample
        self.prj: tp.Optional[_project.Project] = None
        # data
        self._stack: tp.Optional[Array] = None
        self._shape: tp.Optional[tp.Tuple] = None
        self._area: tp.Optional[int] = None
        self._channel_number: tp.Optional[int] = None
        self._probabilities: tp.Optional[Array] = None

        self._cell_mask_o: tp.Optional[Array] = None
        self._cell_mask: tp.Optional[Array] = None

        self._nuclei_mask_o: tp.Optional[Array] = None
        self._nuclei_mask: tp.Optional[Array] = None

        self._cytoplasm_mask: tp.Optional[Array] = None
        self._membrane_mask: tp.Optional[Array] = None
        self._extracellular_mask: tp.Optional[Array] = None

        self._adjacency_graph = None
        self._clusters: tp.Optional[Series] = None

        # Add kwargs as attributes
        self.__dict__.update(kwargs)

    def __repr__(self):
        return (
            "Region"
            + (f" {self.roi_number}" if self.roi_number is not None else "")
            + (f" of sample '{self.sample.name}'" if self.sample is not None else "")
        )

    @classmethod
    def from_stack(
        cls, stack_file: tp.Union[Path, str], make_sample: bool = True, **kwargs
    ) -> ROI:
        if isinstance(stack_file, str):
            stack_file = Path(stack_file)
        stack_file = stack_file.absolute()

        # reason = "Stack file must end with '_full.tiff' for the time being."
        # assert stack_file.endswith("_full.tiff")
        roi_numbers = re.findall(r".*-(\d+)_full\.tiff", stack_file.as_posix())
        if len(roi_numbers) != 1:
            print("Could not determine ROI number.")
            roi_number = None
        else:
            roi_number = int(roi_numbers[0])

        roi = ROI(
            name=stack_file.stem.replace("_full", ""),
            root_dir=stack_file.parent,
            stacks_dir=stack_file.parent,
            roi_number=roi_number,
            **kwargs,
        )
        roi._stack_file = stack_file
        if make_sample:
            roi.sample = _sample.IMCSample.from_rois([roi])
            roi.sample.rois = [roi]
        return roi

    @property
    def channel_labels(self) -> Series:
        """
        Return a Series with a string for each channel in the ROIs stack.
        """
        import tifffile
        from xml.etree import ElementTree

        if self._channel_labels is not None:
            return self._channel_labels

        # Read channel names from OME-TIFF XML metadata
        t = tifffile.TiffFile(self.get_input_filename("stack"))
        if t.is_ome:
            et = ElementTree.fromstring(t.ome_metadata)
            preview = (
                pd.Series([e.get("Name") for e in et.iter() if e.tag.endswith("Channel")])
                .rename("channel")
                .rename_axis("index")
            )
            if not (preview.empty or preview.isnull().all()):
                self._channel_labels = preview
                return self._channel_labels

        # Read channel labels from CSV file
        channel_labels_file = self.get_input_filename("stack").replace_(".tiff", ".csv")
        if not channel_labels_file.exists():
            msg = (
                "File is not OME-TIFF, `channel_labels` was not given upon initialization "
                f"and '{channel_labels_file}' could not be found!"
            )
            raise FileNotFoundError(msg)

        preview = pd.read_csv(channel_labels_file, header=None, squeeze=True)
        if isinstance(preview, pd.Series):
            order = preview.to_frame(name="ChannelName").set_index("ChannelName")
            # read reference
            ref: DataFrame = self.sample.panel_metadata
            ref = ref.loc[
                ref["AcquisitionID"].isin([self.roi_number, str(self.roi_number)])
            ]
            self._channel_labels = (
                order.join(ref.reset_index().set_index("ChannelName"))["index"]
                .reset_index(drop=True)
                .rename("channel")
            )
        else:
            preview = preview.dropna().set_index(0).squeeze().rename("channel")
            preview.index = preview.index.astype(int).rename("index")
            self._channel_labels = preview
        return self._channel_labels

    @property
    def channel_exclude(self) -> Series:
        if self._channel_exclude is not None:
            return self._channel_exclude
        if self.channel_labels is not None:
            self._channel_exclude = pd.Series(
                index=self.channel_labels, dtype="object"
            ).fillna(False)
            return self._channel_exclude

    def set_channel_exclude(self, values: tp.Union[tp.Sequence[str], Series]):
        """
        values: tp.List | Series
            tp.Sequence of channels to exclude.
        """
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
        mtx: Array = self.read_input("stack", permissive=False, set_attribute=False)
        self._shape = mtx.shape
        self._channel_number = mtx.shape[0]
        return mtx

    @property
    def stack_eq(self):
        """Same as `stack` but equalized per channel."""
        if self._stack_eq is not None:
            return self._stack_eq
        return np.asarray([eq(x) for x in self.stack])

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
    def shape(self) -> tp.Tuple[int, ...]:
        """The shape of the image stack."""
        if self._shape is not None:
            return self._shape
        try:
            self._shape = (np.nan,) + self.mask.shape
        except AttributeNotSetError:
            try:
                self._shape = self.stack.shape
            except AttributeNotSetError:
                raise AttributeNotSetError("ROI does not have either stack or mask!")
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
        return res  # [:3, :, :]  # return only first 3 labels

    @property
    def cell_mask_o(self) -> Array:
        """
        An array with unique integers for each cell.
        Original array including cells touching image borders.
        """
        if self._cell_mask_o is not None:
            return self._cell_mask_o
        self._cell_mask_o = self.read_input("cell_mask", set_attribute=False)
        return self._cell_mask_o

    @property
    def cell_mask(self) -> Array:
        """
        An array with unique integers for each cell,
        excluding cells touching image borders.
        """
        if self._cell_mask is not None:
            return self._cell_mask
        self._cell_mask = clear_border(self.cell_mask_o)
        return self._cell_mask

    @property
    def nuclei_mask_o(self) -> Array:
        """
        An array with unique integers for each cell.
        Original array including nuclei of cells touching image borders.
        """
        if self._nuclei_mask_o is not None:
            return self._nuclei_mask_o
        self._nuclei_mask_o = self.read_input("nuclei_mask", set_attribute=False)
        # todo: align numbering with cell mask
        return self._nuclei_mask_o

    @property
    def nuclei_mask(self) -> Array:
        """
        An array with unique integers for each nuclei
        matched with cell mask not touching image borders.
        """
        if self._nuclei_mask is not None:
            return self._nuclei_mask
        nucl = self.nuclei_mask_o
        if self.get_input_filename("cell_mask").exists():
            nucl[~np.isin(nucl, np.unique(self.cell_mask)[1:])] = 0
        self._nuclei_mask = nucl
        return self._nuclei_mask

    @property
    def cytoplasm_mask(self) -> Array:
        """
        An array with unique integers for the cytoplasm of each cell.
        The cytoplasm is defined as the cell area excluding nuclei and membrane.
        """
        if self._cytoplasm_mask is not None:
            return self._cytoplasm_mask
        cyto = self.cell_mask.copy()
        o = (self.nuclei_mask > 0) | (self.membrane_mask > 0)
        cyto[o] = 0
        self._cytoplasm_mask = cyto
        return self._cytoplasm_mask

    @property
    def membrane_mask(self) -> Array:
        """
        An array with unique integers for the membrane of each cell.
        The membrane is the area of cell defined by a fixed border in the cell's border.
        """
        from skimage.segmentation import find_boundaries

        if self._membrane_mask is not None:
            return self._membrane_mask
        membrs = (
            find_boundaries(
                self.cell_mask, MEMBRANE_MASK_AREA, mode="inner", background=0
            ).astype(int)
            * self.cell_mask
        ).astype("uint")
        self._membrane_mask = membrs
        return self._membrane_mask

    @property
    def extracellular_mask(self) -> Array:
        """
        An array with unique integers for the extracellular area of each cell.
        The extracellular area is a fixed amount around the cell, not overlapping other cells.
        """
        from skimage.segmentation import expand_labels

        if self._extracellular_mask is not None:
            return self._extracellular_mask
        extr = expand_labels(self.cell_mask, EXTRACELLULAR_MASK_AREA)
        extr[self.cell_mask > 0] = 0
        self._extracellular_mask = extr
        return self._extracellular_mask

    @property
    def mask(self) -> Array:
        """
        An array with unique integers for the default mask area of each cell.
        """
        mask: Array = getattr(self, self.mask_layer + "_mask")
        return mask

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
                self.get_input_filename("adjacency_graph")
            )
        except FileNotFoundError:
            return None
        return self._adjacency_graph

    def _get_area(self) -> int:
        """Get area of ROI"""
        return np.multiply(*self.shape[1:])  # type: ignore[no-any-return]

    def get_input_filename(self, input_type: str) -> Path:
        """Get path to file with data for ROI.

        Available `input_type` values are:
            - "stack": Multiplexed image stack
            - "channel_labels": Labels of channels (may not exist for OME-TIFF stacks)
            - "ilastik_input": Features extracted by ilastik (usually not available by default)
            - "probabilities": 3 color probability intensities predicted by ilastik
            - "cell_mask": TIFF file with mask for cells
            - "nuclei_mask": TIFF file with mask for nuclei
            - "nuclear_mask": TIFF file with mask for nuclei
            - "cell_type_assignments": CSV file with cell type assignemts for each cell
            - "adjacency_graph": Cell neighborhood graph.
        """
        if hasattr(self, "_" + input_type + "_file"):
            return Path(getattr(self, "_" + input_type + "_file"))
        to_read = {
            "stack": (self.stacks_dir, "_full.tiff"),
            "channel_labels": (self.stacks_dir, "_full.csv"),
            "ilastik_input": (self.stacks_dir, "_ilastik_s2.h5"),
            # "features": (self.stacks_dir, "_ilastik_s2_Features.h5"),
            "probabilities": (self.stacks_dir, "_Probabilities.tiff"),
            # "uncertainty": (ROI_UNCERTAINTY_DIR, "_ilastik_s2_Probabilities_uncertainty.tiff"),
            # "cell_mask": (self.masks_dir, "_ilastik_s2_Probabilities_mask.tiff"),
            "cell_mask": (self.masks_dir, "_full_mask.tiff"),
            "nuclei_mask": (self.masks_dir, "_full_nucmask.tiff"),
            "nuclear_mask": (self.masks_dir, "_full_nucmask.tiff"),
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
        return dir_ / (self.name + suffix)

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
        parameters: tp.Optional[tp.Dict] = None,
    ) -> tp.Optional[Array]:
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
            value = read_image_from_file(self.get_input_filename(key), **parameters)
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
        only_these_keys: tp.Optional[tp.Sequence[str]] = None,
        permissive: bool = False,
        set_attribute: bool = True,
        overwrite: bool = False,
        parameters: tp.Optional[tp.Union[tp.Dict, tp.Sequence]] = None,
    ) -> tp.Optional[tp.Dict[str, Array]]:
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
        channel: tp.Union[int, str],
        red_func: str = "mean",
        log: bool = False,
        equalize: bool = False,
        minmax: bool = False,
        smooth: tp.Optional[int] = None,
        dont_warn: bool = False,
    ) -> tp.Tuple[str, Array, tp.Tuple[float, float]]:
        """
        Get a 2D signal array from a channel name or number.
        If the channel name matches more than one channel, return reduction of channels.

        Parameters
        ----------
        channel : {tp.Union[int, str]}
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
        tp.Tuple[str, Array, tp.Tuple[float, float]]
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
            stack: Array, red_func: str, ex: tp.Union[tp.Sequence[bool], Series] = None
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
        j = (
            partial(gaussian, sigma=smooth, mode="reflect")
            if smooth is not None
            else _idenv
        )
        excluded = self.channel_exclude[self.channel_exclude]

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
                    if not dont_warn:
                        print(excluded_message.format(channel, label))
                arr = stack[self.channel_labels == channel]
            else:
                match = self.channel_labels.str.contains(re.escape((channel)))
                if match.any():
                    label = ", ".join(self.channel_labels[match])
                    if match.sum() == 1:
                        if self.channel_labels[match].squeeze() in excluded:
                            if not dont_warn:
                                print(excluded_message.format(channel, label))
                        arr = stack[match]
                    else:
                        ex = (~match) | self.channel_exclude.values
                        label = ", ".join(self.channel_labels[~ex])
                        if ex.sum() == 1:
                            if not dont_warn:
                                print(excluded_message.format(channel, label))
                            arr = stack[match]
                        else:
                            if not dont_warn:
                                msg = f"Could not find out channel '{channel}' in `roi.channel_labels` but could find '{label}'."
                                print(msg)
                            if ex.all():
                                if not dont_warn:
                                    print(excluded_message.format(channel, label))
                                ex = ~match
                            arr = reduce_channels(stack, "mean", ex)
                            f = g = _idenv
                else:
                    msg = f"Could not find out channel '{channel}' in `sample.channel_labels`."
                    # # LOGGER.error(msg)
                    raise ValueError(msg)
        vminmax = arr.min(), arr.max()
        arr = j(h(g(f((arr)))))
        return label, arr, vminmax

    def _get_channels(
        self, channels: tp.Sequence[tp.Union[int, str]], **kwargs
    ) -> tp.Tuple[str, Array, Array]:
        """
        Convenience function to get signal from various channels.
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
        channel: tp.Union[int, str],
        ax: tp.Optional[Axis] = None,
        equalize: bool = True,
        log: bool = True,
        minmax: bool = True,
        smooth: tp.Optional[int] = None,
        position: tp.Optional[tp.Tuple[tp.Tuple[int, int], tp.Tuple[int, int]]] = None,
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
            channel, log=log, equalize=equalize, minmax=minmax
        )
        p = p.squeeze()

        if _ax is None:
            _, _ax = plt.subplots(1, 1, figsize=(4, 4))
        if position is not None:
            p = p[slice(*position[0][::-1], 1), slice(*position[1][::-1], 1)]
        _ax.imshow(p, rasterized=True, **kwargs)
        if add_scale:
            _add_scale(_ax)
        if add_range:
            _add_minmax(_minmax, _ax)
        _ax.axis("off")
        _ax.set_title(f"{self.name}\n{channel}")
        return _ax

    def plot_channels(
        self,
        channels: tp.Optional[tp.Sequence[str]] = None,
        merged: bool = False,
        axes: tp.Sequence[Axis] = None,
        equalize: bool = None,
        log: bool = True,
        minmax: bool = True,
        smooth: tp.Optional[int] = None,
        position: tp.Optional[tp.Tuple[tp.Tuple[int, int], tp.Tuple[int, int]]] = None,
        add_scale: bool = True,
        add_range: bool = True,
        share_axes: bool = True,
        **kwargs,
    ) -> tp.Optional[Figure]:
        """

        If axes is given it must be length channels

        **kwargs: dict
            Additional keyword arguments will be passed to imc.graphics.merge_channels.
            Pass 'target_colors' to select colors to use when using `merged`.

        """
        # TODO: optimize this by avoiding reading stack for every channel
        if channels is None:
            channels = self.channel_labels.index

        if axes is None:
            n, m = (1, 1) if merged else get_grid_dims(len(channels))
            ar = self.shape[1] / self.shape[2]
            fig, _axes = plt.subplots(
                n,
                m,
                figsize=(m * 4, n * 4 * ar),
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
                list(channels),
                log=log,
                equalize=equalize,
                minmax=minmax,
                smooth=smooth,
            )
            arr2, colors = merge_channels(arr, return_colors=True, **kwargs)

            if position is not None:
                arr2 = arr2[slice(*position[0][::-1], 1), slice(*position[1][::-1], 1)]

            x, y, _ = arr2.shape
            _axes[0].imshow(minmax_scale(arr2))
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
                fontsize=6,
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
                    position=position,
                    add_scale=add_scale,
                    add_range=add_range,
                    **kwargs,
                )
        for _ax in _axes:  # [i + 1 :]
            _ax.axis("off")
        if "fig" in locals():
            fig.subplots_adjust(wspace=0.1, hspace=0.1)
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

    @tp.overload
    def plot_cell_type(self, cluster, ax: None) -> Figure:
        ...

    @tp.overload
    def plot_cell_type(self, cluster, ax: Axis) -> None:
        ...

    def plot_cell_type(
        self,
        cluster,
        ax: tp.Optional[Axis] = None,
        cmap: tp.Optional[str] = "gray",
        add_scale: bool = True,
        position: tp.Optional[tp.Tuple[tp.Tuple[int, int], tp.Tuple[int, int]]] = None,
    ) -> Figure:
        if ax is None:
            fig, _ax = plt.subplots(1, 1, figsize=(4, 4))
        else:
            _ax = ax
        m = self.clusters == cluster
        if m.sum() == 0:
            raise ValueError(f"Cound not find cluster '{cluster}'.")

        arr = cell_labels_to_mask(self.cell_mask, m)
        if position is not None:
            arr = arr[slice(*position[0][::-1], 1), slice(*position[1][::-1], 1)]
        _ax.imshow(arr, cmap=cmap)
        _ax.set_title(cluster)
        _ax.axis("off")

        if add_scale:
            _add_scale(_ax)
        return fig if ax is None else None

    def plot_cell_types(
        self,
        cell_type_assignments: Series = None,
        cell_type_combinations: tp.Optional[
            tp.Union[str, tp.Sequence[tp.Tuple[str, str]]]
        ] = None,
        position: tp.Optional[tp.Tuple[tp.Tuple[int, int], tp.Tuple[int, int]]] = None,
        ax: tp.Union[Axis, tp.Sequence[Axis]] = None,
        palette: tp.Optional[str] = None,
        add_scale: bool = True,
        add_legend: bool = True,
        legend_kwargs: tp.Dict = {},
    ) -> tp.Union[Figure, tp.Sequence[Patch]]:
        """
        If ax is given it must match number of `cell_type_combinations`.
        """
        from imc.utils import is_numeric, sorted_nicely

        if cell_type_assignments is not None:
            clusters = cell_type_assignments
        else:
            clusters = self.clusters
        if isinstance(clusters.index, pd.MultiIndex):
            clusters = clusters.loc[self.name]
        clusters.index = clusters.index.astype(int)

        if not is_numeric(clusters):
            # Replace the strings with a number
            labels = pd.Series(sorted_nicely(np.unique(clusters.values))).astype(str)
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

        if ns.min() == 0:
            ns += 1

        if (ns.iloc[0] != 0) and (ns.index.min() == 0):
            ns.index += 1

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
            axes[i, 0].imshow(self.get_mean_all_channels() * 0.1, cmap=bckgd_cmap)
            # plot each of the cell types with different colors
            res = cell_labels_to_mask(self.cell_mask, clusters)
            rgb, cmap = values_to_rgb_colors(res, from_palette=palette)
            if position is not None:
                rgb = rgb[slice(*position[0][::-1], 1), slice(*position[1][::-1], 1)]
            axes[i, 0].imshow(rgb)
            patches += [mpatches.Patch(color=c, label=l) for l, c in cmap.items()]
            axes[i, 0].axis("off")
            if add_scale:
                _add_scale(axes[i, 0])

        if add_legend:
            _add_legend(patches, axes[-1, 0], **legend_kwargs)
        return fig if ax is None else patches

    def get_distinct_marker_sets(
        self, n_groups: int = 4, group_size: int = 4, save_plot: bool = False
    ) -> tp.Tuple[DataFrame, tp.Dict[int, tp.Sequence[str]]]:
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

        marker_sets: tp.Dict[int, tp.Sequence[str]] = dict()
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
        self, axes: tp.Optional[tp.Sequence[Axis]] = None, add_scale: bool = True
    ) -> tp.Optional[Figure]:
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

        nuclei = self.get_input_filename("nuclei_mask").exists()
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
        _axes[0].imshow(self._get_channel("mean", equalize=True, minmax=True)[1])
        _axes[1].set_title(dna_label)
        _axes[1].imshow(eq(dna))
        _axes[2].set_title("Probabilities")
        _axes[2].imshow(probabilities)
        i = 0
        if nuclei:
            _axes[3 + i].set_title("Nuclei")
            _axes[3 + i].imshow(
                self.nuclei_mask, cmap=get_random_label_cmap(), interpolation="none"
            )
            i += 1
        _axes[3 + i].set_title("Cells")
        _axes[3 + i].imshow(
            self.cell_mask, cmap=get_random_label_cmap(), interpolation="none"
        )
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
        channel_include: tp.Sequence[str] = None,
        channel_exclude: tp.Sequence[str] = None,
        layers: tp.Sequence[str] = ["cell"],
        **kwargs,
    ) -> DataFrame:
        """Quantify intensity of each cell in each channel."""
        from imc.ops.quant import quantify_cell_intensity

        if channel_include is not None:
            kwargs["channel_include"] = self.channel_labels.str.contains(
                channel_include
            ).values
        if channel_exclude is not None:
            kwargs["channel_exclude"] = self.channel_labels.str.contains(
                channel_exclude
            ).values

        stack = self.stack
        _quant = list()
        for layer in layers:
            quant = (
                quantify_cell_intensity(
                    stack,
                    getattr(self, layer + "_mask_o"),
                    **kwargs,
                )
                .rename(columns=self.channel_labels)
                .assign(layer=layer)
            )
            _quant.append(quant)
        quant = pd.concat(_quant)
        quant.index.name = "obj_id"
        if layers == ["cell"]:
            quant = quant.drop(["layer"], axis=1)
        return quant

    def quantify_cell_morphology(
        self, layers: tp.Sequence[str] = ["cell"], **kwargs
    ) -> DataFrame:
        """
        Quantify shape attributes of each cell.
        Additional keyword arguments are passed to
        `imc.ops.quant.quantify_cell_morphology`.
        """
        from imc.ops.quant import quantify_cell_morphology

        _quant = list()
        for layer in layers:
            quant = (
                quantify_cell_morphology(
                    getattr(self, layer + "_mask_o"),
                    **kwargs,
                )
                .rename(columns=self.channel_labels)
                .assign(layer=layer)
            )
            _quant.append(quant)
        quant = pd.concat(_quant)
        quant.index.name = "obj_id"
        if layers == ["cell"]:
            quant = quant.drop(["layer"], axis=1)
        return quant

    def set_clusters(self, clusters: tp.Optional[Series] = None) -> None:
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
