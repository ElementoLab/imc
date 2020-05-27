#! /usr/bin/env python

"""
A class to model a imaging mass cytometry sample.
"""

from __future__ import annotations  # fix the type annotatiton of not yet undefined classes

from typing import Dict, Tuple, List, Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import matplotlib.pyplot as plt  # type: ignore
import matplotlib.patches as mpatches  # type: ignore
import seaborn as sns  # type: ignore

from anndata import AnnData  # type: ignore
from skimage.exposure import equalize_hist as eq  # type: ignore

from imc.data_models.roi import ROI
from imc.types import Path, Figure, Patch, DataFrame, Series, MultiIndexSeries

# from imc import LOGGER
from imc.operations import predict_cell_types_from_reference
from imc.utils import parse_acquisition_metadata
from imc.graphics import get_grid_dims, add_legend, share_axes_by
from imc.exceptions import cast  # TODO: replace with typing.cast

FIG_KWS = dict(dpi=300, bbox_inches="tight")

DEFAULT_ROI_NAME_ATTRIBUTE = "roi_name"
DEFAULT_ROI_NUMBER_ATTRIBUTE = "roi_number"

DEFAULT_TOGGLE_ATTRIBUTE = "toggle"


class IMCSample:
    """

    If `csv_metadata` is given, it will initialize `ROI` objects for each row.

    If `panel_metadata` is given, it will use that
    """

    # sample_number: Optional[str]
    # panorama_number: Optional[str]
    # roi_number: Optional[str]
    # sample_numbers: Optional[List[int]]
    # panorama_numbers: Optional[List[int]]
    # roi_numbers: Optional[List[int]]

    # clusters: Series  # MultiIndex: ['roi', 'obj_id']

    file_types = ["cell_type_assignments"]

    def __init__(
        self,
        sample_name: str,
        root_dir: Path,
        csv_metadata: Optional[Union[Path, DataFrame]] = None,
        roi_name_atribute: str = DEFAULT_ROI_NAME_ATTRIBUTE,
        roi_number_atribute: str = DEFAULT_ROI_NUMBER_ATTRIBUTE,
        panel_metadata: Optional[Union[Path, DataFrame]] = None,
        channel_labels: Optional[Series] = None,
        prj: Optional["Project"] = None,
        **kwargs,
    ):
        self.name: str = str(sample_name)
        self.sample_name: str = sample_name
        self.root_dir = Path(root_dir).absolute()
        self.metadata: Optional[DataFrame] = (
            pd.read_csv(csv_metadata) if isinstance(csv_metadata, str) else csv_metadata
        )
        self.roi_name_atribute = roi_name_atribute
        self.roi_number_atribute = roi_number_atribute
        self._panel_metadata: Optional[DataFrame] = (
            pd.read_csv(panel_metadata, index_col=0)
            if isinstance(panel_metadata, (str, Path))
            else panel_metadata
        )
        # TODO: make sure channel labels conform to internal specification: "Label(Metal\d+)"
        self._channel_labels: Optional[Series] = (
            pd.read_csv(channel_labels, index_col=0, squeeze=True)
            if isinstance(channel_labels, (str, Path))
            else channel_labels
        )
        self.prj = prj
        self.rois: List["ROI"] = list()

        self.anndata: Optional[AnnData] = None
        self._clusters: Optional[MultiIndexSeries] = None

        # Add kwargs as attributes
        self.__dict__.update(kwargs)

        # initialize
        if self.metadata is not None:
            self._initialize_sample_from_annotation()

    def __repr__(self):
        return f"Sample '{self.name}' with {len(self.rois)} ROIs"

    def _initialize_sample_from_annotation(self, toggle: Optional[bool] = None) -> None:
        metadata = pd.DataFrame(self.metadata)  # this makes the type explicit
        if toggle:
            metadata = metadata[metadata[DEFAULT_TOGGLE_ATTRIBUTE]]

        has_numbers = self.roi_number_atribute in metadata.columns

        for i, (_, row) in enumerate(metadata.iterrows(), 1):
            roi = ROI(
                name=row[self.roi_name_atribute],
                roi_number=row[self.roi_number_atribute] if has_numbers else i,
                root_dir=self.root_dir,
                sample=self,
                prj=self.prj,
                **row.drop(["roi_name", "roi_number"], errors="ignore").to_dict(),
            )
            self.rois.append(roi)

    @property
    def n_rois(self) -> int:
        return len(self.rois)

    @property
    def panel_metadata(self) -> DataFrame:
        """Get an order of channel labels from sample panel."""
        if self._panel_metadata is not None:
            return self._panel_metadata
        # read reference
        rpath = Path(
            self.root_dir, "ometiff", self.name, self.name + "_AcquisitionChannel_meta.csv"
        )
        if not rpath.exists():
            msg = (
                "Sample `panel_metadata` was not given upon initialization "
                f"and '{rpath}' could not be found!"
            )
            raise FileNotFoundError(msg)
        self._panel_metadata = parse_acquisition_metadata(rpath)
        return self._panel_metadata

    @property
    def channel_labels(self) -> Series:
        return pd.DataFrame(
            [roi.channel_labels.rename(roi.name) for roi in self.rois]
        ).T.rename_axis(index="chanel", columns="roi")

    @property
    def clusters(self) -> MultiIndexSeries:
        if self._clusters is not None:
            return self._clusters
        self.prj.set_clusters(samples=[self])
        return self._clusters

    def _get_input_filename(self, input_type: str) -> Path:
        """Get path to file with data for Sample.

        Available `input_type` values are:
            - "cell_type_assignments": CSV file with cell type assignemts for each cell and each ROI
        """
        to_read = {
            # "cell": ("cpout", "cell.csv"),
            # "relationships": ("cpout", "Object relationships.csv"),
            "cell_type_assignments": ("single_cell", ".cell_type_assignment_against_reference.csv"),
            # "anndata": ("single_cell", ".cell.mean.all_vars.processed.h5ad")
        }
        dir_, suffix = to_read[input_type]
        return self.root_dir / dir_ / (self.name + suffix)

    def read_all_inputs(
        self,
        rois: Optional[List["ROI"]] = None,
        only_these_keys: Optional[List[str]] = None,
        permissive: bool = False,
        set_attribute: bool = True,
        # overwrite: bool = False,
        **kwargs,
    ) -> Optional[Dict[str, DataFrame]]:
        """
        Wrapper to read the input of each ROI object but also
        sample-specific inputs in addition.

        Will only return IMCSample keys.
        """
        rois = rois or self.rois

        if only_these_keys is None:
            only_these_keys = ROI.file_types + IMCSample.file_types

        roi_keys = [k for k in only_these_keys if k in ROI.file_types]
        if roi_keys:
            for roi in rois:
                roi.read_all_inputs(
                    only_these_keys=only_these_keys,
                    permissive=permissive,
                    set_attribute=set_attribute,
                    **kwargs,
                )
        if only_these_keys is None:
            only_these_keys = self.file_types
        sample_keys = [k for k in only_these_keys if k in IMCSample.file_types]
        if sample_keys:
            if not set_attribute and len(only_these_keys) != 1:
                msg = "With `set_attribute` False, only one in `only_these_keys` can be seleted."
                raise ValueError(msg)

            res = dict()
            for ftype in only_these_keys:
                _path = self._get_input_filename(ftype)
                try:
                    # TODO: logger.info()
                    v = pd.read_csv(_path, index_col=0)
                except FileNotFoundError:
                    if permissive:
                        continue
                    raise
                if set_attribute:
                    # TODO: fix assignemtn to @property
                    setattr(self, ftype, v)
                else:
                    res[ftype] = v
            return res if not set_attribute else None
        return None

    def plot_rois(
        self, channel: Union[str, int], rois: Optional[List["ROI"]] = None
    ) -> Figure:  # List[ROI]
        """Plot a single channel for all ROIs"""
        rois = rois or self.rois

        n, m = get_grid_dims(len(rois))
        fig, axis = plt.subplots(n, m, figsize=(m * 4, n * 4), squeeze=False)
        axis = axis.flatten()
        i = 0  # just in case there are no ROIs
        for i, roi in enumerate(rois):
            roi.plot_channel(channel, ax=axis[i])
        for _ax in axis[i:]:
            _ax.axis("off")
        return fig

    def plot_channels(self, channels, rois: Optional[List["ROI"]] = None):
        raise NotImplementedError

    def plot_cell_types(
        self,
        rois: Optional[List["ROI"]] = None,
        cell_type_combinations: Optional[Union[str, List[Tuple[str, str]]]] = None,
        cell_type_assignments: Optional[DataFrame] = None,
        palette: Optional[str] = "tab20",
    ):
        rois = rois or self.rois

        n = len(cell_type_combinations or [1])
        m = len(rois)
        fig, axes = plt.subplots(n, m, figsize=(3 * m, 3 * n), squeeze=False)
        patches: List[Patch] = list()
        for _ax, roi in zip(np.hsplit(axes, m), rois):
            patches += roi.plot_cell_types(
                cell_type_combinations=cell_type_combinations,
                cell_type_assignments=cell_type_assignments,
                palette=palette,
                ax=_ax,
            )
        add_legend(patches, axes[0, -1])
        return fig

    def plot_probabilities_and_segmentation(self, rois: Optional[List["ROI"]] = None) -> Figure:
        n = len(rois or self.rois)
        fig, axes = plt.subplots(
            n, 5, figsize=(5 * 4, 4 * n), gridspec_kw=dict(wspace=0.05), sharex="row", sharey="row",
        )
        fig.suptitle(self.name)
        for i, roi in enumerate(self.rois):
            roi.plot_probabilities_and_segmentation(axes=axes[i])
        return fig

    def cell_to_anndata(self, red_func="mean", set_attribute: bool = False):
        _df = self.quantify_cell_intensity(func=red_func)
        _an = AnnData(_df.drop("roi", axis=1).sort_index(axis=1))
        _an.obs["roi"] = pd.Categorical(_df["roi"].values)
        _an.raw = _an

        if set_attribute:
            self.anndata = _an
        return _an

    def quantify_cell_intensity(self, rois: Optional[List["ROI"]] = None, **kwargs) -> DataFrame:
        return pd.concat(
            [
                roi.quantify_cell_intensity(**kwargs).assign(roi=roi.name)
                for roi in rois or self.rois
            ]
        )

    def quantify_cell_morphology(self, rois: Optional[List["ROI"]] = None, **kwargs) -> DataFrame:
        return pd.concat(
            [
                roi.quantify_cell_morphology(**kwargs).assign(roi=roi.name)
                for roi in rois or self.rois
            ]
        )

    def set_clusters(
        self, clusters: Optional[MultiIndexSeries] = None, rois: Optional[List["ROI"]] = None,
    ) -> None:
        id_cols = ["roi", "obj_id"]
        if clusters is None:
            self.prj.set_clusters(samples=[self])
        else:
            assert isinstance(clusters.index, pd.MultiIndex)
            assert clusters.index.names == id_cols
            self._clusters = clusters
            "cell_type_assignments"
        for roi in rois or self.rois:
            roi.set_clusters(clusters=self.clusters.loc[roi.name])

    def predict_cell_types_from_reference(self, **kwargs) -> None:
        predict_cell_types_from_reference(self, **kwargs)

    def cell_type_adjancency(
        self, rois: Optional[List["ROI"]] = None, output_prefix: Optional[Path] = None
    ) -> None:
        rois = rois or self.rois

        output_prefix = output_prefix or self.root_dir / "single_cell" / (
            self.name + ".cluster_adjacency_graph."
        )

        # TODO: check default input
        # Plot adjancency for all ROIs next to each other and across
        adj_matrices = pd.concat(
            [
                pd.read_csv(f"{output_prefix}{roi.name}.norm_over_random.csv", index_col=0).assign(
                    roi=roi.roi_number
                )
                for roi in rois
            ]
        )
        # g2 = nx.readwrite.read_gpickle(roi_prefix + "neighbor_graph.gpickle")

        mean_ = adj_matrices.drop("roi", axis=1).groupby(level=0).mean().sort_index(0).sort_index(1)
        adj_matrices = adj_matrices.append(mean_.assign(roi="mean"))

        m = self.n_rois + 1
        nrows = 3
        fig, _ax = plt.subplots(nrows, m, figsize=(4 * m, 4 * nrows), sharex=False, sharey=False)
        v = np.nanstd(adj_matrices.drop("roi", axis=1).values)
        kws = dict(
            cmap="RdBu_r",
            center=0,
            square=True,
            xticklabels=True,
            yticklabels=True,
            vmin=-v,
            vmax=v,
        )
        for i, roi in enumerate(rois):
            _ax[0, i].set_title(roi.name)
            _ax[0, i].imshow(eq(roi.stack.mean(0)))
            _ax[0, i].axis("off")
            __x = (
                adj_matrices.loc[adj_matrices["roi"] == roi.roi_number]
                .drop("roi", axis=1)
                .reindex(index=mean_.index, columns=mean_.index)
                .fillna(0)
            )
            sns.heatmap(__x, ax=_ax[1, i], **kws)
            sns.heatmap(__x - mean_, ax=_ax[2, i], **kws)
        _ax[1, 0].set_ylabel("Observed ratios")
        _ax[2, 0].set_ylabel("Ratio difference to mean")
        _ax[1, -1].set_title("ROI mean")
        sns.heatmap(mean_, ax=_ax[1, -1], **kws)
        _ax[0, -1].axis("off")
        _ax[2, -1].axis("off")
        share_axes_by(_ax[1:], "both")
        fig.savefig(output_prefix + "roi_over_mean_rois.image_clustermap.svg", **FIG_KWS)
