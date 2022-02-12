#! /usr/bin/env python

"""
A class to model a imaging mass cytometry sample.
"""

from __future__ import annotations
import typing as tp
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt  # type: ignore
import matplotlib.patches as mpatches  # type: ignore
import seaborn as sns  # type: ignore

from anndata import AnnData  # type: ignore
from skimage.exposure import equalize_hist as eq  # type: ignore

from imc.data_models.roi import ROI
from imc.types import Path, Figure, Patch, DataFrame, Series, MultiIndexSeries
import imc.data_models.project as _project
import imc.data_models.roi as _roi
from imc.utils import parse_acquisition_metadata
from imc.graphics import get_grid_dims, add_legend, share_axes_by
from imc.exceptions import cast  # TODO: replace with typing.cast

FIG_KWS = dict(dpi=300, bbox_inches="tight")

DEFAULT_SAMPLE_NAME = "sample"
DEFAULT_ROI_NAME_ATTRIBUTE = "roi_name"
DEFAULT_ROI_NUMBER_ATTRIBUTE = "roi_number"

DEFAULT_TOGGLE_ATTRIBUTE = "toggle"


class IMCSample:
    """

    If `metadata` is given, it will initialize `ROI` objects for each row.

    If `panel_metadata` is given, it will use that
    """

    # sample_number: tp.Optional[str]
    # panorama_number: tp.Optional[str]
    # roi_number: tp.Optional[str]
    # sample_numbers: tp.Optional[tp.List[int]]
    # panorama_numbers: tp.Optional[tp.List[int]]
    # roi_numbers: tp.Optional[tp.List[int]]

    # clusters: Series  # MultiIndex: ['roi', 'obj_id']

    file_types = ["cell_type_assignments"]

    def __init__(
        self,
        sample_name: str = DEFAULT_SAMPLE_NAME,
        root_dir: tp.Optional[Path] = None,
        metadata: tp.Optional[tp.Union[Path, DataFrame]] = None,
        subfolder_per_sample: bool = True,
        roi_name_atribute: str = DEFAULT_ROI_NAME_ATTRIBUTE,
        roi_number_atribute: str = DEFAULT_ROI_NUMBER_ATTRIBUTE,
        panel_metadata: tp.Optional[tp.Union[Path, DataFrame]] = None,
        channel_labels: tp.Optional[Series] = None,
        prj: tp.Optional["_project.Project"] = None,
        **kwargs,
    ):
        self.name: str = str(sample_name)
        self.sample_name: str = sample_name
        self.root_dir = Path(root_dir).absolute() if root_dir is not None else None
        self.metadata: tp.Optional[DataFrame] = (
            pd.read_csv(metadata) if isinstance(metadata, (str, Path)) else metadata
        )
        self.subfolder_per_sample = subfolder_per_sample
        self.roi_name_atribute = roi_name_atribute
        self.roi_number_atribute = roi_number_atribute
        self.panel_metadata: tp.Optional[DataFrame] = (
            pd.read_csv(panel_metadata, index_col=0)
            if isinstance(panel_metadata, (str, Path))
            else panel_metadata
        )
        # TODO: make sure channel labels conform to internal specification: "Label(Metal\d+)"
        self._channel_labels: tp.Optional[Series] = (
            pd.read_csv(channel_labels, index_col=0, squeeze=True)
            if isinstance(channel_labels, (str, Path))
            else channel_labels
        )
        self.prj = prj

        self.anndata: tp.Optional[AnnData] = None
        self._clusters: tp.Optional[MultiIndexSeries] = None
        self.quantification = None

        # Add kwargs as attributes
        self.__dict__.update(kwargs)

        if "__no_init__" in kwargs:
            return

        # initialize
        self.rois: tp.List[_roi.ROI] = list()
        self._initialize_sample_from_annotation(values_to_propagate=kwargs.keys())

    def __repr__(self):
        r = len(self.rois)
        return f"Sample '{self.name}' with {r} ROI" + ("" if r == 1 else "s")

    def __getitem__(self, item: int) -> _roi.ROI:
        return self.rois[item]

    def __iter__(self) -> tp.Iterator[_roi.ROI]:
        return iter(self.rois)

    def __len__(self) -> int:
        return len(self.rois)

    @classmethod
    def from_stacks(cls, tiffs: tp.Union[Path, tp.Sequence[Path]], **kwargs) -> IMCSample:
        if isinstance(tiffs, Path):
            tiffs = [tiffs]
        # TODO: assumes all ROIs are from same sample
        rois = [ROI.from_stack(tiff) for tiff in tiffs]
        return IMCSample.from_rois(rois, **kwargs)

    @classmethod
    def from_rois(
        cls, rois: tp.Union[_roi.ROI, tp.Sequence[_roi.ROI]], **kwargs
    ) -> IMCSample:
        if isinstance(rois, _roi.ROI):
            rois = [rois]
        # TODO: assumes all ROIs are from same sample
        name = rois[0].name.split("-")[0]
        if rois[0].root_dir is not None:
            root_dir = rois[0].root_dir.parent
        return IMCSample(sample_name=name, root_dir=root_dir, __no_init__=True, **kwargs)

    def _detect_rois(self) -> DataFrame:
        if self.root_dir is None:
            print(
                f"Sample does not have `root_dir`. Cannot find ROIs for sample '{self.name}'."
            )
            return pd.DataFrame()

        content = (
            self.root_dir.glob(self.name + "*_full.tiff")
            if not self.subfolder_per_sample
            else (self.root_dir / "tiffs").glob("*_full.tiff")
        )
        df = pd.Series(content).to_frame()
        if df.empty:
            print(f"Could not find ROIs for sample '{self.name}'.")
            return df
        df[DEFAULT_ROI_NAME_ATTRIBUTE] = df[0].apply(
            lambda x: x.name.replace("_full.tiff", "")
        )
        try:
            df[DEFAULT_ROI_NUMBER_ATTRIBUTE] = (
                df[DEFAULT_ROI_NAME_ATTRIBUTE].str.extract(r"-(\d+)$")[0].astype(int)
            )
        except ValueError:
            pass
        df = df.sort_values(df.columns.tolist(), ignore_index=True).drop(0, axis=1)
        return df

    def _initialize_sample_from_annotation(
        self, toggle: bool = None, values_to_propagate: tp.Sequence[str] = []
    ) -> None:
        if self.metadata is None:
            metadata = self._detect_rois()
        else:
            metadata = pd.DataFrame(self.metadata)  # this makes the type explicit
        if toggle:
            metadata = metadata[metadata[DEFAULT_TOGGLE_ATTRIBUTE]]

        has_numbers = self.roi_number_atribute in metadata.columns

        for i, (_, row) in enumerate(metadata.iterrows(), 1):
            roi = ROI(
                name=row[self.roi_name_atribute],
                roi_number=row[self.roi_number_atribute] if has_numbers else i,
                root_dir=self.root_dir / "tiffs"
                if self.subfolder_per_sample
                else self.root_dir,
                sample=self,
                prj=self.prj,
                **row.drop(
                    [DEFAULT_ROI_NAME_ATTRIBUTE, DEFAULT_ROI_NUMBER_ATTRIBUTE],
                    errors="ignore",
                ).to_dict(),
                **{k: getattr(self, k) for k in values_to_propagate},
            )
            self.rois.append(roi)

    @property
    def n_rois(self) -> int:
        return len(self.rois)

    @property
    def roi_names(self) -> tp.List[str]:
        return [r.name for r in self]

    @property
    def channel_labels(self) -> tp.Union[Series, DataFrame]:
        labels = pd.DataFrame(
            [roi.channel_labels.rename(roi.name) for roi in self.rois]
        ).T.rename_axis(index="channel", columns="roi")
        if (labels.apply(pd.Series.nunique, axis=1) == 1).all():
            return labels.iloc[:, 0].rename(self.name)
        return labels

    @property
    def channel_names(self) -> tp.Union[Series, DataFrame]:
        names = pd.DataFrame(
            [roi.channel_names.rename(roi.name) for roi in self.rois]
        ).T.rename_axis(index="channel", columns="roi")
        if (names.apply(pd.Series.nunique, axis=1) == 1).all():
            return names.iloc[:, 0].rename(self.name)
        return names

    @property
    def channel_metals(self) -> tp.Union[Series, DataFrame]:
        metals = pd.DataFrame(
            [roi.channel_metals.rename(roi.name) for roi in self.rois]
        ).T.rename_axis(index="channel", columns="roi")
        if (metals.apply(pd.Series.nunique, axis=1) == 1).all():
            return metals.iloc[:, 0].rename(self.name)
        return metals

    @property
    def clusters(self) -> MultiIndexSeries:
        if self._clusters is not None:
            return self._clusters
        try:
            self.prj.set_clusters(samples=[self])
        except KeyError:
            self._clusters = pd.read_csv(
                self.get_input_filename("cell_type_assignments"),
                index_col=[0, 1, 2],
            ).loc[self.name]
            self.set_clusters(self._clusters)
        return self._clusters

    def get_input_filename(self, input_type: str) -> Path:
        """Get path to file with data for Sample.

        Available `input_type` values are:
            - "cell_type_assignments": CSV file with cell type assignemts for each cell and each ROI
        """
        to_read = {
            # "cell": ("cpout", "cell.csv"),
            # "relationships": ("cpout", "Object relationships.csv"),
            "cell_type_assignments": (
                "single_cell",
                ".cell_type_assignment_against_reference.csv",
            ),
            "anndata": ("single_cell", ".single_cell.processed.h5ad"),
        }
        dir_, suffix = to_read[input_type]
        return self.root_dir / dir_ / (self.name + suffix)

    def get(self, attr):
        try:
            return self.__getattribute(attr)
        except AttributeError:
            return None

    def read_all_inputs(
        self,
        rois: tp.Sequence[_roi.ROI] = None,
        only_these_keys: tp.Sequence[str] = None,
        permissive: bool = False,
        set_attribute: bool = True,
        # overwrite: bool = False,
        **kwargs,
    ) -> tp.Optional[tp.Dict[str, DataFrame]]:
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
                _path = self.get_input_filename(ftype)
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
        self, channel: tp.Union[str, int], rois: tp.Sequence[_roi.ROI] = None
    ) -> Figure:  # tp.List[ROI]
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

    def plot_channels(
        self,
        channels: tp.Sequence[str] = ["mean"],
        merged: bool = False,
        rois: tp.Optional[tp.List[_roi.ROI]] = None,
        per_roi: bool = False,
        save: bool = False,
        output_dir: tp.Optional[Path] = None,
        **kwargs,
    ) -> Figure:
        """
        Plot a list of channels for all ROIs.
        """
        rois = rois or self.rois
        if isinstance(channels, str):
            channels = [channels]
        if save:
            output_dir = Path(output_dir or self.prj.results_dir / "qc")
            output_dir.mkdir(exist_ok=True)
            channels_str = ",".join(channels)
            fig_file = output_dir / ".".join([self.name, f"all_rois.{channels_str}.pdf"])
        if per_roi:
            for roi in rois:
                fig = roi.plot_channels(channels, merged=merged, **kwargs)
                if save:
                    fig_file = output_dir / ".".join(
                        [self.name, roi.name, channels_str, "pdf"]
                    )
                    fig.savefig(fig_file, **FIG_KWS)
        else:
            i = 0
            j = 1 if merged else len(channels)
            n, m = get_grid_dims(len(rois)) if merged else get_grid_dims(len(rois) * j)

            fig, axes = plt.subplots(n, m, figsize=(4 * m, 4 * n))
            axes = axes.flatten()
            for roi in rois:
                roi.plot_channels(channels, axes=axes[i : i + j], merged=merged, **kwargs)
                i += j
            for _ax in axes[i:]:
                _ax.axis("off")
            if save:
                fig.savefig(fig_file, **FIG_KWS)
        return fig

    def plot_cell_types(
        self,
        rois: tp.List[_roi.ROI] = None,
        cell_type_combinations: tp.Union[str, tp.List[tp.Tuple[str, str]]] = None,
        cell_type_assignments: DataFrame = None,
        palette: tp.Optional[str] = None,
    ) -> Figure:
        rois = rois or self.rois

        n = len(cell_type_combinations or [1])
        m = len(rois)
        fig, axes = plt.subplots(n, m, figsize=(3 * m, 3 * n), squeeze=False)
        patches: tp.List[Patch] = list()
        for _ax, roi in zip(np.hsplit(axes, m), rois):
            patches += roi.plot_cell_types(
                cell_type_combinations=cell_type_combinations,
                cell_type_assignments=cell_type_assignments,
                palette=palette,
                ax=_ax,
            )
        add_legend(patches, axes[0, -1])
        return fig

    def plot_probabilities_and_segmentation(
        self, rois: tp.List[_roi.ROI] = None, add_scale: bool = True
    ) -> Figure:
        n = len(rois or self.rois)
        fig, axes = plt.subplots(
            n,
            5,
            figsize=(5 * 4, n * 4),
            gridspec_kw=dict(wspace=0.05),
            sharex="row",
            sharey="row",
            squeeze=False,
        )
        fig.suptitle(self.name)
        for i, roi in enumerate(self.rois):
            roi.plot_probabilities_and_segmentation(axes=axes[i], add_scale=add_scale)
        return fig

    def cell_to_anndata(
        self, red_func: str = "mean", set_attribute: bool = False, **kwargs
    ) -> AnnData:
        _df = self.quantify_cell_intensity(func=red_func, **kwargs)
        _an = AnnData(_df.drop("roi", axis=1).sort_index(axis=1))
        _an.obs["roi"] = pd.Categorical(_df["roi"].values)
        _an.raw = _an

        if set_attribute:
            self.anndata = _an
        return _an

    def quantify_cells(
        self,
        intensity: bool = True,
        morphology: bool = True,
        set_attribute: bool = True,
        samples: tp.List["IMCSample"] = None,
        rois: tp.List[_roi.ROI] = None,
    ) -> DataFrame:
        """
        Measure the intensity of each channel in each single cell.
        """
        from imc.ops.quant import quantify_cells_rois

        quantification = quantify_cells_rois(rois or self.rois, intensity, morphology)
        if set_attribute:
            self.quantification = quantification
        return quantification

    def quantify_cell_intensity(
        self,
        rois: tp.Sequence[_roi.ROI] = None,
        **kwargs,
    ) -> DataFrame:
        """
        Measure the intensity of each channel in each single cell.
        """
        from imc.ops.quant import quantify_cell_intensity_rois

        return quantify_cell_intensity_rois(rois or self.rois, **kwargs)

    def quantify_cell_morphology(
        self,
        rois: tp.Sequence[_roi.ROI] = None,
        **kwargs,
    ) -> DataFrame:
        """
        Measure the shape parameters of each single cell.
        """
        from imc.ops.quant import quantify_cell_morphology_rois

        return quantify_cell_morphology_rois(rois or self.rois, **kwargs)

    def cluster_cells(
        self,
        output_prefix: Path = None,
        plot: bool = True,
        set_attribute: bool = True,
        rois: tp.Sequence[_roi.ROI] = None,
        **kwargs,
    ) -> tp.Optional[Series]:
        """
        Derive clusters of single cells based on their channel intensity.
        """
        from imc.ops.clustering import single_cell_analysis

        output_prefix = Path(output_prefix or self.root_dir / "single_cell" / self.name)

        if "quantification" not in kwargs and self.quantification is not None:
            kwargs["quantification"] = self.quantification
        if "cell_type_channels" not in kwargs and self.panel_metadata is not None:
            if "cell_type" in self.panel_metadata.columns:
                kwargs["cell_type_channels"] = self.panel_metadata.query(
                    "cell_type == 1"
                ).index.tolist()

        clusters = single_cell_analysis(
            output_prefix=output_prefix,
            rois=rois or self.rois,
            plot=plot,
            **kwargs,
        )
        # save clusters as CSV in default file
        clusters.reset_index().to_csv(
            self.get_input_filename("cell_type_assignments"), index=False
        )
        if not set_attribute:
            return clusters

        # Set clusters for project and propagate for Samples and ROIs.
        # in principle there was no need to pass clusters here as it will be read
        # however, the CSV roundtrip might give problems in edge cases, for
        # example when the sample name is only integers
        self.set_clusters(clusters.astype(str).loc[self.name])
        return None

    def set_clusters(
        self,
        clusters: MultiIndexSeries = None,
        rois: tp.List[_roi.ROI] = None,
    ) -> None:
        id_cols = ["roi", "obj_id"]
        if clusters is None:
            self.prj.set_clusters(samples=[self])
        else:
            assert isinstance(clusters.index, pd.MultiIndex)
            assert clusters.index.names == id_cols
            self._clusters = clusters
        for roi in rois or self.rois:
            roi.set_clusters(clusters=self.clusters.loc[roi.name].squeeze())

    def predict_cell_types_from_reference(self, **kwargs) -> None:
        from imc.ops.clustering import predict_cell_types_from_reference

        predict_cell_types_from_reference(self, **kwargs)

    def cell_type_adjancency(
        self,
        rois: tp.List[_roi.ROI] = None,
        output_prefix: Path = None,
    ) -> None:
        rois = rois or self.rois

        output_prefix = output_prefix or self.root_dir / "single_cell" / (
            self.name + ".cluster_adjacency_graph."
        )

        # TODO: check default input
        # Plot adjancency for all ROIs next to each other and across
        adj_matrices = pd.concat(
            [
                pd.read_csv(
                    f"{output_prefix}{roi.name}.norm_over_random.csv",
                    index_col=0,
                ).assign(roi=roi.roi_number)
                for roi in rois
            ]
        )
        # g2 = nx.readwrite.read_gpickle(roi_prefix + "neighbor_graph.gpickle")

        mean_ = (
            adj_matrices.drop("roi", axis=1)
            .groupby(level=0)
            .mean()
            .sort_index(0)
            .sort_index(1)
        )
        adj_matrices = adj_matrices.append(mean_.assign(roi="mean"))

        m = self.n_rois + 1
        nrows = 3
        fig, _ax = plt.subplots(
            nrows, m, figsize=(4 * m, 4 * nrows), sharex=False, sharey=False
        )
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
