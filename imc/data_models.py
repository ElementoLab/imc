#! /usr/bin/env python

from __future__ import annotations  # this will fix the type annotatiton of not yet undefined classes
import os
import re

from typing import Dict, Tuple, List, Sequence, Optional, Union, Any  # , cast

import numpy as np
import pandas as pd
import scipy
import scipy.ndimage as ndi
import multiprocessing
import parmap

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from anndata import AnnData
import networkx as nx
from skimage.exposure import equalize_hist as eq
from skimage.segmentation import clear_border

from imc.types import (
    Path,
    Figure, Axis, Patch,
    Array, DataFrame, Series, MultiIndexSeries
)
# from imc import LOGGER
from imc.operations import (
    parse_acquisition_metadata,
    derive_reference_cell_type_labels,
    quantify_cells,
    measure_cell_attributes,
    single_cell_analysis,
    predict_cell_types_from_reference,
    get_adjacency_graph,
    measure_cell_type_adjacency,
    cluster_communities
)
from imc.utils import (
    read_image_from_file,
    sorted_nicely,
    minmax_scale
)
from imc.graphics import (
    colorbar_decorator, add_scale as _add_scale,
    get_grid_dims,
    get_transparent_cmaps,
    add_legend,
    share_axes_by,
    cell_labels_to_mask,
    numbers_to_rgb_colors,
)
from imc.exceptions import cast  # TODO: replace with typing.cast

FIG_KWS = dict(dpi=300, bbox_inches="tight")
sns.clustermap = colorbar_decorator(sns.clustermap)

DEFAULT_PROJECT_NAME = "project"
DEFAULT_SAMPLE_NAME_ATTRIBUTE = "sample_name"
DEFAULT_SAMPLE_GROUPING_ATTRIBUTEs = [DEFAULT_SAMPLE_NAME_ATTRIBUTE]
DEFAULT_TOGGLE_ATTRIBUTE = "toggle"
DEFAULT_PROCESSED_DIR_NAME = Path("processed")
DEFAULT_PRJ_SINGLE_CELL_DIR = Path("single_cell")

DEFAULT_ROI_NAME_ATTRIBUTE = "roi_name"
DEFAULT_ROI_NUMBER_ATTRIBUTE = "roi_number"

# processed directory structure
SUBFOLDERS_PER_SAMPLE = True
ROI_STACKS_DIR = Path("tiffs")
ROI_MASKS_DIR = Path("tiffs")
ROI_UNCERTAINTY_DIR = Path("uncertainty")
ROI_SINGLE_CELL_DIR = Path("single_cell")


# def cast(arg: Optional[GenericType], name: str, obj: str) -> GenericType:
#     """Remove `Optional` from `T`."""
#     if arg is None:
#         raise AttributeNotSetError(f"Attribute '{name}' of '{obj}' cannot be None!")
#     return arg


class Project:
    """
    A class to model a IMC project.

    Parameters
    ----------
    name : :obj:`str`
        Project name. Defaults to "project".
    csv_metadata : :obj:`str`
        Path to CSV metadata sheet.

    Attributes
    ----------
    name : :obj:`str`
        Project name
    csv_metadata : :obj:`str`
        Path to CSV metadata sheet.
    metadata : :class:`pandas.DataFrame`
        Metadata dataframe
    samples : List[:class:`IMCSample`]
        List of IMC sample objects.
    """

    # name: str
    # processed_dir: str
    # sample_name_attribute: str
    # toggle: bool
    # csv_metadata: Optional[str]
    # samples: List["IMCSample"]
    # rois: List["ROI"]  # @property
    def __init__(
        self,
        name: str = DEFAULT_PROJECT_NAME,
        sample_metadata: Optional[Union[Path, DataFrame]] = None,
        sample_name_attribute: str = DEFAULT_SAMPLE_NAME_ATTRIBUTE,
        sample_grouping_attributes: Optional[List[str]] = DEFAULT_SAMPLE_GROUPING_ATTRIBUTEs,
        panel_metadata: Optional[Union[Path, DataFrame]] = None,
        channel_labels: Optional[Union[Path, Series]] = None,
        toggle: bool = True,
        processed_dir: Path = DEFAULT_PROCESSED_DIR_NAME,
        **kwargs,
    ):
        # Initialize
        self.name = name
        self.sample_metadata = (
            pd.read_csv(sample_metadata)
            if isinstance(sample_metadata, (str, Path))
            else sample_metadata
        )
        self.samples: List["IMCSample"] = list()
        self.sample_name_attribute = sample_name_attribute
        self.sample_grouping_attributes = sample_grouping_attributes
        self.panel_metadata: Optional[DataFrame] = (
            pd.read_csv(panel_metadata, index_col=0)
            if isinstance(panel_metadata, (str, Path))
            else panel_metadata
        )
        # TODO: make sure channel labels conform to internal specification: "Label(Metal\d+)"
        self.channel_labels: Optional[Series] = (
            pd.read_csv(channel_labels, index_col=0, squeeze=True)
            if isinstance(channel_labels, (str, Path))
            else channel_labels
        )

        self.toggle = toggle
        self.processed_dir = Path(processed_dir).absolute()
        self.quantification: Optional[DataFrame] = None
        self._clusters: Optional[MultiIndexSeries] = None  # MultiIndex: ['sample', 'roi', 'obj_id']

        if not hasattr(self, "samples"):
            self.samples = list()
        if self.sample_metadata is not None:
            self._initialize_project_from_annotation(**kwargs)

    def __repr__(self):
        return f"Project '{self.name}' with {len(self.samples)} samples"

    def _initialize_project_from_annotation(
        self,
        toggle: Optional[bool] = None,
        sample_grouping_attributes: Optional[List[str]] = None,
        **kwargs,
    ):
        def cols_with_unique_values(dfs) -> set:
            return {col for col in dfs if len(dfs[col].unique()) == 1}

        if (toggle or self.toggle) and ("toggle" in self.sample_metadata.columns):
            # TODO: logger.info("Removing samples without toggle active")
            self.sample_metadata = self.sample_metadata[
                self.sample_metadata[DEFAULT_TOGGLE_ATTRIBUTE]
            ]

        sample_grouping_attributes = (
            sample_grouping_attributes
            or self.sample_grouping_attributes
            or self.sample_metadata.columns.tolist()
        )

        for _, idx in self.sample_metadata.groupby(sample_grouping_attributes).groups.items():
            rows = self.sample_metadata.loc[idx]
            const_cols = cols_with_unique_values(rows)
            row = rows[const_cols].drop_duplicates().squeeze()

            sample = IMCSample(
                sample_name=row[self.sample_name_attribute],
                root_dir=(self.processed_dir / str(row[self.sample_name_attribute]))
                if SUBFOLDERS_PER_SAMPLE
                else self.processed_dir,
                csv_metadata=rows,
                **kwargs,
                **row.drop("sample_name", errors="ignore").to_dict(),
            )
            sample.prj = self
            for roi in sample.rois:
                roi.prj = self
                # If channel labels are given, add them to all ROIs
                roi._channel_labels = self.channel_labels
            self.samples.append(sample)

    def _get_input_filename(self, input_type: str) -> Path:
        """Get path to file with data for Sample.

        Available `input_type` values are:
            - "cell_type_assignments": CSV file with cell type assignemts for each cell and each ROI
        """
        to_read = {
            "h5ad": (DEFAULT_PRJ_SINGLE_CELL_DIR, ".single_cell.processed.h5ad"),
            "cell_cluster_assignments": (
                DEFAULT_PRJ_SINGLE_CELL_DIR,
                ".single_cell.cluster_assignments.csv",
            ),
        }
        dir_, suffix = to_read[input_type]
        return self.processed_dir / dir_ / (self.name + suffix)

    @property
    def rois(self) -> List["ROI"]:
        """
        Return a list of all ROIs of the project samples.
        """
        return [roi for sample in self.samples for roi in sample.rois]

    def plot_channels(
        self,
        channels: List[str] = ["mean"],
        samples: Optional[List["IMCSample"]] = None,
        per_sample: bool = False,
        output_dir: Optional[Path] = None,
    ) -> Figure:
        """
        Plot a list of channels for all Samples/ROIs.
        """
        output_dir = Path(output_dir or self.processed_dir / "qc")
        os.makedirs(output_dir, exist_ok=True)
        if per_sample:
            for sample in samples or self.samples:
                fig = sample.plot_channels(channels)
                # fig.savefig(
                #     output_dir / ".".join([self.name, sample.name, "all_rois.channel_mean.svg"]),
                #     **FIG_KWS,
                # )
        else:
            rois = [roi for sample in samples or self.samples for roi in getattr(sample, "rois")]
            __n, __m = get_grid_dims(len(rois))
            fig, axes = plt.subplots(__n, __m, figsize=(4 * __m, 4 * __n))
            axes = axes.flatten()
            i = 0
            j = len(channels)
            for roi in rois:
                roi.plot_channels(channels, axes=axes[i : i + j])
                i += 1
            for _ax in axes[i:]:
                _ax.axis("off")
            # fig.savefig(output_dir / ".".join([self.name, "all_rois.channel_mean.svg"]), **FIG_KWS)
        return fig

    # TODO: write decorator to get/set default outputdir and handle dir creation
    def plot_probabilities_and_segmentation(
        self,
        samples: Optional[List["IMCSample"]] = None,
        jointly: bool = False,
        output_dir: Optional[Path] = None,
    ):
        samples = samples or self.samples
        for sample in samples:
            sample.read_all_inputs(only_these_keys=["probabilities", "cell_mask", "nuclei_mask"])
        output_dir = Path(output_dir or self.processed_dir / "qc")
        os.makedirs(output_dir, exist_ok=True)
        if not jointly:
            for sample in samples:
                plot_file = output_dir / ".".join(
                    [self.name, sample.name, "all_rois.plot_probabilities_and_segmentation.svg"]
                )
                fig = sample.plot_probabilities_and_segmentation()
                fig.savefig(plot_file, **FIG_KWS)
        else:
            rois = [roi for sample in samples for roi in sample.rois]
            __n = len(rois)
            fig, axes = plt.subplots(__n, 5, figsize=(4 * 5, 4 * __n))
            for i, roi in enumerate(rois):
                roi.plot_probabilities_and_segmentation(axes=axes[i])
            plot_file = output_dir / (
                self.name + ".all_rois.plot_probabilities_and_segmentation.all_rois.svg"
            )
            fig.savefig(plot_file, **FIG_KWS)

    def plot_cell_types(
        self,
        samples: Optional[List["IMCSample"]] = None,
        rois: Optional[List["ROI"]] = None,
        cell_type_combinations: Optional[Union[str, List[Tuple[str, str]]]] = None,
        cell_type_assignments: Optional[DataFrame] = None,
        palette: Optional[str] = "tab20",
    ):
        # TODO: fix compatibility of `cell_type_combinations`.
        samples = samples or self.samples
        rois = rois or self.rois

        __n = len(samples)
        __m = max([sample.n_rois for sample in samples])
        fig, axes = plt.subplots(__n, __m, figsize=(3 * __m, 3 * __n), squeeze=False)
        patches: List[Patch] = list()
        for i, sample in enumerate(samples):
            for j, roi in enumerate([roi for roi in rois if roi in sample.rois]):
                patches += roi.plot_cell_types(
                    cell_type_combinations=cell_type_combinations,
                    cell_type_assignments=cell_type_assignments,
                    palette=palette,
                    ax=axes[i, j],
                )
        add_legend(patches, axes[0, -1])
        for ax in axes.flatten():
            ax.axis("off")
        return fig

    def channel_summary(
        self,
        samples: Optional[List["IMCSample"]] = None,
        rois: Optional[List["ROI"]] = None,
        red_func: str = "mean",
        channel_blacklist: Optional[List[str]] = None,
        plot: bool = True,
        **kwargs,
    ) -> Union[DataFrame, Tuple[DataFrame, Figure]]:
        _res = dict()
        # for sample, _func in zip(samples or self.samples, red_func):
        samples = samples or self.samples
        rois = [r
                for sample in (samples or self.samples)
                for r in sample.rois if r in (rois or sample.rois)]
        for roi in rois:
            _res[roi.name] = pd.Series(
                getattr(roi.stack, red_func)(axis=(1, 2)), index=roi.channel_labels
            )
        res = pd.DataFrame(_res)
        # filter channels out if requested
        if channel_blacklist:
            res = res.loc[res.index[~res.index.str.contains("|".join(channel_blacklist))]]
        res = res / res.mean()

        if plot:
            res = np.log1p(res)
            # calculate mean intensity
            channel_mean = res.mean(axis=1).rename("channel_mean")

            # calculate cell density
            cell_density = pd.Series(
                [roi.cells_per_area_unit() for roi in rois],
                index=[roi.name for roi in rois], name="cell_density")
            if all(cell_density < 0):
                cell_density *= 1000

            def_kwargs = dict(z_score=0, center=0, robust=True, cmap="RdBu_r")
            def_kwargs.update(kwargs)
            # TODO: add {row,col}_colors colorbar to heatmap
            for kws, label, cbar_label in [({}, "", ""), (def_kwargs, ".z_score", " (Z-score)")]:
                plot_file = self.processed_dir \
                    / "qc" / self.name + f".mean_per_channel.clustermap{label}.svg"
                grid = sns.clustermap(
                    res, cbar_kws=dict(label=red_func.capitalize() + cbar_label),
                    row_colors=channel_mean, row_colors_cmap="Greens",
                    col_colors=cell_density, col_colors_cmap="BuPu",
                    metric="correlation",
                    xticklabels=True, yticklabels=True,
                    **kws)
                grid.fig.suptitle("Mean channel intensities")
                grid.savefig(plot_file, dpi=300, bbox_inches="tight")
            grid.fig.grid = grid
            return (res, grid.fig)
        return res

    def image_summary(
            self,
            samples: Optional[List["IMCSample"]] = None,
            rois: List["ROI"] = None
        ):
        raise NotImplementedError
        from imc.utils import lacunarity, fractal_dimension
        rois = [r for r in (rois or self.rois) if r.sample in (samples or self.samples)]
        roi_names = [r.name for r in rois]
        densities = pd.Series(
            {roi.name: roi.cells_per_area_unit() for roi in rois},
            name="cell density")
        lacunarities = pd.Series(
            parmap.map(lacunarity, [roi.cell_mask_o for roi in rois], pm_pbar=True),
            index=roi_names, name="lacunarity")
        fractal_dimensions = pd.Series(
            parmap.map(fractal_dimension, [roi.cell_mask_o for roi in rois], pm_pbar=True),
            index=roi_names, name="fractal_dimension")

        morphos = pd.DataFrame([densities * 1e4, lacunarities, fractal_dimensions]).T

    def channel_correlation(
            self,
            samples: Optional[List["IMCSample"]] = None,
            rois: Optional[List['ROI']] = None
        ) -> Figure:
        """
        Observe the pairwise correlation of channels across ROIs.
        """
        from imc.operations import _correlate_channels__roi
        rois = [r
                for sample in (samples or self.samples)
                for r in sample.rois if r in (rois or sample.rois)]
        res = parmap.map(_correlate_channels__roi, rois, pm_pbar=True)

        labels = rois[0].channel_labels
        xcorr = pd.DataFrame(
            np.asarray(res).mean(0),
            index=labels, columns=labels)

        grid = sns.clustermap(
            xcorr,
            cmap="RdBu_r",
            center=0,
            robust=True,
            metric="correlation",
            cbar_kws=dict(label="Pearson correlation"),
        )
        grid.ax_col_dendrogram.set_title("Pairwise channel correlation\n(pixel level)")
        grid.savefig(self.processed_dir / "qc" / "channel_pairwise_correlation.svg", **FIG_KWS)
        grid.fig.grid = grid
        return grid.fig

    def quantify_cells(
            self,
            samples: Optional[List["IMCSample"]] = None,
            rois: Optional[List["ROI"]] = None,
            morphology: bool = False,
            set_attribute: bool = True,
        ) -> Optional[DataFrame]:
        """
        Measure the intensity of each channel in each single cell.
        """
        if morphology:
            quantification = pd.concat([
                self.quantify_cell_intensity(samples=samples, rois=rois)
                    .drop(['sample', 'roi'], axis=1),
                self.quantify_cell_morphology(samples=samples, rois=rois)], axis=1)
        else:
            quantification = self.quantify_cell_intensity(samples=samples, rois=rois)
        if not set_attribute:
            return quantification
        else:
            self.quantification = quantification
            return None

    def quantify_cell_intensity(
            self,
            samples: Optional[List["IMCSample"]] = None,
            rois: Optional[List["ROI"]] = None,
            **kwargs):
        """
        Measure the intensity of each channel in each single cell.
        """
        from imc.operations import _quantify_cell_intensity__roi
        return pd.concat(
            parmap.map(
                _quantify_cell_intensity__roi,
                [r
                    for sample in (samples or self.samples)
                    for r in sample.rois if r in (rois or sample.rois)],
                pm_pbar=True,
                **kwargs
            )
        )

    def quantify_cell_morphology(
            self,
            samples: Optional[List["IMCSample"]] = None,
            rois: Optional[List["ROI"]] = None,
            **kwargs):
        """
        Measure the shape parameters of each single cell.
        """
        from imc.operations import _quantify_cell_morphology__roi
        return pd.concat(
            parmap.map(
                _quantify_cell_morphology__roi,
                [r
                    for sample in (samples or self.samples)
                    for r in sample.rois if r in (rois or sample.rois)],
                pm_pbar=True,
                **kwargs
            )
        )

    def cluster_cells(
        self,
        samples: Optional[List["IMCSample"]] = None,
        rois: Optional[List["ROI"]] = None,
        output_prefix: Optional[Path] = None,
        plot: bool = True,
        set_attribute: bool = True,
        **kwargs,
    ) -> Optional[Series]:
        """
        Derive clusters of single cells based on their channel intensity.
        """
        output_prefix = Path(
            output_prefix
            or self.processed_dir / "single_cell" / self.name)

        quantification = None
        if 'quantification' in kwargs:
            quantification = kwargs['quantification']
            del kwargs['quantification']
        cell_type_channels = None
        if 'cell_type_channels' in kwargs:
            cell_type_channels = kwargs['cell_type_channels']
            del kwargs['cell_type_channels']
        if self.panel_metadata is not None:
            if "cell_type" in self.panel_metadata.columns:
                cell_type_channels = self.panel_metadata.query("cell_type == 1").index.tolist()

        clusters = single_cell_analysis(
            output_prefix=output_prefix,
            rois=[r for r in (rois or self.rois) if r.sample in (samples or self.samples)],
            quantification=quantification,
            cell_type_channels=cell_type_channels,
            plot=plot,
            **kwargs,
        )
        # save clusters as CSV in default file
        clusters.reset_index().to_csv(
            self._get_input_filename("cell_cluster_assignments"), index=False)
        if not set_attribute:
            return clusters

        # set clusters for project and propagate for Samples and ROIs
        # in principle there was no need to pass clusters here as it will be read
        # however, the CSV serialization might give problems in edge cases, for
        # example when the sample name is only integers
        self.set_clusters(clusters)
        return None

    @property
    def clusters(self):
        if self._clusters is not None:
            return self._clusters
        else:
            self.set_clusters()
            return self._clusters

    def set_clusters(
        self,
        clusters: Optional[MultiIndexSeries] = None,
        samples: Optional[List["IMCSample"]] = None,
        write_to_disk: bool = False,
    ) -> None:
        """
        Set the `clusters` attribute of the project and
        propagate it to the Samples and their ROIs.

        If not given, `clusters` is the output of
        :func:`Project._get_input_filename`("cell_cluster_assignments").
        """
        id_cols = ["sample", "roi", "obj_id"]
        if clusters is None:
            clusters = (
                pd.read_csv(
                    self._get_input_filename("cell_cluster_assignments"),
                    dtype={"sample": str, "roi": str},
                ).set_index(id_cols)
            )["cluster"]  # .astype(str)
        assert isinstance(clusters.index, pd.MultiIndex)
        assert clusters.index.names == id_cols
        self._clusters = clusters
        for sample in samples or self.samples:
            sample.set_clusters(clusters=clusters.loc[sample.name])
        if write_to_disk:
            self._clusters.reset_index().to_csv(
                self._get_input_filename("cell_cluster_assignments"),
                index=False)

    def label_clusters(
        self,
        h5ad_file: Optional[Path] = None,
        output_prefix: Optional[Path] = None,
        **kwargs) -> None:
        """
        Derive labels for each identified cluster
        based on its most abundant markers.
        """
        prefix = self.processed_dir / "single_cell" / self.name
        h5ad_file = Path(h5ad_file or prefix + ".single_cell.processed.h5ad")
        output_prefix = Path(output_prefix or prefix + ".cell_type_reference")
        new_labels = derive_reference_cell_type_labels(h5ad_file, output_prefix, **kwargs)
        self._rename_clusters(new_labels.to_dict())

    def _rename_clusters(self, new_labels: dict, save: bool = True):
        clusters = cast(self.clusters).replace(new_labels)
        if save:
            clusters.reset_index().to_csv(
                self._get_input_filename("cell_cluster_assignments"), index=False)
        self.set_clusters(clusters)

    def sample_comparisons(
        self,
        samples: Optional[List["IMCSample"]] = None,
        sample_attributes: Optional[List[str]] = None,
        output_prefix: Optional[Path] = None,
        cell_type_percentage_threshold: float = 1.0,
    ):
        # TODO: revamp/separate into smaller functions
        import itertools
        from scipy.stats import mannwhitneyu
        from statsmodels.stats.multitest import multipletests

        sample_attributes = sample_attributes or ['name']
        samples = samples or self.samples
        rois = [roi for sample in samples for roi in sample.rois]
        output_prefix = output_prefix or self.processed_dir / "single_cell" / self.name + "."
        output_prefix.parent.mkdir(exist_ok=True)

        # group samples by desired attributes
        sample_df = pd.DataFrame({
                k: v for k, v in sample.__dict__.items()
                if isinstance(v, str)
            } for sample in samples)[['name'] + sample_attributes].set_index('name').rename_axis('sample').reset_index()
        sample_groups = (
            sample_df
            .groupby(sample_attributes)['sample']
            .apply(set)
        )
        sample_roi_df = pd.DataFrame([(roi.name, roi.sample.name) for roi in rois], columns=['roi', 'sample'])

        # Whole channel means
        channel_means: DataFrame = self.channel_summary(plot=False)
        channel_means = channel_means.reset_index().melt(id_vars='channel', var_name="roi").reset_index(drop=True)
        channel_df = channel_means.merge(sample_roi_df).merge(sample_df).sort_values(sample_attributes)

        # cell type abundancy per sample or group of samples
        cluster_counts = self.clusters.groupby(level=['sample', 'roi']).value_counts().rename("cell_count")
        cluster_perc = (cluster_counts.groupby('cluster').sum() / cluster_counts.sum()) * 100
        filtered_clusters = cluster_perc[cluster_perc > cell_type_percentage_threshold].index

        # # absolute
        # # fraction of total
        cluster_df = cluster_counts.reset_index().merge(sample_df).sort_values(sample_attributes)
        cluster_df['cell_perc'] = cluster_df.groupby('roi')['cell_count'].apply(lambda x: (x / x.sum()) * 100)


        # Test difference between channels/clusters
        # # channels
        _res = list()
        for attribute in sample_attributes:
            for channel in channel_df['channel'].unique():
                for group1, group2 in itertools.permutations(channel_df[attribute].unique(), 2):
                    __a = channel_df.query(f"channel == '{channel}' & {attribute} == '{group1}'")['value']
                    __b = channel_df.query(f"channel == '{channel}' & {attribute} == '{group2}'")['value']
                    _am = __a.mean()
                    _bm = __b.mean()
                    means = [_am, _bm, np.log2(__a.mean() / __b.mean())]
                    _res.append([attribute, channel, group1, group2, *means, *mannwhitneyu(__a, __b)])
        cols = ['attribute', 'channel', 'group1', 'group2', 'mean1', 'mean2', 'log2_fold', 'stat', 'p_value']
        channel_stats = pd.DataFrame(_res, columns=cols)
        channel_stats['p_adj'] = multipletests(channel_stats['p_value'], method='fdr_bh')[1]

        # # # remove duplication due to lazy itertools.permutations
        channel_stats['abs_log2_fold'] = channel_stats['log2_fold'].abs()
        channel_stats = (
            channel_stats
            .drop_duplicates(subset=['attribute', 'channel', 'abs_log2_fold', 'p_value'])
            .drop("abs_log2_fold", axis=1)
            .reset_index(drop=True))
        # # #  reorder so taht "Healthy" is in second column always
        for i, row in channel_stats.iterrows():
            if "Healthy" in row['group1']:
                row['group1'] = row['group2']
                row['group2'] = "Healthy"
                row['log2_fold'] = -row['log2_fold']
                channel_stats.loc[i] = row
        # # # save
        channel_stats.to_csv(output_prefix + f"channel_mean.testing_between_attributes.csv", index=False)

        # # clusters
        _res = list()
        for attribute in sample_attributes:
            for cluster in cluster_df['cluster'].unique():
                for group1, group2 in itertools.permutations(cluster_df[attribute].unique(), 2):
                    __a = cluster_df.query(f"cluster == '{cluster}' & {attribute} == '{group1}'")['cell_count']
                    __b = cluster_df.query(f"cluster == '{cluster}' & {attribute} == '{group2}'")['cell_count']
                    _am = __a.mean()
                    _bm = __b.mean()
                    means = [_am, _bm, np.log2(__a.mean() / __b.mean())]
                    _res.append([attribute, cluster, group1, group2, *means, *mannwhitneyu(__a, __b)])
        cols = ['attribute', 'cluster', 'group1', 'group2', 'mean1', 'mean2', 'log2_fold', 'stat', 'p_value']
        cluster_stats = pd.DataFrame(_res, columns=cols)
        cluster_stats['p_adj'] = multipletests(cluster_stats['p_value'], method='fdr_bh')[1]

        # # # remove duplication due to lazy itertools.permutations
        cluster_stats['abs_log2_fold'] = cluster_stats['log2_fold'].abs()
        cluster_stats = (
            cluster_stats
            .drop_duplicates(subset=['attribute', 'cluster', 'abs_log2_fold', 'p_value'])
            .drop("abs_log2_fold", axis=1)
            .reset_index(drop=True))
        # # # reorder so taht "Healthy" is in second column always
        for i, row in cluster_stats.iterrows():
            if "Healthy" in row['group1']:
                row['group1'] = row['group2']
                row['group2'] = "Healthy"
                row['log2_fold'] = -row['log2_fold']
                cluster_stats.loc[i] = row
        # # # save
        cluster_stats.to_csv(output_prefix + f"cell_type_abundance.testing_between_attributes.csv", index=False)


        # Filter out rare cell types if required
        filtered_cluster_df = cluster_df.loc[cluster_df['cluster'].isin(filtered_clusters)]


        # Plot
        # # barplots
        # # # channel means
        __n = len(sample_attributes)
        kwargs = dict(x="value", y="channel", orient="horiz", ci='sd', data=channel_df)  # , estimator=np.std)
        fig, axes = plt.subplots(__n, 2, figsize=(5 * 2, 10 * __n), squeeze=False, sharey="row")
        for i, attribute in enumerate(sample_attributes):
            for axs in axes[i, (0, 1)]:
                sns.barplot(**kwargs, hue=attribute, ax=axs)
            axes[i, 1].set_xscale("log")
            for axs, lab in zip(axes[i, :], ['Channel mean', 'Channel mean (log)']):
                axs.set_xlabel(lab)
        fig.savefig(output_prefix + f"channel_mean.by_{attribute}.barplot.svg", **FIG_KWS)

        # # # clusters
        # # # # plot once for all cell types, another time excluding rare cell types
        __n = len(sample_attributes)
        kwargs = dict(y="cluster", orient="horiz", ci='sd')  # , estimator=np.std)
        for label, pl_df in [("all_clusters", cluster_df), ("filtered_clusters", filtered_cluster_df)]:
            fig, axes = plt.subplots(__n, 3, figsize=(5 * 3, 10 * __n), squeeze=False, sharey="row")
            for i, attribute in enumerate(sample_attributes):
                for axs in axes[i, (0, 1)]:
                    sns.barplot(**kwargs, x="cell_count", hue=attribute, data=pl_df, ax=axs)
                axes[i, 1].set_xscale("log")
                sns.barplot(**kwargs, x="cell_perc", hue=attribute, data=pl_df, ax=axes[i, 2])
                for axs, lab in zip(axes[i, :], ['Cell count', 'Cell count (log)', 'Cell percentage']):
                    axs.set_xlabel(lab)
            fig.savefig(output_prefix + f"cell_type_abundance.by_{attribute}.barplot.svg", **FIG_KWS)

        # # volcano
        # # # channels
        __n = len(sample_attributes)
        __m = channel_stats[['attribute', 'group1', 'group2']].drop_duplicates().groupby('attribute').count().max().max()
        fig, axes = plt.subplots(__n, __m, figsize=(__m * 5, __n * 5), squeeze=False, sharex="row", sharey="row")
        fig.suptitle("Changes in mean channel intensity")
        for i, attribute in enumerate(sample_attributes):
            p = channel_stats.query(f"attribute == '{attribute}'")
            for j, (_, (group1, group2)) in enumerate(p[['group1', 'group2']].drop_duplicates().iterrows()):
                q = p.query(f"group1 == '{group1}' & group2 == '{group2}'")
                y = -np.log10(q['p_value'])
                v = q['log2_fold'].abs().max()
                v *= 1.2
                axes[i, j].scatter(q['log2_fold'], y, c=y, cmap="autumn_r")
                for k, row in q.query('p_value < 0.05').iterrows():
                    axes[i, j].text(
                        row['log2_fold'], -np.log10(row['p_value']), s=row['channel'], fontsize=5,
                        ha='left' if np.random.rand() > 0.5 else 'right')
                axes[i, j].axvline(0, linestyle="--", color="grey")
                title = attribute + f"\n{group1} vs {group2}"
                axes[i, j].set(xlabel="log2(fold-change)", ylabel="-log10(p-value)", title=title) # , xlim=(-v, v))
            for axs in axes[i, j + 1:]:
                axs.axis("off")
        fig.savefig(output_prefix + f"channel_mean.by_{attribute}.volcano.svg", **FIG_KWS)

        # # # clusters
        __n = len(sample_attributes)
        __m = cluster_stats[['attribute', 'group1', 'group2']].drop_duplicates().groupby('attribute').count().max().max()
        fig, axes = plt.subplots(__n, __m, figsize=(__m * 5, __n * 5), squeeze=False, sharex="row", sharey="row")
        fig.suptitle("Changes in cell type composition\nfor each cell type")
        for i, attribute in enumerate(sample_attributes):
            p = cluster_stats.query(f"attribute == '{attribute}'")
            for j, (_, (group1, group2)) in enumerate(p[['group1', 'group2']].drop_duplicates().iterrows()):
                q = p.query(f"group1 == '{group1}' & group2 == '{group2}'")
                y = -np.log10(q['p_value'])
                v = q['log2_fold'].abs().max()
                v *= 1.2
                axes[i, j].scatter(q['log2_fold'], y, c=y, cmap="autumn_r")
                for k, row in q.query('p_value < 0.05').iterrows():
                    axes[i, j].text(
                        row['log2_fold'], -np.log10(row['p_value']), s=row['cluster'], fontsize=5,
                        ha='left' if np.random.rand() > 0.5 else 'right')
                axes[i, j].axvline(0, linestyle="--", color="grey")
                title = attribute + f"\n{group1} vs {group2}"
                axes[i, j].set(xlabel="log2(fold-change)", ylabel="-log10(p-value)", title=title) # , xlim=(-v, v))
            for axs in axes[i, j + 1:]:
                axs.axis("off")
        fig.savefig(output_prefix + f"cell_type_abundance.by_{attribute}.volcano.svg", **FIG_KWS)

        # # heatmap of cell type counts
        cluster_counts = (
            self.clusters
            .reset_index()
            .assign(count=1)
            .pivot_table(index='cluster', columns='roi', aggfunc=sum, values='count', fill_value=0))
        roi_areas = pd.Series(
            [np.multiply(*roi.shape[1:]) for roi in rois],
            index=[roi.name for roi in rois])

        cluster_densities = (cluster_counts / roi_areas) * 1e4
        grid = sns.clustermap(
            cluster_densities, metric="correlation",
            cbar_kws=dict(label="Cells per area unit (x1e4)"))
        grid.savefig(output_prefix + f"cell_type_abundance.by_area.svg", **FIG_KWS)

        grid = sns.clustermap(
            cluster_densities, metric="correlation",
            z_score=0, cmap="RdBu_r", center=0,
            cbar_kws=dict(label="Cells per area unit (Z-score)"))
        grid.savefig(output_prefix + f"cell_type_abundance.by_area.zscore.svg", **FIG_KWS)

    def measure_adjacency(
        self,
        samples: Optional[List["IMCSample"]] = None,
        output_prefix: Optional[Path] = None
    ) -> None:
        """
        Derive cell adjacency graphs for each ROI.
        """
        output_prefix = output_prefix or self.processed_dir / "single_cell" / self.name + "."
        rois = [r for sample in (samples or self.samples) for r in sample.rois]
        __gs = parmap.map(get_adjacency_graph, rois, pm_pbar=True)
        for roi, __g in zip(rois, __gs):
            roi._adjacency_graph = __g

        # TODO: package the stuff below into a function

        # First measure adjacency as odds against background
        freqs = parmap.map(measure_cell_type_adjacency, rois)
        # freqs = [
        #     pd.read_csv(roi.sample.root_dir / "single_cell" / roi.name + ".cluster_adjacency_graph.norm_over_random.csv", index_col=0)
        #     for roi in rois
        # ]

        melted = pd.concat([
            f.reset_index().melt(id_vars='index').assign(roi=roi.name)
            for roi, f in zip(rois, freqs)])

        mean_f = melted.pivot_table(
            index='index', columns='variable', values="value", aggfunc=np.mean)
        sns.clustermap(mean_f, cmap="RdBu_r", center=0, robust=True)

        __v = np.percentile(melted['value'].abs(), 95)
        __n, __m = get_grid_dims(len(freqs))
        fig, axes = plt.subplots(
            __n, __m, figsize=(__m * 5, __n * 5), sharex=True, sharey=True)
        axes = axes.flatten()
        for i, (dfs, roi) in enumerate(zip(freqs, rois)):
            axes[i].set_title(roi.name)
            sns.heatmap(
                dfs, ax=axes[i], cmap="RdBu_r", center=0,
                rasterized=True, square=True, xticklabels=True, yticklabels=True,
                vmin=-__v, vmax=__v
            )
        for axs in axes[i + 1:]:
            axs.axis('off')
        fig.savefig(output_prefix + "adjacency.all_rois.pdf", **FIG_KWS)

    def find_communities(
        self,
        samples: Optional[List["IMCSample"]] = None,
        output_prefix: Optional[Path] = None,
        **kwargs
    ) -> None:
        """
        Find communities and supercommunities of cell types across all images.
        """
        rois = [r for sample in (samples or self.samples) for r in sample.rois]
        cluster_communities(rois=rois, output_prefix=output_prefix, **kwargs)



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
        prj: Optional[Project] = None,
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
        self._clusters: Optional[Series] = None

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
        rpath = Path(self.root_dir, "ometiff", self.name, self.name + "_AcquisitionChannel_meta.csv")
        if not rpath.exists():
            msg = (
                "Sample `panel_metadata` was not given upon initialization "
                f"and '{rpath}' could not be found!"
            )
            raise FileNotFoundError(msg)
        self._panel_metadata = parse_acquisition_metadata(rpath)
        return self._panel_metadata

    @property
    def channel_labels(self):
        return pd.DataFrame(
            [roi.channel_labels.rename(roi.name) for roi in self.rois]
        ).T.rename_axis(index="chanel", columns="roi")

    @property
    def clusters(self):
        if self._clusters is not None:
            return self._clusters
        else:
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
                    __v = pd.read_csv(_path, index_col=0)
                except FileNotFoundError:
                    if permissive:
                        continue
                    raise
                if set_attribute:
                    # TODO: fix assignemtn to @property
                    setattr(self, ftype, __v)
                else:
                    res[ftype] = __v
            return res if not set_attribute else None
        return None

    def plot_rois(
        self, channel: Union[str, int], rois: Optional[List["ROI"]] = None
    ) -> Figure:  # List[ROI]
        """Plot a single channel for all ROIs"""
        rois = rois or self.rois

        __n, __m = get_grid_dims(len(rois))
        fig, axis = plt.subplots(__n, __m, figsize=(__m * 4, __n * 4), squeeze=False)
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

        __n = len(cell_type_combinations or [1])
        __m = len(rois)
        fig, axes = plt.subplots(__n, __m, figsize=(3 * __m, 3 * __n), squeeze=False)
        patches: List[Patch] = list()
        for _ax, roi in zip(np.hsplit(axes, __m), rois):
            patches += roi.plot_cell_types(
                cell_type_combinations=cell_type_combinations,
                cell_type_assignments=cell_type_assignments,
                palette=palette,
                ax=_ax,
            )
        add_legend(patches, axes[0, -1])
        return fig

    def plot_probabilities_and_segmentation(self, rois: Optional[List["ROI"]] = None) -> Figure:
        __n = len(rois or self.rois)
        fig, axes = plt.subplots(
            __n,
            5,
            figsize=(5 * 4, 4 * __n),
            gridspec_kw=dict(wspace=0.05),
            sharex="row",
            sharey="row",
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
            [roi.quantify_cell_intensity(**kwargs).assign(roi=roi.name) for roi in rois or self.rois]
        )

    def quantify_cell_morphology(self, rois: Optional[List["ROI"]] = None, **kwargs) -> DataFrame:
        return pd.concat(
            [
                roi.quantify_cell_morphology(**kwargs).assign(roi=roi.name)
                for roi in rois or self.rois
            ]
        )

    def set_clusters(
        self,
        clusters: Optional[MultiIndexSeries] = None,
        rois: Optional[List["ROI"]] = None,
    ) -> None:
        id_cols = ["roi", "obj_id"]
        if clusters is None:
            cast(self.prj).set_clusters(samples=[self])
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

        output_prefix = output_prefix or self.root_dir / "single_cell" / (self.name + ".cluster_adjacency_graph.")

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

        __m = self.n_rois + 1
        nrows = 3
        fig, _ax = plt.subplots(
            nrows, __m, figsize=(4 * __m, 4 * nrows), sharex=False, sharey=False
        )
        __v = np.nanstd(adj_matrices.drop("roi", axis=1).values)
        kws = dict(
            cmap="RdBu_r",
            center=0,
            square=True,
            xticklabels=True,
            yticklabels=True,
            vmin=-__v,
            vmax=__v,
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
        sample: Optional[IMCSample] = None,
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
        self.prj: Optional[Project] = None
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
            self.channel_labels_file or
            sample.root_dir / self.stacks_dir / (self.name + "_full.csv"))
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
            dims=['channel', 'X', 'Y'],
            coords={"channel": self.channel_labels.values})
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
        else:
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
                permissive=permissive, set_attribute=set_attribute,
                overwrite=overwrite, parameters=params)
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
            __m, __n = get_grid_dims(len(channels))
            fig, _axes = plt.subplots(
                __n, __m, figsize=(__m * 4, __n * 4), squeeze=False,
                sharex=share_axes, sharey=share_axes)
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
                __m = np.empty_like(stack)
                for i in range(__m.shape[0]):
                    __m[i] = stack[i] - stack[i].mean()
                label = f"Channel {channel}"
                arr = getattr(__m, channel)(axis=0)
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
                                "Could not find out channel '%s' in "
                                "`sample.channel_labels` "
                                "but could find '%s'. Returning sum of those.", (channel, names)
                            )
                        order = match.reset_index(drop=True)[match.values].index
                        __m = np.empty((match.sum(),) + stack.shape[1:])
                        j = 0
                        for i in order:
                            __m[j] = stack[i] - stack[i].mean()
                            j += 1
                        label = names
                        arr = getattr(__m, red_func)(axis=0)
                else:
                    msg = f"Could not find out channel '{channel}' " "in `sample.channel_labels`."
                    # # LOGGER.error(msg)
                    raise ValueError(msg)
        if equalize:
            arr = minmax_scale(eq(arr))
        return label, arr

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
        **kwargs
    ) -> Axis:
        """
        Plot a single channel.

        Supports indexing of channels either by name or integer.
        Special strings for :class:`numpy.ndarray` functions can be passed to
        reduce values across channels (first axis). Pass e.g. 'mean' or 'sum'.

        Keyword arguments are passed to :func:`~matplotlib.pyplot.imshow`
        """
        _ax = ax
        channel, __p = self._get_channel(channel, equalize=equalize)
        if log:
            __p += abs(__p.min())
            __p = np.log1p(__p)

        if _ax is None:
            _, _ax = plt.subplots(1, 1, figsize=(4, 4))
        _ax.imshow(__p.squeeze(), rasterized=True, **kwargs)
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
                clusters = clusters.replace(dict(zip(labels, ns.astype(int))))
            else:
                clusters = clusters.replace(dict(zip(labels, np.arange(len(labels)))))
        else:
            labels = sorted(np.unique(clusters.values))

        # simply plot all cell types jointly
        if cell_type_combinations in [None, "all"]:
            combs = [tuple(sorted(clusters.unique()))]
        else:
            combs = cell_type_combinations

        __n = len(combs)
        if ax is None:
            __m = 1
            fig, axes = plt.subplots(
                __n, __m, figsize=(3 * __m, 3 * __n), sharex="col", sharey="col", squeeze=False
            )
        else:
            axes = ax
            if isinstance(ax, np.ndarray) and len(ax) != __n:
                raise ValueError(f"Given axes must be of length of cell_type_combinations ({__n}).")

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
            colors = sns.color_palette(palette, len(labels))
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
            index=self.channel_labels, columns=self.channel_labels)
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

        __c = pd.Series(
            scipy.cluster.hierarchy.fcluster(
                grid.dendrogram_col.linkage, n_groups, criterion="maxclust"),
            index=xcorr.index,
        )

        marker_sets: Dict[int, List[str]] = dict()
        for _sp in range(1, n_groups + 1):
            marker_sets[_sp] = list()
            for i in np.random.choice(np.unique(__c), group_size, replace=True):
                marker_sets[_sp].append(np.random.choice(__c[__c == i].index, 1, replace=True)[0])
        return (xcorr, marker_sets)

    def plot_overlayied_channels_subplots(self, n_groups: int) -> Figure:
        """
        Plot all channels of ROI in `n_groups` combinations, where each combination
        has as little overlap as possible.
        """
        stack = self.stack

        _, marker_sets = self.get_distinct_marker_sets(
            n_groups=n_groups, group_size=int(np.floor(self.channel_number / n_groups)))

        __n, __m = get_grid_dims(n_groups)
        fig, axis = plt.subplots(
            __n, __m, figsize=(6 * __m, 6 * __n), sharex=True, sharey=True, squeeze=False,
        )
        axis = axis.flatten()
        for i, (marker_set, mrks) in enumerate(marker_sets.items()):
            patches = list()
            cmaps = get_transparent_cmaps(len(mrks))
            for _, (__l, __c) in enumerate(zip(mrks, cmaps)):
                __x = stack[self.channel_labels == __l, :, :].squeeze()
                __v = __x.mean() + __x.std() * 2
                axis[i].imshow(
                    __x,
                    cmap=__c,
                    vmin=0,
                    vmax=__v,
                    label=__l,
                    interpolation="bilinear",
                    rasterized=True,
                )
                axis[i].axis("off")
                patches.append(mpatches.Patch(color=__c(256), label=__m))
            axis[i].legend(
                handles=patches,
                bbox_to_anchor=(1.05, 1),
                loc=2,
                borderaxespad=0.0,
                title=marker_set,
            )
        return fig

    def plot_probabilities_and_segmentation(
        self,
        axes: Optional[Sequence[Axis]] = None
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
                1, 5, figsize=(5 * 4, 4),
                gridspec_kw=dict(wspace=0.05), sharex=True, sharey=True
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
            cast(self.sample).set_clusters(clusters=clusters, rois=[self])
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
