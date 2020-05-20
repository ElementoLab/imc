#! /usr/bin/env python

"""
A class to model a imaging mass cytometry project.
"""

from __future__ import annotations  # fix the type annotatiton of not yet undefined classes
import os

from typing import Tuple, List, Optional, Union  # , cast

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import parmap  # type: ignore

import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

from imc.data_models.sample import IMCSample
from imc.types import Path, Figure, Patch, DataFrame, Series, MultiIndexSeries

# from imc import LOGGER
from imc.operations import (
    derive_reference_cell_type_labels,
    single_cell_analysis,
    get_adjacency_graph,
    measure_cell_type_adjacency,
    cluster_communities,
)
from imc.graphics import (
    get_grid_dims,
    add_legend,
)
from imc.exceptions import cast  # TODO: replace with typing.cast

FIG_KWS = dict(dpi=300, bbox_inches="tight")

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
            n, m = get_grid_dims(len(rois))
            fig, axes = plt.subplots(n, m, figsize=(4 * m, 4 * n))
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
            n = len(rois)
            fig, axes = plt.subplots(n, 5, figsize=(4 * 5, 4 * n))
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

        n = len(samples)
        m = max([sample.n_rois for sample in samples])
        fig, axes = plt.subplots(n, m, figsize=(3 * m, 3 * n), squeeze=False)
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
        rois = [
            r
            for sample in (samples or self.samples)
            for r in sample.rois
            if r in (rois or sample.rois)
        ]
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
                index=[roi.name for roi in rois],
                name="cell_density",
            )
            if all(cell_density < 0):
                cell_density *= 1000

            def_kwargs = dict(z_score=0, center=0, robust=True, cmap="RdBu_r")
            def_kwargs.update(kwargs)
            # TODO: add {row,col}_colors colorbar to heatmap
            for kws, label, cbar_label in [({}, "", ""), (def_kwargs, ".z_score", " (Z-score)")]:
                plot_file = (
                    self.processed_dir / "qc" / self.name
                    + f".mean_per_channel.clustermap{label}.svg"
                )
                grid = sns.clustermap(
                    res,
                    cbar_kws=dict(label=red_func.capitalize() + cbar_label),
                    row_colors=channel_mean,
                    col_colors=cell_density,
                    metric="correlation",
                    xticklabels=True,
                    yticklabels=True,
                    **kws,
                )
                grid.fig.suptitle("Mean channel intensities")
                grid.savefig(plot_file, dpi=300, bbox_inches="tight")
            grid.fig.grid = grid
            return (res, grid.fig)
        return res

    def image_summary(self, samples: Optional[List["IMCSample"]] = None, rois: List["ROI"] = None):
        raise NotImplementedError
        from imc.utils import lacunarity, fractal_dimension

        rois = [r for r in (rois or self.rois) if r.sample in (samples or self.samples)]
        roi_names = [r.name for r in rois]
        densities = pd.Series(
            {roi.name: roi.cells_per_area_unit() for roi in rois}, name="cell density"
        )
        lacunarities = pd.Series(
            parmap.map(lacunarity, [roi.cell_mask_o for roi in rois], pm_pbar=True),
            index=roi_names,
            name="lacunarity",
        )
        fractal_dimensions = pd.Series(
            parmap.map(fractal_dimension, [roi.cell_mask_o for roi in rois], pm_pbar=True),
            index=roi_names,
            name="fractal_dimension",
        )

        morphos = pd.DataFrame([densities * 1e4, lacunarities, fractal_dimensions]).T

    def channel_correlation(
        self, samples: Optional[List["IMCSample"]] = None, rois: Optional[List["ROI"]] = None
    ) -> Figure:
        """
        Observe the pairwise correlation of channels across ROIs.
        """
        from imc.operations import _correlate_channels__roi

        rois = [
            r
            for sample in (samples or self.samples)
            for r in sample.rois
            if r in (rois or sample.rois)
        ]
        res = parmap.map(_correlate_channels__roi, rois, pm_pbar=True)

        labels = rois[0].channel_labels
        xcorr = pd.DataFrame(np.asarray(res).mean(0), index=labels, columns=labels)

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
            quantification = pd.concat(
                [
                    self.quantify_cell_intensity(samples=samples, rois=rois).drop(
                        ["sample", "roi"], axis=1
                    ),
                    self.quantify_cell_morphology(samples=samples, rois=rois),
                ],
                axis=1,
            )
        else:
            quantification = self.quantify_cell_intensity(samples=samples, rois=rois)
        if not set_attribute:
            return quantification
        self.quantification = quantification
        return None

    def quantify_cell_intensity(
        self,
        samples: Optional[List["IMCSample"]] = None,
        rois: Optional[List["ROI"]] = None,
        **kwargs,
    ):
        """
        Measure the intensity of each channel in each single cell.
        """
        from imc.operations import _quantify_cell_intensity__roi

        return pd.concat(
            parmap.map(
                _quantify_cell_intensity__roi,
                [
                    r
                    for sample in (samples or self.samples)
                    for r in sample.rois
                    if r in (rois or sample.rois)
                ],
                pm_pbar=True,
                **kwargs,
            )
        )

    def quantify_cell_morphology(
        self,
        samples: Optional[List["IMCSample"]] = None,
        rois: Optional[List["ROI"]] = None,
        **kwargs,
    ):
        """
        Measure the shape parameters of each single cell.
        """
        from imc.operations import _quantify_cell_morphology__roi

        return pd.concat(
            parmap.map(
                _quantify_cell_morphology__roi,
                [
                    r
                    for sample in (samples or self.samples)
                    for r in sample.rois
                    if r in (rois or sample.rois)
                ],
                pm_pbar=True,
                **kwargs,
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
        output_prefix = Path(output_prefix or self.processed_dir / "single_cell" / self.name)

        quantification = None
        if "quantification" in kwargs:
            quantification = kwargs["quantification"]
            del kwargs["quantification"]
        cell_type_channels = None
        if "cell_type_channels" in kwargs:
            cell_type_channels = kwargs["cell_type_channels"]
            del kwargs["cell_type_channels"]
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
            self._get_input_filename("cell_cluster_assignments"), index=False
        )
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
            )[
                "cluster"
            ]  # .astype(str)
        assert isinstance(clusters.index, pd.MultiIndex)
        assert clusters.index.names == id_cols
        self._clusters = clusters
        for sample in samples or self.samples:
            sample.set_clusters(clusters=clusters.loc[sample.name])
        if write_to_disk:
            self._clusters.reset_index().to_csv(
                self._get_input_filename("cell_cluster_assignments"), index=False
            )

    def label_clusters(
        self, h5ad_file: Optional[Path] = None, output_prefix: Optional[Path] = None, **kwargs
    ) -> None:
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
                self._get_input_filename("cell_cluster_assignments"), index=False
            )
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

        sample_attributes = sample_attributes or ["name"]
        samples = samples or self.samples
        rois = [roi for sample in samples for roi in sample.rois]
        output_prefix = output_prefix or self.processed_dir / "single_cell" / self.name + "."
        output_prefix.parent.mkdir(exist_ok=True)

        # group samples by desired attributes
        sample_df = (
            pd.DataFrame(
                {k: v for k, v in sample.__dict__.items() if isinstance(v, str)}
                for sample in samples
            )[["name"] + sample_attributes]
            .set_index("name")
            .rename_axis("sample")
            .reset_index()
        )
        sample_groups = sample_df.groupby(sample_attributes)["sample"].apply(set)
        sample_roi_df = pd.DataFrame(
            [(roi.name, roi.sample.name) for roi in rois], columns=["roi", "sample"]
        )

        # Whole channel means
        channel_means: DataFrame = self.channel_summary(plot=False)
        channel_means = (
            channel_means.reset_index()
            .melt(id_vars="channel", var_name="roi")
            .reset_index(drop=True)
        )
        channel_df = (
            channel_means.merge(sample_roi_df).merge(sample_df).sort_values(sample_attributes)
        )

        # cell type abundancy per sample or group of samples
        cluster_counts = (
            self.clusters.groupby(level=["sample", "roi"]).value_counts().rename("cell_count")
        )
        cluster_perc = (cluster_counts.groupby("cluster").sum() / cluster_counts.sum()) * 100
        filtered_clusters = cluster_perc[cluster_perc > cell_type_percentage_threshold].index

        # # absolute
        # # fraction of total
        cluster_df = cluster_counts.reset_index().merge(sample_df).sort_values(sample_attributes)
        cluster_df["cell_perc"] = cluster_df.groupby("roi")["cell_count"].apply(
            lambda x: (x / x.sum()) * 100
        )

        # Test difference between channels/clusters
        # # channels
        _res = list()
        for attribute in sample_attributes:
            for channel in channel_df["channel"].unique():
                for group1, group2 in itertools.permutations(channel_df[attribute].unique(), 2):
                    a = channel_df.query(f"channel == '{channel}' & {attribute} == '{group1}'")[
                        "value"
                    ]
                    b = channel_df.query(f"channel == '{channel}' & {attribute} == '{group2}'")[
                        "value"
                    ]
                    am = a.mean()
                    bm = b.mean()
                    means = [am, bm, np.log2(a.mean() / b.mean())]
                    _res.append([attribute, channel, group1, group2, *means, *mannwhitneyu(a, b)])
        cols = [
            "attribute",
            "channel",
            "group1",
            "group2",
            "mean1",
            "mean2",
            "log2_fold",
            "stat",
            "p_value",
        ]
        channel_stats = pd.DataFrame(_res, columns=cols)
        channel_stats["p_adj"] = multipletests(channel_stats["p_value"], method="fdr_bh")[1]

        # # # remove duplication due to lazy itertools.permutations
        channel_stats["abs_log2_fold"] = channel_stats["log2_fold"].abs()
        channel_stats = (
            channel_stats.drop_duplicates(
                subset=["attribute", "channel", "abs_log2_fold", "p_value"]
            )
            .drop("abs_log2_fold", axis=1)
            .reset_index(drop=True)
        )
        # # #  reorder so taht "Healthy" is in second column always
        for i, row in channel_stats.iterrows():
            if "Healthy" in row["group1"]:
                row["group1"] = row["group2"]
                row["group2"] = "Healthy"
                row["log2_fold"] = -row["log2_fold"]
                channel_stats.loc[i] = row
        # # # save
        channel_stats.to_csv(
            output_prefix + f"channel_mean.testing_between_attributes.csv", index=False
        )

        # # clusters
        _res = list()
        for attribute in sample_attributes:
            for cluster in cluster_df["cluster"].unique():
                for group1, group2 in itertools.permutations(cluster_df[attribute].unique(), 2):
                    a = cluster_df.query(f"cluster == '{cluster}' & {attribute} == '{group1}'")[
                        "cell_count"
                    ]
                    b = cluster_df.query(f"cluster == '{cluster}' & {attribute} == '{group2}'")[
                        "cell_count"
                    ]
                    am = a.mean()
                    bm = b.mean()
                    means = [am, bm, np.log2(a.mean() / b.mean())]
                    _res.append([attribute, cluster, group1, group2, *means, *mannwhitneyu(a, b)])
        cols = [
            "attribute",
            "cluster",
            "group1",
            "group2",
            "mean1",
            "mean2",
            "log2_fold",
            "stat",
            "p_value",
        ]
        cluster_stats = pd.DataFrame(_res, columns=cols)
        cluster_stats["p_adj"] = multipletests(cluster_stats["p_value"], method="fdr_bh")[1]

        # # # remove duplication due to lazy itertools.permutations
        cluster_stats["abs_log2_fold"] = cluster_stats["log2_fold"].abs()
        cluster_stats = (
            cluster_stats.drop_duplicates(
                subset=["attribute", "cluster", "abs_log2_fold", "p_value"]
            )
            .drop("abs_log2_fold", axis=1)
            .reset_index(drop=True)
        )
        # # # reorder so taht "Healthy" is in second column always
        for i, row in cluster_stats.iterrows():
            if "Healthy" in row["group1"]:
                row["group1"] = row["group2"]
                row["group2"] = "Healthy"
                row["log2_fold"] = -row["log2_fold"]
                cluster_stats.loc[i] = row
        # # # save
        cluster_stats.to_csv(
            output_prefix + f"cell_type_abundance.testing_between_attributes.csv", index=False
        )

        # Filter out rare cell types if required
        filtered_cluster_df = cluster_df.loc[cluster_df["cluster"].isin(filtered_clusters)]

        # Plot
        # # barplots
        # # # channel means
        n = len(sample_attributes)
        kwargs = dict(
            x="value", y="channel", orient="horiz", ci="sd", data=channel_df
        )  # , estimator=np.std)
        fig, axes = plt.subplots(n, 2, figsize=(5 * 2, 10 * n), squeeze=False, sharey="row")
        for i, attribute in enumerate(sample_attributes):
            for axs in axes[i, (0, 1)]:
                sns.barplot(**kwargs, hue=attribute, ax=axs)
            axes[i, 1].set_xscale("log")
            for axs, lab in zip(axes[i, :], ["Channel mean", "Channel mean (log)"]):
                axs.set_xlabel(lab)
        fig.savefig(output_prefix + f"channel_mean.by_{attribute}.barplot.svg", **FIG_KWS)

        # # # clusters
        # # # # plot once for all cell types, another time excluding rare cell types
        n = len(sample_attributes)
        kwargs = dict(y="cluster", orient="horiz", ci="sd")  # , estimator=np.std)
        for label, pl_df in [
            ("all_clusters", cluster_df),
            ("filtered_clusters", filtered_cluster_df),
        ]:
            fig, axes = plt.subplots(n, 3, figsize=(5 * 3, 10 * n), squeeze=False, sharey="row")
            for i, attribute in enumerate(sample_attributes):
                for axs in axes[i, (0, 1)]:
                    sns.barplot(**kwargs, x="cell_count", hue=attribute, data=pl_df, ax=axs)
                axes[i, 1].set_xscale("log")
                sns.barplot(**kwargs, x="cell_perc", hue=attribute, data=pl_df, ax=axes[i, 2])
                for axs, lab in zip(
                    axes[i, :], ["Cell count", "Cell count (log)", "Cell percentage"]
                ):
                    axs.set_xlabel(lab)
            fig.savefig(
                output_prefix + f"cell_type_abundance.by_{attribute}.barplot.svg", **FIG_KWS
            )

        # # volcano
        # # # channels
        n = len(sample_attributes)
        m = (
            channel_stats[["attribute", "group1", "group2"]]
            .drop_duplicates()
            .groupby("attribute")
            .count()
            .max()
            .max()
        )
        fig, axes = plt.subplots(
            n, m, figsize=(m * 5, n * 5), squeeze=False, sharex="row", sharey="row"
        )
        fig.suptitle("Changes in mean channel intensity")
        for i, attribute in enumerate(sample_attributes):
            p = channel_stats.query(f"attribute == '{attribute}'")
            for j, (_, (group1, group2)) in enumerate(
                p[["group1", "group2"]].drop_duplicates().iterrows()
            ):
                q = p.query(f"group1 == '{group1}' & group2 == '{group2}'")
                y = -np.log10(q["p_value"])
                v = q["log2_fold"].abs().max()
                v *= 1.2
                axes[i, j].scatter(q["log2_fold"], y, c=y, cmap="autumn_r")
                for k, row in q.query("p_value < 0.05").iterrows():
                    axes[i, j].text(
                        row["log2_fold"],
                        -np.log10(row["p_value"]),
                        s=row["channel"],
                        fontsize=5,
                        ha="left" if np.random.rand() > 0.5 else "right",
                    )
                axes[i, j].axvline(0, linestyle="--", color="grey")
                title = attribute + f"\n{group1} vs {group2}"
                axes[i, j].set(
                    xlabel="log2(fold-change)", ylabel="-log10(p-value)", title=title
                )  # , xlim=(-v, v))
            for axs in axes[i, j + 1 :]:
                axs.axis("off")
        fig.savefig(output_prefix + f"channel_mean.by_{attribute}.volcano.svg", **FIG_KWS)

        # # # clusters
        n = len(sample_attributes)
        m = (
            cluster_stats[["attribute", "group1", "group2"]]
            .drop_duplicates()
            .groupby("attribute")
            .count()
            .max()
            .max()
        )
        fig, axes = plt.subplots(
            n, m, figsize=(m * 5, n * 5), squeeze=False, sharex="row", sharey="row"
        )
        fig.suptitle("Changes in cell type composition\nfor each cell type")
        for i, attribute in enumerate(sample_attributes):
            p = cluster_stats.query(f"attribute == '{attribute}'")
            for j, (_, (group1, group2)) in enumerate(
                p[["group1", "group2"]].drop_duplicates().iterrows()
            ):
                q = p.query(f"group1 == '{group1}' & group2 == '{group2}'")
                y = -np.log10(q["p_value"])
                v = q["log2_fold"].abs().max()
                v *= 1.2
                axes[i, j].scatter(q["log2_fold"], y, c=y, cmap="autumn_r")
                for k, row in q.query("p_value < 0.05").iterrows():
                    axes[i, j].text(
                        row["log2_fold"],
                        -np.log10(row["p_value"]),
                        s=row["cluster"],
                        fontsize=5,
                        ha="left" if np.random.rand() > 0.5 else "right",
                    )
                axes[i, j].axvline(0, linestyle="--", color="grey")
                title = attribute + f"\n{group1} vs {group2}"
                axes[i, j].set(
                    xlabel="log2(fold-change)", ylabel="-log10(p-value)", title=title
                )  # , xlim=(-v, v))
            for axs in axes[i, j + 1 :]:
                axs.axis("off")
        fig.savefig(output_prefix + f"cell_type_abundance.by_{attribute}.volcano.svg", **FIG_KWS)

        # # heatmap of cell type counts
        cluster_counts = (
            self.clusters.reset_index()
            .assign(count=1)
            .pivot_table(index="cluster", columns="roi", aggfunc=sum, values="count", fill_value=0)
        )
        roi_areas = pd.Series(
            [np.multiply(*roi.shape[1:]) for roi in rois], index=[roi.name for roi in rois]
        )

        cluster_densities = (cluster_counts / roi_areas) * 1e4
        grid = sns.clustermap(
            cluster_densities,
            metric="correlation",
            cbar_kws=dict(label="Cells per area unit (x1e4)"),
        )
        grid.savefig(output_prefix + f"cell_type_abundance.by_area.svg", **FIG_KWS)

        grid = sns.clustermap(
            cluster_densities,
            metric="correlation",
            z_score=0,
            cmap="RdBu_r",
            center=0,
            cbar_kws=dict(label="Cells per area unit (Z-score)"),
        )
        grid.savefig(output_prefix + f"cell_type_abundance.by_area.zscore.svg", **FIG_KWS)

    def measure_adjacency(
        self, samples: Optional[List["IMCSample"]] = None, output_prefix: Optional[Path] = None
    ) -> None:
        """
        Derive cell adjacency graphs for each ROI.
        """
        output_prefix = output_prefix or self.processed_dir / "single_cell" / self.name + "."
        rois = [r for sample in (samples or self.samples) for r in sample.rois]
        gs = parmap.map(get_adjacency_graph, rois, pm_pbar=True)
        for roi, g in zip(rois, gs):
            roi._adjacency_graph = g

        # TODO: package the stuff below into a function

        # First measure adjacency as odds against background
        freqs = parmap.map(measure_cell_type_adjacency, rois)
        # freqs = [
        #     pd.read_csv(roi.sample.root_dir / "single_cell" / roi.name + ".cluster_adjacency_graph.norm_over_random.csv", index_col=0)
        #     for roi in rois
        # ]

        melted = pd.concat(
            [
                f.reset_index().melt(id_vars="index").assign(roi=roi.name)
                for roi, f in zip(rois, freqs)
            ]
        )

        mean_f = melted.pivot_table(
            index="index", columns="variable", values="value", aggfunc=np.mean
        )
        sns.clustermap(mean_f, cmap="RdBu_r", center=0, robust=True)

        v = np.percentile(melted["value"].abs(), 95)
        n, m = get_grid_dims(len(freqs))
        fig, axes = plt.subplots(n, m, figsize=(m * 5, n * 5), sharex=True, sharey=True)
        axes = axes.flatten()
        i = -1
        for i, (dfs, roi) in enumerate(zip(freqs, rois)):
            axes[i].set_title(roi.name)
            sns.heatmap(
                dfs,
                ax=axes[i],
                cmap="RdBu_r",
                center=0,
                rasterized=True,
                square=True,
                xticklabels=True,
                yticklabels=True,
                vmin=-v,
                vmax=v,
            )
        for axs in axes[i + 1 :]:
            axs.axis("off")
        fig.savefig(output_prefix + "adjacency.all_rois.pdf", **FIG_KWS)

    def find_communities(
        self,
        samples: Optional[List["IMCSample"]] = None,
        output_prefix: Optional[Path] = None,
        **kwargs,
    ) -> None:
        """
        Find communities and supercommunities of cell types across all images.
        """
        rois = [r for sample in (samples or self.samples) for r in sample.rois]
        cluster_communities(rois=rois, output_prefix=output_prefix, **kwargs)
