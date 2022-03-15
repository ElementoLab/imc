#! /usr/bin/env python

"""
A class to model a imaging mass cytometry project.
"""

from __future__ import annotations
import os
import pathlib
import typing as tp

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import parmap  # type: ignore

import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from seaborn_extensions import clustermap

from imc.data_models.sample import IMCSample
from imc.data_models.roi import ROI
from imc.types import Path, Figure, Patch, DataFrame, Series, MultiIndexSeries

# from imc import LOGGER
# from imc.ops.clustering import derive_reference_cell_type_labels, single_cell_analysis
from imc.ops.adjacency import get_adjacency_graph, measure_cell_type_adjacency
from imc.ops.community import cluster_communities
from imc.graphics import get_grid_dims, add_legend
from imc.utils import align_channels_by_name
from imc.exceptions import cast  # TODO: replace with typing.cast

FIG_KWS = dict(dpi=300, bbox_inches="tight")

DEFAULT_PROJECT_NAME = "project"
DEFAULT_SAMPLE_NAME_ATTRIBUTE = "sample_name"
DEFAULT_SAMPLE_GROUPING_ATTRIBUTEs = [DEFAULT_SAMPLE_NAME_ATTRIBUTE]
DEFAULT_TOGGLE_ATTRIBUTE = "toggle"
DEFAULT_PROCESSED_DIR_NAME = Path("processed")
DEFAULT_RESULTS_DIR_NAME = Path("results")
DEFAULT_PRJ_SINGLE_CELL_DIR = Path("single_cell")

# processed directory structure
SUBFOLDERS_PER_SAMPLE = True
ROI_STACKS_DIR = Path("tiffs")
ROI_MASKS_DIR = Path("tiffs")
ROI_UNCERTAINTY_DIR = Path("uncertainty")
ROI_SINGLE_CELL_DIR = Path("single_cell")


# def cast(arg: tp.Optional[GenericType], name: str, obj: str) -> GenericType:
#     """Remove `tp.Optional` from `T`."""
#     if arg is None:
#         raise AttributeNotSetError(f"Attribute '{name}' of '{obj}' cannot be None!")
#     return arg


class Project:
    """
    A class to model a IMC project.
    """

    """
    Parameters
    ----------
    metadata : :obj:`str`
        Path to CSV metadata sheet.
    name : :obj:`str`
        Project name. Defaults to "project".

    Attributes
    ----------
    name : :obj:`str`
        Project name
    metadata : :obj:`str`
        Path to CSV metadata sheet.
    metadata : :class:`pandas.DataFrame`
        Metadata dataframe
    samples : tp.List[:class:`IMCSample`]
        tp.List of IMC sample objects.
    """

    def __init__(
        self,
        metadata: tp.Optional[tp.Union[str, Path, DataFrame]] = None,
        name: str = DEFAULT_PROJECT_NAME,
        sample_name_attribute: str = DEFAULT_SAMPLE_NAME_ATTRIBUTE,
        sample_grouping_attributes: tp.Sequence[str] = None,
        panel_metadata: tp.Optional[tp.Union[Path, DataFrame]] = None,
        toggle: bool = True,
        subfolder_per_sample: bool = SUBFOLDERS_PER_SAMPLE,
        processed_dir: Path = DEFAULT_PROCESSED_DIR_NAME,
        results_dir: Path = DEFAULT_RESULTS_DIR_NAME,
        **kwargs,
    ):
        # Initialize
        self.name = name
        self.metadata = (
            pd.read_csv(metadata)
            if isinstance(metadata, (str, pathlib.Path, Path))
            else metadata
        )
        self.samples: tp.Sequence[IMCSample] = list()
        self.sample_name_attribute = sample_name_attribute
        self.sample_grouping_attributes = (
            sample_grouping_attributes or DEFAULT_SAMPLE_GROUPING_ATTRIBUTEs
        )
        self.panel_metadata: tp.Optional[DataFrame] = (
            pd.read_csv(panel_metadata, index_col=0)
            if isinstance(panel_metadata, (str, Path))
            else panel_metadata
        )
        # # TODO: make sure channel labels conform to internal specification: "Label(Metal\d+)"
        # self.channel_labels: tp.Optional[Series] = (
        #     pd.read_csv(channel_labels, index_col=0, squeeze=True)
        #     if isinstance(channel_labels, (str, Path))
        #     else channel_labels
        # )

        self.toggle = toggle
        self.subfolder_per_sample = subfolder_per_sample
        self.processed_dir = Path(processed_dir).expanduser().absolute()
        self.results_dir = Path(results_dir).expanduser().absolute()
        self.quantification: tp.Optional[DataFrame] = None
        self._clusters: tp.Optional[
            MultiIndexSeries
        ] = None  # MultiIndex: ['sample', 'roi', 'obj_id']

        # Add kwargs as attributes
        self.__dict__.update(kwargs)

        if "__no_init__" in kwargs:
            return
        self._initialize_project_from_annotation(**kwargs)

        if not self.rois:
            print(
                "Could not find ROIs for any of the samples. "
                "Either pass metadata with one row per ROI, "
                "or set `processed_dir` in order for ROIs to be discovered, "
                "and make sure select the right project stucture with `subfolder_per_sample`."
            )

        # if self.channel_labels is None:
        #     self.set_channel_labels()

    def __repr__(self):
        s = len(self.samples)
        r = len(self.rois)
        return (
            f"Project '{self.name}' with {s} sample"
            + (" " if s == 1 else "s ")
            + f"and {r} ROI"
            + (" " if r == 1 else "s ")
            + "in total."
        )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def __getitem__(self, item: int) -> IMCSample:
        return self.samples[item]

    def __iter__(self) -> tp.Iterator[IMCSample]:
        return iter(self.samples)

    @classmethod
    def from_stacks(cls, tiffs: tp.Union[Path, tp.Sequence[Path]], **kwargs) -> Project:
        if isinstance(tiffs, Path):
            tiffs = [tiffs]
        return Project.from_rois([ROI.from_stack(s) for s in tiffs], **kwargs)

    @classmethod
    def from_rois(cls, rois: tp.Union[ROI, tp.Sequence[ROI]], **kwargs) -> Project:
        if isinstance(rois, ROI):
            rois = [rois]
        samples = [r.sample for r in rois if r.sample is not None]
        return Project(samples=samples, __no_init__=True, **kwargs)

    @classmethod
    def from_samples(
        cls, samples: tp.Union[IMCSample, tp.Sequence[IMCSample]], **kwargs
    ) -> Project:
        if isinstance(samples, IMCSample):
            samples = [samples]
        return Project(samples=samples, __no_init__=True, **kwargs)

    def _detect_samples(self) -> DataFrame:
        if self.processed_dir is None:
            print("Project does not have `processed_dir`. Cannot find Samples.")
            return pd.DataFrame()

        content = sorted(
            [x for x in self.processed_dir.iterdir() if x.is_dir()]
            if self.subfolder_per_sample
            else self.processed_dir.glob("*_full.tiff")
        )
        df = pd.Series(content, dtype="object").to_frame()
        if df.empty:
            print(f"Could not find any Samples in '{self.processed_dir}'.")
            return df
        df[DEFAULT_SAMPLE_NAME_ATTRIBUTE] = (
            df[0].apply(lambda x: x.name.replace("_full.tiff", ""))
            if self.subfolder_per_sample
            else df[0].map(lambda x: x.name).str.extract(r"(.*)-\d+_full\.tiff")[0]
        )
        return df.drop([0], axis=1)  # .sort_values(DEFAULT_SAMPLE_NAME_ATTRIBUTE)

    def _initialize_project_from_annotation(self, **kwargs) -> None:
        def cols_with_unique_values(dfs: DataFrame) -> set:
            return {col for col in dfs if len(dfs[col].unique()) == 1}

        metadata = self.metadata if self.metadata is not None else self._detect_samples()

        if metadata.empty:
            return

        if self.toggle and ("toggle" in metadata.columns):
            # TODO: logger.info("Removing samples without toggle active")
            metadata = metadata.loc[metadata[DEFAULT_TOGGLE_ATTRIBUTE], :]

        sample_grouping_attributes = (
            self.sample_grouping_attributes or metadata.columns.list()
        )

        for _, idx in metadata.groupby(
            sample_grouping_attributes, sort=False
        ).groups.items():
            rows = metadata.loc[idx]
            const_cols = list(cols_with_unique_values(rows))
            row = rows[const_cols].drop_duplicates().squeeze(axis=0)

            sample = IMCSample(
                sample_name=row[self.sample_name_attribute],
                root_dir=(self.processed_dir / str(row[self.sample_name_attribute]))
                if self.subfolder_per_sample
                else self.processed_dir,
                subfolder_per_sample=self.subfolder_per_sample,
                metadata=rows if rows.shape[0] > 1 else None,
                panel_metadata=self.panel_metadata,
                prj=self,
                **kwargs,
                **row.drop("sample_name", errors="ignore").to_dict(),
            )
            for roi in sample.rois:
                roi.prj = self
                # If channel labels are given, add them to all ROIs
                # roi._channel_labels = self.channel_labels
            self.samples.append(sample)

    @property
    def rois(self) -> tp.List[ROI]:
        """
        Return a tp.List of all ROIs of the project samples.
        """
        return [roi for sample in self.samples for roi in sample.rois]

    @rois.setter
    def rois(self, rois: tp.List[tp.Union[str, ROI]]):
        for sample in self.samples:
            sample.rois = [r for r in sample.rois if r.name in rois or r in rois]

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_rois(self) -> int:
        return len(self.rois)

    @property
    def channel_labels(self) -> tp.Union[Series, DataFrame]:
        return pd.concat([sample.channel_labels for sample in self.samples], axis=1)

    @property
    def channel_names(self) -> tp.Union[Series, DataFrame]:
        return pd.concat([sample.channel_names for sample in self.samples], axis=1)

    @property
    def channel_metals(self) -> tp.Union[Series, DataFrame]:
        return pd.concat([sample.channel_metals for sample in self.samples], axis=1)

    def _get_rois(
        self,
        samples: tp.Optional[tp.Sequence[IMCSample]],
        rois: tp.Optional[tp.Sequence[ROI]],
    ) -> tp.List[ROI]:
        return [
            r
            for sample in (samples or self.samples)
            for r in sample.rois
            if r in (rois or sample.rois)
        ]

    def get_input_filename(self, input_type: str) -> Path:
        """Get path to file with data for Sample.

        Available `input_type` values are:
            - "cell_type_assignments": CSV file with cell type assignemts for each cell and each ROI
        """
        to_read = {
            "h5ad": (
                DEFAULT_PRJ_SINGLE_CELL_DIR,
                ".single_cell.processed.h5ad",
            ),
            "cell_cluster_assignments": (
                DEFAULT_PRJ_SINGLE_CELL_DIR,
                ".single_cell.cluster_assignments.csv",
            ),
        }
        dir_, suffix = to_read[input_type]
        return self.results_dir / dir_ / (self.name + suffix)

    def get_samples(self, sample_names: tp.Union[str, tp.List[str]]):
        if isinstance(sample_names, str):
            sample_names = [sample_names]
        samples = [s for s in self.samples if s.name in sample_names]
        if samples:
            return samples[0] if len(samples) == 1 else samples
        else:
            ValueError(f"Sample '{sample_names}' couldn't be found.")

    def get_rois(self, roi_names: tp.Union[str, tp.List[str]]):
        if isinstance(roi_names, str):
            roi_names = [roi_names]
        rois = [r for r in self.rois if r.name in roi_names]
        if rois:
            return rois[0] if len(rois) == 1 else rois
        else:
            ValueError(f"Sample '{roi_names}' couldn't be found.")

    def plot_channels(
        self,
        channels: tp.Sequence[str] = ["mean"],
        per_sample: bool = False,
        merged: bool = False,
        save: bool = False,
        output_dir: tp.Optional[Path] = None,
        samples: tp.Optional[tp.Sequence[IMCSample]] = None,
        rois: tp.Optional[tp.Sequence[ROI]] = None,
        **kwargs,
    ) -> Figure:
        """
        Plot a tp.List of channels for all Samples/ROIs.
        """
        if isinstance(channels, str):
            channels = [channels]
        output_dir = Path(output_dir or self.results_dir / "qc")
        if save:
            output_dir.mkdir(exist_ok=True)
            channels_str = ",".join(channels)
            fig_file = output_dir / ".".join([self.name, f"all_rois.{channels_str}.pdf"])
        if per_sample:
            for sample in samples or self.samples:
                fig = sample.plot_channels(channels, **kwargs)
                if save:
                    fig_file = output_dir / ".".join(
                        [self.name, sample.name, f"all_rois.{channels_str}.pdf"]
                    )
                    fig.savefig(fig_file, **FIG_KWS)
        else:
            rois = self._get_rois(samples, rois)

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

    # TODO: write decorator to get/set default outputdir and handle dir creation
    def plot_probabilities_and_segmentation(
        self,
        jointly: bool = False,
        output_dir: Path = None,
        samples: tp.Sequence[IMCSample] = None,
        rois: tp.Sequence[ROI] = None,
    ):
        # TODO: adapt to detect whether to plot nuclei mask
        samples = samples or self.samples
        # for sample in samples:
        #     sample.read_all_inputs(only_these_keys=["probabilities", "cell_mask", "nuclei_mask"])
        output_dir = Path(output_dir or self.results_dir / "qc")
        os.makedirs(output_dir, exist_ok=True)
        if not jointly:
            for sample in samples:
                plot_file = output_dir / ".".join(
                    [
                        self.name,
                        sample.name,
                        "all_rois.plot_probabilities_and_segmentation.svg",
                    ]
                )
                fig = sample.plot_probabilities_and_segmentation()
                fig.savefig(plot_file, **FIG_KWS)
        else:
            rois = self._get_rois(samples, rois)
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
        cell_type_combinations: tp.Union[str, tp.List[tp.Tuple[str, str]]] = None,
        cell_type_assignments: DataFrame = None,
        palette: tp.Optional[str] = None,
        samples: tp.List[IMCSample] = None,
        rois: tp.List[ROI] = None,
    ) -> Figure:
        # TODO: fix compatibility of `cell_type_combinations`.
        samples = samples or self.samples
        rois = rois or self.rois

        n = len(samples)
        m = max([sample.n_rois for sample in samples])
        fig, axes = plt.subplots(n, m, figsize=(3 * m, 3 * n), squeeze=False)
        patches: tp.List[Patch] = list()
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
        red_func: str = "mean",
        channel_exclude: tp.Sequence[str] = None,
        plot: bool = True,
        output_prefix: Path = None,
        samples: tp.Sequence[IMCSample] = None,
        rois: tp.Sequence[ROI] = None,
        **kwargs,
    ) -> tp.Union[DataFrame, tp.Tuple[DataFrame, Figure]]:
        """
        Summary statistics on the signal of each channel across ROIs.
        """
        if output_prefix is None:
            output_prefix = self.results_dir / "qc" / self.name
        output_prefix.parent.mkdir()

        # for sample, _func in zip(samples or self.samples, red_func):
        samples = samples or self.samples
        rois = self._get_rois(samples, rois)

        _res = dict()
        for roi in rois:
            _res[roi.name] = pd.Series(
                getattr(roi.stack, red_func)(axis=(1, 2)),
                index=roi.channel_labels,
            )
        res = pd.DataFrame(_res)

        if res.isnull().any().any():
            res = align_channels_by_name(res)

        # filter channels out if requested
        if channel_exclude is not None:
            # to accomodate strings with especial characters like a parenthesis
            # (as existing in the metal), exclude exact matches OR containing strings
            exc = res.index.isin(channel_exclude) | res.index.str.contains(
                "|".join(channel_exclude)
            )
            res = res.loc[res.index[~exc]]
        res = res / res.mean()

        if plot:
            res = np.log1p(res)  # type: ignore[assignment]
            # calculate mean intensity
            channel_mean = res.mean(axis=1).rename("channel_mean")

            # calculate cell density
            cell_density = pd.Series(
                [roi.cells_per_area_unit() for roi in rois],
                index=[roi.name for roi in rois],
                name="cell_density",
            )
            if (cell_density < 0.1).all():
                cell_density *= 1000

            sample_names = pd.Series(
                [roi.sample.name if roi.sample is not None else None for roi in rois],
                index=cell_density.index,
                name="sample",
            )

            def_kwargs = dict(z_score=0, center=0, robust=True, cmap="RdBu_r")
            def_kwargs.update(kwargs)
            # TODO: add {row,col}_colors colorbar to heatmap
            for kws, label, cbar_label in [
                (dict(), "", ""),
                (def_kwargs, ".z_score", " (Z-score)"),
            ]:
                plot_file = output_prefix + f".mean_per_channel.clustermap{label}.svg"
                grid = clustermap(
                    res,
                    cbar_kws=dict(label=red_func.capitalize() + cbar_label),
                    row_colors=channel_mean,
                    col_colors=cell_density.to_frame().join(sample_names),
                    metric="correlation",
                    xticklabels=True,
                    yticklabels=True,
                    **kws,
                )
                grid.fig.suptitle("Mean channel intensities", y=1.05)
                grid.savefig(plot_file, dpi=300, bbox_inches="tight")
            grid.fig.grid = grid
            return (res, grid.fig)
        res.index.name = "channel"
        return res

    def image_summary(
        self,
        samples: tp.Sequence[IMCSample] = None,
        rois: tp.Sequence[ROI] = None,
    ):
        raise NotImplementedError
        from imc.utils import lacunarity, fractal_dimension

        rois = self._get_rois(samples, rois)
        roi_names = [r.name for r in rois]
        densities = pd.Series(
            {roi.name: roi.cells_per_area_unit() for roi in rois},
            name="cell density",
        )
        lacunarities = pd.Series(
            parmap.map(lacunarity, [roi.cell_mask_o for roi in rois], pm_pbar=True),
            index=roi_names,
            name="lacunarity",
        )
        fractal_dimensions = pd.Series(
            parmap.map(
                fractal_dimension,
                [roi.cell_mask_o for roi in rois],
                pm_pbar=True,
            ),
            index=roi_names,
            name="fractal_dimension",
        )

        morphos = pd.DataFrame([densities * 1e4, lacunarities, fractal_dimensions]).T

    def channel_correlation(
        self,
        channel_exclude: tp.List[str] = None,
        samples: tp.List[IMCSample] = None,
        rois: tp.List[ROI] = None,
    ) -> Figure:
        """
        Observe the pairwise correlation of channels across ROIs.
        """
        from imc.ops.quant import _correlate_channels__roi

        (self.results_dir / "qc").mkdir()

        rois = self._get_rois(samples, rois)
        _res = parmap.map(_correlate_channels__roi, rois, pm_pbar=True)

        # handling differnet pannels based on channel name
        # that then makes that concatenating dfs with duplicated names in indeces
        res = pd.concat(
            [x.groupby(level=0).mean().T.groupby(level=0).mean().T for x in _res]
        )
        xcorr = res.groupby(level=0).mean().fillna(0)
        labels = xcorr.index
        if channel_exclude is not None:
            exc = labels.isin(channel_exclude) | labels.str.contains(
                "|".join(channel_exclude)
            )
            xcorr = xcorr.loc[labels[~exc], labels[~exc]]
        xcorr.to_csv(
            self.results_dir / "qc" / self.name + ".channel_pairwise_correlation.csv"
        )

        grid = sns.clustermap(
            xcorr,
            cmap="RdBu_r",
            center=0,
            robust=True,
            xticklabels=True,
            yticklabels=True,
            cbar_kws=dict(label="Pearson correlation"),
        )
        grid.ax_col_dendrogram.set_title("Pairwise channel correlation\n(pixel level)")
        grid.savefig(
            self.results_dir / "qc" / self.name + ".channel_pairwise_correlation.svg",
            **FIG_KWS,
        )
        grid.fig.grid = grid
        return grid.fig

    @tp.overload
    def quantify_cells(
        self,
        layers: tp.List[str],
        intensity: bool,
        intensity_kwargs: tp.Dict[str, tp.Any],
        morphology: bool,
        morphology_kwargs: tp.Dict[str, tp.Any],
        set_attribute: tp.Literal[True],
        samples: tp.List[IMCSample],
        rois: tp.List[ROI],
    ) -> None:
        ...

    @tp.overload
    def quantify_cells(
        self,
        layers: tp.List[str],
        intensity: bool,
        intensity_kwargs: tp.Dict[str, tp.Any],
        morphology: bool,
        morphology_kwargs: tp.Dict[str, tp.Any],
        set_attribute: tp.Literal[False],
        samples: tp.List[IMCSample],
        rois: tp.List[ROI],
    ) -> DataFrame:
        ...

    def quantify_cells(
        self,
        layers: tp.Sequence[str] = ["cell"],
        intensity: bool = True,
        intensity_kwargs: tp.Dict[str, tp.Any] = {},
        morphology: bool = True,
        morphology_kwargs: tp.Dict[str, tp.Any] = {},
        set_attribute: bool = True,
        samples: tp.Sequence[IMCSample] = None,
        rois: tp.Sequence[ROI] = None,
    ) -> tp.Optional[DataFrame]:
        """
        Measure the channel and morphological features of each single cell.
        """
        from imc.ops.quant import quantify_cells_rois

        quantification = quantify_cells_rois(
            self._get_rois(samples, rois),
            layers=layers,
            intensity=intensity,
            intensity_kwargs=intensity_kwargs,
            morphology=morphology,
            morphology_kwargs=morphology_kwargs,
        )
        if not set_attribute:
            return quantification
        self.quantification = quantification
        return None

    def quantify_cell_intensity(
        self,
        samples: tp.Sequence[IMCSample] = None,
        rois: tp.Sequence[ROI] = None,
        **kwargs,
    ) -> DataFrame:
        """
        Measure the intensity of each channel in each single cell.

        Parameters
        ----------
        samples: Sequence[Sample]
            Subset of samples to use.
            Default is all.
        rois:  Sequence[ROI]
            Subset of samples to use.
            Default is all.
        kwargs:
            Additional keyword-arguments will be passed to `imc.ops.quant.quantify_cell_intensity`:
            red_func: str
                Function to reduce values per cell.
                Default is "mean".
            border_objs: bool
                Whether to quantify objects touching image border.
                Default is `False`.
            equalize: bool
                Whether to scale the signal. This is actually a cap on the 98th percentile.
                Default is `True`.
                TODO: change keyword name to 'percentile_scale'.
            scale: bool
                Whether to scale signal to unit space.
                Default is `False`.
            channel_include: Array
                Sequence of channels to include. This is a boolean array matching the ROI channels.
                Default is `None`: all channels.
            channel_exclude: Array
                Sequence of channels to exclude. This is a boolean array matching the ROI channels.
                Default is `None`: no channels.
        """
        from imc.ops.quant import quantify_cell_intensity_rois

        return quantify_cell_intensity_rois(self._get_rois(samples, rois), **kwargs)

    def quantify_cell_morphology(
        self,
        samples: tp.Sequence[IMCSample] = None,
        rois: tp.Sequence[ROI] = None,
        **kwargs,
    ) -> DataFrame:
        """
        Measure the shape parameters of each single cell.

        Parameters
        ----------
        samples: Sequence[Sample]
            Subset of samples to use.
            Default is all.
        rois:  Sequence[ROI]
            Subset of samples to use.
            Default is all.
        kwargs:
            Additional keyword-arguments will be passed to `imc.ops.quant.quantify_cell_morphology`:

            attributes: Sequence[str]
                Attributes to quantify. For extensive list refer to https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
            border_objs: bool
                Whether to quantify objects touching image border.
                Default is `False`.
        """
        from imc.ops.quant import quantify_cell_morphology_rois

        return quantify_cell_morphology_rois(self._get_rois(samples, rois), **kwargs)

    def cluster_cells(
        self,
        output_prefix: Path = None,
        plot: bool = True,
        set_attribute: bool = True,
        samples: tp.Sequence[IMCSample] = None,
        rois: tp.Sequence[ROI] = None,
        **kwargs,
    ) -> tp.Optional[Series]:
        """
        Derive clusters of single cells based on their channel intensity.
        """
        output_prefix = Path(
            output_prefix or self.results_dir / "single_cell" / self.name
        )

        if "quantification" not in kwargs and self.quantification is not None:
            kwargs["quantification"] = self.quantification
        if "cell_type_channels" not in kwargs and self.panel_metadata is not None:
            if "cell_type" in self.panel_metadata.columns:
                kwargs["cell_type_channels"] = self.panel_metadata.query(
                    "cell_type == 1"
                ).index.list()

        clusters = single_cell_analysis(
            output_prefix=output_prefix,
            rois=self._get_rois(samples, rois),
            plot=plot,
            **kwargs,
        )
        # save clusters as CSV in default file
        clusters.reset_index().to_csv(
            self.get_input_filename("cell_cluster_assignments"), index=False
        )
        if not set_attribute:
            return clusters

        # Set clusters for project and propagate for Samples and ROIs.
        # in principle there was no need to pass clusters here as it will be read
        # however, the CSV roundtrip might give problems in edge cases, for
        # example when the sample name is only integers
        self.set_clusters(clusters.astype(str))
        return None

    @property
    def clusters(self):
        if self._clusters is not None:
            return self._clusters
        self.set_clusters()
        return self._clusters

    def set_clusters(
        self,
        clusters: MultiIndexSeries = None,
        write_to_disk: bool = False,
        samples: tp.Sequence[IMCSample] = None,
    ) -> None:
        """
        Set the `clusters` attribute of the project and
        propagate it to the Samples and their ROIs.

        If not given, `clusters` is the output of
        :func:`Project.get_input_filename`("cell_cluster_assignments").
        """
        id_cols = ["sample", "roi", "obj_id"]
        fn = self.get_input_filename("cell_cluster_assignments")
        fn.parent.mkdir()
        if clusters is None:
            clusters = (
                pd.read_csv(
                    fn,
                    dtype={"sample": str, "roi": str},
                ).set_index(id_cols)
            )[
                "cluster"
            ]  # .astype(str)
        assert isinstance(
            clusters.index, pd.MultiIndex
        ), "Series index must be MultiIndexSeries with levels 'sample', 'roi', 'obj_id'"
        assert (
            clusters.index.names == id_cols
        ), "Series level names must be 'sample', 'roi', 'obj_id'"
        self._clusters = clusters
        for sample in samples or self.samples:
            sample.set_clusters(clusters=clusters.loc[sample.name])
        if write_to_disk:
            self._clusters.reset_index().to_csv(fn, index=False)

    def label_clusters(
        self,
        h5ad_file: Path = None,
        output_prefix: Path = None,
        **kwargs,
    ) -> None:
        """
        Derive labels for each identified cluster
        based on its most abundant markers.
        """
        prefix = self.results_dir / "single_cell" / self.name
        h5ad_file = Path(h5ad_file or prefix + ".single_cell.processed.h5ad")
        output_prefix = Path(output_prefix or prefix + ".cell_type_reference")
        new_labels = derive_reference_cell_type_labels(h5ad_file, output_prefix, **kwargs)
        self._rename_clusters(new_labels.to_dict())

    def _rename_clusters(self, new_labels: dict, save: bool = True):
        clusters = cast(self.clusters).replace(new_labels)
        if save:
            clusters.reset_index().to_csv(
                self.get_input_filename("cell_cluster_assignments"),
                index=False,
            )
        self.set_clusters(clusters)

    def sample_comparisons(
        self,
        sample_attributes: tp.Union[
            tp.Sequence[str], tp.Dict[str, tp.Sequence[str]]
        ] = None,
        output_prefix: Path = None,
        channel_exclude: tp.Sequence[str] = None,
        samples: tp.Sequence[IMCSample] = None,
        rois: tp.Sequence[ROI] = None,
        steps: tp.Sequence[str] = ["channel_mean", "cell_type_abundance"],
    ):
        """
        Compare channel intensity and cellular abundance between sample attributes.
        """
        from seaborn_extensions import swarmboxenplot, volcano_plot

        attributes = (
            list(sample_attributes.keys())
            if isinstance(sample_attributes, dict)
            else ["name"]
            if sample_attributes is None
            else sample_attributes
        )
        samples = samples or self.samples
        rois = self._get_rois(samples, rois)
        output_prefix = (
            output_prefix or self.results_dir / "single_cell" / self.name + "."
        )
        output_prefix.parent.mkdir(exist_ok=True)

        # group samples by desired attributes
        sample_df = (
            pd.DataFrame(
                {k: v for k, v in sample.__dict__.items() if isinstance(v, str)}
                for sample in samples
            )[["name"] + attributes]
            .set_index("name")
            .rename_axis("sample")
            .reset_index()
        )
        if isinstance(sample_attributes, dict):
            for attr, order in sample_attributes.items():
                sample_df[attr] = pd.Categorical(
                    sample_df[attr], ordered=True, categories=order
                )

        sample_roi_df = pd.DataFrame(
            [(roi.name, roi.sample.name) for roi in rois],
            columns=["roi", "sample"],
        )

        to_plot = list()

        # Whole channel means
        if "channel_mean" in steps:
            to_plot.append((channel_df, "channel", "channel_mean", "value"))
            channel_means: DataFrame = self.channel_summary(
                plot=False, channel_exclude=channel_exclude
            )
            channel_means.index.name = "channel"
            channel_means = (
                channel_means.reset_index()
                .melt(id_vars="channel", var_name="roi")
                .reset_index(drop=True)
            )
            channel_df = (
                channel_means.merge(sample_roi_df)
                .merge(sample_df)
                .sort_values(attributes)
            )
            channel_df.to_csv(output_prefix + "channel_mean.csv", index=False)

        # cell type abundance per sample or group of samples
        if "cell_type_abundance" in steps:
            to_plot.append(
                (cluster_df, "cluster", "cell_type_abundance_area", "cell_mm2")
            )
            to_plot.append(
                (cluster_df, "cluster", "cell_type_abundance_percentage", "cell_perc")
            )

            cluster_counts = (
                self.clusters.groupby(level=["sample", "roi"])
                .value_counts()
                .rename_axis(["sample", "roi", "cluster"])
                .rename("cell_count")
            )
            # # absolute
            cluster_df = (
                cluster_counts.reset_index().merge(sample_df).sort_values(attributes)
            )
            # # per area
            area = pd.Series({r.name: r.area for r in rois}, name="area").rename_axis(
                "roi"
            )
            cluster_df = cluster_df.merge(area.reset_index())  # type: ignore[union-attr]
            cluster_df["cell_mm2"] = (cluster_df["cell_count"] / cluster_df["area"]) * 1e6
            # # fraction of total
            cluster_df["cell_perc"] = cluster_df.groupby("roi")["cell_count"].apply(
                lambda x: (x / x.sum()) * 100
            )
            cluster_df = cluster_df.drop("area", axis=1)
            cluster_df.to_csv(output_prefix + "cell_type_abundance.csv", index=False)

        # Test difference between channels/clusters
        # # plot boxplots, volcano, and heatmaps
        for df, var, data_type, quant in to_plot:
            for attr in attributes:
                pref = output_prefix + f"{data_type}.{quant}.testing_between_attributes"
                p = df.pivot_table(
                    index=["roi", attr], columns=var, values=quant, fill_value=0
                ).reset_index()

                fig, stats = swarmboxenplot(
                    data=p, x=attr, y=p.columns.drop([attr, "roi"])
                )
                stats.to_csv(pref + ".csv", index=False)
                fig.savefig(pref + ".swarmboxenplot.svg", **FIG_KWS)
                plt.close(fig)

                fig = volcano_plot(stats)
                fig.savefig(pref + ".volcano_plot.svg", **FIG_KWS)
                plt.close(fig)

                pp = p.drop(attr, axis=1).set_index("roi")
                pp = pp.loc[pp.var(1) > 0, pp.var(0) > 0]
                c = p.set_index("roi")[attr]
                grid = clustermap(pp, config="abs", row_colors=c)
                grid.fig.savefig(pref + ".clustermap.abs.svg", **FIG_KWS)
                plt.close(grid.fig)

                grid = clustermap(pp, config="z", row_colors=c)
                grid.fig.savefig(pref + ".clustermap.z.svg", **FIG_KWS)
                plt.close(grid.fig)

    def measure_adjacency(
        self,
        output_prefix: Path = None,
        samples: tp.List[IMCSample] = None,
        rois: tp.List[ROI] = None,
    ) -> None:
        """
        Derive cell adjacency graphs for each ROI.
        """
        output_prefix = (
            output_prefix or self.results_dir / "single_cell" / self.name + "."
        )
        rois = self._get_rois(samples, rois)

        # Get graph for missing ROIs
        _rois = [r for r in rois if r._adjacency_graph is None]
        if _rois:
            gs = parmap.map(get_adjacency_graph, _rois, pm_pbar=True)
            # gs = [get_adjacency_graph(roi) for roi in _rois]
            for roi, g in zip(_rois, gs):
                roi._adjacency_graph = g

        # TODO: package the stuff below into a function

        # First measure adjacency as odds against background
        freqs = parmap.map(measure_cell_type_adjacency, rois, pm_pbar=True)
        # freqs = [measure_cell_type_adjacency(roi) for roi in rois]
        # freqs = [
        #     pd.read_csv(
        #         roi.sample.root_dir / "single_cell" / roi.name
        #         + ".cluster_adjacency_graph.norm_over_random.csv",
        #         index_col=0,
        #     )
        #     for roi in rois
        # ]

        melted = pd.concat(
            [
                f.reset_index()
                .melt(id_vars="index")
                .assign(sample=roi.sample.name, roi=roi.name)
                for roi, f in zip(rois, freqs)
            ]
        )
        melted.to_csv(output_prefix + "adjacency_frequencies.csv")

        mean_f = melted.pivot_table(
            index="index", columns="variable", values="value", aggfunc=np.mean
        )
        sns.clustermap(mean_f, cmap="RdBu_r", center=0, robust=True)

        # v = np.percentile(melted["value"].abs(), 95)
        # n, m = get_grid_dims(len(freqs))
        # fig, axes = plt.subplots(n, m, figsize=(m * 5, n * 5), sharex=True, sharey=True)
        # axes = axes.flatten()
        # i = -1
        # for i, (dfs, roi) in enumerate(zip(freqs, rois)):
        #     axes[i].set_title(roi.name)
        #     sns.heatmap(
        #         dfs,
        #         ax=axes[i],
        #         cmap="RdBu_r",
        #         center=0,
        #         rasterized=True,
        #         square=True,
        #         xticklabels=True,
        #         yticklabels=True,
        #         vmin=-v,
        #         vmax=v,
        #     )
        # for axs in axes[i + 1 :]:
        #     axs.axis("off")
        # fig.savefig(output_prefix + "adjacency.all_rois.pdf", **FIG_KWS)

        return melted

    def find_communities(
        self,
        output_prefix: Path = None,
        samples: tp.List[IMCSample] = None,
        rois: tp.List[ROI] = None,
        **kwargs,
    ) -> None:
        """
        Find communities and supercommunities of cell types across all images.
        """
        rois = self._get_rois(samples, rois)
        cluster_communities(rois=rois, output_prefix=output_prefix, **kwargs)
