"""
Functions for high order operations.
"""

# fix the type annotatiton of not yet undefined classes
from __future__ import annotations
from collections import Counter
import os
import re
from typing import Tuple, List, Optional, Dict, Union, Literal, Any, overload

from ordered_set import OrderedSet
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import parmap

from scipy.cluster.hierarchy import fcluster
import scipy.ndimage as ndi
from scipy.stats import pearsonr
from skimage import exposure
import skimage
from skimage.future import graph
import skimage.io
import skimage.measure
from skimage.restoration import estimate_sigma
from skimage.segmentation import clear_border
from sklearn.linear_model import LinearRegression

from anndata import AnnData
import scanpy as sc
import networkx as nx
import community

from imc import MEMORY
from imc.exceptions import cast
from imc.types import DataFrame, Series, Array, Path, MultiIndexSeries
from imc.utils import (
    read_image_from_file,
    estimate_noise,
    double_z_score,
    minmax_scale,
)
from imc.graphics import get_grid_dims, rasterize_scanpy, add_legend


matplotlib.rcParams["svg.fonttype"] = "none"
FIG_KWS = dict(bbox_inches="tight", dpi=300)
sc.settings.n_jobs = -1


DEFAULT_SINGLE_CELL_RESOLUTION = 1.0
MAX_BETWEEN_CELL_DIST = 4
DEFAULT_COMMUNITY_RESOLUTION = 0.005
DEFAULT_SUPERCOMMUNITY_RESOLUTION = 0.5
# DEFAULT_SUPER_COMMUNITY_NUMBER = 12


def quantify_cell_intensity(
    stack: Union[Array, Path],
    mask: Union[Array, Path],
    red_func: str = "mean",
    border_objs: bool = False,
    equalize: bool = True,
    scale: bool = False,
    channel_include: Optional[Array] = None,
    channel_exclude: Optional[Array] = None,
) -> DataFrame:
    """
    Measure the intensity of each channel in each cell


    Parameters
    ----------
    stack: Union[Array, Path]
        Image to quantify.
    mask: Union[Array, Path]
        Mask to quantify.
    red_func: str
        Function to reduce pixels to object borders. Defaults to 'mean'.
    border_objs: bool
        Whether to quantify objects touching image border. Defaults to False.
    channel_include: :class:`~np.ndarray`
        Boolean array for channels to include.
    channel_exclude: :class:`~np.ndarray`
        Boolean array for channels to exclude.
    """
    from skimage.exposure import equalize_hist as eq

    if isinstance(stack, Path):
        stack = read_image_from_file(stack)
    if isinstance(mask, Path):
        mask = read_image_from_file(mask)
    if not border_objs:
        mask = clear_border(mask)

    if equalize:
        # stack = np.asarray([eq(x) for x in stack])
        _stack = list()
        for x in stack:
            p = np.percentile(x, 98)
            x[x > p] = p
            _stack.append(x)
        stack = np.asarray(_stack)
    if scale:
        stack = np.asarray([minmax_scale(x) for x in stack])

    cells = np.unique(mask)
    # the minus one here is to skip the background "0" label which is also
    # ignored by `skimage.measure.regionprops`.
    n_cells = len(cells) - 1
    n_channels = stack.shape[0]

    if channel_include is None:
        channel_include = np.asarray([True] * n_channels)
    if channel_exclude is None:
        channel_exclude = np.asarray([False] * n_channels)

    res = np.zeros(
        (n_cells, n_channels), dtype=int if red_func == "sum" else float
    )
    for channel in np.arange(stack.shape[0])[
        channel_include & ~channel_exclude
    ]:
        res[:, channel] = [
            getattr(x.intensity_image, red_func)()
            for x in skimage.measure.regionprops(mask, stack[channel])
        ]
    return pd.DataFrame(res, index=cells[1:]).rename_axis(index="obj_id")


def quantify_cell_morphology(
    mask: Union[Array, Path],
    attributes: List[str] = [
        "area",
        "perimeter",
        "major_axis_length",
        # 'minor_axis_length',  in some images I get ValueError
        # just like https://github.com/scikit-image/scikit-image/issues/2625
        # 'orientation',
        # orientation should be random, so I'm not including it
        "eccentricity",
        "solidity",
        "centroid",
    ],
    border_objs: bool = False,
) -> DataFrame:
    if isinstance(mask, Path):
        mask = read_image_from_file(mask)
    if not border_objs:
        mask = clear_border(mask)

    return pd.DataFrame(
        skimage.measure.regionprops_table(mask, properties=attributes),
        index=np.unique(mask)[1:],
    ).rename_axis(index="obj_id")


def _quantify_cell_intensity__roi(roi: "ROI", **kwargs) -> DataFrame:
    assignment = dict(roi=roi.name)
    if roi.sample is not None:
        assignment["sample"] = roi.sample.name
    return roi.quantify_cell_intensity(**kwargs).assign(**assignment)


def _quantify_cell_morphology__roi(roi: "ROI", **kwargs) -> DataFrame:
    assignment = dict(roi=roi.name)
    if roi.sample is not None:
        assignment["sample"] = roi.sample.name
    return roi.quantify_cell_morphology(**kwargs).assign(**assignment)


def _correlate_channels__roi(
    roi: "ROI", labels: str = "channel_names"
) -> DataFrame:
    xcorr = np.corrcoef(roi.stack.reshape((roi.channel_number, -1)))
    np.fill_diagonal(xcorr, 0)
    labs = getattr(roi, labels)
    return pd.DataFrame(xcorr, index=labs, columns=labs)


# def _get_adjacency_graph__roi(roi: "ROI", **kwargs) -> DataFrame:
#     output_prefix = roi.sample.root_dir / "single_cell" / roi.name
#     return get_adjacency_graph(roi.stack, roi.mask, roi.clusters, output_prefix, **kwargs)


def quantify_cell_intensity_rois(
    rois: List["ROI"],
    **kwargs,
) -> DataFrame:
    """
    Measure the intensity of each channel in each single cell.
    """
    return pd.concat(
        parmap.map(_quantify_cell_intensity__roi, rois, pm_pbar=True, **kwargs)
    ).rename_axis(index="obj_id")


def quantify_cell_morphology_rois(
    rois: List["ROI"],
    **kwargs,
) -> DataFrame:
    """
    Measure the shape parameters of each single cell.
    """
    return pd.concat(
        parmap.map(_quantify_cell_morphology__roi, rois, pm_pbar=True, **kwargs)
    ).rename_axis(index="obj_id")


def quantify_cells_rois(
    rois: List["ROI"],
    layers: List[str],
    intensity: bool = True,
    intensity_kwargs: Dict[str, Any] = {},
    morphology: bool = True,
    morphology_kwargs: Dict[str, Any] = {},
) -> DataFrame:
    """
    Measure the intensity of each channel in each single cell.
    """
    quants = list()
    if intensity:
        quants.append(
            quantify_cell_intensity_rois(
                rois=rois, layers=layers, **intensity_kwargs
            )
        )
    if morphology:
        quants.append(
            quantify_cell_morphology_rois(
                rois=rois, layers=layers, **morphology_kwargs
            )
        )

    return (
        # todo: this will fail if there's different layers in intensity and morphology
        pd.concat(
            [quants[0].drop(["sample", "roi"], axis=1), quants[1]], axis=1
        )
        if len(quants) > 1
        else quants[0]
    ).rename_axis(index="obj_id")


def check_channel_axis_correlation(
    arr: Array, channel_labels: List[str], output_prefix: Path
) -> DataFrame:
    # # Plot and regress
    n, m = get_grid_dims(arr.shape[0])
    fig, axis = plt.subplots(
        m, n, figsize=(n * 4, m * 4), squeeze=False, sharex=True, sharey=True
    )

    res = list()
    for channel in range(arr.shape[0]):
        for axs in [0, 1]:
            s = arr[channel].mean(axis=axs)
            order = np.arange(s.shape[0])
            model = LinearRegression()
            model.fit(order[:, np.newaxis] / max(order), s)
            res.append(
                [
                    channel,
                    axs,
                    model.coef_[0],
                    model.intercept_,
                    pearsonr(order, s)[0],
                ]
            )

            axis.flatten()[channel].plot(order, s)
        axis.flatten()[channel].set_title(
            f"{channel_labels[channel]}\nr[X] = {res[-2][-1]:.2f}; r[Y] = {res[-1][-1]:.2f}"
        )

    axis[int(m / 2), 0].set_ylabel("Mean signal along axis")
    axis[-1, int(n / 2)].set_xlabel("Order along axis")
    c = sns.color_palette("colorblind")
    patches = [
        mpatches.Patch(color=c[0], label="X"),
        mpatches.Patch(color=c[1], label="Y"),
    ]
    axis[int(m / 2), -1].legend(
        handles=patches,
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
        title="Axis",
    )
    fig.savefig(output_prefix + "channel-axis_correlation.svg", **FIG_KWS)

    df = pd.DataFrame(
        res, columns=["channel", "axis", "coef", "intercept", "r"]
    )
    df["axis_label"] = df["axis"].replace(0, "X").replace(1, "Y")
    df["channel_label"] = [x for x in channel_labels for _ in range(2)]
    df["abs_r"] = df["r"].abs()
    df.to_csv(output_prefix + "channel-axis_correlation.csv", index=False)
    return df


def fix_signal_axis_dependency(
    arr: Array, channel_labels: List[str], res: DataFrame, output_prefix: Path
) -> Array:
    # res = pd.read_csv(pjoin("processed", "case_b", "plots", "qc", roi + "_channel-axis_correlation.csv"))
    corr_d = np.empty_like(arr)
    for channel in range(arr.shape[0]):
        r = res.query(f"channel == {channel}")
        x = r.query("axis_label == 'X'")["coef"].squeeze()
        xinter = r.query("axis_label == 'X'")["intercept"].squeeze()
        y = r.query("axis_label == 'Y'")["coef"].squeeze()
        yinter = r.query("axis_label == 'Y'")["intercept"].squeeze()
        # to_reg = pd.DataFrame(arr[channel]).reset_index().melt(id_vars='index').rename(columns=dict(index="X", variable="Y"))

        order = np.arange(arr[channel].shape[0])
        dd = arr[channel]
        m = np.ones_like(dd)
        m = m * (order / max(order) * x) + (xinter)
        m = (m.T * (order / max(order) * y)).T + (yinter)
        ddfix = (dd - m) + dd.mean()
        corr_d[channel] = ddfix

        fig, axis = plt.subplots(
            1, 7, sharex=True, sharey=False, figsize=(7 * 3, 3 * 1)
        )
        fig.suptitle(channel_labels[channel])
        axis[0].set_title("Original")
        axis[0].imshow(dd)
        axis[1].set_title("Original, equalized")
        axis[1].imshow(exposure.equalize_hist(dd))
        axis[2].set_title("Bias mask")
        axis[2].imshow(m)
        axis[3].set_title("Bias removed")
        axis[3].imshow(ddfix)
        axis[4].set_title("Bias removed, equalized")
        axis[4].imshow(exposure.equalize_hist(ddfix))
        axis[5].set_title("Channel bias")
        axis[5].plot(order, dd.mean(axis=0), label="Original", alpha=0.5)
        axis[5].plot(order, ddfix.mean(axis=0), label="Bias removed", alpha=0.5)
        axis[5].set_xlabel("Position along X axis")
        axis[5].set_ylabel("Signal along X axis")
        axis[5].legend()
        axis[6].set_title("Channel bias")
        axis[6].plot(order, dd.mean(axis=1), label="Original", alpha=0.5)
        axis[6].plot(order, ddfix.mean(axis=1), label="Bias removed", alpha=0.5)
        axis[6].set_xlabel("Position along Y axis")
        axis[6].set_ylabel("Signal along Y axis")
        axis[6].legend()
        for ax in axis[:-2]:
            ax.axis("off")
        fig.savefig(
            output_prefix
            + f"channel-axis_correlation_removal.{channel_labels[channel]}.demonstration.svg",
            **FIG_KWS,
        )
        plt.close("all")
    return corr_d


@MEMORY.cache
def measure_channel_background(
    rois: List["ROI"], plot: bool = True, output_prefix: Optional[Path] = None
) -> Series:
    # Quantify whole image area
    _imeans: Dict[str, Series] = dict()
    _istds: Dict[str, Series] = dict()
    # Quantify only cell area
    _cmeans: Dict[str, Series] = dict()
    _cstds: Dict[str, Series] = dict()
    for roi in rois:
        stack = roi.stack
        mask = roi.cell_mask
        _imeans[roi.name] = pd.Series(
            stack.mean(axis=(1, 2)), index=roi.channel_labels
        )
        _istds[roi.name] = pd.Series(
            stack.std(axis=(1, 2)), index=roi.channel_labels
        )
        _cmeans[roi.name] = pd.Series(
            [
                np.ma.masked_array(stack[i], mask).mean()
                for i in range(roi.channel_number)
            ],
            index=roi.channel_labels,
        )
        _cstds[roi.name] = pd.Series(
            [
                np.ma.masked_array(stack[i], mask).std()
                for i in range(roi.channel_number)
            ],
            index=roi.channel_labels,
        )
    imeans = pd.DataFrame(_imeans) + 1
    istds = pd.DataFrame(_istds)
    iqv2s = np.sqrt(istds / imeans)
    cmeans = pd.DataFrame(_cmeans) + 1
    cstds = pd.DataFrame(_cstds)
    cqv2s = np.sqrt(cstds / cmeans)

    fore_backg = ((imeans - cmeans) / (cmeans + imeans)).mean(1)
    fore_backg_disp = pd.Series(np.log1p(fore_backg.abs() * 1e4))

    noise = pd.Series(
        np.fromiter(
            map(estimate_noise, [lay for roi in rois for lay in roi.stack]),
            float,
        )
        .reshape(len(rois), -1)
        .mean(0),
        index=cmeans.index,
    )

    sigmas = pd.DataFrame(
        parmap.map(
            estimate_sigma,
            [np.moveaxis(roi.stack, 0, -1) for roi in rois],
            multichannel=True,
        ),
        index=[roi.name for roi in rois],
        columns=rois[0].channel_labels,
    ).T

    # Join all metrics
    metrics = (
        cmeans.mean(1)
        .to_frame(name="cell_mean")
        # .join(cstds.mean(1).rename("cell_std"))
        .join(cqv2s.mean(1).rename("cell_qv2"))
        .join(imeans.mean(1).rename("image_mean"))
        # .join(istds.mean(1).rename("image_std"))
        .join(iqv2s.mean(1).rename("image_qv2"))
        .join(fore_backg_disp.rename("fore_backg"))
        .join(noise.rename("noise"))
        .join(sigmas.mean(1).rename("sigma"))
    )
    metrics_std = (metrics - metrics.min()) / (metrics.max() - metrics.min())

    if not plot:
        return metrics_std.mean(1)

    output_prefix = cast(output_prefix)
    if not output_prefix.endswith("."):
        output_prefix += "."

    metrics.to_csv(output_prefix + "channel_background_noise_measurements.csv")
    metrics_std.to_csv(
        output_prefix + "channel_background_noise_measurements.standardized.csv"
    )

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(2 * 4, 2 * 4), sharex="col")
    axes[0, 0].set_title("Image")
    axes[0, 1].set_title("Cells")

    # plot mean vs std
    for i, (means, stds, qv2s) in enumerate(
        [(imeans, istds, iqv2s), (cmeans, cstds, cqv2s)]
    ):
        mean = means.mean(1)
        std = stds.mean(1)
        qv2 = qv2s.mean(1)

        axes[0, i].set_xlabel("Mean")
        axes[0, i].set_ylabel("Standard deviation")
        axes[0, i].scatter(mean, std, c=fore_backg_disp)
        for channel in means.index:
            lab = "left" if np.random.rand() > 0.5 else "right"
            axes[0, i].text(
                mean.loc[channel], std.loc[channel], channel, ha=lab, fontsize=4
            )
        v = max(mean.max().max(), std.max().max())
        axes[0, i].plot((0, v), (0, v), linestyle="--", color="grey")
        axes[0, i].loglog()

        # plot mean vs qv2
        axes[1, i].set_xlabel("Mean")
        axes[1, i].set_ylabel("Squared coefficient of variation")
        axes[1, i].scatter(mean, qv2, c=fore_backg_disp)
        for channel in means.index:
            lab = "left" if np.random.rand() > 0.5 else "right"
            axes[1, i].text(
                mean.loc[channel], qv2.loc[channel], channel, ha=lab, fontsize=4
            )
        axes[1, i].axhline(1, linestyle="--", color="grey")
        axes[1, i].set_xscale("log")
        if qv2.min() > 0.01:
            axes[1, i].set_yscale("log")
    fig.savefig(output_prefix + "channel_mean_variation_noise.svg", **FIG_KWS)

    fig, axes = plt.subplots(1, 2, figsize=(2 * 4, 4))
    p = fore_backg.sort_values()
    axes[0].scatter(p.rank(), p)
    axes[1].scatter(p.abs().rank(), p.abs())
    axes[1].set_yscale("log")
    axes[0].set_xlabel("Channel rank")
    axes[1].set_xlabel("Channel rank")
    axes[0].set_ylabel("Foreground/Background difference")
    axes[1].set_ylabel("Foreground/Background difference (abs)")
    axes[0].axhline(0, linestyle="--", color="grey")
    axes[1].axhline(0, linestyle="--", color="grey")
    fig.savefig(
        output_prefix + "channel_foreground_background_diff.rankplot.svg",
        **FIG_KWS,
    )

    grid = sns.clustermap(metrics_std, xticklabels=True, yticklabels=True)
    grid.fig.savefig(
        output_prefix + "channel_mean_variation_noise.clustermap.svg", **FIG_KWS
    )
    # TODO: review what metrics should be included in the final decision
    return metrics_std.mean(1)


def anndata_to_cluster_means(
    ann: anndata.AnnData, raw: bool = False, cluster_label: str = "cluster"
) -> DataFrame:
    means = dict()
    obj = ann if not raw else ann.raw
    for cluster in ann.obs[cluster_label].unique():
        clust = ann.obs[cluster_label] == cluster
        means[cluster] = obj[clust, :].X.mean(0)
    mean_expr = pd.DataFrame(means, index=obj.var.index).sort_index(axis=1)
    mean_expr.columns.name = "cluster"
    return mean_expr


def single_cell_analysis(
    output_prefix: Path,
    quantification: Optional[DataFrame] = None,
    rois: Optional[List["ROI"]] = None,
    label_clusters: bool = True,
    plot: bool = True,
    intensity: bool = True,
    morphology: bool = True,
    filter_channels: bool = False,
    cell_type_channels: Optional[List[str]] = None,
    channel_filtering_threshold: float = 0.1,  # 0.12
    channel_include: Optional[List[str]] = None,
    channel_exclude: List[str] = [
        "<EMPTY>",
        "EMPTY",
        "Ar80",
        "Ru9",
        "Ru10",
    ],  # r"Ru\d+", "DNA"
    cluster_min_percentage: float = 1.0,
    leiden_clustering_resolution: float = DEFAULT_SINGLE_CELL_RESOLUTION,
    plot_only_channels: List[str] = None,
) -> MultiIndexSeries:
    """

    cell_type_channels: These channels will be used for clustering cell types.
                        By default all are included. Subject to `channel_include`.
                        `channel_exclude` and outcome of `filter_channels` above
                        `channel_filtering_threshold`.
    channel_include: These channels will always be included for quantification
                     unless `filter_channels` is True and they do not pass
                     `channel_filtering_threshold`.
    channel_exclude: These channels will not be used either for quantification.
    """
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    if not str(output_prefix).endswith("."):
        output_prefix = Path(str(output_prefix) + ".")

    if quantification is None and rois is None:
        raise ValueError("One of `quantification` or `rois` must be given.")
    rois = cast(rois)

    # TODO: check all ROIs have same channels
    channel_labels = rois[0].channel_labels
    if filter_channels:
        print("Filtering channels.")
        metric = measure_channel_background(
            rois, plot=plot, output_prefix=output_prefix
        )
        channel_threshold = metric > channel_filtering_threshold
        filtered_channels = metric[channel_threshold].index.tolist()
    else:
        channel_threshold = pd.Series(
            [True] * len(channel_labels), index=channel_labels
        )
        filtered_channels = channel_labels.tolist()

    if quantification is None:
        print("Quantifying single cells.")
        quantification = quantify_cells_rois(
            rois=rois, intensity=intensity, morphology=morphology
        )

    # Remove excluded channels
    for _ch in channel_exclude:
        quantification = quantification.loc[
            :, ~quantification.columns.str.contains(_ch)
        ]
    # Filter out low QC channels
    if filter_channels:
        # TODO: fileter channels by QC metrics
        pass

    # Keep only include channels
    if channel_include is not None:
        _includes = [
            _ch for _ch in quantification.columns if _ch in channel_include
        ]
        quantification = quantification.loc[:, _includes]

    # Get categoricals
    cats = [x for x in ["sample", "roi"] if x in quantification.columns]

    # Start usual single cell analysis
    ann = AnnData(
        quantification.drop(cats, axis=1)
        .sort_index(axis=1)
        .reset_index(drop=True)
    )
    for cat in cats:
        ann.obs[cat] = pd.Categorical(quantification[cat].values)
    ann.obs["obj_id"] = quantification.index
    ann.obs["label"] = ann.obs[cats].astype(str).apply(", ".join, axis=1)
    ann.raw = ann

    ann.obs["n_counts"] = ann.X.sum(axis=1).astype(int)
    ann.obs["log_counts"] = np.log10(ann.obs["n_counts"])

    # normalize
    sc.pp.normalize_per_cell(ann, counts_per_cell_after=1e4)
    sc.pp.log1p(ann)

    # Create temporary Anndata for cell type discovery
    ann_ct = ann.copy()
    # select only requested channels
    if cell_type_channels is not None:
        _includes = ann.var.index.isin(cell_type_channels)
        ann_ct = ann_ct[:, _includes]
    # remove "batch" effect
    if "sample" in cats and len(ann_ct.obs["sample"].unique()) > 1:
        sc.pp.combat(ann_ct, "sample")

    # dim res
    sc.pp.scale(ann_ct, max_value=10)
    sc.pp.pca(ann_ct)
    sc.pp.neighbors(ann_ct, n_neighbors=8, use_rep="X")
    sc.tl.umap(ann_ct)
    sc.tl.leiden(
        ann_ct, key_added="cluster", resolution=leiden_clustering_resolution
    )

    ann_ct.obs["cluster"] = pd.Categorical(
        ann_ct.obs["cluster"].astype(int) + 1
    )
    ann.obs["cluster"] = ann_ct.obs["cluster"]
    ann.obsm = ann_ct.obsm
    # sc.tl.diffmap(ann)

    # Generate marker-based labels for clusters
    if label_clusters:
        new_labels = derive_reference_cell_type_labels(
            mean_expr=anndata_to_cluster_means(ann),
            cluster_assignments=ann.obs["cluster"],
            cell_type_channels=ann_ct.var.index,
            output_prefix=output_prefix,
            plot=plot,
        )
        new_labels = new_labels.index.astype(str) + " - " + new_labels
        ann.obs["cluster"] = ann.obs["cluster"].replace(new_labels)

    # Test
    sc.tl.rank_genes_groups(
        ann, groupby="cluster", method="logreg", n_genes=ann.shape[1]
    )

    # Save object
    sc.write(output_prefix + "single_cell.processed.h5ad", ann)

    # Save dataframe with cluster assignemnt
    clusters = ann.obs[cats + ["obj_id", "cluster"]]
    clusters.to_csv(output_prefix + "single_cell.cluster_assignments.csv")

    if not plot:
        return clusters.set_index(cats + ["obj_id"])["cluster"]

    # Plot

    # display raw mean values, but in log scale
    # raw = a.raw.copy()
    ann.raw._X = np.log1p(ann.raw.X)

    # # heatmap of all cells
    sc.pl.heatmap(
        ann,
        ann.var.index,
        log=True,
        standard_scale="obs",
        use_raw=False,
        show=False,
        groupby="roi",
    )
    plt.gca().figure.savefig(
        output_prefix + "single_cell.norm_scaled.heatmap.svg", **FIG_KWS
    )

    # randomize cell order in order to prevent "clustering" effects between
    # rois when plotting
    ann = ann[ann.obs.index.to_series().sample(frac=1).values, :]

    variables = cats + ["label", "log_counts", "cluster"]
    sc_kwargs = dict(color=variables, show=False, return_fig=True, use_raw=True)
    fig = sc.pl.pca(ann, **sc_kwargs)
    rasterize_scanpy(fig)
    fig.savefig(output_prefix + "cell.pca.svg", **FIG_KWS)
    fig = sc.pl.umap(ann, **sc_kwargs)
    rasterize_scanpy(fig)
    fig.savefig(output_prefix + "cell.umap.svg", **FIG_KWS)

    # fig = sc.pl.diffmap(ann, **kwargs)
    # rasterize_scanpy(fig)
    # fig.savefig(output_prefix + 'cell.diffmap.svg', **FIG_KWS)

    # Plot differential
    sc.pl.rank_genes_groups(ann, show=False)
    plt.gca().figure.savefig(
        output_prefix + "cell.differential_expression_per_cluster.svg",
        **FIG_KWS,
    )
    # sc.pl.rank_genes_groups_dotplot(ann, n_genes=4)
    # axs = sc.pl.rank_genes_groups_matrixplot(ann, n_genes=1, standard_scale='var', cmap='Blues')

    # Cells per cluster
    counts = ann.obs["cluster"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(2 * 4, 4), sharey=True)
    for axs in axes.flatten():
        sns.barplot(
            counts, counts.index, ax=axs, orient="horiz", palette="magma"
        )
        axs.set_xlabel("Cells")
        axs.set_ylabel("Cluster")
    axes[-1].set_xscale("log")
    fig.savefig(output_prefix + "cell_count_per_cluster.barplot.svg", **FIG_KWS)

    # Plot abundance per cluster
    cluster_counts_per_roi = (
        ann.obs.groupby(["cluster"] + cats)
        .count()
        .iloc[:, 0]
        .rename("ROI")
        .reset_index()
        .pivot_table(index="cluster", columns=cats, fill_value=0)
    )

    fig, axes = plt.subplots(1, 3, figsize=(4 * 3, 4), sharey=True)
    kwargs = dict(robust=True, square=True, xticklabels=True, yticklabels=True)
    for ax, log in zip(axes, [False, True]):
        sns.heatmap(
            cluster_counts_per_roi
            if not log
            else np.log10(1 + cluster_counts_per_roi),
            cbar_kws=dict(
                label="Cells per cluster" + ("" if not log else " (log10)")
            ),
            ax=ax,
            **kwargs,
        )
    sns.heatmap(
        (cluster_counts_per_roi / cluster_counts_per_roi.sum()) * 100,
        cbar_kws=dict(label="Cells per cluster (%)"),
        ax=axes[2],
        **kwargs,
    )
    fig.savefig(
        output_prefix + "cell.counts_per_cluster_per_roi.svg", **FIG_KWS
    )

    # # Plot heatmaps with mean expression per cluster
    mean_expr = anndata_to_cluster_means(ann)
    mean_expr.to_csv(output_prefix + "cell.mean_expression_per_cluster.csv")
    row_means = mean_expr.mean(1).sort_index().rename("channel_mean")
    col_counts = ann.obs["cluster"].value_counts().rename("cells_per_cluster")

    kwargs = dict(
        row_colors=row_means,
        col_colors=col_counts,
        metric="correlation",
        robust=True,
        xticklabels=True,
        yticklabels=True,
        figsize=None if not label_clusters else (10, 15),
    )

    for label1, df in [
        ("all_clusters", mean_expr),
        ("cell_type_channels", mean_expr.loc[ann_ct.var.index, :]),
        (
            "filtered_clusters",
            mean_expr.loc[
                :, (counts / counts.sum()) >= cluster_min_percentage / 100
            ],
        ),
    ]:
        for label2, label3, kwargs2 in [
            ("", "", {}),
            (
                "row_zscore.",
                "\n(Row Z-score)",
                dict(z_score=0, cmap="RdBu_r", center=0),
            ),
            (
                "col_zscore.",
                "\n(Column Z-score)",
                dict(z_score=1, cmap="RdBu_r", center=0),
            ),
            (
                "double_zscore.",
                "\n(Double Z-score)",
                dict(cmap="RdBu_r", center=0),
            ),
        ]:
            grid = sns.clustermap(
                df if label2 != "double_zscore." else double_z_score(df),
                cbar_kws=dict(label="Mean expression" + label3),
                **kwargs,
                **kwargs2,
            )
            grid.savefig(
                output_prefix
                + f"cell.mean_expression_per_cluster.{label1}.{label2}svg",
            )
            # **FIG_KWS)

    # # these take really long to be saved
    # markers = ann_ct.var.index.tolist() if plot_only_channels is None else plot_only_channels
    # sc_kwargs = dict(color=variables + markers, show=False, return_fig=True, use_raw=True)
    # fig = sc.pl.pca(ann, **sc_kwargs)
    # rasterize_scanpy(fig)
    # fig.savefig(output_prefix + "cell.pca.all_channels.pdf", **FIG_KWS)
    # fig = sc.pl.umap(ann, **sc_kwargs)
    # rasterize_scanpy(fig)
    # fig.savefig(output_prefix + "cell.umap.all_channels.pdf", **FIG_KWS)

    return clusters.set_index(cats + ["obj_id"])["cluster"]


def derive_reference_cell_type_labels(
    h5ad_file: Optional[Path] = None,
    mean_expr: Optional[DataFrame] = None,
    cluster_assignments: Optional[Series] = None,
    cell_type_channels: Optional[List[str]] = None,
    output_prefix: Optional[Path] = None,
    plot: bool = True,
    std_threshold: float = 1.5,
    cluster_min_percentage: float = 0.0,
) -> Series:
    if plot and output_prefix is None:
        raise ValueError("If `plot` if True, `output_prefix` must be given.")

    if plot:
        if output_prefix is None:
            raise ValueError("If `plot` then `output_prefix` must be given.")
        output_prefix.parent.mkdir(exist_ok=True)
        if not output_prefix.endswith("."):
            output_prefix += "."

        if h5ad_file is None and cluster_assignments is None:
            raise ValueError(
                "If `h5ad_file` is not given and `plot` is True, "
                " `cluster_assignments` must be given too."
            )

    # get cell type clusters
    if h5ad_file is not None and mean_expr is None:
        ann = sc.read(h5ad_file)
        cats = [x for x in ["roi", "sample"] if x in ann.obs.columns]
        cluster = ann.obs[cats + ["cluster"]]
        cluster.index = cluster.index.astype(int)
        cluster = cluster.sort_index()
        fractions = cluster["cluster"].value_counts()

        # # get cluster means
        mean_expr = anndata_to_cluster_means(ann)
    elif mean_expr is not None:
        mean_expr = cast(mean_expr)
        fractions = cluster_assignments.value_counts()

    if cell_type_channels is None:
        cell_type_channels = mean_expr.index.tolist()

    # Remove clusters below a certain percentage if requested
    fractions = fractions[
        (fractions / fractions.sum()) > (cluster_min_percentage / 100)
    ].rename("Cells per cluster")

    # make sure indexes match
    _mean_expr = mean_expr.reindex(fractions.index, axis=1)

    # doubly Z-scored matrix
    mean_expr_z = double_z_score(_mean_expr)

    # use a simple STD threshold for "positiveness"
    v = mean_expr_z.values.flatten()
    if std_threshold is None:
        v1 = get_threshold_from_gaussian_mixture(v)
    else:
        v1 = v.std() * std_threshold

    # label each cluster on positiveness for each marker
    labels = {x: "" for x in mean_expr_z.columns}
    for clust in mean_expr_z.columns:
        __s = mean_expr_z[clust].squeeze().sort_values(ascending=False)
        # _sz = (__s - __s.mean()) / __s.std()
        _sz = __s.loc[cell_type_channels]
        for i in _sz[_sz >= v1].index:
            labels[clust] += i + ", "

    # convert from marker positive to named cell types
    # in the absense of cell names, one could just label cell types as positive
    act_labels = {k: re.sub(r"\(.*", "+", k) for k in _mean_expr.index}
    assign = {
        ch: OrderedSet(
            propert for marker, propert in act_labels.items() if marker in label
        )
        for ch, label in labels.items()
    }
    new_labels = (
        pd.Series({k: ", ".join(v) for k, v in assign.items()})
        .sort_index()
        .rename("cell_type")
        .rename_axis("cluster")
    )
    to_replace = {k: str(k) + " - " + v for k, v in new_labels.items()}

    mean_expr_z_l = mean_expr_z.rename(columns=to_replace)
    fractions_l = fractions.copy()
    fractions_l.index = fractions_l.index.to_series().replace(to_replace)
    if output_prefix is not None:
        mean_expr_z_l.to_csv(
            output_prefix + "cell_type_assignement.reference_cluster_labels.csv"
        )

    if not plot:
        return new_labels
    output_prefix = cast(output_prefix)

    fig, axs = plt.subplots(1, 1, figsize=(3, 3))
    axs.set_title("Distribution of mean expressions")
    sns.distplot(v, ax=axs)
    axs.axvline(v.mean(), linestyle="--", color="grey")
    axs.axvline(v1, linestyle="--", color="red")
    fig.savefig(
        output_prefix
        + "mean_expression_per_cluster.both_z.threshold_position.svg"
    )

    cmeans = mean_expr.mean(1).rename("Channel mean")

    t = mean_expr_z >= v1
    kwargs = dict(
        metric="correlation",
        robust=True,
        xticklabels=True,
        yticklabels=True,
        row_colors=cmeans,
        col_colors=fractions,
    )
    opts = [
        (mean_expr, "original", dict()),
        (
            mean_expr_z,
            "both_z",
            dict(
                center=0,
                cmap="RdBu_r",
                cbar_kws=dict(label="Mean expression (Z-score)"),
            ),
        ),
        (
            t.loc[t.any(1), t.any(0)],
            "both_z.thresholded",
            dict(
                cmap="binary",
                linewidths=1,
                cbar_kws=dict(label="Mean expression (Z-score)"),
            ),
        ),
    ]
    for df, label, kwargs2 in opts:
        grid = sns.clustermap(df, **kwargs, **kwargs2)
        grid.savefig(output_prefix + f"mean_expression_per_cluster.{label}.svg")

    # replot now with labels
    figsize = grid.fig.get_size_inches()
    figsize[1] *= 1.2
    t = mean_expr_z_l >= v1
    kwargs = dict(
        metric="correlation",
        robust=True,
        xticklabels=True,
        yticklabels=True,
        center=0,
        cmap="RdBu_r",
        cbar_kws=dict(label="Mean expression (Z-score)"),
        row_colors=cmeans,
        col_colors=fractions_l,
    )
    opts = [
        (mean_expr_z_l, "labeled.both_z", dict()),
        (t.loc[t.any(1), t.any(0)], "labeled.both_z.thresholded", dict()),
    ]
    for df, label, kwargs2 in opts:
        grid = sns.clustermap(df, **kwargs, **kwargs2)
        grid.savefig(output_prefix + f"mean_expression_per_cluster.{label}.svg")

    # pairwise cluster correlation
    grid = sns.clustermap(
        mean_expr_z_l.corr(),
        center=0,
        cmap="RdBu_r",
        cbar_kws=dict(label="Pearson correlation"),
        metric="correlation",
        robust=True,
        xticklabels=True,
        yticklabels=True,
        row_colors=fractions_l,
        col_colors=fractions_l,
    )
    grid.savefig(
        output_prefix
        + "mean_expression_per_cluster.labeled.both_z.correlation.svg",
        **FIG_KWS,
    )

    return new_labels


# def add_extra_colorbar_to_clustermap(
#     data: Series, grid, cmap="inferno", location="columns", **kwargs
# ):
#     # get position to add new axis in existing figure
#     # # get_position() returns ((x0, y0), (x1, y1))
#     heat = grid.ax_heatmap.get_position()

#     if location == "columns":
#         width = 0.025
#         orientation = "vertical"
#         dend = grid.ax_col_dendrogram.get_position()
#         bbox = [[heat.x1, dend.y0], [heat.x1 + width, dend.y1]]
#     else:
#         height = 0.025
#         orientation = "horizontal"
#         dend = grid.ax_row_dendrogram.get_position()
#         bbox = [[dend.x0, dend.y0 - height], [dend.x1, dend.y0]]

#     ax = grid.fig.add_axes(matplotlib.transforms.Bbox(bbox))
#     norm = matplotlib.colors.Normalize(vmin=data.min(), vmax=data.max())
#     cb1 = matplotlib.colorbar.ColorbarBase(
#         ax, cmap=plt.get_cmap(cmap), norm=norm, orientation=orientation, label=data.name
#     )


def predict_cell_types_from_reference(
    sample: "IMCSample",
    reference_csv: str = None,
    h5ad_file: Path = None,
    output_prefix: Path = None,
    plot: bool = True,
):
    from imc.utils import get_mean_expression_per_cluster

    output_prefix = output_prefix or (
        sample.root_dir / "single_cell" / sample.name + "."
    )
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    default_ref = (
        sample.prj.processed_dir
        / "single_cell"
        / "cell_type_reference.cell_type_assignement.reference_cluster_labels.csv"
    )
    ref = pd.read_csv(reference_csv or default_ref, index_col=0)

    default_h5ad = (
        sample.root_dir
        / "single_cell"
        / (sample.name + ".cell.mean.all_vars.processed.h5ad")
    )
    query_a = sc.read(h5ad_file or default_h5ad)
    query_means = get_mean_expression_per_cluster(query_a)
    query_means = (query_means - query_means.mean(0)) / query_means.std(0)
    query_means = ((query_means.T - query_means.mean(1)) / query_means.std(1)).T
    corrwithref = pd.DataFrame(
        query_means.corrwith(ref[ct]).rename(ct) for ct in ref.columns
    ).rename_axis(columns="Query clusters", index="Reference cell types")

    query_col_fractions = (
        query_a.obs["cluster"].value_counts().rename("Cells per cluster")
    )
    side = corrwithref.shape[0] * 0.33
    grid = sns.clustermap(
        corrwithref,
        center=0,
        cmap="RdBu_r",
        cbar_kws=dict(label="Mean expression"),
        metric="correlation",
        robust=True,
        xticklabels=True,
        yticklabels=True,
        figsize=(max(map(len, corrwithref.index)) * 0.15 + side, side),
        col_colors=query_col_fractions,
    )
    grid.savefig(
        output_prefix
        + "cell_type_assignment_against_reference.correlation.svg",
        **FIG_KWS,
    )

    # simply assign to argmax for now
    # TODO: add further customization to cell type assignment
    pred_cell_type_labels = {
        x: x + " - " + corrwithref[x].idxmax() for x in corrwithref.columns
    }

    side = query_means.shape[0] * 0.33
    grid = sns.clustermap(
        query_means.rename(columns=pred_cell_type_labels).rename_axis(
            "Predicted cell types"
        ),
        center=0,
        cmap="RdBu_r",
        cbar_kws=dict(label="Mean expression"),
        metric="correlation",
        robust=True,
        xticklabels=True,
        yticklabels=True,
        figsize=(max(map(len, query_means.index)) * 0.15 + side, side),
        col_colors=query_col_fractions,
    )
    grid.savefig(
        output_prefix + "cluster_means.predicted_labels.svg", **FIG_KWS
    )

    # export cell type labels for each cell object
    cell_type_assignments = (
        query_a.obs[["roi", "cluster"]]
        .replace(pred_cell_type_labels)
        .reset_index()
        .sort_values(["roi", "index"])
        .set_index("index")
    )
    cell_type_assignments.to_csv(
        output_prefix + "cell_type_assignment_against_reference.csv"
    )
    # cell_type_assignments = pd.read_csv(output_prefix + 'cell_type_assignment_against_reference.csv', index_col=0)
    return cell_type_assignments


# def merge_clusterings(samples: List["IMCSample"]):

#     means = dict()
#     for sample in samples:
#         ann = sc.read(
#             sample.root_dir / "single_cell" / (sample.name + ".single_cell.processed.h5ad")
#         )
#         mean = anndata_to_cluster_means(ann, raw=False, cluster_label="cluster")
#         mean.columns = sample.name + " - " + mean.columns.str.extract(r"^(\d+) - .*")[0]
#         means[sample.name] = mean

#     _vars = set([y for x in means.values() for y in x.index.tolist()])
#     variables = [v for v in _vars if all([v in var.index for var in means.values()])]
#     means = {k: v.loc[variables].apply(minmax_scale, axis=1) for k, v in means.items()}

#     index = [y for x in means.values() for y in x.columns]
#     res = pd.DataFrame(index=index, columns=index, dtype=float)
#     for s1, m1 in means.items():
#         for s2, m2 in means.items():
#             for c2 in m2:
#                 res.loc[m1.columns, c2] = m1.corrwith(m2[c2])

#     res2 = res.copy()
#     np.fill_diagonal(res2.values, np.nan)
#     intra = list()
#     for sample in samples:
#         intra += (
#             res2.loc[res2.index.str.contains(sample.name), res2.index.str.contains(sample.name)]
#             .values.flatten()
#             .tolist()
#         )
#     inter = list()
#     for s1 in samples:
#         for s2 in samples:
#             if s1 == s2:
#                 continue
#         inter += (
#             res2.loc[res2.index.str.contains(s1.name), res2.index.str.contains(s2.name)]
#             .values.flatten()
#             .tolist()
#         )

#     disp = res.loc[
#         res.index.str.contains("|".join([x.name for x in samples[:-1]])),
#         res.index.str.contains("|".join([x.name for x in samples[1:]])),
#     ]
#     sns.clustermap(disp, center=0, cmap="RdBu_r", xticklabels=True, yticklabels=True)


def get_adjacency_graph(
    roi: "ROISample",
    output_prefix: Optional[Path] = None,
    max_dist: int = MAX_BETWEEN_CELL_DIST,
):
    clusters = roi.clusters
    if clusters is None:
        print("ROI does not have assigned clusters.")

    output_prefix = Path(
        output_prefix or roi.sample.root_dir / "single_cell" / (roi.name + ".")
    )
    if not output_prefix.endswith("."):
        output_prefix += "."
    os.makedirs(output_prefix.parent, exist_ok=True)

    mask = roi.cell_mask

    # align mask with cell type assignment (this is only to remove border cells)
    if clusters is not None:
        mask[~np.isin(mask, roi.clusters.index)] = 0

    # Get the closest cell of each background point dependent on `max_dist`
    # # first measure the distance of each background point to the closest cell
    background = mask == 0
    d = ndi.distance_transform_edt(
        background, return_distances=True, return_indices=False
    )

    background = background & (d <= max_dist)
    i, j = ndi.distance_transform_edt(
        background, return_distances=False, return_indices=True
    )
    mask = mask[i, j]

    # Simply use mean of channels as distance
    image_mean = exposure.equalize_hist(roi.stack.mean(axis=0))

    # Construct adjacency graph based on cell distances

    g = graph.rag_mean_color(image_mean, mask, mode="distance")
    # remove background node (unfortunately it can't be masked beforehand)
    if 0 in g.nodes:
        g.remove_node(0)

    fig, ax = plt.subplots(1, 1)
    lc = graph.show_rag(
        mask,
        g,
        (image_mean * 255).astype(int),
        ax=ax,
        img_cmap="viridis",
        edge_cmap="Reds",
        edge_width=1,
    )
    ax.axis("off")
    fig.colorbar(lc, fraction=0.03, ax=ax)
    ax.get_children()[0].set_rasterized(True)
    ax.get_children()[-2].set_rasterized(True)
    fig.savefig(output_prefix + "neighbor_graph.svg", **FIG_KWS)

    # add cluster label atrtribute
    if clusters is not None:
        nx.set_node_attributes(g, roi.clusters.to_dict(), name="cluster")
        nx.set_node_attributes(
            g, roi.clusters.index.to_series().to_dict(), name="obj_id"
        )
    # save graph
    nx.readwrite.write_gpickle(g, output_prefix + "neighbor_graph.gpickle")
    return g


def measure_cell_type_adjacency(
    roi: "ROISample",
    method: str = "random",
    adjacency_graph: Optional[nx.Graph] = None,
    n_iterations: int = 100,
    inf_replace_method: str = "min",
    output_prefix: Optional[Path] = None,
    plot: bool = True,
    save: bool = True,
) -> DataFrame:
    output_prefix = output_prefix or (
        roi.sample.root_dir / "single_cell" / roi.name + "."
    )
    if not output_prefix.endswith("."):
        output_prefix += "."

    cluster_counts = roi.clusters.value_counts()

    if adjacency_graph is None:
        adjacency_graph = roi.adjacency_graph

    adj, order = nx.linalg.attrmatrix.attr_matrix(
        adjacency_graph, node_attr="cluster"
    )
    order = pd.Series(order).astype(
        roi.clusters.dtype
    )  #  passing dtype at instantiation gives warning
    freqs = pd.DataFrame(adj, order, order).sort_index(0).sort_index(1)
    if save:
        freqs.to_csv(output_prefix + "cluster_adjacency_graph.frequencies.csv")

    if method == "random":
        norm_freqs = correct_interaction_background_random(
            roi, freqs, "cluster", n_iterations, save, output_prefix
        )
    elif method == "pharmacoscopy":
        norm_freqs = correct_interaction_background_pharmacoscopy(
            freqs, cluster_counts, roi.clusters.shape[0], inf_replace_method
        )
    if save:
        norm_freqs.to_csv(
            output_prefix + "cluster_adjacency_graph.norm_over_random.csv"
        )

    if not plot:
        return norm_freqs
    v = norm_freqs.values.std() * 2
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(4 * 2, 4))
    kws = dict(
        cmap="RdBu_r", center=0, square=True, xticklabels=True, yticklabels=True
    )
    sns.heatmap(norm_freqs, robust=True, ax=axes[0], **kws)
    kws2 = dict(vmin=-v, vmax=v, cbar_kws=dict(label="Log odds interaction"))
    sns.heatmap(norm_freqs, ax=axes[1], **kws, **kws2)
    fig.savefig(
        output_prefix + "cluster_adjacency_graph.norm_over_random.heatmap.svg",
        **FIG_KWS,
    )
    del kws["square"]
    try:
        grid = sns.clustermap(norm_freqs, **kws, **kws2)
        grid.savefig(
            output_prefix
            + "cluster_adjacency_graph.norm_over_random.clustermap.svg",
            **FIG_KWS,
        )
    except FloatingPointError:
        pass
    return norm_freqs


def correct_interaction_background_random(
    roi: "ROI",
    freqs: DataFrame,
    attribute,
    n_iterations: int,
    save: bool,
    output_prefix: Union[str, Path],
):
    values = {
        x: roi.adjacency_graph.nodes[x][attribute]
        for x in roi.adjacency_graph.nodes
    }
    shuffled_freqs = list()
    for _ in tqdm(range(n_iterations)):
        g2 = roi.adjacency_graph.copy()
        shuffled_attr = pd.Series(values).sample(frac=1)
        shuffled_attr.index = values
        nx.set_node_attributes(g2, shuffled_attr.to_dict(), name=attribute)
        rf, rl = nx.linalg.attrmatrix.attr_matrix(g2, node_attr=attribute)
        rl = pd.Series(rl, dtype=roi.clusters.dtype)
        shuffled_freqs.append(
            pd.DataFrame(rf, index=rl, columns=rl).sort_index(0).sort_index(1)
        )
    shuffled_freq = pd.concat(shuffled_freqs)
    if save:
        shuffled_freq.to_csv(
            output_prefix
            + f"cluster_adjacency_graph.random_frequencies.all_iterations_{n_iterations}.csv"
        )
    shuffled_freq = shuffled_freq.groupby(level=0).sum().sort_index(1)
    if save:
        shuffled_freq.to_csv(
            output_prefix + "cluster_adjacency_graph.random_frequencies.csv"
        )

    fl = np.log1p((freqs / freqs.values.sum()) * 1e6)
    sl = np.log1p((shuffled_freq / shuffled_freq.values.sum()) * 1e6)
    # make sure both contain all edges/nodes
    fl = fl.reindex(sl.index, axis=0).reindex(sl.index, axis=1).fillna(0)
    sl = sl.reindex(fl.index, axis=0).reindex(fl.index, axis=1).fillna(0)
    return fl - sl


def correct_interaction_background_pharmacoscopy(
    frequency_matrix: DataFrame,
    cluster_counts: Series,
    total_cells: int,
    inf_replace_method: Optional[str] = "min_symmetric",
):
    c = np.log(total_cells)
    fa = np.log(frequency_matrix.sum().sum()) - c
    norms = pd.DataFrame()
    for ct1 in frequency_matrix.index:
        for ct2 in frequency_matrix.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                o = np.log(frequency_matrix.loc[ct1, ct2]) - np.log(
                    frequency_matrix.loc[ct1].sum()
                )
                if o == 0:
                    norms.loc[ct1, ct2] = 0.0
                    continue
                f1 = np.log(cluster_counts.loc[ct1]) - c
                f2 = np.log(cluster_counts.loc[ct2]) - c

            norms.loc[ct1, ct2] = o - (f1 + f2 + fa)
    if inf_replace_method is None:
        return norms

    # three ways to replace -inf (cell types with no event touching):
    # # 1. replace with lowest non-inf value (dehemphasize the lower bottom - lack of touching)
    if inf_replace_method == "min":
        norm_freqs = norms.replace(
            -np.inf, norms[norms != (-np.inf)].min().min()
        )
    # # 2. replace with minus highest (try to )
    if inf_replace_method == "max":
        norm_freqs = norms.replace(-np.inf, -norms.max().max())
    # # 3. One of the above + make symmetric by  X @ X.T + Z-score
    if inf_replace_method == "min_symmetric":
        norm_freqs = norms.replace(
            -np.inf, norms[norms != (-np.inf)].min().min()
        )
        norm_freqs = norm_freqs @ norm_freqs.T
        norm_freqs = (
            norm_freqs - norm_freqs.values.mean()
        ) / norm_freqs.values.std()
    if inf_replace_method == "max_symmetric":
        norm_freqs = norms.replace(
            -np.inf, norms[norms != (-np.inf)].max().max()
        )
        norm_freqs = norm_freqs @ norm_freqs.T
        norm_freqs = (
            norm_freqs - norm_freqs.values.mean()
        ) / norm_freqs.values.std()
    return norm_freqs


def find_communities(
    roi: "ROI",
    community_resolution: float = DEFAULT_COMMUNITY_RESOLUTION,
    plot: bool = True,
) -> Tuple[Series, Tuple]:
    # def networkx_to_igraph(graph):
    #     import igraph as ig
    #     g = ig.Graph(edges=list(graph.edges))
    #     # If the original graph has non-consecutive integer labels,
    #     # igraph will create a node for the non existing vertexes.
    #     # These can simply be removed from the graph.
    #     nodes = pd.Series(list(graph.nodes))
    #     vertexes = pd.Series(range(len(g.vs)))
    #     g.delete_vertices(vertexes[~vertexes.isin(nodes)].values)
    #     return g

    def get_community_members(partition: Dict) -> Dict:
        counts = Counter(partition)
        # {com: members}
        comms: Dict[int, set] = dict()
        for com in counts.keys():
            comms[com] = set()
        for n, com in partition.items():
            comms[com].add(n)
        return comms

    def get_community_cell_type_composition(roi: "ROI", partition: Series):
        cts = dict()
        for com, members in get_community_members(partition).items():
            # cts[f"{roi.sample.name} - {roi.roi_number} - {com}"] = \
            cts[com] = roi.clusters.loc[members].value_counts()
        return (
            pd.DataFrame(cts)
            .fillna(0)
            .rename_axis(index="cell_type", columns="community")
            .astype(int)
        )

    # Community finding in graph (overclustering)
    roi_output_prefix = (
        roi.sample.root_dir / "single_cell" / (roi.name + ".communities.")
    )

    # TODO: use leiden instead of louvain
    # g = networkx_to_igraph(roi.adjacency_graph)
    # p = partitions[roi] = pd.Series(
    #     la.find_partition(
    #         g, la.RBConfigurationVertexPartition,
    #         resolution_parameter=community_resolution).membership,
    #     name="community", index=roi.adjacency_graph.nodes).sort_index()
    partition = pd.Series(
        community.best_partition(
            roi.adjacency_graph, resolution=community_resolution
        ),  # , weight="expr_weight")
        name="community",
    ).sort_index()
    n = partition.value_counts().shape[0]
    tqdm.write(f"Found {n} communities for ROI {roi}.")
    partition += 1
    partition.to_csv(roi_output_prefix + "graph_partition.csv")
    comps = (
        get_community_cell_type_composition(roi, partition)
        .T.assign(sample=roi.sample.name, roi=roi.name)
        .set_index(["sample", "roi"], append=True)
    )
    comps.index = comps.index.reorder_levels(["sample", "roi", "community"])

    if plot:
        # get cell type counts per community
        comps_s = comps.reset_index(level=["sample", "roi"], drop=True)
        percent = (comps_s.T / comps_s.sum(1)) * 100
        grid = sns.clustermap(
            percent, metric="correlation", cbar_kws=dict(label="% of cell type")
        )
        grid.savefig(roi_output_prefix + "cell_type_composition.svg", **FIG_KWS)
        grid = sns.clustermap(
            percent,
            z_score=1,
            cmap="RdBu_r",
            center=0,
            metric="correlation",
            cbar_kws=dict(label="% of cell type (Z-score)"),
        )
        grid.savefig(
            roi_output_prefix + "cell_type_composition.zscore.svg", **FIG_KWS
        )
    return partition, comps


def cluster_communities(
    rois: List["ROI"],
    output_prefix: Optional[Path] = None,
    supercommunity_resolution: float = DEFAULT_SUPERCOMMUNITY_RESOLUTION,
) -> Series:
    output_prefix = output_prefix or (
        rois[0].prj.processed_dir
        / "single_cell"
        / (rois[0].prj.name + ".communities.")
    )
    output_prefix = cast(output_prefix)

    res = parmap.map(find_communities, rois)
    partitions = {k: v[0] for k, v in zip(rois, res)}
    composition = (
        pd.concat([v[1] for v in res]).fillna(0).astype(int).sort_index()
    )
    composition.to_csv(
        output_prefix + ".all_communities.cell_type_composition.csv"
    )

    print(f"Found {composition.shape[0]} communities across all ROIs.")

    composition = pd.read_csv(
        output_prefix + ".all_communities.cell_type_composition.csv",
        index_col=[0, 1, 2],
    )

    # Cluster communities by leiden clustering based on cell type composition
    a = AnnData(composition)
    sc.pp.log1p(a)
    sc.pp.neighbors(a)
    sc.tl.leiden(
        a, resolution=supercommunity_resolution, key_added="supercommunity"
    )
    n_scomms = len(a.obs["supercommunity"].unique())
    print(f"Found {n_scomms} supercommunities.")
    # Make supercommunities 1-based (to distinguish from masks where 0 == background)
    a.obs["supercommunity"] = pd.Categorical(
        a.obs["supercommunity"].astype(int) + 1
    )
    sc.tl.umap(a)
    sc.pp.pca(a)

    # DataFrame(cell vs [celltype, community, supercommunity])
    _assignments = list()
    for roi in rois:
        # {cell: cell type}
        if roi.clusters.dtype == "int" and roi.clusters.min() == 0:
            c1 = (
                roi.clusters + 1
            )  # TODO: this +1 should be removed when clustering is re-run since the new implm
        else:
            c1 = roi.clusters
        # {cell: community}
        c2 = pd.Series(partitions[roi], name="community").rename_axis(
            index="obj_id"
        )
        scomm = a.obs.loc[(roi.sample.name, roi.name), "supercommunity"].astype(
            int
        )
        assert c2.value_counts().shape[0] == scomm.shape[0]
        c3 = c2.replace(scomm.to_dict()).rename("supercommunity")
        assert c3.max() <= n_scomms
        assert c1.shape == c2.shape == c3.shape
        assert (c1.index == c2.index).all()
        assert (c2.index == c3.index).all()
        c = c1.to_frame().join(c2).join(c3)
        assert roi.clusters.shape[0] == c.shape[0]
        c["sample"] = roi.sample.name
        c["roi"] = roi.roi_number
        _assignments.append(c)
    assignments = pd.concat(_assignments).set_index(
        ["sample", "roi"], append=True
    )
    assignments.index = assignments.index.reorder_levels(
        ["sample", "roi", "obj_id"]
    )

    # Further merge supercommunities if distant by less than X% of composition
    # TODO: revise supercommunity merging
    max_supercommunity_difference = 10.0
    comp = assignments.assign(count=1).pivot_table(
        index="supercommunity",
        columns="cluster",
        values="count",
        aggfunc=sum,
        fill_value=0,
    )

    perc = (comp.T / comp.sum(1)).T * 100
    diffs = pd.DataFrame(
        np.sqrt(abs(perc.values - perc.values[:, None]).sum(axis=2)),
        index=perc.index,
        columns=perc.index,
    )
    grid = sns.clustermap(diffs)
    repl = pd.Series(
        dict(
            zip(
                grid.data.columns,
                fcluster(
                    grid.dendrogram_col.linkage,
                    t=max_supercommunity_difference,
                    criterion="distance",
                ),
            )
        )
    ).sort_index()

    comp.index = comp.index.to_series().replace(repl)
    comp = comp.groupby(level=0).sum()

    assignments["supercommunity"] = assignments["supercommunity"].replace(repl)

    # check name/number supercommunities is sorted on the abundance of their cell types
    s = (
        assignments["supercommunity"]
        .value_counts()
        .sort_values(ascending=False)
    )
    assignments["supercommunity"] = assignments["supercommunity"].replace(
        dict(zip(s.index, np.arange(1, len(s))))
    )

    # save final assignments
    assignments.to_csv(
        output_prefix + "cell_type.community.supercommunities.csv"
    )

    # Visualize
    # # visualize initial communities in clustermap, PCA or UMAP
    perc = (composition.T / composition.sum(1)).T * 100
    grid = sns.clustermap(perc, metric="correlation", rasterized=True)
    grid.savefig(
        output_prefix
        + "communities.cell_type_composition.leiden_clustering.clustermap_viz.svg",
        **FIG_KWS,
    )
    grid = sns.clustermap(
        np.log1p(composition),
        row_linkage=grid.dendrogram_row.linkage,
        col_linkage=grid.dendrogram_col.linkage,
        metric="correlation",
        row_colors=plt.get_cmap("tab20")(a.obs["supercommunity"].astype(int)),
        rasterized=True,
    )
    grid.savefig(
        output_prefix
        + "communities.cell_type_composition.leiden_clustering.clustermap_viz.counts.svg",
        **FIG_KWS,
    )
    for method in ["pca", "umap"]:
        fig = getattr(sc.pl, method)(
            a,
            color=["supercommunity"] + a.var.index.tolist(),
            return_fig=True,
            show=False,
        )
        fig.savefig(
            output_prefix
            + f"communities.cell_type_composition.leiden_clustering.{method}_viz.svg",
            **FIG_KWS,
        )

    # # visualize the rediction of supercommunities based on difference thresh
    grid = sns.clustermap(
        diffs,
        col_colors=plt.get_cmap("tab20")(repl.values),
        row_colors=plt.get_cmap("tab20")(repl.values),
        cbar_kws=dict(label="Sqrt(Sum(diff))"),
    )
    grid.savefig(
        output_prefix + "supercommunities.reduction_by_diff.clustermap.svg",
        **FIG_KWS,
    )

    # assignments = pd.read_csv(output_prefix + "cell_type.community.supercommunities.csv", index_col=[0, 1, 2])
    # # cell type vs {community, supercommunity}
    for var_ in ["community", "supercommunity"]:
        supercts = assignments.assign(count=1).pivot_table(
            index="cluster",
            columns=var_,
            values="count",
            aggfunc=sum,
            fill_value=0,
        )
        perc_supercts = (supercts / supercts.sum()) * 100

        grid = sns.clustermap(
            perc_supercts,
            metric="correlation",
            rasterized=True,
            cbar_kws=dict(label="% of supercommunity"),
        )
        grid.savefig(
            output_prefix + f"{var_}.cell_type_composition.svg", **FIG_KWS
        )
        grid = sns.clustermap(
            perc_supercts,
            z_score=1,
            cmap="RdBu_r",
            center=0,
            metric="correlation",
            rasterized=True,
            cbar_kws=dict(label="% of supercommunity (Z-score)"),
        )
        grid.savefig(
            output_prefix + f"{var_}.cell_type_composition.zscore.svg",
            **FIG_KWS,
        )

    leg_kws = dict(bbox_to_anchor=(0, -0.05))

    vars_ = ["cluster", "community", "supercommunity"]
    n = len(rois)
    m = len(vars_)
    patches: Dict[str, List] = dict()
    fig, axes = plt.subplots(
        n, m, figsize=(4 * m, 4 * n), squeeze=False, sharex="row", sharey="row"
    )
    for i, roi in enumerate(rois):
        for j, var_ in enumerate(vars_):
            if i == 0:
                patches[var_] = list()
            p = roi.plot_cell_types(
                ax=axes[i, j, np.newaxis, np.newaxis],
                cell_type_assignments=assignments.loc[
                    (roi.sample.name, roi.roi_number), var_
                ],
                palette="nipy_spectral",
            )
            patches[var_] += p
    for j, var_ in enumerate(vars_):
        if var_ == "community":
            continue
        add_legend(
            patches[var_], axes[-1, j], **leg_kws
        )  # label="Super community",
        _z = zip(
            axes[0].squeeze(),
            ["Cell types", "Communities", "Super communities"],
        )
    for axs, lab in _z:
        axs.set_title(lab)
    # TODO: limit rasterization to main image
    for axs in axes.flat:
        axs.set_rasterized(True)
    fig.savefig(
        output_prefix + "communities_supercommunities.all_rois.svg", **FIG_KWS
    )

    return assignments["supercommunity"]

    # # # roi vs supercommunity
    # rs = (
    #     assignments
    #     .assign(count=1)
    #     .reset_index()
    #     .pivot_table(columns=['sample', 'roi'], index='supercommunity', values='count', aggfunc=sum, fill_value=0))
    # rs = rs / rs.sum()


@overload
def get_best_mixture_number(
    x: Series,
    min_mix: int,
    max_mix: int,
    subsample_if_needed: bool,
    n_iters: int,
    metrics: List[str],
    red_func: str,
    return_prediction: Literal[False],
) -> int:
    ...


@overload
def get_best_mixture_number(
    x: Series,
    min_mix: int,
    max_mix: int,
    subsample_if_needed: bool,
    n_iters: int,
    metrics: List[str],
    red_func: str,
    return_prediction: Literal[True],
) -> Tuple[int, Array]:
    ...


def get_best_mixture_number(
    x: Series,
    min_mix: int = 2,
    max_mix: int = 6,
    subsample_if_needed: bool = True,
    n_iters: int = 3,
    metrics: List[str] = [
        "silhouette_score",
        "calinski_harabasz_score",
        "davies_bouldin_score",
    ],
    red_func: str = "mean",
    return_prediction: bool = False,
) -> Union[int, Tuple[int, Array]]:
    from sklearn.mixture import GaussianMixture
    import sklearn.metrics

    def get_means(num: Series, pred: Union[Series, Array]) -> Series:
        return num.groupby(pred).mean().sort_values()

    def replace_pred(x: Series, y: Union[Series, Array]) -> Series:
        means = get_means(x, y)
        repl = dict(zip(means.index, range(len(means))))
        y2 = pd.Series(y, index=x.index).replace(repl)
        new_means = get_means(x, y2.values)
        assert all(new_means.index == range(len(new_means)))
        return y2

    xx = (
        x.sample(n=10_000) if subsample_if_needed and x.shape[0] > 10_000 else x
    )

    if isinstance(xx, pd.Series):
        xx = xx.values.reshape((-1, 1))

    mi = range(min_mix, max_mix)
    mixes = pd.DataFrame(index=metrics, columns=mi)
    for i in tqdm(mi):
        mix = GaussianMixture(i)
        # mix.fit_predict(x)
        for f in metrics:
            func = getattr(sklearn.metrics, "davies_bouldin_score")
            mixes.loc[f, i] = np.mean(
                [func(xx, mix.fit_predict(xx)) for _ in range(n_iters)]
            )
        # mixes[i] = np.mean([silhouette_score(x, mix.fit_predict(x)) for _ in range(iters)])
    mixes.loc["davies_bouldin_score"] = 1 / mixes.loc["davies_bouldin_score"]

    # return best
    # return np.argmax(mixes.values()) + min_mix  # type: ignore
    best = mixes.columns[int(getattr(np, red_func)(mixes.apply(np.argmax, 1)))]
    if not return_prediction:
        return best  # type: ignore

    # now train with full data
    mix = GaussianMixture(best)
    return (best, replace_pred(x, mix.fit_predict(x.values.reshape((-1, 1)))))


def get_threshold_from_gaussian_mixture(
    x: Series, y: Optional[Series] = None, n_components: int = 2
) -> Array:
    def get_means(num: Series, pred: Union[Series, Array]) -> Series:
        return num.groupby(pred).mean().sort_values()

    def replace_pred(x: Series, y: Union[Series, Array]) -> Series:
        means = get_means(x, y)
        repl = dict(zip(means.index, range(len(means))))
        y2 = pd.Series(y, index=x.index).replace(repl)
        new_means = get_means(x, y2.values)
        assert all(new_means.index == range(len(new_means)))
        return y2

    x = x.sort_values()

    if y is None:
        from sklearn.mixture import GaussianMixture  # type: ignore

        mix = GaussianMixture(n_components=n_components)
        xx = x.values.reshape((-1, 1))
        y = mix.fit_predict(xx)
    else:
        y = y.reindex(x.index).values
    y = replace_pred(x, y).values
    thresh = x.loc[((y[:-1] < y[1::])).tolist() + [False]].reset_index(
        drop=True
    )
    assert len(thresh) == (n_components - 1)
    return thresh


def get_probability_of_gaussian_mixture(
    x: Series, n_components: int = 2, population=-1
) -> Series:
    from sklearn.mixture import GaussianMixture  # type: ignore

    x = x.sort_values()
    mix = GaussianMixture(n_components=n_components)
    xx = x.values.reshape((-1, 1))
    mix.fit(xx)
    means = pd.Series(mix.means_.squeeze()).sort_values()
    # assert (means.index == range(n_components)).all()
    # order components by mean
    p = mix.predict_proba(xx)[:, means.index]
    # take requested population
    p = p[:, population]
    return pd.Series(p, index=x.index).sort_index()


def fit_gaussian_mixture(
    x: Union[Series, DataFrame], n_mixtures: Union[int, List[int]] = None
) -> Union[Series, DataFrame]:
    # TODO: paralelize
    from sklearn.mixture import GaussianMixture

    if isinstance(x, pd.Series):
        x = x.to_frame()
    if isinstance(n_mixtures, int):
        n_mixtures = [n_mixtures] * x.shape[1]
    expr_thresh = x.astype(int)

    def get_means(num, pred):
        return num.groupby(pred).mean().sort_values()

    def replace_pred(x, y):
        means = get_means(x, y)
        repl = dict(zip(range(len(means)), means.index))
        y2 = y.replace(repl)
        new_means = get_means(x, y2)
        assert all(new_means.index == range(len(new_means)))
        return y2

    for i, ch in enumerate(x.columns):
        if n_mixtures is None:
            n_best = get_best_mixture_number(x, return_prediction=False)  # type: ignore[call-overload]
            mix = GaussianMixture(n_best)
        else:
            mix = GaussianMixture(n_mixtures[i])
        _x = x.loc[:, ch]
        x2 = _x.values.reshape((-1, 1))
        mix.fit(x2)
        y = pd.Series(mix.predict(x2), index=x.index, name="class")
        expr_thresh[ch] = replace_pred(_x, y)
    return expr_thresh.squeeze()


def get_population(
    ser: Series, population: int = -1, plot=False, ax=None, **kwargs
) -> pd.Index:
    if population == -1:
        operator = np.greater_equal
    elif population == 0:
        operator = np.less_equal
    else:
        raise ValueError(
            "Chosen population must be '0' (lowest) or '-1' (highest)."
        )

    # Make sure index is unique
    if not ser.index.is_monotonic:
        ser = ser.reset_index(drop=True)

    # Work only in positive space
    xx = ser  # + abs(ser.min())
    done = False
    while not done:
        try:
            n, y = get_best_mixture_number(xx, return_prediction=True, **kwargs)
        except ValueError:  # "Number of labels is 1. Valid values are 2 to n_samples - 1 (inclusive)"
            continue
        done = True
    print(f"Chosen mixture of {n} distributions.")
    done = False
    while not done:
        try:
            thresh = get_threshold_from_gaussian_mixture(xx, n_components=n)
        except AssertionError:
            continue
        done = True

    sel = operator(xx, thresh.iloc[population]).values

    if plot:
        ax = plt.gca() if ax is None else ax
        sns.distplot(xx, kde=False, ax=ax)
        sns.distplot(xx.loc[sel], kde=False, ax=ax)
        [ax.axvline(q, linestyle="--", color="grey") for q in thresh]
        ax = None
    return sel


def stack_to_probabilities(
    stack: Array,
    channel_labels: Series,
    nuclear_channels: Optional[List[str]] = None,
    cytoplasm_channels: Optional[List[str]] = None,
    log: bool = True,
    # scale: bool = True,
) -> Array:
    """
    Very simple way to go from a channel stack to nuclei, cytoplasm and background probabilities.
    """
    from skimage.exposure import equalize_hist as eq

    # nuclear_channels = ["DNA", "Histone", "pCREB", "cKIT", "pSTAT3"]
    nuclear_channels = nuclear_channels or ["DNA", "Histone"]
    _nuclear_channels = channel_labels[
        channel_labels.str.contains("|".join(nuclear_channels))
    ]
    if cytoplasm_channels is None:
        _cytoplasm_channels = channel_labels[
            ~channel_labels.isin(_nuclear_channels)
        ]
    else:
        _cytoplasm_channels = channel_labels[
            channel_labels.str.contains("|".join(cytoplasm_channels))
        ]

    if log:
        stack = np.log1p(stack)

    # if scale:
    #     stack = saturize(stack)

    # # mean of nuclear signal
    ns = stack[_nuclear_channels.index].mean(0)
    # # mean of cytoplasmatic signal
    cs = stack[_cytoplasm_channels.index].mean(0)

    # # normalize
    ns = minmax_scale(ns)
    cs = minmax_scale(eq(cs, 256 * 4))

    # # convert into probabilities
    pn = ns
    pb = 1 - minmax_scale(pn + cs)
    # pb = (pb - pb.min()) / pb.max()
    # pc = minmax_scale(cs - (pn + pb))
    pc = 1 - (pn + pb)
    rgb = np.asarray([pn, pc, pb])

    # pnf = ndi.gaussian_filter(pn, 1)
    # pcf = ndi.gaussian_filter(pc, 1)
    # pbf = ndi.gaussian_filter(pb, 1)
    # rgb = np.asarray([pnf, pcf, pbf])
    # return saturize(rgb)
    return np.clip(rgb, 0, 1)

    # # pp = ndi.zoom(roi.probabilities, (1, 0.5, 0.5))
    # # pp = pp / pp.max()
    # p = stack_to_probabilities(roi.stack, roi.channel_labels)
    # fig, axes = plt.subplots(1, 4, sharex=True, sharey=True)
    # # axes[0].imshow(np.moveaxis(pp / pp.max(), 0, -1))
    # axes[1].imshow(np.moveaxis(p, 0, -1))
    # from skimage.exposure import equalize_hist as eq
    # axes[2].imshow(minmax_scale(eq(roi._get_channel("mean")[1])))
    # axes[3].imshow(roi.mask)


def save_probabilities(probs, output_tiff):
    import tifffile

    tifffile.imsave(
        output_tiff, np.moveaxis((probs * 2 ** 16).astype("uint16"), 0, -1)
    )


# def probabilities_to_mask(arr: Array, nuclei_diameter_range=(5, 30)) -> Array:
#     import h5py
#     import skimage.filters
#     import centrosome
#     import scipy
#     from skimage.filters import threshold_local
#     from skimage.exposure import equalize_hist as eq
#     from centrosome.threshold import get_threshold

#     def size_fn(size, is_foreground):
#         return size < nuclei_diameter_range[1] * nuclei_diameter_range[1]

#     nuclei, cyto, backgd = arr

#     # make cyto
#     nuc_cyto = eq(nuclei + cyto)

#     # smooth nuclei

#     # # typical diameter
#     5, 30

#     # # discard objects outside diameter range
#     False

#     # # discard touching border
#     True

#     sn = skimage.filters.sobel(nuclei)
#     snt = nuclei > skimage.filters.threshold_otsu(nuclei)
#     sntr = skimage.morphology.remove_small_objects(snt)
#     sntrd = ~skimage.morphology.dilation(~sntr)


#     skimage.segmentation.flood_fill(nuc_cyto

#     skimage.segmentation.watershed(scipy.ndimage.morphology.distance_transform_edt(sntr))


#     # # local thresholding
#     nuc = eq(nuclei)
#     # # # threshold smoothing scale 0
#     # # # threshold correction factor 1.2
#     # # # threshold bounds 0.0, 1.0
#     lt, gt = get_threshold(
#         "Otsu",
#         "Adaptive",
#         nuc,
#         threshold_range_min=0,
#         threshold_range_max=1.0,
#         threshold_correction_factor=1.2,
#         adaptive_window_size=50,
#     )

#     binary_image = (nuc >= lt) & (nuc >= gt)

#     # remove small objects
#     skimage.morphology.remove_small_objects(binary_image, min_size=min_size)

#     # # # measure variance and entropy in foreground vs background

#     # fill holes inside foreground

#     binary_image = centrosome.cpmorphology.fill_labeled_holes(binary_image, size_fn=size_fn)

#     # label
#     labeled_image, object_count = scipy.ndimage.label(binary_image, np.ones((3, 3), bool))
#     return labeled_image


# import numpy as np
# import cv2
# from matplotlib import pyplot as plt

# gray = (nuclei.copy() * 255).astype('u8')
# gray = cv2.cvtColor(np.moveaxis(pr, 0, -1),cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# #noise removal
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# # sure background area
# sure_bg = cv2.dilate(opening,kernel,iterations=3)

# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)

# # Marker labelling
# ret, markers = cv2.connectedComponents(sure_fg)

# # Add one to all labels so that sure background is not 0, but 1
# markers = markers+1

# # Now, mark the region of unknown with zero
# markers[unknown==255] = 0
