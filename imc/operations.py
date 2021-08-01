"""
Functions for high order operations.
"""

# fix the type annotatiton of not yet undefined classes
from __future__ import annotations
import os, re, json, typing as tp
from collections import Counter

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
import imc.data_models.roi as _roi
import imc.data_models.sample as _sample
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
    stack: tp.Union[Array, Path],
    mask: tp.Union[Array, Path],
    red_func: str = "mean",
    border_objs: bool = False,
    equalize: bool = True,
    scale: bool = False,
    channel_include: Array = None,
    channel_exclude: Array = None,
) -> DataFrame:
    """
    Measure the intensity of each channel in each cell


    Parameters
    ----------
    stack: tp.Union[Array, Path]
        Image to quantify.
    mask: tp.Union[Array, Path]
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

    res = np.zeros((n_cells, n_channels), dtype=int if red_func == "sum" else float)
    for channel in np.arange(stack.shape[0])[channel_include & ~channel_exclude]:
        res[:, channel] = [
            getattr(x.intensity_image, red_func)()
            for x in skimage.measure.regionprops(mask, stack[channel])
        ]
    return pd.DataFrame(res, index=cells[1:]).rename_axis(index="obj_id")


def quantify_cell_morphology(
    mask: tp.Union[Array, Path],
    attributes: tp.Sequence[str] = [
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

    return (
        pd.DataFrame(
            skimage.measure.regionprops_table(mask, properties=attributes),
            index=np.unique(mask)[1:],
        )
        .rename_axis(index="obj_id")
        .rename(columns={"centroid-0": "X_centroid", "centroid-1": "Y_centroid"})
    )


def _quantify_cell_intensity__roi(roi: _roi.ROI, **kwargs) -> DataFrame:
    assignment = dict(roi=roi.name)
    if roi.sample is not None:
        assignment["sample"] = roi.sample.name
    return roi.quantify_cell_intensity(**kwargs).assign(**assignment)


def _quantify_cell_morphology__roi(roi: _roi.ROI, **kwargs) -> DataFrame:
    assignment = dict(roi=roi.name)
    if roi.sample is not None:
        assignment["sample"] = roi.sample.name
    return roi.quantify_cell_morphology(**kwargs).assign(**assignment)


def _correlate_channels__roi(roi: _roi.ROI, labels: str = "channel_names") -> DataFrame:
    xcorr = np.corrcoef(roi.stack.reshape((roi.channel_number, -1)))
    np.fill_diagonal(xcorr, 0)
    labs = getattr(roi, labels)
    return pd.DataFrame(xcorr, index=labs, columns=labs)


# def _get_adjacency_graph__roi(roi: _roi.ROI, **kwargs) -> DataFrame:
#     output_prefix = roi.sample.root_dir / "single_cell" / roi.name
#     return get_adjacency_graph(roi.stack, roi.mask, roi.clusters, output_prefix, **kwargs)


def quantify_cell_intensity_rois(
    rois: tp.Sequence[_roi.ROI],
    **kwargs,
) -> DataFrame:
    """
    Measure the intensity of each channel in each single cell.
    """
    return pd.concat(
        parmap.map(_quantify_cell_intensity__roi, rois, pm_pbar=True, **kwargs)
    ).rename_axis(index="obj_id")


def quantify_cell_morphology_rois(
    rois: tp.Sequence[_roi.ROI],
    **kwargs,
) -> DataFrame:
    """
    Measure the shape parameters of each single cell.
    """
    return pd.concat(
        parmap.map(_quantify_cell_morphology__roi, rois, pm_pbar=True, **kwargs)
    ).rename_axis(index="obj_id")


def quantify_cells_rois(
    rois: tp.Sequence[_roi.ROI],
    layers: tp.Sequence[str],
    intensity: bool = True,
    intensity_kwargs: tp.Dict[str, tp.Any] = {},
    morphology: bool = True,
    morphology_kwargs: tp.Dict[str, tp.Any] = {},
) -> DataFrame:
    """
    Measure the intensity of each channel in each single cell.
    """
    quants = list()
    if intensity:
        quants.append(
            quantify_cell_intensity_rois(rois=rois, layers=layers, **intensity_kwargs)
        )
    if morphology:
        quants.append(
            quantify_cell_morphology_rois(rois=rois, layers=layers, **morphology_kwargs)
        )

    return (
        # todo: this will fail if there's different layers in intensity and morphology
        pd.concat(
            # ignore because a ROI is not obliged to have a Sample
            [quants[0].drop(["sample", "roi"], axis=1, errors="ignore"), quants[1]],
            axis=1,
        )
        if len(quants) > 1
        else quants[0]
    ).rename_axis(index="obj_id")


def check_channel_axis_correlation(
    arr: Array, channel_labels: tp.Sequence[str], output_prefix: Path
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

    df = pd.DataFrame(res, columns=["channel", "axis", "coef", "intercept", "r"])
    df["axis_label"] = df["axis"].replace(0, "X_centroid").replace(1, "Y_centroid")
    df["channel_label"] = [x for x in channel_labels for _ in range(2)]
    df["abs_r"] = df["r"].abs()
    df.to_csv(output_prefix + "channel-axis_correlation.csv", index=False)
    return df


def fix_signal_axis_dependency(
    arr: Array, channel_labels: tp.Sequence[str], res: DataFrame, output_prefix: Path
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

        fig, axis = plt.subplots(1, 7, sharex=True, sharey=False, figsize=(7 * 3, 3 * 1))
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


def channel_stats(roi: _roi.ROI, channels: tp.Sequence[str] = None):
    if channels is None:
        channels = roi.channel_labels.tolist()
    stack = roi._get_channels(channels)[1]
    mask = roi.cell_mask == 0
    res = dict()
    res["wmeans"] = pd.Series(stack.mean(axis=(1, 2)), index=channels)
    res["wstds"] = pd.Series(stack.std(axis=(1, 2)), index=channels)
    res["cmeans"] = pd.Series(
        [np.ma.masked_array(stack[i], mask=mask).mean() for i in range(len(channels))],
        index=channels,
    )
    res["cstds"] = pd.Series(
        [np.ma.masked_array(stack[i], mask=mask).std() for i in range(len(channels))],
        index=channels,
    )
    res["emeans"] = pd.Series(
        [np.ma.masked_array(stack[i], mask=~mask).mean() for i in range(len(channels))],
        index=channels,
    )
    res["estds"] = pd.Series(
        [np.ma.masked_array(stack[i], mask=~mask).std() for i in range(len(channels))],
        index=channels,
    )
    res["noises"] = pd.Series([estimate_noise(ch) for ch in stack], index=channels)
    res["sigmas"] = pd.Series(
        estimate_sigma(np.moveaxis(stack, 0, -1), multichannel=True), index=channels
    )
    return res


@MEMORY.cache
def measure_channel_background(
    rois: tp.Sequence[_roi.ROI], plot: bool = True, output_prefix: Path = None
) -> Series:
    from imc.utils import align_channels_by_name
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if plot:
        assert (
            output_prefix is not None
        ), "If `plot` is True, `output_prefix` must be given."

    _channels = pd.DataFrame(
        {r.name: r.channel_labels[~r.channel_exclude.values] for r in rois}
    )
    channels = align_channels_by_name(_channels).dropna().iloc[:, 0].tolist()
    roi_names = [r.name for r in rois]

    res = parmap.map(channel_stats, rois, channels=channels, pm_pbar=True)

    wmeans = pd.DataFrame((x["wmeans"] for x in res), index=roi_names).T
    wstds = pd.DataFrame((x["wstds"] for x in res), index=roi_names).T
    wqv2s = np.sqrt(wstds / wmeans)
    cmeans = pd.DataFrame((x["cmeans"] for x in res), index=roi_names).T
    cstds = pd.DataFrame((x["cstds"] for x in res), index=roi_names).T
    cqv2s = np.sqrt(cstds / cmeans)
    emeans = pd.DataFrame((x["emeans"] for x in res), index=roi_names).T
    estds = pd.DataFrame((x["estds"] for x in res), index=roi_names).T
    eqv2s = np.sqrt(estds / emeans)
    fore_backg: DataFrame = np.log(cmeans / emeans)
    # fore_backg_disp = np.log1p(((cmeans / emeans) / (cmeans + emeans))).mean(1)
    noises = pd.DataFrame((x["noises"] for x in res), index=roi_names).T
    sigmas = pd.DataFrame((x["sigmas"] for x in res), index=roi_names).T

    # Join all metrics
    metrics = (
        wmeans.mean(1)
        .to_frame(name="image_mean")
        .join(wstds.mean(1).rename("image_std"))
        .join(wqv2s.mean(1).rename("image_qv2"))
        .join(cmeans.mean(1).rename("cell_mean"))
        .join(cstds.mean(1).rename("cell_std"))
        .join(cqv2s.mean(1).rename("cell_qv2"))
        .join(emeans.mean(1).rename("extra_mean"))
        .join(estds.mean(1).rename("extra_std"))
        .join(eqv2s.mean(1).rename("extra_qv2"))
        .join(fore_backg.mean(1).rename("fore_backg"))
        .join(noises.mean(1).rename("noise"))
        .join(sigmas.mean(1).rename("sigma"))
    ).rename_axis(index="channel")
    metrics_std = (metrics - metrics.min()) / (metrics.max() - metrics.min())

    if not plot:
        # Invert QV2
        sel = metrics_std.columns.str.contains("_qv2")
        metrics_std.loc[:, sel] = 1 - metrics_std.loc[:, sel]
        # TODO: better decision on which metrics matter
        return metrics_std.mean(1)

    output_prefix = cast(output_prefix)
    if not output_prefix.endswith("."):
        output_prefix += "."

    metrics.to_csv(output_prefix + "channel_background_noise_measurements.csv")
    metrics = pd.read_csv(
        output_prefix + "channel_background_noise_measurements.csv", index_col=0
    )

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(3 * 4.1, 2 * 4), sharex="col")
    axes[0, 0].set_title("Whole image")
    axes[0, 1].set_title("Cells")
    axes[0, 2].set_title("Extracellular")
    for i, (means, stds, qv2s) in enumerate(
        [(wmeans, wstds, wqv2s), (cmeans, cstds, cqv2s), (emeans, estds, eqv2s)]
    ):
        # plot mean vs variance
        mean = means.mean(1)
        std = stds.mean(1) ** 2
        qv2 = qv2s.mean(1)
        fb = fore_backg.mean(1)

        axes[0, i].set_xlabel("Mean")
        axes[0, i].set_ylabel("Variance")
        pts = axes[0, i].scatter(mean, std, c=fb)
        if i == 2:
            div = make_axes_locatable(axes[0, i])
            cax = div.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(pts, cax=cax)

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
        axes[1, i].scatter(mean, qv2, c=fb)
        for channel in means.index:
            lab = "left" if np.random.rand() > 0.5 else "right"
            axes[1, i].text(
                mean.loc[channel], qv2.loc[channel], channel, ha=lab, fontsize=4
            )
        axes[1, i].axhline(1, linestyle="--", color="grey")
        axes[1, i].set_xscale("log")
        # if qv2.min() > 0.01:
        #     axes[1, i].set_yscale("log")
    fig.savefig(output_prefix + "channel_mean_variation_noise.svg", **FIG_KWS)

    fig, axes = plt.subplots(1, 2, figsize=(2 * 6.2, 4))
    p = fore_backg.mean(1).sort_values()
    r1 = p.rank()
    r2 = p.abs().rank()
    axes[0].scatter(r1, p)
    axes[1].scatter(r2, p.abs())
    for i in p.index:
        axes[0].text(r1.loc[i], p.loc[i], s=i, rotation=90, ha="center", va="bottom")
        axes[1].text(
            r2.loc[i], p.abs().loc[i], s=i, rotation=90, ha="center", va="bottom"
        )
    axes[1].set_yscale("log")
    axes[0].set_xlabel("Channel rank")
    axes[1].set_xlabel("Channel rank")
    axes[0].set_ylabel("Cellular/extracellular difference")
    axes[1].set_ylabel("Cellular/extracellular difference (abs)")
    axes[0].axhline(0, linestyle="--", color="grey")
    axes[1].axhline(0, linestyle="--", color="grey")
    fig.savefig(
        output_prefix + "channel_foreground_background_diff.rankplot.svg",
        **FIG_KWS,
    )

    grid = sns.clustermap(
        metrics_std,
        xticklabels=True,
        yticklabels=True,
        metric="correlation",
        cbar_kws=dict(label="Variable (min-max)"),
    )
    grid.fig.savefig(
        output_prefix + "channel_mean_variation_noise.clustermap.svg", **FIG_KWS
    )

    # Invert QV2
    sel = metrics_std.columns.str.contains("_qv2")
    metrics_std.loc[:, sel] = 1 - metrics_std.loc[:, sel]
    # TODO: better decision on which metrics matter
    return metrics_std.mean(1)


def anndata_to_cluster_means(
    ann: AnnData, raw: bool = False, cluster_label: str = "cluster"
) -> DataFrame:
    means = dict()
    obj = ann if not raw else ann.raw
    for cluster in ann.obs[cluster_label].unique():
        clust = ann.obs[cluster_label] == cluster
        means[cluster] = obj[clust, :].X.mean(0)
    mean_expr = pd.DataFrame(means, index=obj.var.index).sort_index(axis=1)
    mean_expr.columns.name = "cluster"
    return mean_expr


def phenotyping(
    a: tp.Union[AnnData, Path],
    channels_include: tp.Sequence[str] = None,
    channels_exclude: tp.Sequence[str] = None,
    filter_cells: bool = True,
    z_score: bool = True,
    z_score_per: str = "roi",
    z_score_cap: float = 3.0,
    remove_batch: bool = True,
    batch_variable: str = "sample",
    dim_res_algos: tp.Sequence[str] = ("umap",),
    clustering_method: str = "leiden",
    clustering_resolutions: tp.Sequence[float] = (1.0,),
) -> AnnData:
    import anndata

    # Checks
    reason = f"Can only Z-score values per 'roi' or 'sample'. '{z_score_per}' is not supported."
    assert z_score_per in ["sample", "roi"], reason
    reason = f"Clustering method '{clustering_method}' is not supported."
    assert clustering_method in ["leiden", "parc"]
    reason = "Can only use 'pca', 'umap', 'diffmap', or 'pymde' in `dim_res_algos`."
    assert all(x in ["pca", "umap", "diffmap", "pymde"] for x in dim_res_algos), reason
    if "pymde" in dim_res_algos:
        import pymde
    if clustering_method == "parc":
        from parc import PARC

    if isinstance(a, Path):
        print(f"Reading h5ad file: '{a}'.")
        a = sc.read(a)

    if "sample" not in a.obs.columns:
        a.obs["sample"] = a.obs["roi"].str.extract(r"(.*)-\d+")[0].fillna("")
    if a.raw is None:
        a.raw = a

    # Add morphological variables to obs
    sel = a.var.index.str.contains(r"\(")
    v = a.var.index[~sel]
    for col in v:
        a.obs[col] = a[:, col].X
    a = a[:, sel]

    # Filter out channels
    if channels_exclude is not None:
        a = a[:, ~a.var.index.isin(channels_exclude)]
    if channels_include is not None:
        a = a[:, channels_include]

    # # reduce DNA chanels to one, and move to obs
    dnas = a.var.index[a.var.index.str.contains(r"DNA\d")]
    a.obs["DNA"] = a[:, dnas].X.mean(1)
    a = a[:, ~a.var.index.isin(dnas)]

    # Filter out cells
    if filter_cells:
        if "solidity" not in a.obs.columns:
            print(
                "Could not filter cells based on solidity likely because morphological quantification was not performed!"
            )
        else:
            exclude = a.obs["solidity"] == 1
            p = (exclude).sum() / a.shape[0] * 100
            print(f"Filtered out {exclude.sum()} cells ({p:.2f} %)")

    # Scaling/Normalization
    print("Performing data scaling/normalization.")
    sc.pp.log1p(a)
    if z_score:
        _ads = list()
        for roi_name in a.obs["roi"].unique():
            a2 = a[a.obs["roi"] == roi_name, :]
            sc.pp.scale(a2, max_value=z_score_cap)
            a2.X[a2.X < -z_score_cap] = -z_score_cap
            # print(a2.X.min(), a2.X.max())
            _ads.append(a2)
        a = anndata.concat(_ads)
        sc.pp.scale(a)
    if remove_batch:
        if a.obs[batch_variable].nunique() > 1:
            sc.pp.combat(a, batch_variable)
            sc.pp.scale(a)

    # Dimensionality reduction
    print("Performing dimensionality reduction.")
    sc.pp.pca(a)
    if remove_batch:
        sc.external.pp.bbknn(a, batch_key=batch_variable)
    else:
        sc.pp.neighbors(a)
    if "umap" in dim_res_algos:
        sc.tl.umap(a, gamma=25)
    if "diffmap" in dim_res_algos:
        sc.tl.diffmap(a)
    if "pymde" in dim_res_algos:
        a.obsm["X_pymde"] = pymde.preserve_neighbors(a.X, embedding_dim=2).embed().numpy()
        a.obsm["X_pymde2"] = (
            pymde.preserve_neighbors(
                a.X,
                embedding_dim=2,
                attractive_penalty=pymde.penalties.Quadratic,
                repulsive_penalty=None,
            )
            .embed()
            .numpy()
        )

    # Clustering
    print("Performing clustering.")
    if clustering_method == "leiden":
        for res in clustering_resolutions:
            sc.tl.leiden(a, resolution=res, key_added=f"cluster_{res}")
            a.obs[f"cluster_{res}"] = pd.Categorical(
                a.obs[f"cluster_{res}"].astype(int) + 1
            )
    elif clustering_method == "parc":
        from parc import PARC

        for res in clustering_resolutions:
            p = PARC(
                a.X,
                neighbor_graph=a.obsp["connectivities"],
                random_seed=42,
                resolution_parameter=res,
            )
            p.run_PARC()
            a.obs[f"cluster_{res}"] = pd.Categorical(pd.Series(p.labels) + 1)

    print("Finished phenotyping.")
    return a


def plot_phenotyping(
    a: tp.Union[AnnData, Path],
    output_prefix: Path,
    tech_channels: tp.Sequence[str] = None,
    dim_res_algos: tp.Sequence[str] = ("umap",),
    clustering_resolutions: tp.Sequence[float] = None,
):
    from matplotlib.backends.backend_pdf import PdfPages
    from imc.graphics import add_centroids
    from seaborn_extensions import clustermap

    figkws = dict(dpi=300, bbox_inches="tight")

    # Read in
    if isinstance(a, Path):
        print(f"Reading h5ad file: '{a}'.")
        a = sc.read(a)
    a = a[a.obs.sample(frac=1).index]

    # Checks
    if output_prefix.is_dir():
        output_prefix = output_prefix / "phenotypes."
    if not output_prefix.endswith("."):
        output_prefix += "."
    output_prefix.parent.mkdir()

    if "sample" not in a.obs.columns:
        a.obs["sample"] = a.obs["roi"].str.extract(r"(.*)-\d+")[0].fillna("")

    if tech_channels is None:
        tech_channels = [
            "DNA",
            "eccentricity",
            "solidity",
            "area",
            "perimeter",
            "major_axis_length",
        ]
        tech_channels = [c for c in tech_channels if c in a.obs.columns]

    if clustering_resolutions is None:
        clustering_resolutions = (
            a.obs.columns[a.obs.columns.str.contains("cluster_")]
            .str.extract(r"cluster_(.*)$")[0]
            .astype(float)
        )

    # Plot projections
    non_tech_channels = a.var.index[~a.var.index.isin(tech_channels)].tolist()
    vmax = (
        [None]
        + np.percentile(a.raw[:, non_tech_channels].X, 95, axis=0).tolist()
        + np.percentile(a.obs[tech_channels], 95, axis=0).tolist()
        # + [None]
        + ([None] * len(clustering_resolutions))
    )
    color = (
        ["sample"]
        + non_tech_channels
        + tech_channels
        # + ["topological_domain"]
        + [f"cluster_{res}" for res in clustering_resolutions]
    )
    for algo in tqdm(dim_res_algos):
        f = output_prefix + f"{algo}.pdf"
        with PdfPages(f) as pdf:
            for i, col in enumerate(color):
                fig = sc.pl.embedding(
                    a,
                    basis=algo,
                    color=col,
                    show=False,
                    vmax=vmax[i],
                    use_raw=True,
                ).figure
                rasterize_scanpy(fig)
                if i >= len(color) - len(clustering_resolutions):
                    res = clustering_resolutions[i - len(color)]
                    add_centroids(a, res=res, ax=fig.axes[0], algo=algo)
                plt.figure(fig)
                pdf.savefig(**figkws)
                plt.close(fig)

        # Plot ROIs separately
        f = output_prefix + f"{algo}.sample_roi.pdf"
        projf = getattr(sc.pl, algo)
        fig = projf(a, color=["sample", "roi"], show=False)[0].figure
        rasterize_scanpy(fig)
        fig.savefig(f, **figkws)
        plt.close(fig)

    # Plot average phenotypes
    for res in tqdm(clustering_resolutions):
        df = a.to_df()[non_tech_channels].join(a.obs[tech_channels])

        # Drop variables with no variance
        v = df.var()
        if (v == 0).any():
            df = df.drop(v.index[v == 0], axis=1)

        cluster_means = df.groupby(a.obs[f"cluster_{res}"].values).mean()

        cell_counts = a.obs[f"cluster_{res}"].value_counts().rename("Cells per cluster")

        cell_percs = ((cell_counts / cell_counts.sum()) * 100).rename("Cells (%)")

        op = output_prefix + f"cluster_means.{res}_res."
        kws = dict(
            row_colors=cell_percs.to_frame().join(cell_counts),
            figsize=(10, 6 * res),
        )
        grid = clustermap(cluster_means, **kws)
        grid.savefig(op + "abs.svg")
        plt.close(grid.fig)

        grid = clustermap(cluster_means, **kws, config="z")
        grid.savefig(op + "zscore.svg")
        plt.close(grid.fig)

        # To plot topological domains:
        # df = (a.obs[args.sc_topo.columns.drop(["domain", "topological_domain"])]).replace(
        #     {"False": False, "True": True, "nan": np.nan}
        # )
        # topo_means = df.groupby(a.obs[f"cluster_{res}"].values).mean()
        # topo_means = topo_means.loc[:, topo_means.sum() > 0]

        # g = clustermap(
        #     topo_means.loc[cluster_means.index[grid.dendrogram_row.reordered_ind]],
        #     figsize=(3, 6 * res),
        #     config="z",
        #     row_cluster=False,
        #     cmap="PuOr_r",
        # )
        # g.savefig(op + "abs.topologic.svg")

        # g = clustermap(
        #     topo_means.loc[cluster_means.index[grid.dendrogram_row.reordered_ind]],
        #     figsize=(3, 6 * res),
        #     config="z",
        #     row_cluster=False,
        #     cmap="PuOr_r",
        # )
        # g.savefig(op + "zscore.topologic.svg")

        # grid = clustermap(cluster_means, **kws, config="z", row_cluster=False)
        # grid.savefig(op + "zscore.sorted.svg")
        # g = clustermap(
        #     topo_means,
        #     figsize=(3, 6 * res),
        #     config="z",
        #     row_cluster=False,
        #     cmap="PuOr_r",
        # )
        # g.savefig(op + "zscore.sorted.topologic.svg")
        # plt.close("all")


def single_cell_analysis(
    output_prefix: Path,
    quantification: DataFrame = None,
    rois: tp.List["ROI"] = None,
    label_clusters: bool = True,
    plot: bool = True,
    intensity: bool = True,
    morphology: bool = True,
    filter_channels: bool = False,
    cell_type_channels: tp.List[str] = None,
    channel_filtering_threshold: float = 0.1,  # 0.05
    channel_include: tp.List[str] = None,
    channel_exclude: tp.Sequence[str] = [
        "<EMPTY>",
        "EMPTY",
        "Ar80",
        "Ru9",
        "Ru10",
    ],  # r"Ru\d+", "DNA"
    cluster_min_percentage: float = 1.0,
    leiden_clustering_resolution: float = DEFAULT_SINGLE_CELL_RESOLUTION,
    plot_only_channels: tp.Sequence[str] = None,
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
        metric = measure_channel_background(rois, plot=plot, output_prefix=output_prefix)
        channel_threshold = metric > channel_filtering_threshold
        filtered_channels = metric[channel_threshold].index.tolist()
    else:
        channel_threshold = pd.Series([True] * len(channel_labels), index=channel_labels)
        filtered_channels = channel_labels.tolist()

    if quantification is None:
        print("Quantifying single cells.")
        quantification = quantify_cells_rois(
            rois=rois, intensity=intensity, morphology=morphology
        )

    # Remove excluded channels
    for _ch in channel_exclude:
        quantification = quantification.loc[:, ~quantification.columns.str.contains(_ch)]
    # Filter out low QC channels
    if filter_channels:
        # TODO: fileter channels by QC metrics
        pass

    # Keep only include channels
    if channel_include is not None:
        _includes = [_ch for _ch in quantification.columns if _ch in channel_include]
        quantification = quantification.loc[:, _includes]

    # Get categoricals
    cats = [x for x in ["sample", "roi"] if x in quantification.columns]

    # Start usual single cell analysis
    ann = AnnData(
        quantification.drop(cats, axis=1).sort_index(axis=1).reset_index(drop=True)
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
    sc.tl.leiden(ann_ct, key_added="cluster", resolution=leiden_clustering_resolution)

    ann_ct.obs["cluster"] = pd.Categorical(ann_ct.obs["cluster"].astype(int) + 1)
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
    sc.tl.rank_genes_groups(ann, groupby="cluster", method="logreg", n_genes=ann.shape[1])

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
        sns.barplot(counts, counts.index, ax=axs, orient="horiz", palette="magma")
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
            cluster_counts_per_roi if not log else np.log10(1 + cluster_counts_per_roi),
            cbar_kws=dict(label="Cells per cluster" + ("" if not log else " (log10)")),
            ax=ax,
            **kwargs,
        )
    sns.heatmap(
        (cluster_counts_per_roi / cluster_counts_per_roi.sum()) * 100,
        cbar_kws=dict(label="Cells per cluster (%)"),
        ax=axes[2],
        **kwargs,
    )
    fig.savefig(output_prefix + "cell.counts_per_cluster_per_roi.svg", **FIG_KWS)

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
            mean_expr.loc[:, (counts / counts.sum()) >= cluster_min_percentage / 100],
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
                output_prefix + f"cell.mean_expression_per_cluster.{label1}.{label2}svg",
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
    h5ad_file: Path = None,
    mean_expr: DataFrame = None,
    cluster_assignments: Series = None,
    cell_type_channels: tp.List[str] = None,
    output_prefix: Path = None,
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
        output_prefix + "mean_expression_per_cluster.both_z.threshold_position.svg"
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
        output_prefix + "mean_expression_per_cluster.labeled.both_z.correlation.svg",
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
    sample: _sample.IMCSample,
    reference_csv: str = None,
    h5ad_file: Path = None,
    output_prefix: Path = None,
    plot: bool = True,
):
    from imc.utils import get_mean_expression_per_cluster

    output_prefix = output_prefix or (sample.root_dir / "single_cell" / sample.name + ".")
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
        output_prefix + "cell_type_assignment_against_reference.correlation.svg",
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
    grid.savefig(output_prefix + "cluster_means.predicted_labels.svg", **FIG_KWS)

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


# def merge_clusterings(samples: tp.Sequence["IMCSample"]):

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
    roi: _roi.ROI,
    output_prefix: Path = None,
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
        nx.set_node_attributes(g, roi.clusters.index.to_series().to_dict(), name="obj_id")
    # save graph
    nx.readwrite.write_gpickle(g, output_prefix + "neighbor_graph.gpickle")
    return g


def measure_cell_type_adjacency(
    roi: _roi.ROI,
    method: str = "random",
    adjacency_graph: nx.Graph = None,
    n_iterations: int = 100,
    inf_replace_method: str = "min",
    output_prefix: Path = None,
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

    adj, order = nx.linalg.attrmatrix.attr_matrix(adjacency_graph, node_attr="cluster")
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
        norm_freqs.to_csv(output_prefix + "cluster_adjacency_graph.norm_over_random.csv")

    if not plot:
        return norm_freqs
    v = norm_freqs.values.std() * 2
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(4 * 2, 4))
    kws = dict(cmap="RdBu_r", center=0, square=True, xticklabels=True, yticklabels=True)
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
            output_prefix + "cluster_adjacency_graph.norm_over_random.clustermap.svg",
            **FIG_KWS,
        )
    except FloatingPointError:
        pass
    return norm_freqs


def correct_interaction_background_random(
    roi: _roi.ROI,
    freqs: DataFrame,
    attribute,
    n_iterations: int,
    save: bool,
    output_prefix: tp.Union[str, Path],
):
    values = {
        x: roi.adjacency_graph.nodes[x][attribute] for x in roi.adjacency_graph.nodes
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
    inf_replace_method: tp.Optional[str] = "min_symmetric",
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
        norm_freqs = norms.replace(-np.inf, norms[norms != (-np.inf)].min().min())
    # # 2. replace with minus highest (try to )
    if inf_replace_method == "max":
        norm_freqs = norms.replace(-np.inf, -norms.max().max())
    # # 3. One of the above + make symmetric by  X @ X.T + Z-score
    if inf_replace_method == "min_symmetric":
        norm_freqs = norms.replace(-np.inf, norms[norms != (-np.inf)].min().min())
        norm_freqs = norm_freqs @ norm_freqs.T
        norm_freqs = (norm_freqs - norm_freqs.values.mean()) / norm_freqs.values.std()
    if inf_replace_method == "max_symmetric":
        norm_freqs = norms.replace(-np.inf, norms[norms != (-np.inf)].max().max())
        norm_freqs = norm_freqs @ norm_freqs.T
        norm_freqs = (norm_freqs - norm_freqs.values.mean()) / norm_freqs.values.std()
    return norm_freqs


def find_communities(
    roi: _roi.ROI,
    community_resolution: float = DEFAULT_COMMUNITY_RESOLUTION,
    plot: bool = True,
) -> tp.Tuple[Series, tp.Tuple]:
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

    def get_community_members(partition: tp.Dict) -> tp.Dict:
        counts = Counter(partition)
        # {com: members}
        comms: tp.Dict[int, set] = dict()
        for com in counts.keys():
            comms[com] = set()
        for n, com in partition.items():
            comms[com].add(n)
        return comms

    def get_community_cell_type_composition(roi: _roi.ROI, partition: Series):
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
    roi_output_prefix = roi.sample.root_dir / "single_cell" / (roi.name + ".communities.")

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
        grid.savefig(roi_output_prefix + "cell_type_composition.zscore.svg", **FIG_KWS)
    return partition, comps


def cluster_communities(
    rois: tp.Sequence[_roi.ROI],
    output_prefix: Path = None,
    supercommunity_resolution: float = DEFAULT_SUPERCOMMUNITY_RESOLUTION,
) -> Series:
    output_prefix = output_prefix or (
        rois[0].prj.processed_dir / "single_cell" / (rois[0].prj.name + ".communities.")
    )
    output_prefix = cast(output_prefix)

    res = parmap.map(find_communities, rois)
    partitions = {k: v[0] for k, v in zip(rois, res)}
    composition = pd.concat([v[1] for v in res]).fillna(0).astype(int).sort_index()
    composition.to_csv(output_prefix + ".all_communities.cell_type_composition.csv")

    print(f"Found {composition.shape[0]} communities across all ROIs.")

    composition = pd.read_csv(
        output_prefix + ".all_communities.cell_type_composition.csv",
        index_col=[0, 1, 2],
    )

    # Cluster communities by leiden clustering based on cell type composition
    a = AnnData(composition)
    sc.pp.log1p(a)
    sc.pp.neighbors(a)
    sc.tl.leiden(a, resolution=supercommunity_resolution, key_added="supercommunity")
    n_scomms = len(a.obs["supercommunity"].unique())
    print(f"Found {n_scomms} supercommunities.")
    # Make supercommunities 1-based (to distinguish from masks where 0 == background)
    a.obs["supercommunity"] = pd.Categorical(a.obs["supercommunity"].astype(int) + 1)
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
        c2 = pd.Series(partitions[roi], name="community").rename_axis(index="obj_id")
        scomm = a.obs.loc[(roi.sample.name, roi.name), "supercommunity"].astype(int)
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
    assignments = pd.concat(_assignments).set_index(["sample", "roi"], append=True)
    assignments.index = assignments.index.reorder_levels(["sample", "roi", "obj_id"])

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
    s = assignments["supercommunity"].value_counts().sort_values(ascending=False)
    assignments["supercommunity"] = assignments["supercommunity"].replace(
        dict(zip(s.index, np.arange(1, len(s))))
    )

    # save final assignments
    assignments.to_csv(output_prefix + "cell_type.community.supercommunities.csv")

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
        grid.savefig(output_prefix + f"{var_}.cell_type_composition.svg", **FIG_KWS)
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
    patches: tp.Dict[str, tp.List] = dict()
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
        add_legend(patches[var_], axes[-1, j], **leg_kws)  # label="Super community",
        _z = zip(
            axes[0].squeeze(),
            ["Cell types", "Communities", "Super communities"],
        )
    for axs, lab in _z:
        axs.set_title(lab)
    # TODO: limit rasterization to main image
    for axs in axes.flat:
        axs.set_rasterized(True)
    fig.savefig(output_prefix + "communities_supercommunities.all_rois.svg", **FIG_KWS)

    return assignments["supercommunity"]

    # # # roi vs supercommunity
    # rs = (
    #     assignments
    #     .assign(count=1)
    #     .reset_index()
    #     .pivot_table(columns=['sample', 'roi'], index='supercommunity', values='count', aggfunc=sum, fill_value=0))
    # rs = rs / rs.sum()


@tp.overload
def get_best_mixture_number(
    x: Series,
    min_mix: int,
    max_mix: int,
    subsample_if_needed: bool,
    n_iters: int,
    metrics: tp.Sequence[str],
    red_func: str,
    return_prediction: tp.Literal[False],
) -> int:
    ...


@tp.overload
def get_best_mixture_number(
    x: Series,
    min_mix: int,
    max_mix: int,
    subsample_if_needed: bool,
    n_iters: int,
    metrics: tp.Sequence[str],
    red_func: str,
    return_prediction: tp.Literal[True],
) -> tp.Tuple[int, Array]:
    ...


def get_best_mixture_number(
    x: Series,
    min_mix: int = 2,
    max_mix: int = 6,
    subsample_if_needed: bool = True,
    n_iters: int = 3,
    metrics: tp.Sequence[str] = [
        "silhouette_score",
        "calinski_harabasz_score",
        "davies_bouldin_score",
    ],
    red_func: str = "mean",
    return_prediction: bool = False,
) -> tp.Union[int, tp.Tuple[int, Array]]:
    from sklearn.mixture import GaussianMixture
    import sklearn.metrics

    def get_means(num: Series, pred: tp.Union[Series, Array]) -> Series:
        return num.groupby(pred).mean().sort_values()

    def replace_pred(x: Series, y: tp.Union[Series, Array]) -> Series:
        means = get_means(x, y)
        repl = dict(zip(means.index, range(len(means))))
        y2 = pd.Series(y, index=x.index).replace(repl)
        new_means = get_means(x, y2.values)
        assert all(new_means.index == range(len(new_means)))
        return y2

    xx = x.sample(n=10_000) if subsample_if_needed and x.shape[0] > 10_000 else x

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
    x: Series, y: Series = None, n_components: int = 2
) -> Array:
    def get_means(num: Series, pred: tp.Union[Series, Array]) -> Series:
        return num.groupby(pred).mean().sort_values()

    def replace_pred(x: Series, y: tp.Union[Series, Array]) -> Series:
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
    thresh = x.loc[((y[:-1] < y[1::])).tolist() + [False]].reset_index(drop=True)
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
    x: tp.Union[Series, DataFrame], n_mixtures: tp.Union[int, tp.List[int]] = None
) -> tp.Union[Series, DataFrame]:
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
            n_best = get_best_mixture_number(x, return_prediction=False)  # type: ignore[call-tp.overload]
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
        raise ValueError("Chosen population must be '0' (lowest) or '-1' (highest).")

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
    nuclear_channels: tp.Sequence[str] = None,
    cytoplasm_channels: tp.Sequence[str] = None,
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
        _cytoplasm_channels = channel_labels[~channel_labels.isin(_nuclear_channels)]
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


def save_probabilities(probs: Array, output_tiff: Path):
    import tifffile

    tifffile.imsave(output_tiff, np.moveaxis((probs * 2 ** 16).astype("uint16"), 0, -1))


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


def label_domains(
    rois: tp.Sequence[_roi.ROI],
    output_dir: Path,
    export: bool = True,
    domains: tp.Sequence[str] = ["T", "S", "A", "L", "V", "E"],
    **kwargs,
) -> None:
    """
    Draw shapes outying topological domains in tissue.
    This step is done manually using the `labelme` program.

    $ labelme --autosave --labels metadata/labelme_labels.txt
    """
    if export:
        export_images_for_topological_labeling(rois, output_dir, **kwargs)

    labels_f = (output_dir).mkdir() / "labelme_labels.txt"
    with open(labels_f, "w") as handle:
        handle.write("\n".join(domains))
    os.system(f"labelme --autosave --labels {labels_f} {output_dir}")


def export_images_for_topological_labeling(
    rois: tp.Sequence[_roi.ROI],
    output_dir: Path,
    channels: tp.Sequence[str] = ["mean"],
    overwrite: bool = False,
) -> None:
    """
    Export PNGs for labeling with `labelme`.
    """
    for roi in tqdm(rois):
        f = output_dir / roi.name + ".png"
        if not overwrite and f.exists():
            continue
        array = roi._get_channels(channels, minmax=True, equalize=True)[1].squeeze()
        if array.ndim > 2:
            array = np.moveaxis(array, 0, -1)
        matplotlib.image.imsave(f, array)


def collect_domains(
    input_dir: Path, rois: tp.Sequence[_roi.ROI] = None, output_file: Path = None
) -> tp.Dict[str, tp.Dict]:
    if rois is not None:
        roi_names = [r.name for r in rois]

    filenames = list(input_dir.glob("*.json"))
    if rois is not None:
        filenames = [f for f in filenames if f.stem in roi_names]

    topo_annots = dict()
    for filename in tqdm(filenames):
        annot_f = filename.replace_(".png", ".json")
        if not annot_f.exists():
            continue
        with open(annot_f, "r") as handle:
            annot = json.load(handle)
        topo_annots[filename.stem] = annot["shapes"]
    if output_file is not None:
        with open(output_file, "w") as handle:
            json.dump(topo_annots, handle, indent=4)
    return topo_annots


def illustrate_domains(
    topo_annots: tp.Dict[str, tp.Dict],
    rois: tp.Sequence[_roi.ROI],
    output_dir: Path,
    channels: tp.Sequence[str],
    domain_exclude: tp.Sequence[str] = None,
    cleanup: bool = False,
    cmap_str: str = "Set3",
) -> None:
    """
    Illustrate annotated topological domains of each ROI.
    """
    from imc.utils import polygon_to_mask
    from imc.graphics import legend_without_duplicate_labels
    from shapely.geometry import Polygon

    if domain_exclude is None:
        domain_exclude = []

    labels = list(set(geom["label"] for n, j in topo_annots.items() for geom in j))
    label_color = dict(zip(labels, sns.color_palette(cmap_str)))
    label_order = dict(zip(labels, range(1, len(labels) + 1)))
    cmap = plt.get_cmap(cmap_str)(range(len(labels) + 1))
    cmap[0] = [0, 0, 0, 1]

    for roi_name in tqdm(topo_annots):
        shapes = topo_annots[roi_name]
        roi = [r for r in rois if r.name == roi_name][0]
        annot_mask = np.zeros(roi.shape[1:])
        for shape in shapes:
            if shape["label"] in domain_exclude:
                continue
            region = polygon_to_mask(shape["points"], roi.shape[1:][::-1])
            annot_mask[region > 0] = label_order[shape["label"]]

        ar = roi.shape[1] / roi.shape[2]

        fig, axes = plt.subplots(
            1, 2, figsize=(2 * 4, 4 * ar), gridspec_kw=dict(wspace=0, hspace=0)
        )
        axes[0].set(title=roi.name)
        roi.plot_channels(channels, axes=[axes[0]], merged=True)

        shape_types: Counter[str] = Counter()
        for shape in shapes:
            label: str = shape["label"]
            if label in domain_exclude:
                continue
            shape_types[label] += 1
            c = Polygon(shape["points"]).centroid
            axes[1].text(
                c.x,
                c.y,
                s=f"{label}{shape_types[label]}",
                ha="center",
                va="center",
            )
            axes[0].plot(
                *np.asarray(shape["points"] + [shape["points"][0]]).T,
                label=label,
                color=cmap[label_order[label]],
            )

        m = annot_mask == 0
        annot_mask += 1
        annot_mask[m] = 0
        axes[1].imshow(
            annot_mask,
            cmap=cmap_str,
            vmin=1,
            vmax=len(label_color) + 1,
            interpolation="none",
        )
        axes[1].set(title="Manual annotations")
        legend_without_duplicate_labels(axes[0], title="Domain:")
        for ax in axes:
            ax.axis("off")
        fig.savefig(
            output_dir / roi.name + ".annotations.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    cmd = f"""pdftk
    {output_dir}/*.annotations.pdf
    cat
    output 
    {output_dir}/topological_domain_annotations.pdf"""
    os.system(cmd.replace("\n", " "))

    if cleanup:
        files = output_dir.glob("*.annotations.pdf")
        for file in files:
            file.unlink()


def get_domains_per_cell(
    topo_annots: tp.Dict[str, tp.Dict],
    rois: tp.Sequence[_roi.ROI],
    exclude_domains: tp.Sequence[str] = None,
    remaining_domain: str = "background",
) -> DataFrame:
    """
    Generate annotation of topological domain each cell is contained in
    based on manual annotated masks.

    Parameters
    ----------
    topo_annots: dict
        Dictionary of annotations for each ROI.
    rois: list
        List of ROI objects.
    exclude_domains: list[str]
        Domains to ignore
    exclude_domains: list[str]
        Domains to ignore
    """
    from imc.utils import polygon_to_mask

    if exclude_domains is None:
        exclude_domains = []

    _full_assigns = list()
    for roi_name, shapes in tqdm(topo_annots.items()):
        roi = [r for r in rois if r.name == roi_name][0]
        mask = roi.mask
        cells = np.unique(mask)[1:]
        td_count: Counter[str] = Counter()
        regions = list()
        _assigns = list()
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            if label in exclude_domains:
                continue
            td_count[label] += 1
            points += [points[0]]
            region = polygon_to_mask(points, roi.shape[1:][::-1])
            regions.append(region)
            assign = (
                pd.Series(np.unique(mask[(mask > 0) & region]), name="obj_id")
                .to_frame()
                .assign(
                    roi=roi.name,
                    sample=roi.sample.name,
                    domain_id=f"{label}{td_count[label]}",
                )
            )
            _assigns.append(assign)

        ## if remaining_domain explicetely annotated, skip
        if remaining_domain in td_count:
            print(f"ROI '{roi.name}' has been manually annotated with remaining domains.")
            _full_assigns += _assigns
            continue

        ## add alveolar region as remaining not overlapping with existing regions
        print(f"ROI '{roi.name}' will be annotated with '{remaining_domain}' by default.")
        # remain = ~polygon_to_mask(np.concatenate(polys), roi.shape[1:][::-1]) # <- wrong
        remain = ~np.asarray(regions).sum(0).astype(bool)
        existing = np.sort(pd.concat(_assigns)["obj_id"].unique())
        remain = remain & (~np.isin(mask, existing))

        assign = (
            pd.Series(np.unique(mask[remain]), name="obj_id")
            .drop(0, errors="ignore")
            .to_frame()
            .assign(
                roi=roi.name,
                sample=roi.sample.name,
                domain_id=remaining_domain + "1",
            )
        )
        _assigns.append(assign)
        _full_assigns += _assigns

        # # To visualize:
        # c = (
        #     pd.concat(_assigns)
        #     .set_index(["roi", "obj_id"])["domain_id"]
        #     .rename("cluster")
        # )
        # c = c.str.replace(r"\d", "", regex=True)
        # domains = set(assigns['topological_domain'])
        # fig = roi.plot_cell_types(
        #     c.replace({k + 1: f"{v} - {k}" for k, v in zip(range(len(domains)), domains)})
        # )

    assigns = pd.concat(_full_assigns)
    assigns["topological_domain"] = assigns["domain_id"].str.replace(
        r"\d", "", regex=True
    )

    # reduce duplicated annotations but for cells annotated with background, make that the primary annotation
    id_cols = ["sample", "roi", "obj_id"]
    assigns = (
        assigns.groupby(id_cols).apply(
            lambda x: x
            if (x.shape[0] == 1)
            else x.loc[x["topological_domain"] == remaining_domain, :]
            if (x["topological_domain"] == remaining_domain).any()
            else x
        )
        # .drop(id_cols, axis=1)
        .reset_index(level=-1, drop=True)
    ).set_index(id_cols)

    # make sure there are no cells with more than one domain that is background
    tpc = assigns.groupby(id_cols)["domain_id"].nunique()
    cells = tpc.index
    assert not assigns.loc[cells[tpc > 1]].isin([remaining_domain]).any().any()

    assigns = (
        assigns.reset_index()
        .drop_duplicates(subset=id_cols)
        .set_index(id_cols)
        .sort_index()
    )

    # expand domains
    for domain in assigns["topological_domain"].unique():
        assigns[domain] = assigns["topological_domain"] == domain

    return assigns


@tp.overload
def get_domain_areas(
    topo_annots: tp.Dict[str, tp.Dict],
    rois: tp.Sequence[_roi.ROI],
    per_domain: tp.Literal[False],
) -> tp.Dict[Path, float]:
    ...


@tp.overload
def get_domain_areas(
    topo_annots: tp.Dict[str, tp.Dict],
    rois: tp.Sequence[_roi.ROI],
    per_domain: tp.Literal[True],
) -> DataFrame:
    ...


def get_domain_areas(
    topo_annots: tp.Dict[str, tp.Dict],
    rois: tp.Sequence[_roi.ROI] = None,
    per_domain: bool = False,
) -> tp.Union[tp.Dict[Path, float], DataFrame]:
    """
    Get area of airways per image in microns.
    """
    from shapely.geometry import Polygon

    mpp = 1  # scale
    if rois is not None:
        roi_names = [r.name for r in rois]
        topo_annots = {k: v for k, v in topo_annots.items() if k in roi_names}

    _areas = list()
    for roi_name, shapes in tqdm(topo_annots.items()):
        count: Counter[str] = Counter()
        for shape in shapes:
            label = shape["label"]
            count[label] += 1
            a = Polygon(shape["points"]).area
            _areas.append([roi_name, label + str(count[label]), a * mpp])

    areas = (
        pd.DataFrame(_areas)
        .rename(columns={0: "filename", 1: "domain_domain_obj", 2: "area"})
        .set_index("filename")
    )
    if not per_domain:
        areas = areas.groupby("filename")["area"].sum().to_dict()
    return areas


def get_domain_mask(
    topo_annot: tp.Dict,
    roi: _roi.ROI,
    exclude_domains: tp.Sequence[str],
    fill_remaining: str = None,
    per_domain: bool = False,
) -> Array:
    """ """
    import tifffile
    from imc.utils import polygon_to_mask

    _, h, w = roi.shape
    masks = list()
    region_types = list()
    region_names = list()
    count: Counter[str] = Counter()
    for shape in topo_annot:
        shape["points"] += [shape["points"][0]]
        region = polygon_to_mask(shape["points"], (w, h))
        label = shape["label"]
        count[label] += 1
        masks.append(region)
        region_types.append(label)
        region_names.append(label + str(count[label]))

    for_mask = np.asarray(
        [m for l, m in zip(region_types, masks) if l not in exclude_domains]
    ).sum(0)
    if fill_remaining is not None:
        masks += [for_mask == 0]
        region_types += [fill_remaining]
        for_mask[for_mask == 0] = -1
    exc_mask = np.asarray(
        [m for l, m in zip(region_types, masks) if l in exclude_domains]
    ).sum(0)
    mask: Array = (
        ((for_mask != 0) & ~(exc_mask != 0))
        if isinstance(exc_mask, np.ndarray)
        else for_mask
    ).astype(bool)

    if per_domain:
        nmask = np.empty_like(mask, dtype="object")
        for r, l in zip(masks, region_types):
            if l not in exclude_domains:
                nmask[mask & r] = l
        mask = np.ma.masked_array(nmask, mask=nmask == None)

    return mask
