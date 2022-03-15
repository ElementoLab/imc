"""
Functions for single-cell adjacency.
"""

import typing as tp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy.ndimage as ndi
from skimage import exposure
from skimage.future import graph
import networkx as nx

import imc.data_models.roi as _roi
from imc.types import DataFrame, Series, Path

FIG_KWS = dict(bbox_inches="tight", dpi=300)
MAX_BETWEEN_CELL_DIST = 4


def get_adjacency_graph(
    roi: _roi.ROI,
    output_prefix: Path = None,
    max_dist: int = MAX_BETWEEN_CELL_DIST,
) -> graph:
    """
    Derive a spatial representation of cells in image using a graph.

    Parameters
    ----------
    roi: imc.ROI
        ROI object to derive graph for.

    output_prefix: typing.Path
        Prefix to output file with graph.
        Defaults to sample root dir / 'single_cell'.

    max_dist: int
        Maximum distance to consider physical interaction between cells (graph edges)

    Returns
    -------
    networkx.Graph
        Adjacency graph for cells in ROI.
    """
    clusters = roi.clusters
    if clusters is None:
        print("ROI does not have assigned clusters.")

    output_prefix = Path(
        output_prefix or roi.sample.root_dir / "single_cell" / (roi.name + ".")
    )
    if not output_prefix.endswith("."):
        output_prefix += "."
    output_prefix.parent.mkdir()

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
    stack = roi.stack
    if hasattr(roi, "channel_exclude"):
        stack = stack[~roi.channel_exclude]
    image_mean = np.asarray([exposure.equalize_hist(x) for x in stack]).mean(0)
    image_mean = (image_mean - image_mean.min()) / (
        np.percentile(image_mean, 98) - image_mean.min()
    )

    # Construct adjacency graph based on cell distances
    g = graph.rag_mean_color(image_mean, mask, connectivity=2, mode="distance")
    # g = skimage.future.graph.RAG(mask, connectivity=2)
    # remove background node (unfortunately it can't be masked beforehand)
    if 0 in g.nodes:
        g.remove_node(0)

    fig, ax = plt.subplots(1, 1)
    i = (image_mean * 255).astype("uint8")
    i = np.moveaxis(np.asarray([i, i, i]), 0, -1)
    lc = graph.show_rag(
        mask.astype("uint32"),
        g,
        i,
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
    plt.close(fig)

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
    """
    Derive an aggregated measure of adjacency betwen cell types for one ROI.

    Parameters
    ----------
    roi: imc.ROI
        ROI object to derive graph for.

    method: str
        Method to normalize interactions by.
        - 'random': generate empirical background of expected interactions based on cell type abundance by randomization (permutation of cell type identities).
        - 'pharmacoscopy': method with analytical solution from Vladimer et al (10.1038/nchembio.2360). Not recommended for small images.
        Default is 'random'.

    adjacency_graph: networkx.Graph
        Adjacency graph per cell for ROI.
        By default, and if not given will be the `ROI.adjacency_graph` attribute.

    n_iterations: int
        Number of permutations to run when `method` == 'random'.
        Defaults to 100.

    inf_replace_method: str
        If `method` == 'pharmacoscopy', how to handle cases where interactions are not observed.

    output_prefix: typing.Path
        Prefix to output file with graph.
        Defaults to sample root dir / 'single_cell'.

    plot: bool
        Whether to plot visualizations.
        Default is `True`.

    save: bool
        Whether to save output to disk.
        Default is `True`.

    Returns
    -------
    pandas.DataFrame
        DataFrame of cell type interactions normalized by `method`.
    """
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
    freqs = pd.DataFrame(adj, order, order).sort_index(axis=0).sort_index(axis=1)
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
    plt.close(fig)
    del kws["square"]
    try:
        grid = sns.clustermap(norm_freqs, **kws, **kws2)
        grid.savefig(
            output_prefix + "cluster_adjacency_graph.norm_over_random.clustermap.svg",
            **FIG_KWS,
        )
        plt.close(grid.fig)
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
            pd.DataFrame(rf, index=rl, columns=rl).sort_index(axis=0).sort_index(axis=1)
        )
    shuffled_freq = pd.concat(shuffled_freqs)
    if save:
        shuffled_freq.to_csv(
            output_prefix
            + f"cluster_adjacency_graph.random_frequencies.all_iterations_{n_iterations}.csv"
        )
    shuffled_freq = shuffled_freq.groupby(level=0).sum().sort_index(axis=1)
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
