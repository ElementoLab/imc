"""
Functions for community detection.
"""

import typing as tp
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import parmap
from anndata import AnnData
import scanpy as sc
import community

import imc.data_models.roi as _roi
from imc.exceptions import cast
from imc.types import Series, Path
from imc.graphics import add_legend


FIG_KWS = dict(bbox_inches="tight", dpi=300)

DEFAULT_SINGLE_CELL_RESOLUTION = 1.0
MAX_BETWEEN_CELL_DIST = 4
DEFAULT_COMMUNITY_RESOLUTION = 0.005
DEFAULT_SUPERCOMMUNITY_RESOLUTION = 0.5
# DEFAULT_SUPER_COMMUNITY_NUMBER = 12


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
    from scipy.cluster.hierarchy import fcluster

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
