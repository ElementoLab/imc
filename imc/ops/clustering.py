"""
Functions for single-cell clustering.
"""

import os, re, json, typing as tp

from ordered_set import OrderedSet
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from anndata import AnnData
import scanpy as sc

import imc.data_models.sample as _sample
from imc.exceptions import cast
from imc.types import DataFrame, Series, Path, MultiIndexSeries
from imc.utils import minmax_scale
from imc.graphics import rasterize_scanpy


FIG_KWS = dict(bbox_inches="tight", dpi=300)
sc.settings.n_jobs = -1


DEFAULT_SINGLE_CELL_RESOLUTION = 1.0


def anndata_to_cluster_means(
    ann: AnnData, cluster_label: str, raw: bool = False
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
            mean_expr=anndata_to_cluster_means(ann, "cluster"),
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
    mean_expr = anndata_to_cluster_means(ann, "cluster")
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
    std_threshold: float = None,
    cluster_min_percentage: float = 0.0,
) -> Series:
    from seaborn_extensions import clustermap

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
        mean_expr = anndata_to_cluster_means(ann, "cluster")
    elif mean_expr is not None:
        mean_expr = cast(mean_expr)
        fractions = cluster_assignments.value_counts()

    mean_expr = mean_expr.rename_axis(index="Channel", columns="Cluster")

    if cell_type_channels is None:
        cell_type_channels = mean_expr.index.tolist()

    # Remove clusters below a certain percentage if requested
    fractions = (
        fractions[(fractions / fractions.sum()) > (cluster_min_percentage / 100)]
        .rename("Cells per cluster")
        .rename_axis("Cluster")
    )

    # make sure indexes match
    _mean_expr = mean_expr.reindex(fractions.index, axis=1)

    # doubly Z-scored matrix
    mean_expr_z = double_z_score(_mean_expr)

    # use a simple STD threshold for "positiveness"
    v = mean_expr_z.values.flatten()
    if std_threshold is None:
        v1 = get_threshold_from_gaussian_mixture(pd.Series(v)).squeeze()
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

    cmeans = mean_expr.mean(1).rename("Channel mean").rename_axis("Channel")

    t = mean_expr_z >= v1
    kwargs = dict(
        metric="correlation",
        robust=True,
        xticklabels=True,
        yticklabels=True,
        row_colors=cmeans.to_frame(),
        col_colors=fractions.to_frame(),
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
        df = df.loc[df.var(1) > 0, df.var() > 0]
        grid = clustermap(df, **kwargs, **kwargs2)
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
        df = df.loc[df.var(1) > 0, df.var() > 0]
        grid = clustermap(df, **kwargs, **kwargs2)
        grid.savefig(output_prefix + f"mean_expression_per_cluster.{label}.svg")

    # pairwise cluster correlation
    grid = clustermap(
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


# # # roi vs supercommunity
# rs = (
#     assignments
#     .assign(count=1)
#     .reset_index()
#     .pivot_table(columns=['sample', 'roi'], index='supercommunity', values='count', aggfunc=sum, fill_value=0))
# rs = rs / rs.sum()
