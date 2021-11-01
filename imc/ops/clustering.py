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
import imc.data_models.roi as _roi
from imc.exceptions import cast
from imc.types import DataFrame, Series, Path, MultiIndexSeries
from imc.utils import minmax_scale, double_z_score
from imc.graphics import rasterize_scanpy


FIG_KWS = dict(bbox_inches="tight", dpi=300)
sc.settings.n_jobs = -1


DEFAULT_CELL_TYPE_REFERENCE = (
    "https://gist.github.com/afrendeiro/4aa133c2fcb5eb0152957b11ec753b74/raw",
    Path(".imc.cell_type_reference.yaml"),
)


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
                pdf.savefig(**FIG_KWS)
                plt.close(fig)

        # Plot ROIs separately
        f = output_prefix + f"{algo}.sample_roi.pdf"
        projf = getattr(sc.pl, algo)
        fig = projf(a, color=["sample", "roi"], show=False)[0].figure
        rasterize_scanpy(fig)
        fig.savefig(f, **FIG_KWS)
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


def predict_cell_types_from_reference(
    quant: tp.Union[AnnData, DataFrame, Path],
    output_prefix: Path,
    covariates: DataFrame,
    method: str = "astir",
    astir_reference: Path = None,
    astir_parameters: tp.Dict[str, tp.Any] = {},
):
    import anndata
    import yaml
    from imc.utils import download_file

    # Get dataframe with expression
    if isinstance(quant, Path):
        if quant.endswith("csv") or quant.endswith("csv.gz"):
            quant = pd.read_csv(quant, index_col=0)
        elif quant.endswith(".h5ad"):
            quant = anndata.read(quant)
    elif isinstance(quant, anndata.AnnData):
        quant = quant.to_df()

    # Remove metal label from column names
    quant.columns = quant.columns.str.extract(r"(.*)\(.*")[0].fillna(
        quant.columns.to_series().reset_index(drop=True)
    )

    if method != "astir":
        raise NotImplementedError("Only the `astir` method is currently supported.")

    # Prepare reference dictionary
    if astir_reference is not None:
        reference = yaml.safe_load(astir_reference.open())
    else:
        # if not DEFAULT_CELL_TYPE_REFERENCE[1].exists():
        download_file(DEFAULT_CELL_TYPE_REFERENCE[0], DEFAULT_CELL_TYPE_REFERENCE[1])
        ref = yaml.safe_load(DEFAULT_CELL_TYPE_REFERENCE[1].open())
        reference = dict()
        reference["cell_types"] = unroll_reference_dict(ref["cell_types"], False)
        reference["cell_states"] = unroll_reference_dict(ref["cell_states"], False)
        reference = filter_reference_based_on_available_markers(reference, quant.columns)

    res = astir(
        input_expr=quant,
        marker_dict=reference,
        design=covariates,
        output_prefix=output_prefix,
        **astir_parameters,
    )
    return res


def astir(
    input_expr: DataFrame,
    marker_dict: tp.Dict[str, tp.List[str]],
    design: DataFrame,
    output_prefix: Path,
    batch_size: int = None,
    max_epochs: int = 200,
    learning_rate: float = 2e-3,
    initial_epochs: int = 3,
    device: str = "cpu",
    plot: bool = True,
):
    from astir import Astir
    import torch

    if output_prefix.is_dir():
        output_prefix = output_prefix / "astir."
        output_prefix.parent.mkdir()

    ast = Astir(input_expr, marker_dict, design)
    ast._device = torch.device("cpu")
    if batch_size is None:
        batch_size = ast.get_type_dataset().get_exprs_df().shape[0] // 100

    params = dict(
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_init_epochs=initial_epochs,
    )
    res = pd.DataFrame(index=input_expr.index)
    if "cell_types" in marker_dict:
        ast.fit_type(**params)
        _t = ast.get_celltypes()
        res = res.join(_t)
        _tp = ast.get_celltype_probabilities()
        _tp.columns = _tp.columns + "_probability"
        res = res.join(_tp)
        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(4, 2))
            ax.plot(ast.get_type_losses(), label="loss")
            ax.legend()
            ax.set(xlabel="Epochs", ylabel="Loss")
            fig.savefig(output_prefix + "cell_type.loss.svg", **FIG_KWS)
            plt.close(fig)
    if "cell_states" in marker_dict:
        ast.fit_state(**params)
        _s = ast.get_cellstates()
        res = res.join(_s)
        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(4, 2))
            ax.plot(ast.get_state_losses(), label="loss")
            ax.legend()
            ax.set(xlabel="Epochs", ylabel="Loss")
            fig.savefig(output_prefix + "cell_state.loss.svg", **FIG_KWS)
            plt.close(fig)
    ast.save_models(output_prefix + "fitted_model.hdf5")
    return res


def unroll_reference_dict(
    x: tp.Dict,
    name_with_predecessors: bool = True,
    max_depth: int = -1,
    _cur_depth: int = 0,
    _predecessors: tp.List[str] = [],
) -> tp.Dict:
    from copy import deepcopy

    x = deepcopy(x)
    new = dict()
    for k, v in x.items():
        if "markers" in v:
            name = " - ".join(_predecessors + [k]) if name_with_predecessors else k
            if v["markers"] != [None]:
                new[name] = v["markers"]
                v.pop("markers")
        if (
            isinstance(v, dict)
            and (len(v) > 0)
            and ((_cur_depth < max_depth) or max_depth == -1)
        ):
            new.update(
                unroll_reference_dict(
                    v,
                    name_with_predecessors=name_with_predecessors,
                    max_depth=max_depth,
                    _cur_depth=_cur_depth + 1,
                    _predecessors=_predecessors + [k],
                )
            )
    return new


def filter_reference_based_on_available_markers(
    x: tp.Dict, markers: tp.Sequence[str]
) -> tp.Dict:
    def _filter(x2):
        inter = dict()
        for k, v in x2.items():
            n = list(filter(lambda i: i in markers, v))
            if n:
                inter[k] = n
        return inter

    new = dict()
    new["cell_types"] = _filter(x["cell_types"])
    new["cell_states"] = _filter(x["cell_states"])
    return new
