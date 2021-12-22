#!/usr/bin/env python

"""
Phenotype cells.
"""

import sys
import typing as tp

import pandas as pd

from imc.ops.clustering import (
    phenotyping,
    # plot_phenotyping,
    predict_cell_types_from_reference,
)
from imc.scripts import build_cli
from imc.utils import filter_kwargs_by_callable


def main(cli: tp.Sequence[str] = None) -> int:
    parser = build_cli("phenotype")
    args = parser.parse_args(cli)
    print("Starting phenotyping step!")

    args.channels_include = (
        args.channels_include.split(",") if args.channels_include is not None else None
    )
    args.channels_exclude = args.channels_exclude.split(",")
    args.dim_res_algos = args.dim_res_algos.split(",")
    args.clustering_resolutions = list(map(float, args.clustering_resolutions.split(",")))
    args.output_dir.mkdir()

    if args.compute:
        print(f"Phenotyping quantified cells in '{args.a}'.")
        pkwargs = filter_kwargs_by_callable(args.__dict__, phenotyping)
        a = phenotyping(**pkwargs)
        a.write(args.output_dir / "processed.h5ad")
        # Save for project:
        # prj.get_input_filename("cell_cluster_assignments")

        # Cell type identity
        # TODO: connect options to CLI
        print("Matching expression to reference cell types.")
        df = a.raw.to_adata().to_df()[a.var.index[~a.var.index.str.contains("EMPTY")]]
        df = df.loc[:, df.var() > 0]
        cov = pd.get_dummies(a.obs[args.batch_variable])
        preds = predict_cell_types_from_reference(df, args.output_dir, covariates=cov)
        a.obs = a.obs.join(preds)
        a.write(args.output_dir / "processed.h5ad")

        # grid = clustermap(a.to_df().groupby(a.obs['cell_type']).mean())
        # grid = clustermap(a.obs.corr(), cmap='RdBu_r', center=0)

    # if args.plot:
    #     print(f"Plotting phenotypes in directory '{args.output_dir}'.")
    #     output_prefix = args.output_dir / "phenotypes."
    #     if args.compute:
    #         args.a = a
    #     pkwargs = filter_kwargs_by_callable(args.__dict__, plot_phenotyping)
    #     plot_phenotyping(output_prefix=output_prefix, **pkwargs)

    print("Finished phenotyping step.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
