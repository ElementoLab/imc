#!/usr/bin/env python

"""
Phenotype cells.
"""

import sys
import typing as tp

from imc.operations import phenotyping, plot_phenotyping
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

    if args.plot:
        print(f"Plotting phenotypes in directory '{args.output_dir}'.")
        output_prefix = args.output_dir / "phenotypes."
        if args.compute:
            args.a = a
        pkwargs = filter_kwargs_by_callable(args.__dict__, plot_phenotyping)
        plot_phenotyping(output_prefix=output_prefix, **pkwargs)

    print("Finished phenotyping step.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
