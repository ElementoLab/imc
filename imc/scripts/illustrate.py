#!/usr/bin/env python

"""
Illustrate IMC data.
"""

import sys
import typing as tp

from tqdm import tqdm
import matplotlib.pyplot as plt
import scanpy as sc

from imc import Project, ROI
from imc.ops.clustering import phenotyping, plot_phenotyping
from imc.scripts import build_cli, find_tiffs, find_h5ad
from imc.utils import filter_kwargs_by_callable

figkws = dict(dpi=300, bbox_inches="tight")


def main(cli: tp.Sequence[str] = None) -> int:
    parser = build_cli("illustrate")
    args = parser.parse_args(cli)

    if args.tiffs is None:
        args.tiffs = find_tiffs()
    if len(args.tiffs) == 0:
        raise ValueError("Input files were not provided and could not be found!")

    if args.h5ad is None:
        args.h5ad = find_h5ad()
    if args.h5ad is None:
        if args.clusters:
            print(
                "No h5ad file was provided and it could not be found. "
                "Not illustrating clusters."
            )
            args.clusters = False
        if args.cell_types:
            print(
                "No h5ad file was provided and it could not be found. "
                "Not illustrating cell types."
            )
            args.cell_types = False

    print("Starting illustration step!")

    args.channels_include = (
        args.channels_include.split(",") if args.channels_include is not None else None
    )
    args.channels_exclude = args.channels_exclude.split(",")
    args.output_dir.mkdir()

    prj = Project.from_stacks(args.tiffs)
    if args.stacks:
        dir_ = (args.output_dir / "stacks").mkdir()
        print(f"Plotting full image stacks in directory '{dir_}'.")
        for roi in tqdm(prj.rois):
            f = dir_ / roi.name + ".full_stack.pdf"
            if f.exists() and not args.overwrite:
                continue
            fig = roi.plot_channels()
            fig.savefig(f, **figkws)
            plt.close(fig)

    if args.channels:
        dir_ = (args.output_dir / "channels").mkdir()
        print(f"Plotting channels for all images jointly in directory '{dir_}'.")
        for ch in tqdm(prj.rois[0].channel_labels):
            f = dir_ / ch + ".rois.pdf"
            if f.exists() and not args.overwrite:
                continue
            fig = prj.plot_channels([ch])
            fig.savefig(f, **figkws)
            plt.close(fig)

    id_cols = ["sample", "roi", "obj_id"]
    if args.clusters:
        dir_ = (args.output_dir / "clusters").mkdir()
        print(f"Plotting cluster illustrations in directory '{dir_}'.")

        a = sc.read(args.h5ad)
        clusters = a.obs.columns[a.obs.columns.str.contains("cluster_")]
        for cluster in tqdm(clusters):
            f = dir_ / f"clustering_illustrations.{cluster}.pdf"
            if f.exists() and not args.overwrite:
                continue
            # TODO: plot markers next to clusters, or overlay
            prj.set_clusters(a.obs.set_index(id_cols)[cluster].rename("cluster"))
            fig = prj.plot_cell_types()
            for ax in fig.axes[1:]:
                ax.legend_.set_visible(False)
            fig.savefig(f, **figkws)
            plt.close(fig)

    if args.cell_types:
        dir_ = (args.output_dir / "cell_type").mkdir()
        print(f"Plotting cell_type illustrations in directory '{dir_}'.")

        a = sc.read(args.h5ad)
        cts = a.obs.columns[a.obs.columns.str.contains("cluster_")].intersection(
            a.obs.columns[a.obs.columns.str.contains("_label")]
        )
        for ct in tqdm(cts):
            f = dir_ / f"cell_type_illustrations.{ct}.pdf"
            if f.exists() and not args.overwrite:
                continue
            # TODO: plot markers next to cell types, or overlay
            prj.set_clusters(a.obs.set_index(id_cols)[ct].rename("cluster"))
            fig = prj.plot_cell_types()
            for ax in fig.axes[1:]:
                ax.legend_.set_visible(False)
            fig.savefig(f, **figkws)
            plt.close(fig)

    print("Finished illustration step.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
