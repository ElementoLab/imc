#!/usr/bin/env python

"""
Quantify images in stacks.
"""

import sys
import argparse
import typing as tp
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

from imc import ROI
from imc.types import Path, Series, Array
from imc.ops.quant import quantify_cells_rois
from imc.scripts import build_cli, find_tiffs


def main(cli: tp.Sequence[str] = None) -> int:
    parser = build_cli("quantify")
    args = parser.parse_args(cli)
    if not args.tiffs:
        args.tiffs = find_tiffs()
        if not args.tiffs:
            print("Input files were not provided and cannot be found!")
            return 1

    fs = "\n\t- " + "\n\t- ".join([f.as_posix() for f in args.tiffs])
    print(f"Starting quantification step for {len(args.tiffs)} TIFF files:{fs}!")

    # Prepare ROI objects
    rois = list()
    for tiff in args.tiffs:
        roi = ROI.from_stack(tiff)
        roi.set_channel_exclude(args.channel_exclude.split(","))
        rois.append(roi)

    quant = quantify_cells_rois(
        rois, args.layers.split(","), morphology=args.morphology
    ).reset_index()

    # reorder columns for nice effect
    ext = ["roi", "obj_id"] + (["X_centroid", "Y_centroid"] if args.morphology else [])
    rem = [x for x in quant.columns if x not in ext]
    quant = quant[ext + rem]

    if args.output is None:
        f = Path("processed").mkdir() / "quantification.csv"
        quant.to_csv(f, index=False)
        print(f"Wrote CSV file to '{f.absolute()}'.")
    else:
        quant.to_csv(args.output, index=False)
        print(f"Wrote CSV file to '{args.output.absolute()}'.")

    if args.output_h5ad:
        from anndata import AnnData
        import scanpy as sc

        v = len(str(quant["obj_id"].max()))
        idx = quant["roi"] + "-" + quant["obj_id"].astype(str).str.zfill(v)
        quant.index = idx

        cols = ["sample", "roi", "obj_id", "X_centroid", "Y_centroid", "layer"]
        cols = [c for c in cols if c in quant.columns]
        ann = AnnData(quant.drop(cols, axis=1, errors="ignore"), obs=quant[cols])
        f = Path("processed").mkdir() / "quantification.h5ad"
        sc.write(f, ann)
        print(f"Wrote h5ad file to '{f.absolute()}'.")

    print("Finished quantification step.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
