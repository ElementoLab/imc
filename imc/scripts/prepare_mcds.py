#!/usr/bin/env python

"""
Convert MCD files to TIFF and Sample/ROI structure.
"""

import sys
import argparse
import typing as tp

import pandas as pd

from imc.types import Path
from imc.utils import mcd_to_dir, plot_panoramas_rois
from imc.scripts import build_cli


def main(cli: tp.Sequence[str] = None) -> int:
    parser = build_cli("prepare")
    args = parser.parse_args(cli)

    if not args.pannel_csvs:
        args.pannel_csvs = [None] * len(args.mcd_files)
    elif len(args.pannel_csvs) == 1:
        args.pannel_csvs = args.pannel_csvs * len(args.mcd_files)
    else:
        assert len(args.mcd_files) == len(args.pannel_csvs)

    if (args.sample_names is None) or (len(args.mcd_files) != len(args.sample_names)):
        args.sample_names = [None] * len(args.mcd_files)

    fs = "\n\t- " + "\n\t- ".join([f.as_posix() for f in args.mcd_files])
    print(f"Starting analysis of {len(args.mcd_files)} MCD files: {fs}!")

    for mcd_file, pannel_csv, sample_name in zip(
        args.mcd_files, args.pannel_csvs, args.sample_names
    ):
        sargs = args.__dict__.copy()
        del sargs["mcd_files"]
        del sargs["pannel_csvs"]
        del sargs["root_output_dir"]
        del sargs["sample_names"]
        sargs["mcd_file"] = mcd_file
        sargs["pannel_csv"] = pannel_csv
        sargs["sample_name"] = sample_name
        sargs["output_dir"] = args.root_output_dir / mcd_file.stem
        sargs = {k: v for k, v in sargs.items() if v is not None}

        print(f"Started analyzing '{mcd_file}'.")
        mcd_to_dir(**sargs)

        # Plot ROI positions on panoramas and slide
        plot_panoramas_rois(
            yaml_spec=mcd_file.replace_(".mcd", ".session_metadata.yaml"),
            output_prefix=args.root_output_dir / mcd_file.stem / mcd_file.stem + ".",
            panorama_image_prefix=args.root_output_dir / mcd_file.stem / "Panorama_",
            save_roi_arrays=False,
        )

        print(f"Finished processing '{mcd_file}'.")

    print("Finished with all files!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
