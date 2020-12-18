#!/usr/bin/env python

"""
Inspect MCD files, reporting on their basic statistics, saving
metadata as YAML files, and panel information as CSV files.
"""

import sys
import argparse
from typing import List

import pandas as pd

from imc.types import Path
from imc.utils import mcd_to_dir
from imc.scripts import cli_config


def main(cli: List[str] = None) -> int:
    parser = get_args()
    args = parser.parse_args(cli)

    if len(args.pannel_csvs) == 1:
        args.pannel_csvs = args.pannel_csvs * len(args.mcd_files)
    else:
        assert len(args.mcd_files) == len(args.pannel_csvs)

    if len(args.mcd_files) != len(args.sample_names):
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
        print(f"Finished processing '{mcd_file}'.")

    print("Finished with all files!")
    return 0


def get_args() -> argparse.ArgumentParser:
    _help = "MCD files to process."
    parser = argparse.ArgumentParser(**cli_config["subcommands"]["prepare"])  # type: ignore[index]
    parser.add_argument(dest="mcd_files", nargs="+", type=Path, help=_help)
    _help = "Either one file or one for each MCD file."
    parser.add_argument(
        "-p",
        "--panel-csv",
        dest="pannel_csvs",
        nargs="+",
        type=Path,
        help=_help,
    )
    parser.add_argument(
        "-o",
        "--root-output-dir",
        dest="root_output_dir",
        default="processed",
        type=Path,
    )
    parser.add_argument(
        "-n", "--sample-name", dest="sample_names", nargs="+", type=str
    )
    parser.add_argument(
        "--partition-panels", dest="partition_panels", action="store_true"
    )
    parser.add_argument(
        "--filter-full", dest="filter_full", action="store_true"
    )
    parser.add_argument("--ilastik", dest="ilastik_output", action="store_true")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true")
    parser.add_argument(
        "--no-empty-rois", dest="allow_empty_rois", action="store_false"
    )
    parser.add_argument("--only-crops", dest="only_crops", action="store_true")
    parser.add_argument("--n-crops", dest="n_crops", type=int)
    parser.add_argument("--crop-width", dest="crop_width", type=int)
    parser.add_argument("--crop-height", dest="crop_height", type=int)
    parser.add_argument(
        "-k",
        "--keep-original-names",
        dest="keep_original_roi_names",
        action="store_true",
    )
    return parser


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
