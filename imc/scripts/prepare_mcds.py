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


def main(cli: List[str] = None) -> int:
    parser = get_args()
    args = parser.parse_args(cli)

    fs = "\n\t- " + "\n\t- ".join([f.as_posix() for f in args.mcd_files])
    print(f"Starting analysis of {len(args.mcd_files)} MCD files: {fs}!")

    for mcd_file in args.mcd_files:
        print(f"Started analyzing '{mcd_file}'.")
        mcd_to_dir(**{k: v for k, v in args.__dict__.items() if v is not None})
        print(f"Finished processing '{mcd_file}'.")

    print("Finished with all files!")
    return 0


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="mcd_files", nargs="+", type=Path)
    parser.add_argument(dest="pannel_csv", type=Path)
    parser.add_argument("-o", "--output-dir", dest="output_dir", type=Path)
    parser.add_argument("-n", "--sample-name", dest="sample_name", type=str)
    parser.add_argument(
        "-p", "--partition-panels", dest="partition_panels", action="store_true"
    )
    parser.add_argument(
        "--filter-full", dest="filter_full", action="store_true"
    )
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
