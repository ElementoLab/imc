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
from imc.scripts.inspect_mcds import main as inspect
from imc.scripts.prepare_mcds import main as prepare
from imc.scripts.segment_stacks import main as segment
from imc.utils import mcd_to_dir
from imc.scripts import cli_config


def main(cli: List[str] = None) -> int:
    parser = get_args()
    main_args, cmd_args = parser.parse_known_args()

    if main_args.command == "inspect":
        inspect(cmd_args)
    elif main_args.command == "prepare":
        prepare(cmd_args)
    elif main_args.command == "segment":
        segment(cmd_args)
    return 0


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(**cli_config["main"])  # type: ignore[index]

    subparsers = parser.add_subparsers(dest="command", required=True)

    for cmd in cli_config["subcommands"]:
        subparsers.add_parser(cmd, add_help=False)
    return parser


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
