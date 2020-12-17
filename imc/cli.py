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
from imc.utils import mcd_to_dir


def main(cli: List[str] = None) -> int:
    parser = get_args()
    main_args, cmd_args = parser.parse_known_args()

    if main_args.command == "inspect":
        inspect(cmd_args)
    elif main_args.command == "prepare":
        prepare(cmd_args)
    return 0


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")
    inspect_parser = subparsers.add_parser("inspect", add_help=False)
    prepare_parser = subparsers.add_parser("prepare", add_help=False)

    return parser


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
