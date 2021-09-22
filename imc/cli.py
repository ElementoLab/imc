#!/usr/bin/env python

"""
Inspect MCD files, reporting on their basic statistics, saving
metadata as YAML files, and panel information as CSV files.
"""

import sys
import argparse
from argparse import RawTextHelpFormatter
import typing as tp

from imc._version import version
from imc.scripts.process import main as process
from imc.scripts.inspect_mcds import main as inspect
from imc.scripts.prepare import main as prepare
from imc.scripts.predict import main as predict
from imc.scripts.segment_stacks import main as segment
from imc.scripts.quantify import main as quantify
from imc.scripts.phenotype import main as phenotype
from imc.scripts.illustrate import main as illustrate
from imc.scripts.view import main as view

cli_config: tp.Dict[str, tp.Any]
from imc.scripts import cli_config


def main(cli: tp.Sequence[str] = None) -> int:
    parser = get_args()
    parser.add_argument("-v", "--version", action="version", version=version)
    main_args, cmd_args = parser.parse_known_args(cli)

    if main_args.command not in cli_config["subcommands"]:
        raise ValueError(f"Command '{main_args.command}' not known!")
    return eval(main_args.command)(cmd_args)


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(**cli_config["main"], formatter_class=RawTextHelpFormatter)  # type: ignore[index]

    subparsers = parser.add_subparsers(dest="command", required=True)

    for cmd in cli_config["subcommands"]:
        subparsers.add_parser(cmd, add_help=False)
    return parser


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
