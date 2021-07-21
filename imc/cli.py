#!/usr/bin/env python

"""
Inspect MCD files, reporting on their basic statistics, saving
metadata as YAML files, and panel information as CSV files.
"""

import sys
import argparse
import typing as tp

from imc.scripts.process import main as process
from imc.scripts.inspect_mcds import main as inspect
from imc.scripts.prepare_mcds import main as prepare
from imc.scripts.predict import main as predict
from imc.scripts.segment_stacks import main as segment
from imc.scripts.quantify import main as quantify
from imc.scripts.phenotype import main as phenotype
from imc.scripts.view import main as view

cli_config: tp.Dict[str, tp.Any]
from imc.scripts import cli_config


def main(cli: tp.Sequence[str] = None) -> int:
    parser = get_args()
    main_args, cmd_args = parser.parse_known_args(cli)

    if main_args.command == "process":
        process(cmd_args)
    elif main_args.command == "inspect":
        inspect(cmd_args)
    elif main_args.command == "prepare":
        prepare(cmd_args)
    elif main_args.command == "predict":
        predict(cmd_args)
    elif main_args.command == "segment":
        segment(cmd_args)
    elif main_args.command == "quantify":
        quantify(cmd_args)
    elif main_args.command == "phenotype":
        phenotype(cmd_args)
    elif main_args.command == "view":
        view(cmd_args)
    else:
        print("Command not known!")
        return 1
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
