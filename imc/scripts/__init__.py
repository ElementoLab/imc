from typing import List, Dict
import argparse

from imc.types import Path


def build_cli(cmd) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(**cli_config["subcommands"][cmd])  # type: ignore[index]
    parser = build_params(parser, cli_config["subcommand_arguments"][cmd])  # type: ignore[index]
    return parser


def build_params(
    parser: argparse.ArgumentParser, config: List[Dict[str, Dict[str, str]]]
) -> argparse.ArgumentParser:
    for opt in config:
        args = opt.get("args") or {}
        kwargs = opt.get("kwargs") or {}
        parser.add_argument(*args, **kwargs)  # type: ignore[arg-type]
    return parser


epilog = "https://github.com/ElementoLab/imc"
cli_config = {
    "main": {
        "prog": "imc",
        "description": "A package for the analysis of Imaging Mass Cytometry data.",
        "epilog": epilog,
    },
    "subcommands": {
        "inspect": {
            "prog": "imc inspect",
            "description": "Inspect MCD files and extract metadata.",
            "epilog": epilog,
        },
        "prepare": {
            "prog": "imc prepare",
            "description": "Prepare project directory from MCD files.",
            "epilog": epilog,
        },
        "segment": {
            "prog": "imc segment",
            "description": "Segment image stacks.",
            "epilog": epilog,
        },
    },
    "subcommand_arguments": {
        "inspect": [
            {"kwargs": {"dest": "mcd_files", "nargs": "+", "type": Path}},
            {
                "args": ["--no-write"],
                "kwargs": {"dest": "no_write", "action": "store_true"},
            },
            {
                "args": [
                    "-o",
                    "--output-prefix",
                ],
                "kwargs": {
                    "dest": "output_prefix",
                    "default": "mcd_files",
                    "type": Path,
                },
            },
        ],
        "prepare": [
            {
                "kwargs": {
                    "dest": "mcd_files",
                    "nargs": "+",
                    "type": Path,
                    "help": "MCD files to process.",
                }
            },
            {
                "args": ["-p", "--panel-csv"],
                "kwargs": {
                    "dest": "pannel_csvs",
                    "nargs": "+",
                    "type": Path,
                    "help": "Either one file or one for each MCD file.",
                },
            },
            {
                "args": ["-o", "--root-output-dir"],
                "kwargs": {
                    "dest": "root_output_dir",
                    "default": "processed",
                    "type": Path,
                },
            },
            {
                "args": ["-n", "--sample-name"],
                "kwargs": {"dest": "sample_names", "nargs": "+", "type": str},
            },
            {
                "args": ["--partition-panels"],
                "kwargs": {"dest": "partition_panels", "action": "store_true"},
            },
            {
                "args": ["--filter-full"],
                "kwargs": {"dest": "filter_full", "action": "store_true"},
            },
            {
                "args": ["--overwrite"],
                "kwargs": {"dest": "overwrite", "action": "store_true"},
            },
            {
                "args": ["--no-empty-rois"],
                "kwargs": {"dest": "allow_empty_rois", "action": "store_false"},
            },
            {
                "args": ["--ilastik"],
                "kwargs": {"dest": "ilastik_output", "action": "store_true"},
            },
            {
                "args": ["--ilastik-compartment"],
                "kwargs": {
                    "dest": "ilastik_compartment",
                    "choices": ["nuclear", "cytoplasm", "both", None],
                    "default": None,
                    "help": "Whether to prepare a ilastik image stack based on cellular compartments. If given, the values in '--panel-csv' will be ignored.",
                },
            },
            {
                "args": ["--only-crops"],
                "kwargs": {"dest": "only_crops", "action": "store_true"},
            },
            {
                "args": ["--no-stacks"],
                "kwargs": {"dest": "export_stacks", "action": "store_false"},
            },
            {
                "args": ["--no-panoramas"],
                "kwargs": {"dest": "export_panoramas", "action": "store_false"},
            },
            {
                "args": ["--n-crops"],
                "kwargs": {"dest": "n_crops", "type": int},
            },
            {
                "args": ["--crop-width"],
                "kwargs": {"dest": "crop_width", "type": int},
            },
            {
                "args": ["--crop-height"],
                "kwargs": {"dest": "crop_height", "type": int},
            },
            {
                "args": ["-k", "--keep-original-names"],
                "kwargs": {
                    "dest": "keep_original_roi_names",
                    "action": "store_true",
                },
            },
        ],
        "segment": [
            {
                "kwargs": {
                    "dest": "tiffs",
                    "nargs": "+",
                    "type": Path,
                    "help": "TIFF files with array stack.",
                },
            },
            {
                "args": ["-p", "--from-probabilities"],
                "kwargs": {
                    "dest": "from_probabilities",
                    "action": "store_true",
                },
            },
            {
                "args": ["-m", "--model"],
                "kwargs": {
                    "choices": ["stardist", "deepcell", "cellpose"],
                    "default": "stardist",
                },
            },
            {
                "args": ["-c", "--compartment"],
                "kwargs": {
                    "choices": ["nuclear", "cytoplasm", "both"],
                    "default": "nuclear",
                },
            },
            {
                "args": ["-e", "--channel-exclude"],
                "kwargs": {
                    "default": "",
                    "help": "Comma-delimited list of channels to exclude from stack.",
                },
            },
            {"args": ["--output-mask-suffix"], "kwargs": {"default": ""}},
            {
                "args": ["--no-save"],
                "kwargs": {"dest": "save", "action": "store_false"},
            },
            {
                "args": ["--overwrite"],
                "kwargs": {
                    "dest": "overwrite",
                    "action": "store_true",
                    "help": "Whether to overwrite outputs.",
                },
            },
            {
                "args": ["--no-plot"],
                "kwargs": {
                    "dest": "plot",
                    "action": "store_false",
                    "help": "Whether plots demonstrating segmentation should be made.",
                },
            },
            {
                "args": ["--no-post-processing"],
                "kwargs": {
                    "dest": "postprocessing",
                    "action": "store_false",
                    "help": "Whether postprocessing of DeepCell segmentation for compartment 'both' should not be performed.",
                },
            },
        ],
    },
}
