import os
import sys
import logging
import typing as tp
import argparse
import getpass
import json
from argparse import RawTextHelpFormatter

from imc.types import Path

# Suppress tensorflow inital output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
logging.getLogger("tensorflow").setLevel(logging.FATAL)

# Suppress tensorflow GPU use
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# TODO: move to global package __init__/config
if sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
    try:
        DEFAULT_IMC_DIR = Path("~/.imc").expanduser().mkdir()
    except Exception:
        DEFAULT_IMC_DIR = Path("./.imc").absolute().mkdir()
        print(
            f"Could not create directory '~/.imc', will use '{DEFAULT_IMC_DIR.as_posix()}'. "
            "Using a resource directory in the user's home directory improves "
            "management of resources. Make sure your home directory is writable."
        )
elif sys.platform.startswith("darwin"):
    # home = Path(f"/User/{getpass.getuser()}")
    try:
        DEFAULT_IMC_DIR = Path("/Applications/.imc").mkdir()
    except Exception:
        DEFAULT_IMC_DIR = Path("./.imc").absolute().mkdir()
        print(
            f"Could not create directory '/User/{getpass.getuser()}/.imc', "
            "will use '{DEFAULT_IMC_DIR.as_posix()}'. "
            "Using a resource directory in the user's home directory improves "
            "management of resources. Make sure your home directory is writable."
        )
elif sys.platform.startswith("win") or sys.platform.startswith("cygwin"):
    raise NotImplementedError("Windows support is not yet available!")
else:
    print(
        "Warning, OS could not be easily identified. Using default dir ~/.imc to store "
        "resources but that might not work!"
    )
    DEFAULT_IMC_DIR = Path("~/.imc").expanduser().mkdir()

DEFAULT_LIB_DIR = (DEFAULT_IMC_DIR / "lib").mkdir()
DEFAULT_MODELS_DIR = (DEFAULT_IMC_DIR / "models").mkdir()


def build_cli(cmd: tp.Sequence[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(**cli_config["subcommands"][cmd], formatter_class=RawTextHelpFormatter)  # type: ignore[index]
    parser = build_params(parser, cli_config["subcommand_arguments"][cmd])  # type: ignore[index]
    return parser


def build_params(
    parser: argparse.ArgumentParser, config: tp.List[tp.Dict[str, tp.Dict[str, str]]]
) -> argparse.ArgumentParser:
    for opt in config:
        args = opt.get("args") or {}
        kwargs = opt.get("kwargs") or {}
        parser.add_argument(*args, **kwargs)  # type: ignore[arg-type]
    return parser


def find_mcds() -> tp.List[Path]:
    return sorted(list(Path().glob("**/*.mcd")))


def find_tiffs() -> tp.List[Path]:
    return sorted(list(Path().glob("**/*_full.tiff")))


def find_h5ad() -> tp.Optional[Path]:
    import numpy as np

    h5ads = [p.as_posix() for p in Path().glob("**/*.h5ad")]
    if not h5ads:
        return None
    return Path(h5ads[np.argmax(list(map(len, h5ads)))])


json_example = {
    "segment": ["--model", "stardist"],
    "quantify": ["--no-morphology", "--layers", "nuclei"],
    "phenotype": [
        "--dim-res-algos",
        "umap,diffmap",
        "--clustering-resolutions",
        "0.5,1.0",
    ],
}
process_step_order = ["inspect", "prepare", "predict", "segment", "quantify", "phenotype"]

epilog = "https://github.com/ElementoLab/imc"
cli_config = {
    "main": {
        "prog": "imc",
        "description": "A package for the analysis of Imaging Mass Cytometry data.\n"
        "Select one of the available commands and use '--help' after each to see all available options.\n\n",
        "epilog": epilog,
    },
    "subcommands": {
        "process": {
            "prog": "imc process",
            "description": "Process raw IMC files end-to-end (raw signal to cell phenotypes).\n"
            "Supports MCD, TIFF or TXT as input.",
            "epilog": epilog,
        },
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
        "predict": {
            "prog": "imc predict",
            "description": "Output cellular probabilities from image stacks.",
            "epilog": epilog,
        },
        "segment": {
            "prog": "imc segment",
            "description": "Segment image stacks.",
            "epilog": epilog,
        },
        "quantify": {
            "prog": "imc quantify",
            "description": "Quantify channel intensity in segmented cells.",
            "epilog": epilog,
        },
        "phenotype": {
            "prog": "imc phenotype",
            "description": "Phenotype cells in quantification matrix.",
            "epilog": epilog,
        },
        "illustrate": {
            "prog": "imc illustrate",
            "description": "Illustrate IMC data.",
            "epilog": epilog,
        },
        "view": {
            "prog": "imc view",
            "description": "Visualize an image stack interactively using a matplotlib frontend.",
            "epilog": epilog,
        },
    },
    "subcommand_arguments": {
        "process": [
            {
                "kwargs": {
                    "dest": "files",
                    "nargs": "*",
                    "type": str,
                    "help": "Input files to process. Can be MCD, TIFF, or TXT.",
                }
            },
            {
                "args": ["-s", "--steps"],
                "kwargs": {
                    "dest": "steps",
                    "default": None,
                    "help": "Comma-delimited list of processing steps to perform.\n"
                    "By default this will be:\n\t- "
                    + "\n\t- ".join(process_step_order)
                    + "\nunless overwridden by this option.",
                },
            },
            {
                "args": ["--start-step"],
                "kwargs": {
                    "dest": "start_step",
                    "default": None,
                    "help": "Step at which to start processing. "
                    "Default is to run all steps.",
                },
            },
            {
                "args": ["--stop-step"],
                "kwargs": {
                    "dest": "stop_step",
                    "default": None,
                    "help": "Step at which to stop processing. "
                    "Default is to run all steps.",
                },
            },
            {
                "args": ["-c", "--config"],
                "kwargs": {
                    "dest": "config",
                    "default": None,
                    "type": Path,
                    "help": "Path to JSON file with configuration for the various steps.\n"
                    "Each step will be run with the options provided.\n"
                    "Keys are steps and values are lists of command-line options.\n"
                    "\nFor example:\n$> cat config.json\n"
                    + json.dumps(json_example, indent=4)
                    + "\n$> imc process -c config.json file1.mcd file2.mcd",
                },
            },
            {
                "args": ["--verbose"],
                "kwargs": {
                    "dest": "quiet",
                    "action": "store_false",
                    "help": "Whether to output more detail of the process.",
                },
            },
        ],
        "inspect": [
            {
                "kwargs": {
                    "dest": "mcd_files",
                    "nargs": "*",
                    "default": None,
                    "type": Path,
                    "help": "Input files in MCD format to be inspected. "
                    "Several files can be provided in a space-separated manner.",
                }
            },
            {
                "args": ["--no-write"],
                "kwargs": {
                    "dest": "no_write",
                    "action": "store_true",
                    "help": "Whether to not write output. " "Default is to do so.",
                },
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
                    "help": "Prefix to use for output files. " "Default is 'mcd_files'.",
                },
            },
        ],
        "prepare": [
            {
                "kwargs": {
                    "dest": "input_files",
                    "nargs": "+",
                    "type": Path,
                    "help": "Files to process. Either MCD of TIFF. "
                    "Several files can be given space-separated.",
                },
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
                    "help": "Path of directory to store output files. "
                    "Default is the directory 'processed' in the current directory.",
                },
            },
            {
                "args": ["--output-format"],
                "kwargs": {
                    "dest": "output_format",
                    "choices": ["tiff", "ome-tiff"],
                    "default": "ome-tiff",
                    "help": "The data format of the output TIFF files. "
                    "One of 'tiff' and 'ome-tiff'.",
                },
            },
            {
                "args": ["--compression-level"],
                "kwargs": {
                    "dest": "compression_level",
                    "type": int,
                    "default": 3,
                    "help": "The level of compression of output TIFF files (0-9). "
                    "Higher compression creates smaller files, but takes longer to read and write.",
                },
            },
            {
                "args": ["-n", "--sample-name"],
                "kwargs": {
                    "dest": "sample_names",
                    "nargs": "+",
                    "type": str,
                    "help": "Name of samples. One value per input file provided (space separated). "
                    "Not required, but can be used to overwrite name derived from file name.",
                },
            },
            {
                "args": ["--partition-panels"],
                "kwargs": {
                    "dest": "partition_panels",
                    "action": "store_true",
                    "help": "Whether to separate ROIs with different panels. "
                    "Default is not to.",
                },
            },
            {
                "args": ["--filter-full"],
                "kwargs": {"dest": "filter_full", "action": "store_true"},
            },
            {
                "args": ["--overwrite"],
                "kwargs": {
                    "dest": "overwrite",
                    "action": "store_true",
                    "help": "Whether to overwrite existing files. " "Default is not to.",
                },
            },
            {
                "args": ["--no-empty-rois"],
                "kwargs": {
                    "dest": "allow_empty_rois",
                    "action": "store_false",
                    "help": "Whether to remove empty ROIs (with no ablation data). "
                    "Default is to skip them.",
                },
            },
            {
                "args": ["--ilastik"],
                "kwargs": {
                    "dest": "ilastik_output",
                    "action": "store_true",
                    "help": "Whether to prepare input files for ilastik. "
                    "Defalt is not to.",
                },
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
                "kwargs": {
                    "dest": "only_crops",
                    "action": "store_true",
                    "help": "Whether to only export crop tiles for ilastik training. "
                    "Default it to do export crop tiles and TIFF stacks.",
                },
            },
            {
                "args": ["--no-stacks"],
                "kwargs": {
                    "dest": "export_stacks",
                    "action": "store_false",
                    "help": "Whether to not export full image stacks as TIFFs. "
                    "Default it to do so.",
                },
            },
            {
                "args": ["--no-panoramas"],
                "kwargs": {
                    "dest": "export_panoramas",
                    "action": "store_false",
                    "help": "Whether to not export panorama images for the whole slide. "
                    "Default it to do so.",
                },
            },
            {
                "args": ["--n-crops"],
                "kwargs": {
                    "dest": "n_crops",
                    "type": int,
                    "help": "Number of crop tiles made for ilastik training.",
                },
            },
            {
                "args": ["--crop-width"],
                "kwargs": {
                    "dest": "crop_width",
                    "type": int,
                    "help": "Width of crop tiles used for ilastik training.",
                },
            },
            {
                "args": ["--crop-height"],
                "kwargs": {
                    "dest": "crop_height",
                    "type": int,
                    "help": "Height of crop tiles used for ilastik training.",
                },
            },
            {
                "args": ["-k", "--keep-original-names"],
                "kwargs": {
                    "dest": "keep_original_roi_names",
                    "action": "store_true",
                    "help": "Whether to keep original names for ROIs from MCD file. "
                    "Default is to number them according to acquisition order.",
                },
            },
            {
                "args": ["--verbose"],
                "kwargs": {
                    "dest": "quiet",
                    "action": "store_false",
                    "help": "Whether to output additional information to stderr. "
                    "Default is not to.",
                },
            },
        ],
        "predict": [
            {
                "kwargs": {
                    "dest": "tiffs",
                    "nargs": "*",
                    "default": None,
                    "type": Path,
                    "help": "TIFF files with array stack.",
                }
            },
            {
                "args": ["-l", "--lib-dir"],
                "kwargs": {
                    "dest": "lib_dir",
                    "default": DEFAULT_LIB_DIR,
                    "type": Path,
                    "help": "Directory to store external software (e.g. ilastik).",
                },
            },
            {
                "args": ["-m", "--models-dir"],
                "kwargs": {
                    "dest": "models_dir",
                    "default": DEFAULT_MODELS_DIR,
                    "type": Path,
                    "help": "Directory to store static models.",
                },
            },
            {
                "args": ["--ilastik-version"],
                "kwargs": {
                    "dest": "ilastik_version",
                    "choices": ["1.3.3post2"],
                    "default": "1.3.3post2",
                    "help": "Version of ilastik software to use. "
                    "Currently only '1.3.3post2' is valid.",
                },
            },
            {
                "args": ["-v", "--model-version"],
                "kwargs": {
                    "dest": "ilastik_model_version",
                    "choices": ["20210302"],
                    "default": "20210302",
                    "help": "Version of ilastik model to use in compartment prediction."
                    "Currently only '20210302' is available.",
                },
            },
            {
                "args": ["--verbose"],
                "kwargs": {
                    "dest": "quiet",
                    "action": "store_false",
                    "help": "Whether to output additional information to stderr. "
                    "Default is not to.",
                },
            },
            {
                "args": ["--custom-model"],
                "kwargs": {
                    "dest": "custom_model",
                    "type": Path,
                    "help": "Path to an existing ilastik model to use.",
                },
            },
            {
                "args": ["--overwrite"],
                "kwargs": {"dest": "overwrite", "action": "store_true"},
                "help": "Whether to overwrite existing output files. "
                "Default is not to.",
            },
            {
                "args": ["--no-cleanup"],
                "kwargs": {
                    "dest": "cleanup",
                    "action": "store_false",
                    "help": "Whether to not cleanup ilastik input files. "
                    "Default is to clean them.",
                },
            },
        ],
        "segment": [
            {
                "kwargs": {
                    "dest": "tiffs",
                    "nargs": "*",
                    "default": None,
                    "type": Path,
                    "help": "TIFF files with array stack.",
                },
            },
            {
                "args": ["-p", "--from-probabilities"],
                "kwargs": {
                    "dest": "from_probabilities",
                    "action": "store_true",
                    "help": "Whether to use a probability file as the input for segmentation. "
                    "That's the output of `imc predict`.",
                },
            },
            {
                "args": ["-m", "--model"],
                "kwargs": {
                    "choices": ["stardist", "deepcell", "cellpose"],
                    "default": "stardist",
                    "help": "Which model to use for segmentation. "
                    "Default is 'stardist'.",
                },
            },
            {
                "args": ["-c", "--compartment"],
                "kwargs": {
                    "choices": ["nuclear", "cytoplasm", "both"],
                    "default": "nuclear",
                    "help": "Which cellular compartment to segment. "
                    "Default is 'nuclear'.",
                },
            },
            {
                "args": ["-e", "--channel-exclude"],
                "kwargs": {
                    "default": "",
                    "help": "Comma-delimited list of channels to exclude from stack.",
                },
            },
            {
                "args": ["--output-mask-suffix"],
                "kwargs": {
                    "default": "",
                    "help": "An optional additional suffix for the output mask.",
                },
            },
            {
                "args": ["--no-save"],
                "kwargs": {
                    "dest": "save",
                    "action": "store_false",
                    "help": "Whether to not save segmentation.",
                },
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
                    "help": "Whether postprocessing of DeepCell segmentation when using compartment 'both' should not be performed.",
                },
            },
            {
                "args": ["--verbose"],
                "kwargs": {
                    "dest": "quiet",
                    "action": "store_false",
                    "help": "Whether to output more detail of the process.",
                },
            },
        ],
        "quantify": [
            {
                "kwargs": {
                    "dest": "tiffs",
                    "nargs": "*",
                    "default": None,
                    "type": Path,
                    "help": "TIFF files with array stack."
                    "If not given will infer from files in current directory.",
                }
            },
            {
                "args": ["--no-morphology"],
                "kwargs": {
                    "dest": "morphology",
                    "action": "store_false",
                    "help": "Whether to quantify object morphology or not. "
                    "Default is to quantify.",
                },
            },
            {
                "args": ["--layers"],
                "kwargs": {
                    "dest": "layers",
                    "default": "cell",
                    "help": "Comma-delimited list of layers of segmentation to quantify. "
                    "Default is 'cell'.",
                },
            },
            {
                "args": ["-e", "--channel-exclude"],
                "kwargs": {
                    "default": "",
                    "help": "Comma-delimited list of channels to exclude from stack.",
                },
            },
            {
                "args": ["--output"],
                "kwargs": {
                    "dest": "output",
                    "help": "Output file with quantification. "
                    "Default is 'processed/quantification.csv'.",
                },
            },
            {
                "args": ["--not-h5ad"],
                "kwargs": {
                    "dest": "output_h5ad",
                    "action": "store_false",
                    "help": "Don't output quantification in h5ad format. "
                    "Default is to write h5ad.",
                },
            },
            {
                "args": ["--overwrite"],
                "kwargs": {"dest": "overwrite", "action": "store_true"},
            },
        ],
        "phenotype": [
            {
                "kwargs": {
                    "dest": "a",
                    "type": Path,
                    "help": "Path to H5ad file with quantification.",
                }
            },
            {
                "args": ["--output-dir"],
                "kwargs": {
                    "default": "results/phenotyping",
                    "type": Path,
                    "dest": "output_dir",
                    "help": "Output directory. " "Default is 'results/phenotyping'.",
                },
            },
            {
                "args": ["-i", "--channel-include"],
                "kwargs": {
                    "dest": "channels_include",
                    "default": None,
                    "help": "Comma-delimited list of channels to include from stack.",
                },
            },
            {
                "args": ["-e", "--channel-exclude"],
                "kwargs": {
                    "dest": "channels_exclude",
                    "default": "80ArAr(ArAr80),89Y(Y89),120Sn(Sn120),127I(I127),131Xe(Xe131),138Ba(Ba138),140Ce(Ce140),190BCKG(BCKG190),202Hg(Hg202),208Pb(Pb208),209Bi(Bi209)",
                    "help": "Comma-delimited list of channels to exclude from stack.",
                },
            },
            {
                "args": ["--no-filter-cells"],
                "kwargs": {
                    "dest": "filter_cells",
                    "action": "store_false",
                    "help": "Whether to filter dubious cells out." " Default is True.",
                },
            },
            {
                "args": ["--no-z-score"],
                "kwargs": {
                    "dest": "z_score",
                    "action": "store_false",
                    "help": "Whether to Z-score transform the intensity values. "
                    "Default is True.",
                },
            },
            {
                "args": ["--z-score-by"],
                "kwargs": {
                    "dest": "z_score_per",
                    "default": "roi",
                    "choices": ["roi", "sample"],
                    "help": "Whether to z-score values per ROI or per Sample. "
                    "Default is 'roi'.",
                },
            },
            {
                "args": ["--z-score-cap"],
                "kwargs": {
                    "dest": "z_score_cap",
                    "default": 3.0,
                    "help": "Absolute value to cap z-scores at." " Default is '3.0'.",
                },
            },
            {
                "args": ["--no-remove-batch"],
                "kwargs": {
                    "dest": "remove_batch",
                    "action": "store_false",
                    "help": "Whether to remove batch effects." " Default is True.",
                },
            },
            {
                "args": ["--batch-variable"],
                "kwargs": {
                    "dest": "batch_variable",
                    "default": "sample",
                    "help": "Which variable to use for batch effect removal. "
                    "Default is 'sample'.",
                },
            },
            {
                "args": ["--dim-res-algos"],
                "kwargs": {
                    "dest": "dim_res_algos",
                    "default": "umap",
                    "help": "Comma-delimited string with algorithms to use for dimensionality reduction. "
                    "Choose a combination from 'umap', 'diffmap', 'pymde'. "
                    "Default is only 'umap'.",
                },
            },
            {
                "args": ["--clustering-method"],
                "kwargs": {
                    "dest": "clustering_method",
                    "default": "leiden",
                    "choices": ["leiden", "parc"],
                    "help": "Method to use for cell clustering. "
                    "Default is 'leiden'. 'parc' requires that package to be installed.",
                },
            },
            {
                "args": ["--clustering-resolutions"],
                "kwargs": {
                    "dest": "clustering_resolutions",
                    "default": "0.5,1.0,1.5,2.5",
                    "help": "Comma-delimited list of floats with various resolutions to do clustering at. "
                    "Default is '0.5,1.0,1.5,2.5'.",
                },
            },
            {
                "args": ["--no-compute"],
                "kwargs": {
                    "dest": "compute",
                    "action": "store_false",
                    "help": "Whether to compute phenotypes." " Default is True.",
                },
            },
            {
                "args": ["--no-plot"],
                "kwargs": {
                    "dest": "plot",
                    "action": "store_false",
                    "help": "Whether to plot phenotypes." " Default is True.",
                },
            },
        ],
        "illustrate": [
            {
                "kwargs": {
                    "dest": "tiffs",
                    "nargs": "*",
                    "default": None,
                    "type": Path,
                    "help": "TIFF files with array stack. "
                    "If not given will infer from files in current directory.",
                },
            },
            {
                "args": ["--h5ad"],
                "kwargs": {
                    "dest": "h5ad",
                    "default": None,
                    "type": Path,
                    "help": "Path to H5ad file with quantification. Will look for default files.",
                },
            },
            {
                "args": ["--output-dir"],
                "kwargs": {
                    "default": "results/illustration",
                    "type": Path,
                    "dest": "output_dir",
                    "help": "Output directory. " "Default is 'results/illustration'.",
                },
            },
            {
                "args": ["--overwrite"],
                "kwargs": {
                    "dest": "overwrite",
                    "action": "store_true",
                    "help": "Whether to overwrite existing files or not."
                    " Default is False.",
                },
            },
            {
                "args": ["-i", "--channel-include"],
                "kwargs": {
                    "dest": "channels_include",
                    "default": None,
                    "help": "Comma-delimited list of channels to include from stack.",
                },
            },
            {
                "args": ["-e", "--channel-exclude"],
                "kwargs": {
                    "dest": "channels_exclude",
                    "default": "80ArAr(ArAr80),89Y(Y89),120Sn(Sn120),127I(I127),131Xe(Xe131),138Ba(Ba138),140Ce(Ce140),190BCKG(BCKG190),202Hg(Hg202),208Pb(Pb208),209Bi(Bi209)",
                    "help": "Comma-delimited list of channels to exclude from stack.",
                },
            },
            {
                "args": ["--no-stacks"],
                "kwargs": {
                    "dest": "stacks",
                    "action": "store_false",
                    "help": "Whether to plot full image stacks for each ROI."
                    " Default is True.",
                },
            },
            {
                "args": ["--no-channels"],
                "kwargs": {
                    "dest": "channels",
                    "action": "store_false",
                    "help": "Whether to plot each channel jointly for all ROIs."
                    " Default is True.",
                },
            },
            {
                "args": ["--no-clusters"],
                "kwargs": {
                    "dest": "clusters",
                    "action": "store_false",
                    "help": "Whether to plot cluster identities of cells on top of channels."
                    " Default is True.",
                },
            },
            {
                "args": ["--no-cell-type"],
                "kwargs": {
                    "dest": "cell_types",
                    "action": "store_false",
                    "help": "Whether to plot cell type identities of cells on top of channels."
                    " Default is True.",
                },
            },
        ],
        "view": [
            {
                "kwargs": {
                    "dest": "input_files",
                    "nargs": "*",
                    "default": None,
                    "type": Path,
                    "help": "MCD, or TIFF files with array stack. MCD requires --napari option.",
                }
            },
            {
                "args": ["-u", "--up-key"],
                "kwargs": {
                    "default": "w",
                    "help": "Key to get previous channel." " Default is 'w'.",
                },
            },
            {
                "args": ["-d", "--down-key"],
                "kwargs": {
                    "default": "s",
                    "help": "Key to get next channel." " Default is 's'.",
                },
            },
            {
                "args": ["-l", "--log-key"],
                "kwargs": {
                    "default": "l",
                    "help": "Key to toggle log transformation." " Default is 'l'.",
                },
            },
            {
                "args": ["--napari"],
                "kwargs": {
                    "dest": "napari",
                    "action": "store_true",
                    "help": "Use napari and napari-imc to view MCD files. "
                    "Default is not to use napari. "
                    "This requires napari package to be installed.",
                },
            },
            {
                "args": ["--kwargs"],
                "kwargs": {
                    "dest": "kwargs",
                    "default": None,
                    "help": "Additional parameters for plot customization passed in the form 'key1=value1,key2=value2'. "
                    "For example '--kwargs \"cmap=RdBu_r,vmin=0,vmax=3\"'.",
                },
            },
        ],
    },
}
