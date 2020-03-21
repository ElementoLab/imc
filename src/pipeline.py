#!/usr/bin/env python

import os
from os.path import join as pjoin
import sys
from argparse import ArgumentParser
import subprocess
import logging
import re
import urllib.request
import time

from imctools.scripts import ometiff2analysis
from imctools.scripts import ome2micat
from imctools.scripts import probablity2uncertainty
from imctools.scripts import convertfolder2imcfolder
from imctools.scripts import exportacquisitioncsv


STEPS = [
    "download_test_data",
    "prepare",
    "train",
    "predict",
    "segment",
    "quantify",
    "uncertainty",
]

STEPS_INDEX = dict(enumerate(STEPS))

#  DIRS = ['base', 'input', 'analysis', 'ilastik', 'ome', 'cp', 'histocat' 'uncertainty']


def main():
    global args
    global logger
    logger = setup_logger()

    logger.info("Starting pipeline")
    args = get_cli_arguments()

    for name, path in args.dirs.items():
        if name not in ["input"]:
            os.makedirs(path, exist_ok=True)

    # This is a major security concern
    try:
        args.step = STEPS_INDEX[int(args.step)]
    finally:
        if args.step == "all":
            for step in STEPS:
                logger.info(f"Doing '{args.step}' step.")
                eval(step)()
                logger.info(f"Done with '{args.step}' step.")
        else:
            logger.info(f"Doing '{args.step}' step.")
            eval(args.step)()
            logger.info(f"Done with '{args.step}' step.")


def setup_logger(level=logging.DEBUG):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.Formatter.converter = time.gmtime

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_cli_arguments():
    parser = ArgumentParser()
    out = pjoin(os.curdir, "pipeline")

    # Software

    # if cellprofiler locations do not exist, clone to some default location
    parser.add_argument(
        "--cellprofiler-pipeline-path",
        dest="cellprofiler_pipeline_path",
        default=None,  # "src/ImcSegmentationPipeline/"
    )
    parser.add_argument(
        "--cellprofiler-plugin-path",
        dest="cellprofiler_plugin_path",
        default=None,  # "/home/afr/Documents/workspace/clones/ImcPluginsCP"
    )
    parser.add_argument(
        "--ilastik-path",
        dest="ilastik_sh_path",
        default="src/ilastik-1.3.3post2-Linux/run_ilastik.sh",
    )
    parser.add_argument(
        "--docker-image",
        dest="docker_image",
        default=None,  # "afrendeiro:cellprofiler"
    )

    # Input
    parser.add_argument("--file-regexp", dest="file_regexp", default=".*.zip")
    parser.add_argument("--csv-pannel", dest="csv_pannel", default=None)
    parser.add_argument(
        "--csv-pannel-metal", dest="csv_pannel_metal", default="Metal Tag"
    )
    parser.add_argument(
        "--csv-pannel-ilastik", dest="csv_pannel_ilastik", default="ilastik"
    )
    parser.add_argument(
        "--csv-pannel-full", dest="csv_pannel_full", default="full"
    )
    parser.add_argument(
        "-i", "--input-dirs", nargs="+", dest="input_dirs", default=None
    )

    # Pre-trained model for classification (ilastik)
    parser.add_argument(
        "-m", "--ilastik-model", dest="ilastik_model", default=None
    )
    # /home/afr/projects/data/fluidigm_example_data/fluidigm_example_data.ilp

    # Pipeline steps
    choices = STEPS + [str(x) for x in range(len(STEPS))]
    parser.add_argument(
        "-s", "--step", dest="step", default="all", choices=choices
    )
    parser.add_argument(dest="output_dir", default=out)

    # Parse and complete with derived info
    args = parser.parse_args()
    dirs = dict()
    args.output_dir = os.path.abspath(args.output_dir)
    dirs["base"] = args.output_dir
    dirs["input"] = args.input_dirs or [pjoin(args.output_dir, "input_data")]
    dirs["analysis"] = pjoin(dirs["base"], "tiffs")
    dirs["ilastik"] = pjoin(dirs["base"], "ilastik")
    dirs["ome"] = pjoin(dirs["base"], "ometiff")
    dirs["cp"] = pjoin(dirs["base"], "cpout")
    dirs["histocat"] = pjoin(dirs["base"], "histocat")
    dirs["uncertainty"] = pjoin(dirs["base"], "uncertainty")
    args.dirs = dirs

    args.csv_pannel = args.csv_pannel or pjoin(
        args.input_dirs[0], "example_pannel.csv"
    )

    args.suffix_mask = "_mask.tiff"
    args.suffix_probablities = "_Probabilities"
    args.list_analysis_stacks = [
        (args.csv_pannel_ilastik, "_ilastik", 1),
        (args.csv_pannel_full, "_full", 0),
    ]

    return args


def check_requirements(func):
    def inner():
        if args.docker_image is None:
            get_docker_image_or_pull()
        if args.cellprofiler_plugin_path is None:
            get_zanotelli_code("cellprofiler_plugin_path", "ImcPluginsCP")
        if args.cellprofiler_pipeline_path is None:
            get_zanotelli_code(
                "cellprofiler_pipeline_path", "ImcSegmentationPipeline"
            )
        func()

    return inner


def get_zanotelli_code(arg, repo):
    if repo not in ["ImcSegmentationPipeline", "ImcPluginsCP"]:
        raise ValueError("")
    _dir = pjoin("src", repo)
    if not os.path.exists(_dir):
        url = f"https://github.com/BodenmillerGroup/{repo} {_dir}"
        cmd = f"git clone {url}"
        run_shell_command(cmd)
    else:
        setattr(args, arg, os.path.abspath(pjoin(os.path.curdir, "src", repo)))


def get_docker_image_or_pull():
    def check_image():
        try:
            # check if exists
            out = (
                subprocess.check_output("docker images".split(" "))
                .decode()
                .strip()
            )
            for line in out.split("\n")[1:]:
                if line.split(" ")[0] == default_docker_image:
                    return True
        except FileNotFoundError:
            logger.error("Docker installation not detected.")
            raise
        except IndexError:
            pass
        return False

    default_docker_image = "afrendeiro/cellprofiler"  # "cellprofiler/cellprofiler"
    if not check_image():
        logger.debug("Found docker image.")
        # build
        logger.debug("Did not find cellprofiler docker image. Will build.")
        cmd = f"docker pull {default_docker_image}"
        run_shell_command(cmd)
    args.docker_image = default_docker_image


def get_ilastik(version="1.3.3"):
    os.chdir("src")
    url = "https://files.ilastik.org/"
    file = f"ilastik-{version}post2-Linux.tar.bz2"
    run_shell_command(f"wget {url + file}")
    run_shell_command(f"tar xfz {file}")
    os.chdir("..")


def download_test_data():
    output_dir = args.dirs["input"][0]
    os.makedirs(os.path.abspath(output_dir), exist_ok=True)
    drop_root = "https://www.dropbox.com/s/"
    end = ".zip?dl=1"
    example_pannel_url = (
        "https://raw.githubusercontent.com/BodenmillerGroup/"
        "ImcSegmentationPipeline/development/config/example_pannel.csv"
    )
    urls = [
        ("example_pannel.csv", example_pannel_url),
        (
            "20170905_Fluidigmworkshopfinal_SEAJa.zip",
            drop_root
            + "awyq9p7n7dexgyt/20170905_Fluidigmworkshopfinal_SEAJa"
            + end,
        ),
        (
            "20170906_FluidigmONfinal_SE.zip",
            drop_root + "0pdt1ke4b07v7zd/20170906_FluidigmONfinal_SE" + end,
        ),
    ]

    for fn, url in urls:
        fn = pjoin(output_dir, fn)
        if os.path.exists(fn) is False:
            urllib.request.urlretrieve(url, fn)


def prepare():
    def export_acquisition():
        re_fn = re.compile(args.file_regexp)

        for fol in args.dirs["input"]:
            for fn in os.listdir(fol):
                if re_fn.match(fn):
                    fn_full = pjoin(fol, fn)
                    print(fn_full)
                    convertfolder2imcfolder.convert_folder2imcfolder(
                        fn_full, out_folder=args.dirs["ome"], dozip=False
                    )
        exportacquisitioncsv.export_acquisition_csv(
            args.dirs["ome"], fol_out=args.dirs["cp"]
        )

    def prepare_histocat():
        if not (os.path.exists(args.dirs["histocat"])):
            os.makedirs(args.dirs["histocat"])
        for fol in os.listdir(args.dirs["ome"]):
            ome2micat.omefolder2micatfolder(
                pjoin(args.dirs["ome"], fol),
                args.dirs["histocat"],
                dtype="uint16",
            )

        for fol in os.listdir(args.dirs["ome"]):
            sub_fol = pjoin(args.dirs["ome"], fol)
            for img in os.listdir(sub_fol):
                if not img.endswith(".ome.tiff"):
                    continue
                basename = img.rstrip(".ome.tiff")
                print(img)
                for (col, suffix, addsum) in args.list_analysis_stacks:
                    ometiff2analysis.ometiff_2_analysis(
                        pjoin(sub_fol, img),
                        args.dirs["analysis"],
                        basename + suffix,
                        pannelcsv=args.csv_pannel,
                        metalcolumn=args.csv_pannel_metal,
                        usedcolumn=col,
                        addsum=addsum,
                        bigtiff=False,
                        pixeltype="uint16",
                    )

    @check_requirements
    def prepare_ilastik():
        cmd = f"""docker run \
            --name cellprofiler_prepare_ilastik --rm \
            -v {args.dirs['base']}:/data:rw \
            -v {args.cellprofiler_plugin_path}:/ImcPluginsCP:ro \
            -v {args.cellprofiler_pipeline_path}:/ImcSegmentationPipeline:ro \
            -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
            -e DISPLAY=$DISPLAY \
            {args.docker_image} \
                --run-headless --run \
                --plugins-directory /ImcPluginsCP/plugins/ \
                --pipeline /ImcSegmentationPipeline/cp3_pipelines/1_prepare_ilastik.cppipe \
                -i /{args.dirs['analysis'].replace(args.dirs['base'], 'data')}/ \
                -o /{args.dirs['ilastik'].replace(args.dirs['base'], 'data')}/"""
        run_shell_command(cmd)

    export_acquisition()
    prepare_histocat()
    prepare_ilastik()


def train():
    cmd = f"""{args.ilastik_sh_path}"""
    run_shell_command(cmd)


def predict():
    cmd = f"""{args.ilastik_sh_path} \
        --headless \
        --project={args.ilastik_model} \
        {args.dirs['analysis']}/*_s2.h5"""
    run_shell_command(cmd)


@check_requirements
def segment():
    cmd = f"""docker run \
    --name cellprofiler_segment --rm \
    -v {args.dirs['base']}:/data:rw \
    -v {args.cellprofiler_plugin_path}:/ImcPluginsCP:ro \
    -v {args.cellprofiler_pipeline_path}:/ImcSegmentationPipeline:ro \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -e DISPLAY=$DISPLAY \
    {args.docker_image} \
        --run-headless --run \
        --plugins-directory /ImcPluginsCP/plugins/ \
        --pipeline /ImcSegmentationPipeline/cp3_pipelines/2_segment_ilastik.cppipe \
        -i /{args.dirs['analysis'].replace(args.dirs['base'], 'data')}/ \
        -o /{args.dirs['analysis'].replace(args.dirs['base'], 'data')}/"""
    run_shell_command(cmd)


@check_requirements
def quantify():
    """
    For this step, the number of channels should be updated in the pipeline file (line 126 and 137).
    """
    cmd = f"""docker run \
    --name cellprofiler_quantify --rm \
    -v {args.dirs['base']}:/data:rw \
    -v {args.cellprofiler_plugin_path}:/ImcPluginsCP:ro \
    -v {args.cellprofiler_pipeline_path}:/ImcSegmentationPipeline:ro \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -e DISPLAY=$DISPLAY \
    {args.docker_image} \
        --run-headless --run \
        --plugins-directory /ImcPluginsCP/plugins/ \
        --pipeline /ImcSegmentationPipeline/cp3_pipelines/3_measure_mask_basic.cppipe \
        -i /{args.dirs['analysis'].replace(args.dirs['base'], 'data')}/ \
        -o /{args.dirs['cp'].replace(args.dirs['base'], 'data')}"""
    run_shell_command(cmd)


def uncertainty():
    """
    This would require LZW decompression which is given by the `imagecodecs`
    Python library (has extensive system-level dependencies in Ubuntu)."""
    for fn in os.listdir(args.dirs["ilastik"]):
        if fn.endswith(args.suffix_probablities + ".tiff"):
            print(fn)
            probablity2uncertainty.probability2uncertainty(
                pjoin(args.dirs["ilastik"], fn), args.dirs["uncertainty"]
            )

    for fn in os.listdir(args.dirs["analysis"]):
        if fn.endswith(args.suffix_probablities + ".tiff"):
            print(fn)
            probablity2uncertainty.probability2uncertainty(
                pjoin(args.dirs["analysis"], fn), args.dirs["uncertainty"]
            )

    for fol in os.listdir(args.dirs["ome"]):
        ome2micat.omefolder2micatfolder(
            pjoin(args.dirs["ome"], fol),
            args.dirs["histocat"],
            fol_masks=args.dirs["analysis"],
            mask_suffix=args.suffix_mask,
            dtype="uint16",
        )


def run_shell_command(cmd):
    cmd = re.findall(r"\S+", cmd)
    logger.debug(f"Running command: {' '.join(cmd)}")
    subprocess.call(cmd)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
