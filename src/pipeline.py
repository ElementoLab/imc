#!/usr/bin/env python

import os
from glob import glob
from os.path import join as pjoin
import sys
from argparse import ArgumentParser
import subprocess
import logging
import re
import urllib.request
import textwrap

from colorama import Fore
import pandas as pd
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

DOCKER_IMAGE = "afrendeiro/cellprofiler"  # "cellprofiler/cellprofiler"

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
    except (ValueError, IndexError):
        pass
    finally:
        if args.step == "all":
            for step in STEPS[1:]:
                logger.info("Doing '%s' step." % step)
                eval(step)()
                logger.info("Done with '%s' step." % step)
        else:
            logger.info("Doing '%s' step." % args.step)
            eval(args.step)()
            logger.info("Done with '%s' step." % args.step)


def setup_logger(level=logging.DEBUG):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(Fore.BLUE + "%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_cli_arguments():
    parser = ArgumentParser()
    out = pjoin(os.curdir, "pipeline")

    # Software
    choices = ['docker', 'singularity']
    parser.add_argument("--container", dest="containerized", default=None, choices=choices)
    parser.add_argument("--docker-image", dest="docker_image", default=DOCKER_IMAGE)

    # # if cellprofiler locations do not exist, clone to some default location
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
        default=None,  # "src/ilastik-1.3.3post2-Linux/run_ilastik.sh",
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

    parser.add_argument("--overwrite", action="store_true")

    # Pipeline steps
    choices = STEPS + [str(x) for x in range(len(STEPS))]
    parser.add_argument(
        "-s", "--step", dest="step", default="all", choices=choices
    )
    parser.add_argument(
        "-d", "--dry-run", dest="dry_run", action="store_true"
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

    if args.containerized is not None:
        dirbind = {"docker": "-v", "singularity": '-B'}
        args.dirbind = dirbind[args.containerized]

    if args.csv_pannel is None:
        if args.input_dirs is not None:
            args.csv_pannel = pjoin(args.input_dirs[0], "example_pannel.csv")

    # Update channel number with pannel for quantification step
    if args.csv_pannel is not None:
        # with open(args.csv_pannel, "r") as handle:
        #     args.channel_number = len(handle.read().strip().split("\n")) - 1
        args.channel_number = pd.read_csv(args.csv_pannel).query("full == 1").shape[0]

    args.suffix_mask = "_mask.tiff"
    args.suffix_probablities = "_Probabilities"
    args.list_analysis_stacks = [
        (args.csv_pannel_ilastik, "_ilastik", 1),
        (args.csv_pannel_full, "_full", 0),
    ]

    return args


def check_requirements(func):
    def docker_or_singularity():
        import shutil
        for run in ['docker', 'singularity']:
            if shutil.which("docker"):
                logger.debug("Selecting %s as container runner." % run)
                return run
        raise ValueError("Neither docker or singularity are available!")

    def inner():
        if args.containerized is not None:
            if args.containerized == 'docker':
                if args.docker_image != DOCKER_IMAGE:
                    get_docker_image_or_pull()
            elif args.containerized == 'docker':
                args.docker_image = "docker://" + args.docker_image
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
        raise ValueError("Please choose only one of the two available repos.")
    _dir = os.path.abspath(pjoin(os.path.curdir, "src", repo))
    if not os.path.exists(_dir):
        url = f"https://github.com/BodenmillerGroup/{repo} {_dir}"
        cmd = f"git clone {url}"
        run_shell_command(cmd)
    setattr(args, arg, _dir)


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
                if line.split(" ")[0] == DOCKER_IMAGE:
                    return True
        except FileNotFoundError:
            logger.error("Docker installation not detected.")
            raise
        except IndexError:
            pass
        return False

    if not check_image():
        logger.debug("Found docker image.")
        # build
        logger.debug("Did not find cellprofiler docker image. Will build.")
        cmd = f"docker pull {DOCKER_IMAGE}"
        run_shell_command(cmd)
    args.docker_image = DOCKER_IMAGE


def check_ilastik(func):
    def get_ilastik(version="1.3.3"):
        url = "https://files.ilastik.org/"
        file = f"ilastik-{version}post2-Linux.tar.bz2"
        run_shell_command(f"wget -o {pjoin('src', file)} {url + file}")
        run_shell_command(f"tar xf {pjoin('src', file)}")

    def inner():
        def_ilastik_sh_path = pjoin("src", "ilastik-1.3.3post2-Linux", "run_ilastik.sh")
        if args.ilastik_sh_path is None:
            if not os.path.exists(def_ilastik_sh_path):
                get_ilastik()
            args.ilastik_sh_path = def_ilastik_sh_path
        func()

    return inner


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
                    if args.dry_run:
                        continue
                    convertfolder2imcfolder.convert_folder2imcfolder(
                        fn_full, out_folder=args.dirs["ome"], dozip=False
                    )
        if args.dry_run:
            return
        exportacquisitioncsv.export_acquisition_csv(
            args.dirs["ome"], fol_out=args.dirs["cp"]
        )

    def prepare_histocat():
        if not os.path.exists(args.dirs["histocat"]):
            os.makedirs(args.dirs["histocat"])
        for fol in os.listdir(args.dirs["ome"]):
            if args.dry_run:
                continue
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
                    if args.dry_run:
                        continue
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
        if args.containerized:
            extra = (
                "--name cellprofiler_prepare_ilastik --rm"
                if args.containerized == 'docker'
                else "")
            cmd = f"""
        {args.containerized} run \\
        {extra} \\
            {args.dirbind} {args.dirs['base']}:/data:rw \\
            {args.dirbind} {args.cellprofiler_plugin_path}:/ImcPluginsCP:ro \\
            {args.dirbind} {args.cellprofiler_pipeline_path}:/ImcSegmentationPipeline:ro \\
            {args.docker_image} \\
                --run-headless --run \\
                --plugins-directory /ImcPluginsCP/plugins/ \\
                --pipeline /ImcSegmentationPipeline/cp3_pipelines/1_prepare_ilastik.cppipe \\
                -i /{args.dirs['analysis'].replace(args.dirs['base'], 'data')}/ \\
                -o /{args.dirs['ilastik'].replace(args.dirs['base'], 'data')}/"""
        else:
            cmd = f"""
            cellprofiler \\
                --run-headless --run \\
                --plugins-directory {args.cellprofiler_plugin_path}/plugins/ \\
                --pipeline {args.cellprofiler_pipeline_path}/cp3_pipelines/1_prepare_ilastik.cppipe \\
                -i {args.dirs['analysis']}/ \\
                -o {args.dirs['ilastik']}/"""

        # {args.dirbind} /tmp/.X11-unix:/tmp/.X11-unix:ro \\
        # -e DISPLAY=$DISPLAY \\
        run_shell_command(cmd)

    export_acquisition()
    prepare_histocat()
    prepare_ilastik()


@check_ilastik
def train():
    """Inputs are the files in ilastik/*.h5"""
    if args.step == 'all' and args.ilastik_model is not None:
        logger.info("Pre-trained model provided. Skipping training step.")
    else:
        logger.info("No model provided. Launching interactive ilastik session.")
        cmd = f"""{args.ilastik_sh_path}"""
        run_shell_command(cmd)


@check_ilastik
def predict():
    cmd = f"""{args.ilastik_sh_path} \\
        --headless \\
        --export_source probabilities \\
        --project {args.ilastik_model} \\
        """
    # Shell expansion of input files won't happen in subprocess call
    cmd += " ".join(glob(f"{args.dirs['analysis']}/*_s2.h5"))
    run_shell_command(cmd)


@check_requirements
def segment():
    extra = (
        "--name cellprofiler_segment --rm"
        if args.containerized == 'docker'
        else "")

    if args.containerized:
        cmd = f"""{args.containerized} run \\
        {extra} \\
        {args.dirbind} {args.dirs['base']}:/data:rw \\
        {args.dirbind} {args.cellprofiler_plugin_path}:/ImcPluginsCP:ro \\
        {args.dirbind} {args.cellprofiler_pipeline_path}:/ImcSegmentationPipeline:ro \\
        {args.docker_image} \\
            --run-headless --run \\
            --plugins-directory /ImcPluginsCP/plugins/ \\
            --pipeline /ImcSegmentationPipeline/cp3_pipelines/2_segment_ilastik.cppipe \\
            -i /{args.dirs['analysis'].replace(args.dirs['base'], 'data')}/ \\
            -o /{args.dirs['analysis'].replace(args.dirs['base'], 'data')}/"""
    else:
        cmd = f"""
        cellprofiler \\
            --run-headless --run \\
            --plugins-directory {args.cellprofiler_plugin_path}/plugins/ \\
            --pipeline {args.cellprofiler_pipeline_path}/cp3_pipelines/2_segment_ilastik.cppipe \\
            -i {args.dirs['analysis']}/ \\
            -o {args.dirs['analysis']}/"""
    run_shell_command(cmd)


@check_requirements
def quantify():
    # For this step, the number of channels should be updated
    # in the pipeline file (line 126 and 137).
    pipeline_file = pjoin(
        args.cellprofiler_pipeline_path,
        "cp3_pipelines",
        "3_measure_mask_basic.cppipe",
    )
    new_pipeline_file = pipeline_file.replace(".cppipe", ".new.cppipe")

    default_channel_number = r"\xff\xfe2\x004\x00"
    new_channel_number = (
        str(str(args.channel_number).encode("utf-16"))
        .replace("b'", "")
        .replace("'", "")
    )
    logger.info(f"Changing the channel number to {args.channel_number}.")

    with open(pipeline_file, "r") as ihandle:
        with open(new_pipeline_file, "w") as ohandle:
            c = ihandle.read()
            cc = c.replace(default_channel_number, new_channel_number)
            ohandle.write(cc)

    if args.containerized:
        extra = (
            "--name cellprofiler_quantify --rm"
            if args.containerized == 'docker'
            else "")
        cmd = f"""{args.containerized} run \\
        {extra} \\
        {args.dirbind} {args.dirs['base']}:/data:rw \\
        {args.dirbind} {args.cellprofiler_plugin_path}:/ImcPluginsCP:ro \\
        {args.dirbind} {args.cellprofiler_pipeline_path}:/ImcSegmentationPipeline:ro \\
        {args.docker_image} \\
            --run-headless --run \\
            --plugins-directory /ImcPluginsCP/plugins/ \\
            --pipeline /ImcSegmentationPipeline/cp3_pipelines/3_measure_mask_basic.new.cppipe \\
            -i /{args.dirs['analysis'].replace(args.dirs['base'], 'data')}/ \\
            -o /{args.dirs['cp'].replace(args.dirs['base'], 'data')}"""
    else:
        cmd = f"""
        cellprofiler
            --run-headless --run \\
            --plugins-directory {args.cellprofiler_plugin_path}/plugins/ \\
            --pipeline {args.cellprofiler_pipeline_path}/cp3_pipelines/3_measure_mask_basic.new.cppipe \\
            -i {args.dirs['analysis']}/ \\
            -o {args.dirs['cp']}"""

    run_shell_command(cmd)

    os.remove(new_pipeline_file)


def uncertainty():
    """
    This requires LZW decompression which is given by the `imagecodecs`
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
    logger.debug("Running command:\n%s" % textwrap.dedent(cmd) + "\n")
    # cmd = cmd)
    c = re.findall(r"\S+", cmd.replace("\\\n", ""))
    if not args.dry_run:
        subprocess.call(c)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
