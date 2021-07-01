#!/usr/bin/env python

"""
Generate probablity maps for each image.
"""

import sys
import typing as tp

from imc import ROI
from imc.types import Path
from imc.scripts import build_cli


def main(cli: tp.List[str] = None) -> int:
    """Generate probability maps for each ROI using ilastik."""
    parser = build_cli("predict")
    args = parser.parse_args(cli)

    fs = "\n\t- " + "\n\t- ".join([f.as_posix() for f in args.tiffs])
    print(f"Starting analysis of {len(args.tiffs)} TIFF files: {fs}!")

    # Prepare ROI objects
    rois = list()
    for tiff in args.tiffs:
        roi = ROI.from_stack(tiff)
        out = roi._get_input_filename("probabilities")
        if not args.overwrite and out.exists():
            continue
        rois.append(roi)

    if not rois:
        print("All output predictions exist. Skipping prediction step.")
        return 0

    # Get resources
    ilastik_sh = get_ilastik(args.lib_dir)
    if args.custom_model is None:
        model_ilp = get_model(args.models_dir, args.ilastik_model_version)
    else:
        model_ilp = args.custom_model

    # Predict
    tiff_files = [roi._get_input_filename("ilastik_input") for roi in rois]
    predict(tiff_files, ilastik_sh, model_ilp)

    for roi in rois:
        _in = roi.root_dir / roi.name + "_ilastik_s2_Probabilities.tiff"
        if _in.exists():
            _in.rename(roi._get_input_filename("probabilities"))

    print("Finished with all files!")
    return 0


def predict(tiff_files: tp.Sequence[Path], ilastik_sh: Path, model_ilp: Path) -> int:
    """
    Use a trained ilastik model to classify pixels in an IMC image.
    """
    cmd = f"""{ilastik_sh} \\
        --headless \\
        --readonly \\
        --export_source probabilities \\
        --project {model_ilp} \\
        """
    # Shell expansion of input files won't happen in subprocess call
    cmd += " ".join([x.replace_(" ", r"\ ").as_posix() for x in tiff_files])
    return run_shell_command(cmd)


def get_ilastik(lib_dir: Path, version: str = "1.3.3post2") -> Path:
    """Download ilastik software."""
    import tarfile

    os = "Linux"

    url = "https://files.ilastik.org/"
    file = f"ilastik-{version}-{os}.tar.bz2"

    f = lib_dir / f"ilastik-{version}-{os}" / "run_ilastik.sh"
    if not f.exists():
        lib_dir.mkdir()
        print("Downloading ilastik archive.")
        download_file(url + file, lib_dir / file)
        print("Extracting ilastik archive.")
        with tarfile.open(lib_dir / file, "r:bz2") as tar:
            tar.extractall(lib_dir)
    return f


def get_model(models_dir: Path, version: str = "20210302") -> Path:
    """Download pre-trained ilastik model."""
    import tarfile

    versions = {
        "20210302": "https://wcm.box.com/shared/static/1q41oshxe76b1uzt1b12etbq3l5dyov4.ilp"
    }

    url = versions[version]
    file = f"pan_dataset.{version}.ilp"

    f = models_dir / file
    if not f.exists():
        models_dir.mkdir()
        print("Downloading ilastik model.")
        download_file(url, f)
    return f


def download_file(url: str, output_file: tp.Union[Path, str], chunk_size=1024) -> None:
    """
    Download a file and write to disk in chunks (not in memory).

    Parameters
    ----------
    url : :obj:`str`
        URL to download from.
    output_file : :obj:`str`
        Path to file as output.
    chunk_size : :obj:`int`
        Size in bytes of chunk to write to disk at a time.
    """
    import shutil
    import urllib.request as request
    from contextlib import closing
    import requests

    if url.startswith("ftp://"):

        with closing(request.urlopen(url)) as r:
            with open(output_file, "wb") as f:
                shutil.copyfileobj(r, f)
    else:
        response = requests.get(url, stream=True)
        with open(output_file, "wb") as outfile:
            outfile.writelines(response.iter_content(chunk_size=chunk_size))


def run_shell_command(cmd: str, dry_run: bool = False) -> int:
    """
    Run a system command.

    Will detect whether a separate shell is required.
    """
    import textwrap
    import subprocess
    import re

    # in case the command has unix pipes or bash builtins,
    # the subprocess call must have its own shell
    # this should only occur if cellprofiler is being run uncontainerized
    # and needs a command to be called prior such as conda activate, etc
    symbol = any([x in cmd for x in ["&", "&&", "|"]])
    source = cmd.startswith("source")
    shell = bool(symbol or source)
    print(
        "Running command:\n",
        " in shell" if shell else "",
        textwrap.dedent(cmd) + "\n",
    )
    c = re.findall(r"\S+", cmd.replace("\\\n", ""))
    if not dry_run:
        if shell:
            print("Running command in shell.")
            code = subprocess.call(cmd, shell=shell)
        else:
            code = subprocess.call(c, shell=shell)
        if code != 0:
            print(
                "Process for command below failed with error:\n'%s'\nTerminating pipeline.\n",
                textwrap.dedent(cmd),
            )
            sys.exit(code)
        if not shell:
            pass
            # usage = resource.getrusage(resource.RUSAGE_SELF)
            # print(
            #     "Maximum used memory so far: {:.2f}Gb".format(
            #         usage.ru_maxrss / 1e6
            #     )
            # )
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
