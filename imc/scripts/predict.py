#!/usr/bin/env python

"""
Generate probablity maps for each pixel in each image.
"""

import sys
import typing as tp

from imc import ROI
from imc.types import Path
from imc.scripts import build_cli, find_tiffs
from imc.utils import download_file, run_shell_command


def main(cli: tp.Sequence[str] = None) -> int:
    """Generate probability maps for each ROI using ilastik."""
    parser = build_cli("predict")
    args = parser.parse_args(cli)
    if not args.tiffs:
        args.tiffs = find_tiffs()
        if not args.tiffs:
            print("Input files were not provided and cannot be found!")
            return 1

    fs = "\n\t- " + "\n\t- ".join([f.as_posix() for f in args.tiffs])
    print(f"Starting predict step for {len(args.tiffs)} TIFF files:{fs}!")

    # Prepare ROI objects
    rois = list()
    for tiff in args.tiffs:
        roi = ROI.from_stack(tiff)
        out = roi.get_input_filename("probabilities")
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
    print("Starting ilastik pixel classification.")
    tiff_files = [roi.get_input_filename("ilastik_input") for roi in rois]
    predict_with_ilastik(tiff_files, ilastik_sh, model_ilp, args.quiet)

    for roi in rois:
        _in = roi.root_dir / roi.name + "_ilastik_s2_Probabilities.tiff"
        if _in.exists():
            _in.rename(roi.get_input_filename("probabilities"))

    if args.cleanup:
        for roi in rois:
            roi.get_input_filename("ilastik_input").unlink()

    print("Finished predict step!")
    return 0


def predict_with_ilastik(
    tiff_files: tp.Sequence[Path], ilastik_sh: Path, model_ilp: Path, quiet: bool = True
) -> int:
    """
    Use a trained ilastik model to classify pixels in an IMC image.
    """
    quiet_arg = "\n        --redirect_output /dev/null \\" if quiet else ""
    cmd = f"""{ilastik_sh} \\
        --headless \\
        --readonly \\
        --export_source probabilities \\{quiet_arg}
        --project {model_ilp} \\
        """
    # Shell expansion of input files won't happen in subprocess call
    cmd += " ".join([x.replace_(" ", r"\ ").as_posix() for x in tiff_files])
    return run_shell_command(cmd, quiet=True)


def get_ilastik(lib_dir: Path, version: str = "1.3.3post2") -> Path:
    """Download ilastik software."""
    import tarfile

    base_url = "https://files.ilastik.org/"

    if sys.platform.startswith("linux"):
        _os = "Linux"
        file = f"ilastik-{version}-{_os}.tar.bz2"
        f = lib_dir / f"ilastik-{version}-{_os}" / "run_ilastik.sh"
    elif sys.platform.startswith("darwin"):
        _os = "OSX"
        file = f"ilastik-{version}-{_os}.tar.bz2"
        f = (
            lib_dir
            / f"ilastik-{version}-{_os}.app"
            / "Contents"
            / "ilastik-release"
            / "run_ilastik.sh"
        )
    else:
        raise NotImplementedError(
            "ilastik command line use is only available for Linux and MacOS!"
        )

    if not f.exists():
        lib_dir.mkdir()
        print("Downloading ilastik archive.")
        download_file(base_url + file, lib_dir / file)
        print("Extracting ilastik archive.")
        with tarfile.open(lib_dir / file, "r:bz2") as tar:
            tar.extractall(lib_dir)
        (lib_dir / file).unlink()
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


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
