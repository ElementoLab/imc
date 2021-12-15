#!/usr/bin/env python

"""
Process raw IMC files end-to-end.
"""

import sys
import typing as tp
import json
from collections import defaultdict
import time
import warnings

from urlpath import URL

from imc.types import Path
from imc.scripts import build_cli, find_mcds, find_tiffs
from imc.scripts.inspect_mcds import main as inspect
from imc.scripts.prepare import main as prepare
from imc.scripts.predict import main as predict
from imc.scripts.segment_stacks import main as segment
from imc.scripts.quantify import main as quantify
from imc.scripts.phenotype import main as phenotype
from imc.utils import download_file


DATA_DIR = Path("data")
PROCESSED_DIR = Path("processed")
MCD_FILE_ENDINGS = (".mcd", ".MCD")
TIFF_FILE_ENDINGS = (".tiff", ".TIFF", ".tif", ".TIF")
TXT_FILE_ENDINGS = (".txt", ".TXT")
DEFAULT_STEP_ARGS = {
    "prepare": ["--ilastik", "--n-crops", "0", "--ilastik-compartment", "nuclear"],
    "segment": ["--from-probabilities", "--model", "deepcell", "--compartment", "both"],
}
process_step_order = ["inspect", "prepare", "predict", "segment", "quantify", "phenotype"]
opts = defaultdict(list)
for k, v in DEFAULT_STEP_ARGS.items():
    opts[k] = v


def main(cli: tp.Sequence[str] = None) -> int:
    parser = build_cli("process")
    args = parser.parse_args(cli)

    if args.quiet:
        warnings.filterwarnings("ignore")

    if not args.files:
        print(
            "No input files were given, "
            "searching for MCD files under current directory."
        )
        args.files = find_mcds()
        if not args.files:
            print("No MCD files found. Searching for TIFF files.")
            args.files = find_tiffs()
            if not args.files:
                print(
                    "No input files could be found. Specify them manually: "
                    "`imc process $FILE`."
                )
                return 1

    # If provided URLs, download files
    urls = list(map(URL, filter(is_url, args.files)))
    args.files = list(filter(lambda x: not is_url(x), args.files))
    args.files = [Path(x).absolute().resolve() for x in args.files]

    missing = [f for f in args.files if not f.exists()]
    if missing:
        fs = "\n\t- ".join(map(str, missing))
        print(f"Could not find the following input files:\n\t- {fs}")
        return 1

    for url in urls:
        print("Given URLs as input, will download.")
        if url.name.endswith(MCD_FILE_ENDINGS):
            f = DATA_DIR.mkdir() / url.name
        elif url.name.endswith(TIFF_FILE_ENDINGS):
            f = PROCESSED_DIR.mkdir() / url.name
        elif url.name.endswith(TXT_FILE_ENDINGS):
            f = DATA_DIR.mkdir() / url.name
        if not f.exists():
            print(f"Downloading file '{url}' into '{f}'.")
            download_file(url.as_posix(), f)
            print("Completed.")
        args.files.append(f)

    # Figure out which steps are going to be done
    if args.steps is None:
        args.steps = process_step_order
    else:
        args.steps = args.steps.split(",")
        assert all(x in process_step_order for x in args.steps)
    if args.start_step is not None:
        args.steps = args.steps[args.steps.index(args.start_step) :]
    if args.stop_step is not None:
        args.steps = args.steps[: args.steps.index(args.stop_step) + 1]

    # Load config
    if args.config is not None:
        with open(args.config) as h:
            opts.update(json.load(h))

    fs = "\n\t- " + "\n\t- ".join([f.as_posix() for f in args.files])
    print(f"Starting processing of {len(args.files)} files:{fs}!")
    steps_s = "\n\t- ".join(args.steps)
    print(f"Will do following steps:\n\t- {steps_s}\n")
    time.sleep(1)

    mcds = [file for file in args.files if file.endswith(MCD_FILE_ENDINGS)]
    mcds_s = list(map(str, mcds))
    tiffs = [file for file in args.files if file.endswith(TIFF_FILE_ENDINGS)]
    tiffs_s = list(map(str, tiffs))
    txts = [file for file in args.files if file.endswith(TXT_FILE_ENDINGS)]
    txts_s = list(map(str, txts))
    if "inspect" in args.steps and mcds:
        inspect(opts["inspect"] + mcds_s)
    if "prepare" in args.steps:
        prepare(opts["prepare"] + mcds_s + tiffs_s + txts_s)

    # Now run remaining for all
    new_tiffs = list()
    for mcd in mcds:
        new_tiffs += list(
            (PROCESSED_DIR / mcd.stem / "tiffs").glob(f"{mcd.stem}*_full.tiff")
        )
    for txt in txts:
        name = txt.name.replace(".txt", "")
        tiff_f = PROCESSED_DIR / name / "tiffs" / name + "_full.tiff"
        new_tiffs += [tiff_f]
    tiffs = sorted(list(map(str, set(tiffs + new_tiffs))))

    s_parser = build_cli("segment")
    s_args = s_parser.parse_args(opts["segment"] + tiffs)
    reason = (
        f"Skipping predict step as segmentation model '{s_args.model}' does not need it."
    )
    if "predict" in args.steps:
        if s_args.model == "deepcell":
            out = predict(opts["predict"] + tiffs)
            if out:
                return out
        else:
            print(reason)
    if "segment" in args.steps:
        segment(opts["segment"] + tiffs)
    if "quantify" in args.steps:
        quantify(opts["quantify"] + tiffs)
    h5ad_f = "processed/quantification.h5ad"
    if "phenotype" in args.steps:
        phenotype(opts["phenotype"] + [h5ad_f])

    print("Finished processing!")
    return 0


def is_url(x: str) -> bool:
    from urllib.parse import urlparse

    if isinstance(x, Path):
        x = x.as_posix()

    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except:
        return False


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
