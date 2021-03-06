#!/usr/bin/env python

"""
Process raw IMC files end-to-end.
"""

import sys
import typing as tp
from collections import defaultdict

from imc.types import Path
from imc.scripts import build_cli
from imc.scripts.inspect_mcds import main as inspect
from imc.scripts.prepare_mcds import main as prepare
from imc.scripts.predict import main as predict
from imc.scripts.segment_stacks import main as segment
from imc.scripts.quantify import main as quantify


PROCESSED_DIR = Path("processed")
MCD_FILE_ENDINGS = (".mcd", ".MCD")
TIFF_FILE_ENDINGS = (".tiff", ".TIFF", ".tif", ".TIF")
DEFAULT_STEP_ARGS = {
    "prepare": ["--ilastik", "--n-crops", "0", "--ilastik-compartment", "nuclear"],
    "segment": ["--from-probabilities", "--model", "deepcell", "--compartment", "both"],
}
defaults = defaultdict(list)
for k, v in DEFAULT_STEP_ARGS.items():
    defaults[k] = v


def main(cli: tp.Sequence[str] = None) -> int:
    parser = build_cli("process")
    args = parser.parse_args(cli)

    fs = "\n\t- " + "\n\t- ".join([f.as_posix() for f in args.files])
    print(f"Starting processing of {len(args.files)} files: {fs}!")

    # If given MCD files, run inspect and prepare steps
    mcds = [file for file in args.files if file.endswith(MCD_FILE_ENDINGS)]
    mcds_s = list(map(str, mcds))
    inspect(defaults["inspect"] + mcds_s)
    prepare(defaults["prepare"] + mcds_s)

    # Now run remaining for all
    given_tiffs = [file for file in args.files if file.endswith(TIFF_FILE_ENDINGS)]
    new_tiffs = list()
    for file in mcds:
        new_tiffs += list(
            (PROCESSED_DIR / file.stem / "tiffs").glob(f"{file.stem}*_full.tiff")
        )
    tiffs = list(map(str, set(given_tiffs + new_tiffs)))

    predict(defaults["predict"] + tiffs)
    segment(defaults["segment"] + tiffs)
    quantify(defaults["quantify"] + tiffs)

    print("Finished processing!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)

# imc process data/20200629_NL1915A.mcd processed/20200629_NL1915A/tiffs/20200629_NL1915A-01_full.tiff
