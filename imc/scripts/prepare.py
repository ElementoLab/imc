#!/usr/bin/env python

"""
Convert MCD files to TIFF and Sample/ROI structure.
"""

import sys
import typing as tp

import pandas as pd
import numpy as np
import tifffile

from imc import ROI
from imc.segmentation import prepare_stack
from imc.utils import mcd_to_dir, plot_panoramas_rois, stack_to_ilastik_h5, txt_to_tiff
from imc.scripts import build_cli


MCD_FILE_ENDINGS = (".mcd", ".MCD")
TIFF_FILE_ENDINGS = (".tiff", ".TIFF", ".tif", ".TIF")
TXT_FILE_ENDINGS = (".txt", ".TXT")


def main(cli: tp.Sequence[str] = None) -> int:
    parser = build_cli("prepare")
    args = parser.parse_args(cli)

    if not args.pannel_csvs:
        args.pannel_csvs = [None] * len(args.input_files)
    elif len(args.pannel_csvs) == 1:
        args.pannel_csvs = args.pannel_csvs * len(args.input_files)
    else:
        assert len(args.input_files) == len(args.pannel_csvs)

    if (args.sample_names is None) or (len(args.input_files) != len(args.sample_names)):
        args.sample_names = [None] * len(args.input_files)

    mcds = [file for file in args.input_files if file.endswith(MCD_FILE_ENDINGS)]
    tiffs = [file for file in args.input_files if file.endswith(TIFF_FILE_ENDINGS)]
    txts = [file for file in args.input_files if file.endswith(TXT_FILE_ENDINGS)]
    if mcds and (tiffs or txts):
        raise ValueError(
            "Mixture of MCD and TIFFs/TXTs were given. "
            "Not yet supported, please run prepare step for each file type separately."
        )

    fs = "\n\t- " + "\n\t- ".join([f.as_posix() for f in args.input_files])
    print(f"Starting prepare step for {len(args.input_files)} files:{fs}!")

    for mcd_file, pannel_csv, sample_name in zip(
        mcds, args.pannel_csvs, args.sample_names
    ):
        sargs = args.__dict__.copy()
        del sargs["input_files"]
        del sargs["pannel_csvs"]
        del sargs["root_output_dir"]
        del sargs["sample_names"]
        sargs["mcd_file"] = mcd_file
        sargs["pannel_csv"] = pannel_csv
        sargs["sample_name"] = sample_name
        sargs["output_dir"] = args.root_output_dir / mcd_file.stem
        sargs = {k: v for k, v in sargs.items() if v is not None}

        print(f"Started analyzing '{mcd_file}'.")
        mcd_to_dir(**sargs)

        # Plot ROI positions on panoramas and slide
        plot_panoramas_rois(
            yaml_spec=mcd_file.replace_(".mcd", ".session_metadata.yaml"),
            output_prefix=args.root_output_dir / mcd_file.stem / mcd_file.stem + ".",
            panorama_image_prefix=args.root_output_dir / mcd_file.stem / "Panorama_",
            save_roi_arrays=False,
            overwrite=args.overwrite,
        )
        print(f"Finished with '{mcd_file}'.")

    for txt in txts:
        tiff_f = txt.replace_(".txt", "_full.tiff")
        txt_to_tiff(txt, tiff_f, write_channel_labels=True)
        tiffs.append(tiff_f)

    for tiff in tiffs:
        roi = ROI.from_stack(tiff)
        stack_file = tiff.replace_("_full.tiff", "_ilastik_s2.h5")
        if stack_file.exists() and (not args.overwrite):
            s = prepare_stack(roi.stack, roi.channel_labels)
            stack_to_ilastik_h5(s[np.newaxis, ...], stack_file)

    print("Finished prepare step!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
