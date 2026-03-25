#!/usr/bin/env python

"""
Quantify images in stacks.
"""
import sys
import typing as tp

import numpy as np
import anndata

from imc import ROI
from imc.types import Path
from imc.ops.quant import quantify_cells_rois
from imc.scripts import build_cli, find_tiffs

def _make_anndata(quant, morphology):
    """Convert a quantification DataFrame to an AnnData object."""
    v = len(str(quant["obj_id"].max()))
    idx = quant["roi"] + "-" + quant["obj_id"].astype(str).str.zfill(v)
    quant.index = idx

    cols = [
        "sample", "roi", "obj_id", "X_centroid", "Y_centroid", "layer",
        "area", "perimeter", "minor_axis_length", "major_axis_length",
        "eccentricity", "solidity"
    ]
    cols = [c for c in cols if c in quant.columns]
    ann = anndata.AnnData(
        quant.drop(cols, axis=1, errors="ignore").astype(float), obs=quant[cols]
    )
    if "X_centroid" in ann.obs.columns:
        ann.obsm["spatial"] = ann.obs[["Y_centroid", "X_centroid"]].values
    return ann


def main(cli: tp.Sequence[str] = None) -> int:
    parser = build_cli("quantify")
    args = parser.parse_args(cli)
    args.morphology = True
    if not args.tiffs:
        args.tiffs = sorted(find_tiffs())
        if not args.tiffs:
            print("Input files were not provided and cannot be found!")
            return 1

    fs = "\n\t- " + "\n\t- ".join([f.as_posix() for f in args.tiffs])
    print(f"Starting quantification step for {len(args.tiffs)} TIFF files:{fs}!")

    # Prepare ROI objects
    rois = list()
    for tiff in args.tiffs:
        roi = ROI.from_stack(tiff)
        roi.set_channel_exclude(args.channel_exclude.split(","))
        rois.append(roi)

    missing = [r.name for r in rois if not r.get_input_filename("stack").exists()]
    if missing:
        m = "\n\t- ".join(missing)
        error = f"Not all stacks exist! Missing:\n\t- {m}"
        raise ValueError(error)
    missing = [r.name for r in rois if not r.get_input_filename("cell_mask").exists()]
    if missing:
        m = "\n\t- ".join(missing)
        error = f"Not all cell masks exist! Missing:\n\t- {m}"
        raise ValueError(error)

    # Process each TIFF individually so each gets its own output file
    for tiff, roi in zip(args.tiffs, rois):
        print(f"Quantifying '{roi.name}'...")
        quant = quantify_cells_rois(
            [roi], args.layers.split(","), morphology=args.morphology
        ).reset_index()

        # reorder columns for nice effect
        ext = ["roi", "obj_id"] + (["X_centroid", "Y_centroid"] if args.morphology else [])
        rem = [x for x in quant.columns if x not in ext]
        quant = quant[ext + rem]

        # Determine output path
        if args.output is not None:
            # -o flag: use as output directory, write <roi_name>_quantification files there
            out_dir = Path(args.output).mkdir()
            stem = tiff.stem.replace("_full", "")
            csv_path = out_dir / f"{stem}_quantification.csv.gz"
        else:
            # Default: write next to the input TIFF
            stem = tiff.stem.replace("_full", "")
            csv_path = tiff.parent / f"{stem}_quantification.csv.gz"

        quant.to_csv(csv_path, index=False)
        print(f"Wrote CSV file to '{csv_path.absolute()}'.")

        if args.output_h5ad:
            ann = _make_anndata(quant, args.morphology)
            h5ad_path = csv_path.replace_(".csv.gz", ".h5ad")
            ann.write(h5ad_path)
            print(f"Wrote h5ad file to '{h5ad_path.absolute()}'.")
            ann2 = anndata.read_h5ad(h5ad_path)
            assert np.allclose(ann.X, ann2.X)

    print("Finished quantification step.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
