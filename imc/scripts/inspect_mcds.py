#!/usr/bin/env python

"""
Inspect MCD files, reporting on their basic statistics, saving
metadata as YAML files, and panel information as CSV files.
"""

import sys
import yaml
import argparse
from collections import OrderedDict
from typing import List, Tuple, Any

import pandas as pd

from imctools.io.mcd.mcdparser import McdParser

from imc.types import Path, DataFrame, Args
from imc.utils import cleanup_channel_names, build_channel_name
from imc.scripts import cli_config


def main(cli: List[str] = None) -> int:
    parser = get_args()
    args = parser.parse_args(cli)

    fs = "\n\t- " + "\n\t- ".join([f.as_posix() for f in args.mcd_files])
    print(f"Starting analysis of {len(args.mcd_files)} MCD files: {fs}!")

    # Inspect each MCD
    metas = dict()
    _chs = list()
    for mcd_file in args.mcd_files:
        meta, ch = inspect_mcd(mcd_file, args)
        metas[mcd_file.as_posix()] = meta
        _chs.append(ch.assign(mcd_file=mcd_file))

    # Dump joint metadata
    if not args.no_write:
        yaml.dump(
            encode(metas),
            open(args.output_prefix + ".all_mcds.yaml", "w"),
            indent=4,
            default_flow_style=False,
            sort_keys=False,
        )

    # Save joint panel info
    # join panels and reorder columns
    channels = pd.concat(_chs)
    channels = channels.reset_index().reindex(
        ["mcd_file", "channel"] + ch.columns.tolist(), axis=1
    )
    # check if more than one panel present
    n_panels = channels.groupby("mcd_file")["channel"].sum().nunique()
    if n_panels == 1:
        print("All MCD files use same panel.")
    else:
        print(f"MCD files use different panels, {n_panels} in total.")

    if not args.no_write:
        channels.to_csv(
            args.output_prefix + ".all_mcds.channel_labels.csv", index=False
        )

    print("Finished with all files!")
    return 0


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(**cli_config["subcommands"]["inspect"])  # type: ignore[index]
    parser.add_argument(dest="mcd_files", nargs="+", type=Path)
    parser.add_argument("--no-write", dest="no_write", action="store_true")
    parser.add_argument(
        "-o",
        "--output-prefix",
        dest="output_prefix",
        default="mcd_files",
        type=Path,
    )
    return parser


def inspect_mcd(mcd_file: Path, args: Args) -> Tuple[DataFrame, DataFrame]:
    cols = [
        "Target",
        "Metal_Tag",
        "Atom",
        "full",
        "ilastik",
    ]
    exclude_channels = ["EMPTY", "190BCKG", "80Ar", "89Y", "127I", "124Xe"]

    print(f"Started analyzing '{mcd_file}'!")

    mcd = McdParser(mcd_file)
    session = mcd.session

    # get channel labels
    ac_ids = session.acquisition_ids
    labels = pd.DataFrame(
        {
            ac_id: cleanup_channel_names(
                session.acquisitions[ac_id].channel_labels
            )
            for ac_id in ac_ids
        }
    )
    metals = pd.DataFrame(
        {ac_id: session.acquisitions[ac_id].channel_names for ac_id in ac_ids}
    )
    channel_names = labels.replace({None: "<EMPTY>"}) + "(" + metals + ")"

    same_channels = bool(
        channel_names.nunique(1).replace(0, 1).all()
    )  # np.bool is not serializable

    if same_channels:
        print("\t * All ROIs have the same markers/metals.")
        ch = channel_names.iloc[:, 0].rename("channel")
        ids = ch.str.extract(r"(?P<Target>.*)\((?P<Metal_Tag>.*)\)")
        ids.index = ch

        annot = pd.DataFrame(ids, columns=cols)
        annot["Atom"] = annot["Metal_Tag"].str.extract(r"(\d+)")[0]
        annot["full"] = (
            ~annot.index.str.contains("|".join(exclude_channels))
        ).astype(int)
        annot["ilastik"] = (
            annot.index.str.contains("DNA") | annot.index.str.startswith("CD")
        ).astype(int)
        if not args.no_write:
            annot.to_csv(mcd_file.replace_(".mcd", ".channel_labels.csv"))
    else:
        annot = pd.DataFrame(columns=cols)
        print("\t * ROIs have different markers/metals.")

    # Save some metadata
    meta = session.get_csv_dict()
    meta["n_slides"] = len(session.slides)
    print(f"\t * Contains {meta['n_slides']} slides.")
    meta["n_panoramas"] = len(session.panoramas)
    print(f"\t * Contains {meta['n_panoramas']} panoramas.")
    meta["n_ROIs"] = len(session.acquisition_ids)
    print(f"\t * Contains {meta['n_ROIs']} ROIs.")
    meta["ROI_numbers"] = session.acquisition_ids
    meta["all_ROIs_same_channels"] = same_channels
    meta["consensus_channels"] = (
        channel_names.iloc[:, 0].to_dict() if same_channels else None
    )
    meta["panoramas"] = {
        p: v.get_csv_dict() for p, v in session.panoramas.items()
    }
    meta["acquisitions"] = {
        a: ac.get_csv_dict() for a, ac in session.acquisitions.items()
    }
    meta.update(session.metadata)
    if not args.no_write:
        yaml.dump(
            encode(meta),
            open(mcd_file.replace_(".mcd", ".session_metadata.yaml"), "w"),
            indent=4,
            default_flow_style=False,
            sort_keys=False,
        )

    mcd.close()
    print(f"Finished with '{mcd_file}'!")
    return meta, annot


def encode(obj: Any) -> Any:
    """
    For serializing to JSON or YAML with no special Python object references.

    No roundtrip!
    """
    if isinstance(obj, bool):
        return str(obj).lower()
    if isinstance(obj, (list, tuple)):
        return [encode(item) for item in obj]
    if isinstance(obj, (dict, OrderedDict)):
        return {encode(key): encode(value) for key, value in obj.items()}
    return obj


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
