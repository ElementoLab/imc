#!/usr/bin/env python

"""
View multiplexed TIFF files interactively.
"""

import sys
import time
import typing as tp

import matplotlib.pyplot as plt

from imc import ROI
from imc.graphics import InteractiveViewer
from imc.scripts import build_cli


def main(cli: tp.Sequence[str] = None) -> int:
    parser = build_cli("view")
    args = parser.parse_args(cli)
    if len(args.input_files) == 0:
        print("Input files were not provided and could not be found!")
        return 1

    kwargs = {}
    if args.kwargs is not None:
        print(args.kwargs)
        params = [x.split("=") for x in args.kwargs.split(",")]
        kwargs = {y[0]: y[1] for y in params}

    fs = "\n\t- " + "\n\t- ".join([f.as_posix() for f in args.input_files])
    print(f"Starting viewers for {len(args.input_files)} files: {fs}!")

    if args.napari:
        assert all(
            f.endswith(".mcd") for f in args.input_files
        ), "If using napari input must be MCD files!"
        import napari

        viewer = napari.Viewer()
        viewer.open(args.input_files)
        napari.run()
        return 0

    assert all(
        f.endswith((".tiff", ".tif")) for f in args.input_files
    ), "Input must be TIFF files!"

    # Prepare ROI objects
    rois = [ROI.from_stack(tiff) for tiff in args.input_files]

    # Generate viewer instances
    viewers = list()
    for roi in rois:
        view = InteractiveViewer(
            roi,
            up_key=args.up_key,
            down_key=args.down_key,
            log_key=args.log_key,
            **kwargs,
        )
        viewers.append(view)

    print(
        f"Press '{args.up_key}' and '{args.down_key}' to scroll through image channels."
        + f" '{args.log_key}' to toggle logarithmic transformation."
    )
    time.sleep(2)
    for view in viewers:
        view.fig.show()
    plt.show(block=True)

    print("Terminating!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
