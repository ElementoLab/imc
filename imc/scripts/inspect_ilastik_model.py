#! /usr/bin/env python

import argparse
import sys
import typing as tp

import h5py
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from imc.types import Path, Array
from imc.graphics import get_grid_dims


matplotlib.rcParams["svg.fonttype"] = "none"
FIG_KWS = dict(dpi=300, bbox_inches="tight")


cli = ["_models/utuc-imc/utuc-imc.ilp"]


def main(cli: tp.List[str] = None) -> int:
    args = parse_arguments().parse_args(cli)

    inspect_ilastik_model(args.model_path)

    if args.plot:
        plot_training_data(args.model_path, args.channels_to_plot)

    if args.extract:
        extract_training_data(args.model_path, args.labels_output_file)

    if args.convert:
        convert_model_data(
            args.model_path,
            args.converted_model_output,
            args.channels_to_retain,
        )

    return 0


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Extract
    parser.add_argument(
        "-e",
        "--extract",
        dest="extract",
        action="store_true",
        help="Whether to extract training labels from ilastik file into numpy array.",
    )
    parser.add_argument(
        "--labels-output",
        dest="labels_output_file",
        default=None,
        type=Path,
        help="Path to file storing numpy array with training labels."
        " If not given will be same as model with different suffix.",
    )

    # Plot
    parser.add_argument(
        "-p",
        "--plot",
        dest="plot",
        action="store_true",
        help="Whether training set examples should be plotted.",
    )
    parser.add_argument(
        "--channels-to-plot",
        dest="channels_to_plot",
        choices=["mean", "last"],
        default="mean",
        help="Which channels to plot. One of 'mean' or 'last'.",
    )

    # Convert
    parser.add_argument(
        "-c",
        "--convert",
        dest="convert",
        action="store_true",
        help="Whether to convert ilastik model to new file by changing the input channels.",
    )
    parser.add_argument(
        "--keep-channels",
        dest="channels_to_retain",
        nargs="+",
        type=int,
        help="Channel numbers to retain in new model.",
    )
    parser.add_argument(
        "--converted-model-output",
        dest="converted_model_output",
        type=Path,
        help="Path to new model output file.",
    )
    parser.add_argument(dest="model_path", type=Path)

    return parser


def inspect_ilastik_model(model_path: Path) -> None:
    print(f"Ilastik model '{model_path}'.")

    f = h5py.File(model_path.as_posix(), mode="r")

    # Input files
    # f['Input Data']['infos']['lane0000']['Raw Data']['filePath'][()].decode()
    n_input = len(f["Input Data"]["infos"])
    training_files = [
        f["Input Data"]["infos"]["lane" + str(x).zfill(4)]["Raw Data"]["filePath"][
            ()
        ].decode()
        for x in range(n_input)
    ]

    print(f"Model was trained with {n_input} files.")

    # Feature matrix
    fv = f["FeatureSelections"]["SelectionMatrix"][()]  # values
    fx = f["FeatureSelections"]["FeatureIds"][()]  # x = derivative
    fy = f["FeatureSelections"]["Scales"][()]  # y = sigma
    feature_matrix = pd.DataFrame(
        fv,
        index=pd.Series(fx, name="Feature").str.decode("utf8"),
        columns=pd.Series(fy, name="Sigma"),
    )
    used = feature_matrix.values.sum()
    total = np.multiply(*feature_matrix.shape)
    print(f"{used}/{total} of the possible feature combinations used.")
    print("Here is the feature matrix:")
    print(feature_matrix, "\n")

    # Pixel classification
    # labels = [x.decode() for x in f['PixelClassification']['LabelNames'][()]]
    # 3 labels (3 classes?)
    # 35 blocks (35 inputs)
    # values, shape=(x, y, 1)
    annots = [len(x) for x in f["PixelClassification"]["LabelSets"].values()]
    filled_annots = [x for x in annots if x != 0]
    print(f"{len(filled_annots)}/{n_input} of the input files were labeled.")

    f.close()


def plot_training_data(
    model_path: Path,
    channels_to_plot: tp.Union[tp.Literal["mean"], tp.Literal["last"]] = "mean",
) -> None:
    from imc.segmentation import normalize

    f = h5py.File(model_path.as_posix(), mode="r")
    n_input = len(f["Input Data"]["infos"])
    annots = [len(x) for x in f["PixelClassification"]["LabelSets"].values()]
    training_files = [
        f["Input Data"]["infos"]["lane" + str(x).zfill(4)]["Raw Data"]["filePath"][
            ()
        ].decode()
        for x in range(n_input)
    ]

    # Plot labels on top of sum of channels
    n, m = get_grid_dims(len(annots))
    fig, axes = plt.subplots(
        m, n, figsize=(n * 3, m * 3), gridspec_kw=dict(wspace=0, hspace=0.05)
    )
    axes = axes.ravel()

    # get colormap depending on what channels are being plotted
    if channels_to_plot == "mean":
        cmap = matplotlib.colors.ListedColormap(
            np.asarray(sns.color_palette("tab10"))[np.asarray([-1, 1, 3])]
        )
    else:
        cmap = matplotlib.colors.ListedColormap(
            np.asarray(sns.color_palette("tab10"))[np.asarray([-4, -6, 3])]
        )

    # plot
    for i in range(n_input):
        if training_files[i].startswith("Input Data"):
            train_arr = f[training_files[i]]
        else:
            train_file = model_path.parent / training_files[i].replace(
                "/stacked_channels", ""
            )
            train_arr = h5py.File(train_file, mode="r")["stacked_channels"]

        train_arr = train_arr[()]
        train_arr[pd.isnull(train_arr)] = 0

        if channels_to_plot == "mean":
            train_arr = normalize(train_arr).mean(-1)
        else:
            train_arr = normalize(train_arr[..., -1])
        training_file_shape = train_arr.shape

        axes[i].imshow(train_arr, rasterized=True)  # , cmap='inferno')
        # axes[i].set_title(image)
        axes[i].axis("off")

        # Now for each block, get coordinates and plot
        label_arr = np.zeros(training_file_shape, dtype=float)
        # label_arr = scipy.sparse.lil_matrix(training_file_shape)
        b = f["PixelClassification"]["LabelSets"]["labels" + str(i).zfill((3))]
        for j, label in enumerate(b):
            # get start-end coordinates within training image
            d = b["block" + str(j).zfill(4)]
            pos = dict(d.attrs)["blockSlice"].replace("[", "").replace("]", "").split(",")
            xs, ys, zs = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in pos]
            arr = d[()].squeeze()
            # now fill the image with the labeled pixels
            label_arr[slice(*xs), slice(*ys)] = arr
        label_arr = np.ma.masked_array(label_arr, label_arr == 0)
        axes[i].imshow(label_arr, cmap=cmap, vmin=1, vmax=3, rasterized=True)
    fig.savefig(
        model_path.replace_(".ilp", f".training_data.{channels_to_plot}.pdf"),
        bbox_inches="tight",
        dpi=300,
    )

    f.close()


def extract_training_data(
    model_path: Path, output_path: Path = None
) -> tp.Tuple[Array, Array]:
    # Extract training labels for preservation independent of model

    if output_path is None:
        output_path = model_path.replace_(".ilp", ".training_data.npz")

    fi = h5py.File(model_path.as_posix(), mode="r")

    n_input = len(fi["Input Data"]["infos"])
    training_files = [
        fi["Input Data"]["infos"]["lane" + str(x).zfill(4)]["Raw Data"]["filePath"][
            ()
        ].decode()
        for x in range(n_input)
    ]

    # Extract arrays
    _signals = list()
    _labels = list()
    for i, file in enumerate(training_files):
        if file.startswith("Input Data"):
            train_arr = fi[file]
        else:
            train_file = model_path.parent / file.replace("/stacked_channels", "")
            train_arr = h5py.File(train_file, mode="r")["stacked_channels"]
        shape = train_arr.shape[:-1]

        # Now for each block, get coordinates and assemble
        label_arr = np.zeros(shape, dtype=float)
        b = fi["PixelClassification"]["LabelSets"]["labels" + str(i).zfill((3))]
        for j, _ in enumerate(b):
            # get start-end coordinates within training image
            d = b["block" + str(j).zfill(4)]
            pos = dict(d.attrs)["blockSlice"].replace("[", "").replace("]", "").split(",")
            xs, ys, _ = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in pos]
            arr = d[()].squeeze()
            # now fill the image with the labeled pixels
            label_arr[slice(*xs), slice(*ys)] = arr

        _signals.append(train_arr[()])
        _labels.append(label_arr)
    fi.close()

    # Save as numpy array
    signals = np.asarray(_signals)
    labels = np.asarray(_labels)
    np.savez_compressed(output_path, x=signals, y=labels)
    return (signals, labels)


def convert_model_data(
    input_model_path: Path,
    output_model_path: Path,
    channels_to_retain: tp.List[int] = [-1],
) -> None:
    # For now this will assume all files were copied into H5 model
    # TODO: implement copying of h5 files with suffix if referenced to disk paths

    # After this, model should be reloaded in ilastik,
    # change one pixel in the training data and re-train

    if output_model_path is None:
        output_model_path = input_model_path.replace_(".ilp", ".converted.ilp")

    with open(output_model_path, "wb") as handle:
        handle.write(open(input_model_path, "rb").read())

    f = h5py.File(output_model_path.as_posix(), mode="r+")

    shape = [v.shape for k, v in f["Input Data"]["local_data"].items()][0]
    print(f"Current shape of input data: {shape}")

    # Change shape of input data
    for k, v in f["Input Data"]["local_data"].items():
        del f["Input Data"]["local_data"][k]
        from imc.segmentation import normalize

        f["Input Data"]["local_data"][k] = normalize(v[()][..., channels_to_retain])

    shape = [v.shape for k, v in f["Input Data"]["local_data"].items()][0]
    print(f"Current shape of input data: {shape}")

    f.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
