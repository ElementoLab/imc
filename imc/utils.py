#! /usr/bin/env python

"""
Convenience utilities for the package.
"""

import re
import json
import typing as tp

import numpy as np
import pandas as pd
import scipy
import scipy.ndimage as ndi

import matplotlib
import matplotlib.pyplot as plt

import h5py
from anndata import AnnData

import skimage
from skimage.exposure import equalize_hist as eq
import skimage.io
import skimage.measure
from skimage.transform import resize
import tifffile
from tqdm import tqdm

from imctools.io.mcd.mcdparser import McdParser

from imc.types import DataFrame, Series, Array, Path, GenericType


matplotlib.rcParams["svg.fonttype"] = "none"
FIG_KWS = dict(bbox_inches="tight", dpi=300)


MAX_BETWEEN_CELL_DIST = 4
DEFAULT_COMMUNITY_RESOLUTION = 0.005
DEFAULT_SUPERCOMMUNITY_RESOLUTION = 0.5
# DEFAULT_SUPER_COMMUNITY_NUMBER = 12


def filter_kwargs_by_callable(
    kwargs: tp.Dict[str, tp.Any], callabl: tp.Callable, exclude: tp.List[str] = None
) -> tp.Dict[str, tp.Any]:
    """Filter a dictionary keeping only the keys which are part of a function signature."""
    from inspect import signature

    args = signature(callabl).parameters.keys()
    return {k: v for k, v in kwargs.items() if (k in args) and k not in (exclude or [])}


def build_channel_name(
    labels: tp.Dict[int, tp.Tuple[str, tp.Optional[str]]],
    metals: tp.Dict[int, tp.Tuple[str, tp.Optional[str]]],
) -> Series:
    return (
        cleanup_channel_names(pd.Series(labels)) + "(" + pd.Series(metals) + ")"
    ).rename("channel")


def cleanup_channel_names(series: tp.Union[Series, tp.List]) -> Series:
    """Standardize channel naming using a set of defined rules."""
    to_replace = [
        ("-", ""),
        ("_", ""),
        (" ", ""),
        ("/", ""),
        ("INFgamma", "IFNgamma"),
        ("pHistone", "pH"),
        ("cmycp67", "cMYCp67"),
    ]
    _series = pd.Series(series) if not isinstance(series, pd.Series) else series
    for k, v in to_replace:
        _series = _series.str.replace(k, v)
    _series = _series.replace("", np.nan).fillna("<EMPTY>")
    _series[_series.str.contains(r"\(").values] = (
        _series.str.extract(r"(.*)\(", expand=False).dropna().values
    )
    return _series


def parse_acquisition_metadata(
    acquisition_csv: Path,
    acquisition_id: tp.Union[str, int] = None,
    reference: tp.Union[Path, DataFrame] = None,
    filter_full: bool = False,
) -> DataFrame:
    acquired = pd.read_csv(acquisition_csv)
    acquired = acquired.loc[
        ~acquired["ChannelLabel"].isin(["X", "Y", "Z"]), :
    ].drop_duplicates()
    if acquisition_id is not None:
        acquired = acquired.loc[
            acquired["AcquisitionID"].isin([acquisition_id, str(acquisition_id)])
        ]
        acquired = acquired[["ChannelLabel", "ChannelName"]]
    else:
        acquired = acquired[["AcquisitionID", "ChannelLabel", "ChannelName"]]

    # remove parenthesis from metal column
    acquired["ChannelName"] = (
        acquired["ChannelName"].str.replace("(", "").str.replace(")", "")
    )

    # clean up the channel name
    acquired["ChannelLabel"] = cleanup_channel_names(acquired["ChannelLabel"])
    acquired.index = acquired["ChannelLabel"] + "(" + acquired["ChannelName"] + ")"

    if reference is None:
        return acquired

    # Check matches, report missing
    _reference = (
        pd.read_csv(reference, index_col=0)
        if not isinstance(reference, pd.DataFrame)
        else reference
    )
    __c = acquired.index.isin(_reference.index)
    if not __c.all():
        miss = "\n - ".join(acquired.loc[~__c, "ChannelLabel"])
        raise ValueError(
            f"Given reference panel '{acquisition_csv}'"
            f" is missing the following channels: \n - {miss}"
        )

    # align and sort by acquisition
    joint_panel = acquired.join(reference)

    # make sure order of ilastik channels is same as the original panel
    # this important in order for the channels to always be the same
    # and the ilastik models to be reusable
    assert all(
        _reference.query("ilastik == True").index
        == joint_panel.query("ilastik == True").index
    )

    if filter_full:
        joint_panel = joint_panel.loc[joint_panel["full"].isin([1, "1", True, "TRUE"])]
    return joint_panel


def metal_order_to_channel_labels(
    metal_csv: Path, channel_metadata: Path, roi_number: tp.Union[str, int]
):

    order = (
        pd.read_csv(metal_csv, header=None, squeeze=True)
        .to_frame(name="ChannelName")
        .set_index("ChannelName")
    )
    # read reference
    ref = parse_acquisition_metadata(channel_metadata)
    ref = ref.loc[ref["AcquisitionID"].isin([roi_number, str(roi_number)])]
    return (
        order.join(ref.reset_index().set_index("ChannelName"))["index"]
        .reset_index(drop=True)
        .rename("channel")
    )


def align_channels_by_name(res: DataFrame, channel_axis=0) -> DataFrame:
    if channel_axis not in [0, 1]:
        raise ValueError("Axis must be one of 0 or 1.")
    if res.isnull().any().any():
        print("Matrix contains NaN values, likely various pannels.")
        if channel_axis == 0:
            miss = res.index[res.isnull().any(axis=1)]
        else:
            miss = res.columns[res.isnull().any(axis=0)]
        # if there's an even number of channels with NaNs
        if len(miss) % 2 == 0:
            ex = miss.str.extract(r"^(.*)\(")[0]
            # if all channel *names* come in pairs
            if (ex.value_counts() == 2).all():
                print("Found matching channel names in different metals, will align.")
                # try to match channel swaps
                for ch in ex.unique():
                    original = miss[miss.str.startswith(ch)]
                    chs = "-".join(original.str.extract(r"^.*\((.*)\)")[0].tolist())
                    new_ch_name = ch + "(" + chs + ")"
                    # add joined values
                    if channel_axis == 0:
                        res.loc[new_ch_name] = (
                            res.loc[original].T.stack().reset_index(level=1, drop=True)
                        )
                    else:
                        res.loc[:, new_ch_name] = res.loc[:, original].T.stack().values
                    # drop original rows
                    res = res.drop(original, axis=channel_axis)
            else:
                print(
                    "Not all channels are in pairs - cannot find out how to relate them."
                )
        else:
            print(
                "Found an odd number of channels missing - cannot find out how to relate them."
            )

    return res


def is_datetime(x: Series) -> bool:
    if "datetime" in x.dtype.name:
        return True
    return False


def is_numeric(x: tp.Union[Series, tp.Any]) -> bool:
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    if (
        x.dtype.name
        in [
            "float",
            "float32",
            "float64",
            "int",
            "int8",
            "int16",
            "int32",
            "int64",
            "Int64",
        ]
        or is_datetime(x)
    ):
        return True
    if x.dtype.name in ["object", "string", "boolean", "bool"]:
        return False
    if x.dtype.name == "category":
        if len(set(type(i) for i in x)) != 1:
            raise ValueError("Series contains mixed types. Cannot transfer to color!")
        return is_numeric(x.iloc[0])
    raise ValueError(f"Cannot transfer data type '{x.dtype}' to color!")


def sorted_nicely(iterable: tp.Sequence[GenericType]) -> tp.Sequence[GenericType]:
    """
    Sort an iterable in the way that humans expect.

    Parameters
    ----------
    l : iterable
        tp.Sequence to be sorted

    Returns
    -------
    iterable
        Sorted iterable
    """

    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    if isinstance(iterable[0], (int, np.int32, np.int64)):
        return np.sort(iterable)

    return sorted(iterable, key=alphanum_key)


def read_image_from_file(file: Path, equalize: bool = False) -> Array:
    """
    Read images from a tiff or hdf5 file into a numpy array.
    Channels, if existing will be in first array dimension.
    If `equalize` is :obj:`True`, convert to float type bounded at [0, 1].
    """
    if not file.exists():
        raise FileNotFoundError(f"Could not find file: '{file}")
    if file.as_posix().endswith((".ome.tiff", ".tiff", ".ome.tif", ".tif")):
        arr = tifffile.imread(file)
    elif file.endswith(".h5"):
        with h5py.File(file, "r") as __f:
            arr = np.asarray(__f[list(__f.keys())[0]])

    if len(arr.shape) == 3:
        if min(arr.shape) == arr.shape[-1]:
            arr = np.moveaxis(arr, -1, 0)
    if equalize:
        arr = eq(arr)
    return arr


def write_image_to_file(
    arr: Array,
    channel_labels: tp.Sequence,
    output_prefix: Path,
    file_format: str = "png",
) -> None:
    if len(arr.shape) != 3:
        skimage.io.imsave(output_prefix + "." + "channel_mean" + "." + file_format, arr)
    else:
        __s = np.multiply(eq(arr.mean(axis=0)), 255).astype(np.uint8)
        skimage.io.imsave(output_prefix + "." + "channel_mean" + "." + file_format, __s)
        for channel, label in tqdm(enumerate(channel_labels), total=arr.shape[0]):
            skimage.io.imsave(
                output_prefix + "." + label + "." + file_format,
                np.multiply(arr[channel], 256).astype(np.uint8),
            )


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
    from urllib import request
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


def run_shell_command(cmd: str, dry_run: bool = False, quiet: bool = False) -> int:
    """
    Run a system command.

    Will detect whether a separate shell is required.
    """
    import sys
    import subprocess
    import textwrap

    # in case the command has unix pipes or bash builtins,
    # the subprocess call must have its own shell
    # this should only occur if cellprofiler is being run uncontainerized
    # and needs a command to be called prior such as conda activate, etc
    symbol = any(x in cmd for x in ["&", "&&", "|"])
    source = cmd.startswith("source")
    shell = bool(symbol or source)
    if not quiet:
        print(
            "Running command:\n",
            " in shell" if shell else "",
            textwrap.dedent(cmd) + "\n",
        )
    if not dry_run:
        if shell:
            if not quiet:
                print("Running command in shell.")
            code = subprocess.call(cmd, shell=shell)
        else:
            # Allow spaces in file names
            c = re.findall(r"\S+", cmd.replace(r"\ ", "__space__").replace("\\\n", ""))
            c = [x.replace("__space__", " ") for x in c]
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


def downcast_int(arr: Array, kind: str = "u") -> Array:
    """
    Downcast numpy array of integers dependent
    on largest number in array compatible with smaller bit depth.
    """
    assert kind in ["u", "i"]
    if kind == "u":
        assert arr.min() >= 0
    m = arr.max()
    for i in [8, 16, 32, 64]:
        if m <= (2 ** i - 1):
            return arr.astype(f"{kind}int{i}")
    return arr


def minmax_scale(x, by_channel=True):
    """
    Scale array to 0-1 range.

    x: np.ndarray
        Array to scale
    by_channel: bool
        Whether to perform scaling by the smallest dimension (channel).
        Defaults to `True`.
    """
    if by_channel and (x.ndim == 3):
        i = np.argmin(x.shape)
        if i != 0:
            x = np.moveaxis(x, i, 0)
        x = np.asarray([minmax_scale(y) for y in x])
        return np.moveaxis(x, 0, i)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x - x.min()) / (x.max() - x.min())


def estimate_noise(i):
    """https://stackoverflow.com/a/25436112/1469535"""
    import math
    from scipy.signal import convolve2d

    h, w = i.shape
    m = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
    sigma = np.sum(np.sum(np.absolute(convolve2d(i, m))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (w - 2) * (h - 2))
    return sigma


def fractal_dimension(Z, threshold=0.9):
    """https://gist.github.com/viveksck/1110dfca01e4ec2c608515f0d5a5b1d1"""
    # Only for 2d image
    assert len(Z.shape) == 2

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k),
            axis=1,
        )

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k * k))[0])

    # Transform Z into a binary array
    Z = Z < threshold

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2 ** np.floor(np.log(p) / np.log(2))

    # Extract the exponent
    n = int(np.log(n) / np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2 ** np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


def lacunarity(image, box_size=30):
    """
    From here: https://satsense.readthedocs.io/en/latest/_modules/satsense/features/lacunarity.html
    Calculate the lacunarity value over an image.

    The calculation is performed following these papers:

    Kit, Oleksandr, and Matthias Luedeke. "Automated detection of slum area
    change in Hyderabad, India using multitemporal satellite imagery."
    ISPRS journal of photogrammetry and remote sensing 83 (2013): 130-137.

    Kit, Oleksandr, Matthias Luedeke, and Diana Reckien. "Texture-based
    identification of urban slums in Hyderabad, India using remote sensing
    data." Applied Geography 32.2 (2012): 660-667.
    """
    kernel = np.ones((box_size, box_size))
    accumulator = scipy.signal.convolve2d(image, kernel, mode="valid")
    mean_sqrd = np.mean(accumulator) ** 2
    if mean_sqrd == 0:
        return 0.0
    return np.var(accumulator) / mean_sqrd + 1


def get_canny_edge_image(image: Array, mask: tp.Optional[Array], radius=30, sigma=0.5):
    """Compute Canny edge image."""
    from skimage.filters.rank import equalize
    from skimage.morphology import disk
    from skimage.feature import canny

    inverse_mask = ~mask
    result = equalize(image, selem=disk(radius), mask=inverse_mask)
    result = canny(result, sigma=sigma, mask=inverse_mask)
    return np.ma.array(result, mask=mask)


def mcd_to_dir(
    mcd_file: Path,
    pannel_csv: Path = None,
    ilastik_output: bool = True,
    ilastik_channels: tp.Sequence[str] = None,
    ilastik_compartment: str = None,
    output_dir: Path = None,
    output_format: str = "tiff",
    overwrite: bool = False,
    compression_level: int = 3,
    sample_name: str = None,
    partition_panels: bool = False,
    filter_full: bool = True,
    min_area: int = 10000,
    export_stacks: bool = True,
    export_panoramas: bool = True,
    keep_original_roi_names: bool = False,
    allow_empty_rois: bool = True,
    only_crops: bool = False,
    n_crops: int = 5,
    crop_width: int = 500,
    crop_height: int = 500,
) -> None:
    """"""
    # TODO: add optional rotation of images if y > x

    def get_dataframe_from_channels(mcd):
        return pd.DataFrame(
            [mcd.get_acquisition_channels(x) for x in session.acquisition_ids],
            index=session.acquisition_ids,
        )

    def all_channels_equal(mcd):
        chs = get_dataframe_from_channels(mcd)
        return all(
            [(chs[c].value_counts() == mcd.n_acquisitions).all() for c in chs.columns]
        )

    def get_panel_partitions(mcd):
        chs = get_dataframe_from_channels(mcd)

        partitions = {k: set(k) for k in chs.drop_duplicates().index}
        for p in partitions:
            for _, row in chs.iterrows():
                print(p, row.name)
                if (row == chs.loc[list(partitions[p])[0]]).all():
                    partitions[p] = partitions[p].union(set([row.name]))
        return partitions.values()

    def clip_hot_pixels(img, hp_filter_shape=(3, 3), hp_threshold=0.0001):
        """
        From https://github.com/BodenmillerGroup/ImcPluginsCP/blob/master/plugins/smoothmultichannel.py#L416
        """
        if hp_filter_shape[0] % 2 != 1 or hp_filter_shape[1] % 2 != 1:
            raise ValueError("Invalid hot pixel filter shape: %s" % str(hp_filter_shape))
        hp_filter_footprint = np.ones(hp_filter_shape)
        hp_filter_footprint[int(hp_filter_shape[0] / 2), int(hp_filter_shape[1] / 2)] = 0
        max_img = ndi.maximum_filter(img, footprint=hp_filter_footprint, mode="reflect")
        hp_mask = img - max_img > hp_threshold
        img = img.copy()
        img[hp_mask] = max_img[hp_mask]
        return img

    if partition_panels:
        raise NotImplementedError("Partitioning sample per panel is not implemented yet.")

    if pannel_csv is None and ilastik_channels is None and ilastik_compartment is None:
        raise ValueError(
            "One of `pannel_csv`, `ilastik_channels` or `ilastik_compartment` must be given!"
        )

    #
    if (
        ilastik_compartment is None
        and pannel_csv is not None
        and ilastik_channels is None
    ):
        panel = pd.read_csv(pannel_csv, index_col=0)
        ilastik_channels = panel.query("ilastik == 1").index.tolist()

    H5_YXC_AXISTAG = json.dumps(
        {
            "axes": [
                {
                    "key": "y",
                    "typeFlags": 2,
                    "resolution": 0,
                    "description": "",
                },
                {
                    "key": "x",
                    "typeFlags": 2,
                    "resolution": 0,
                    "description": "",
                },
                {
                    "key": "c",
                    "typeFlags": 1,
                    "resolution": 0,
                    "description": "",
                },
            ]
        }
    )

    if output_dir is None:
        output_dir = mcd_file.parent / "imc_dir"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Export panoramas
    if not only_crops and export_panoramas:
        get_panorama_images(
            mcd_file,
            output_file_prefix=output_dir / "Panorama",
            overwrite=overwrite,
        )

    # Parse MCD
    mcd = McdParser(mcd_file)
    session = mcd.session

    if sample_name is None:
        sample_name = session.name.replace(" ", "_")

    for i, ac_id in enumerate(session.acquisition_ids):
        print(ac_id, end="\t")
        try:
            ac = mcd.get_acquisition_data(ac_id)
        except Exception as e:  # imctools.io.abstractparserbase.AcquisitionError
            if allow_empty_rois:
                print(e)
                continue
            raise e

        if ac.image_data is None:
            continue

        if np.multiply(*ac.image_data.shape[1:]) < min_area:
            print(
                f"\nROI {ac_id} has less than the minimum area: {min_area}. Skipping.\n"
            )
            continue

        # Get output prefix
        if keep_original_roi_names:
            prefix = Path(session.name.replace(" ", "_") + "_ac")
        else:
            prefix = Path(sample_name + "-" + str(i + 1).zfill(2))

        # Skip if not overwrite
        file_ending = "tiff"
        if (prefix + "_full." + file_ending).exists() and not overwrite:
            print("TIFF images exist and overwrite is set to `False`. Continuing.")
            continue

        # Filter channels
        channel_labels = build_channel_name(ac.channel_labels, ac.channel_names)
        if filter_full:
            # remove background and empty channels
            # TODO: find way to do this more systematically
            channel_labels = channel_labels[
                ~(
                    channel_labels.str.contains(r"^\d")
                    | channel_labels.str.contains("<EMPTY>")
                    | channel_labels.str.contains("_EMPTY_")
                )
            ].reset_index(drop=True)

        p = output_dir / "tiffs" / prefix + "_full."
        if (not only_crops) and export_stacks:
            if (not (p + file_ending).exists()) or overwrite:
                # Filter hot pixels
                ac._image_data = np.asarray([clip_hot_pixels(x) for x in ac.image_data])

                # Save full image
                (output_dir / "tiffs").mkdir()
                if output_format == "tiff":
                    ac.save_tiff(
                        p + file_ending,
                        names=channel_labels.str.extract(r"\((.*)\)")[0],
                        compression=compression_level,
                    )
                elif output_format == "ome-tiff":
                    write_ometiff(
                        arr=ac._image_data,
                        labels=channel_labels.tolist(),
                        output_path=p + file_ending,
                        compression_level=compression_level,
                        description="; ".join(
                            [f"{k}={v}" for k, v in ac.acquisition.metadata.items()]
                        ),
                    )

        # Save channel labels for the stack
        if not only_crops and ((overwrite) or not (p + "csv").exists()) and export_stacks:
            channel_labels.to_csv(p + "csv")

        if not ilastik_output:
            continue

        # Prepare ilastik data
        ilastik_input = output_dir / "tiffs" / prefix + "_ilastik_s2.h5"
        if (not ilastik_input.exists()) or overwrite:
            if ilastik_compartment is None:
                # Get index of ilastik channels
                to_exp = channel_labels[channel_labels.isin(ilastik_channels)]
                to_exp_ind = [
                    ac.channel_masses.index(y)
                    for y in to_exp.str.extract(r".*\(..(\d+)\)")[0]
                ]
                assert to_exp_ind == to_exp.index.tolist()
                full = ac.image_data[to_exp_ind]
                nchannels = len(to_exp)
            else:
                # Or nuclear/cytoplasmic
                from imc.segmentation import prepare_stack

                full = prepare_stack(ac.image_data, channel_labels, ilastik_compartment)
                if len(full.shape) == 2:
                    full = full[np.newaxis, ...]
                nchannels = 2 if ilastik_compartment == "both" else 1

            # Make input for ilastik training
            stack_to_ilastik_h5(full, ilastik_input)

        # # random crops
        # # # make sure height/width are smaller or equal to acquisition dimensions

        if n_crops > 0:
            (output_dir / "ilastik").mkdir()
            s = tuple(x * 2 for x in full.shape[1:])
            if (full.shape[1] < crop_width) or (full.shape[0] < crop_height):
                msg = (
                    "Image is smaller than the requested crop size for ilastik training."
                )
                print(msg)
                continue

        for _ in range(n_crops):
            x = np.random.choice(range(s[0] - crop_width))
            y = np.random.choice(range(s[1] - crop_height))
            crop = full[x : (x + crop_width), y : (y + crop_height), :]
            assert crop.shape == (crop_width, crop_height, nchannels)
            with h5py.File(
                output_dir / "ilastik" / prefix
                + f"_ilastik_x{x}_y{y}_w{crop_width}_h{crop_height}.h5",
                mode="w",
            ) as handle:
                d = handle.create_dataset("stacked_channels", data=crop)
                d.attrs["axistags"] = H5_YXC_AXISTAG

    print("")  # add a newline to the tabs
    mcd.close()

    # all_channels_equal(mcd)
    # partitions = get_panel_partitions(mcd)
    # for partition_id, partition in enumerate(partitions, start=1):
    #     for ac_id in partition:
    #         ac = mcd.get_imc_acquisition(ac_id)
    #         ac.save_image(pjoin(output_dir, f"partition_{partition_id}", ""))


def write_ometiff(
    arr: Array,
    labels: tp.Sequence[str],
    output_path: tp.Union[Path, str],
    compression_level: int = 3,
    description: str = None,
    **tiff_kwargs,
) -> None:
    """
    Write DataArray to a multi-page OME-TIFF file.

    Parameters
    ----------
    arr: np.ndarray
        Array of dimensions CYX.
    output_path: str | pathlib.Path
        File to write TIFF file to.
    **kwargs:
        Additional arguments to tifffile.imwrite.
    """
    # TODO: Add to OME XML: ROI number
    output_path = Path(output_path)
    image_name = output_path.stem.replace("_full", "")
    labels = pd.Series(labels).str.replace("<", "_").str.replace(">", "_").tolist()

    # Generate standard OME-XML
    channels_xml = "".join(
        [
            f"""
                <Channel
                        ID="Channel:0:{i}"
                        Name="{channel}"
                        SamplesPerPixel="1"/>"""
            for i, channel in enumerate(labels)
        ]
    )
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <OME
            xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
        <Image
                Name="{image_name}"
                ID="Image:0"
                Description="{description or ''}">
            <Pixels
                    Type="float"
                    BigEndian="false"
                    DimensionOrder="XYZCT"
                    ID="Pixels:0"
                    Interleaved="false"
                    SizeX="{arr.shape[2]}"
                    SizeY="{arr.shape[1]}"
                    SizeZ="1"
                    SizeC="{arr.shape[0]}"
                    SizeT="1"
                    PhysicalSizeX="1.0"
                    PhysicalSizeXUnit="µm"
                    PhysicalSizeY="1.0"
                    PhysicalSizeYUnit="µm">
                    {channels_xml}
                <TiffData/>
            </Pixels>
        </Image>
    </OME>
    """
    output_path.parent.mkdir()

    # Notes:
    ## resolution: 1 um/px = 25400 px/inch;
    ## even though DimensionOrder is XYZCT, it is read as CYX.
    tifffile.imwrite(
        output_path,
        data=arr,
        description=xml,
        contiguous=True,
        photometric="minisblack",
        resolution=(25400, 25400, "inch"),
        metadata={"Channel": {"Name": labels}},
        compress=compression_level,
        ome=True,
        **tiff_kwargs,
    )

    # To validate the XML:
    # # Write to disk:
    # with open(output_path.replace_(".tiff", ".xml"), "w") as handle:
    #     handle.write(xml)
    # # Validate:
    # $> xmlvalid file.xml


def stack_to_ilastik_h5(stack: Array, output_file: Path = None) -> Array:
    H5_YXC_AXISTAG = json.dumps(
        {
            "axes": [
                {
                    "key": "y",
                    "typeFlags": 2,
                    "resolution": 0,
                    "description": "",
                },
                {
                    "key": "x",
                    "typeFlags": 2,
                    "resolution": 0,
                    "description": "",
                },
                {
                    "key": "c",
                    "typeFlags": 1,
                    "resolution": 0,
                    "description": "",
                },
            ]
        }
    )

    # Make input for ilastik training
    # # zoom 2x
    s = tuple(x * 2 for x in stack.shape[1:])
    full = np.moveaxis(np.asarray([resize(x, s) for x in stack]), 0, -1)
    if output_file is None:
        return full

    # # Save input for ilastik prediction
    with h5py.File(output_file, mode="w") as handle:
        d = handle.create_dataset("stacked_channels", data=full)
        d.attrs["axistags"] = H5_YXC_AXISTAG
    return full


def stack_to_probabilities(
    stack: Array,
    channel_labels: Series,
    nuclear_channels: tp.Sequence[str] = None,
    cytoplasm_channels: tp.Sequence[str] = None,
    log: bool = True,
    # scale: bool = True,
) -> Array:
    """
    Very simple way to go from a channel stack to nuclei, cytoplasm and background probabilities.
    """
    from skimage.exposure import equalize_hist as eq

    # nuclear_channels = ["DNA", "Histone", "pCREB", "cKIT", "pSTAT3"]
    nuclear_channels = nuclear_channels or ["DNA", "Histone"]
    _nuclear_channels = channel_labels[
        channel_labels.str.contains("|".join(nuclear_channels))
    ]
    if cytoplasm_channels is None:
        _cytoplasm_channels = channel_labels[~channel_labels.isin(_nuclear_channels)]
    else:
        _cytoplasm_channels = channel_labels[
            channel_labels.str.contains("|".join(cytoplasm_channels))
        ]

    if log:
        stack = np.log1p(stack)

    # if scale:
    #     stack = saturize(stack)

    # # mean of nuclear signal
    ns = stack[_nuclear_channels.index].mean(0)
    # # mean of cytoplasmatic signal
    cs = stack[_cytoplasm_channels.index].mean(0)

    # # normalize
    ns = minmax_scale(ns)
    cs = minmax_scale(eq(cs, 256 * 4))

    # # convert into probabilities
    pn = ns
    pb = 1 - minmax_scale(pn + cs)
    # pb = (pb - pb.min()) / pb.max()
    # pc = minmax_scale(cs - (pn + pb))
    pc = 1 - (pn + pb)
    rgb = np.asarray([pn, pc, pb])

    # pnf = ndi.gaussian_filter(pn, 1)
    # pcf = ndi.gaussian_filter(pc, 1)
    # pbf = ndi.gaussian_filter(pb, 1)
    # rgb = np.asarray([pnf, pcf, pbf])
    # return saturize(rgb)
    return np.clip(rgb, 0, 1)

    # # pp = ndi.zoom(roi.probabilities, (1, 0.5, 0.5))
    # # pp = pp / pp.max()
    # p = get_input_filename(roi.stack, roi.channel_labels)
    # fig, axes = plt.subplots(1, 4, sharex=True, sharey=True)
    # # axes[0].imshow(np.moveaxis(pp / pp.max(), 0, -1))
    # axes[1].imshow(np.moveaxis(p, 0, -1))
    # from skimage.exposure import equalize_hist as eq
    # axes[2].imshow(minmax_scale(eq(roi._get_channel("mean")[1])))
    # axes[3].imshow(roi.mask)


def save_probabilities(probs: Array, output_tiff: Path):
    import tifffile

    tifffile.imsave(output_tiff, np.moveaxis((probs * 2 ** 16).astype("uint16"), 0, -1))


def txt_to_tiff(
    txt_file: Path, tiff_file: Path, write_channel_labels: bool = True
) -> None:
    """
    Convert a Fluidigm TXT file to a TIFF file.

    Parameters
    ----------
    txt_file :
        Input text file from Fluidigm.

    tiff_file :
        Path to output file.

    write_channel_labels :
        Whether to write a file with labels for the channel names.
    """
    df = pd.read_table(txt_file)
    df = df.drop(
        ["Start_push", "End_push", "Pushes_duration", "Z"], axis=1, errors="ignore"
    )
    df = df.pivot_table(index="X", columns="Y")[df.columns.drop(["X", "Y"])]
    chs = df.columns.get_level_values(0).unique()
    stack = np.asarray([df[c].values for c in chs])

    tifffile.imwrite(tiff_file, stack)
    if write_channel_labels:
        pd.DataFrame({"channel_label": chs}).to_csv(tiff_file.replace_(".tiff", ".csv"))


def plot_panoramas_rois(
    yaml_spec: Path,
    output_prefix: Path,
    panorama_image_prefix: tp.Optional[Path] = None,
    save_roi_arrays: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Plot the location of panoramas and ROIs of a IMC sample.

    yaml_spec: tp.Union[str, pathlib.Path]
        Path to YAML file containing the spec of the acquired sample.
    output_prefix: tp.Union[str, pathlib.Path]
        Prefix path to output the joint image and arrays if `save_roi_arrays` is `True`.
    panorama_image_prefix: tp.Union[str, pathlib.Path]
        Prefix of images of panoramas captured by the Hyperion instrument.
    save_roi_arrays: bool
        Whether to output arrays containing the images captured by the Hyperion instrument
        in the locations of the ROIs.
    """
    import PIL
    import yaml
    import imageio
    from matplotlib.patches import Rectangle
    import seaborn as sns

    def get_pano_coords(pan):
        x1, y1 = float(pan["SlideX1PosUm"]), float(pan["SlideY1PosUm"])
        x3, y3 = float(pan["SlideX3PosUm"]), float(pan["SlideY3PosUm"])
        return tuple(map(int, [x1, y1, x3 - x1, y3 - y1]))

    def get_roi_coords(roi):
        # start positions are in a different unit for some reason
        x1, x2 = float(acq["ROIStartXPosUm"]) / 1000, float(acq["ROIEndXPosUm"])
        y1, y2 = float(acq["ROIStartYPosUm"]) / 1000, float(acq["ROIEndYPosUm"])
        x = min(x1, x2)
        y = min(y1, y2)
        width = max(x2, x1) - x
        height = max(y1, y2) - y
        return tuple(map(int, [x, y, width, height]))

    if save_roi_arrays:
        assert (
            panorama_image_prefix is not None
        ), "If `save_arrays`, provide a `panorama_image_prefix`."

    output_file = output_prefix + "joint_slide_panorama_ROIs.png"
    if output_file.exists() and (not overwrite):
        return

    PIL.Image.MAX_IMAGE_PIXELS = None  # to be able to read PNG file of any size

    spec = yaml.safe_load(yaml_spec.open())
    w, h = int(spec["Slide"][0]["WidthUm"]), int(spec["Slide"][0]["HeightUm"])
    fkws = dict(bbox_inches="tight")

    aspect_ratio = w / h
    fig, ax = plt.subplots(figsize=(4 * aspect_ratio, 4))

    colors = sns.color_palette("tab20c")

    pano_pos = dict()
    pano_imgs = dict()
    for i, pano in enumerate(spec["Panorama"]):
        # # Last panorama is not a real one (ROIs)
        if pano["Description"] == "ROIs":
            continue

        x, y, width, height = get_pano_coords(pano)
        pano_pos[pano["ID"]] = (x, y, width, height)

        # Try to read panorama image
        try:
            # In case acquisition was made
            if pano["Description"].endswith(".jpg"):
                f = panorama_image_prefix + f"{i + 1}.png"
            else:
                i2 = int(pano["Description"].split("_")[1])
                f = panorama_image_prefix + f"{i2}.png"
            pano_img = imageio.imread(f)[..., :3]
            pano_imgs[pano["ID"]] = pano_img
            # print(f"Read image file for panorama '{i + 1}'")
            ax.imshow(pano_img, extent=(x, x + width, y, y + height))
        except (FileNotFoundError, IndexError, ValueError, TypeError):
            # IndexError in case description can't be split in two
            # ValueError in case i2 can't be made an int
            # TypeError in case 'panorama_image_prefix' is None
            print(f"Could not find image file for panorama '{i + 1}'")

        # Plot rectangles
        rect = Rectangle((x, y), width, height, facecolor="none", edgecolor=colors[i])
        ax.add_patch(rect)
        ax.text(
            (x + width / 2),
            (y + height),
            s=f"Panorama '{pano['ID']}'",
            ha="center",
            va="bottom",
            color=colors[i],
        )
        ax.scatter((x + width / 2), (y + height), marker="^", color=colors[i])

    slide_area = (
        float(spec["Panorama"][0]["SlideX2PosUm"])
        - float(spec["Panorama"][0]["SlideX1PosUm"])
    ) * (
        float(spec["Panorama"][0]["SlideY2PosUm"])
        - float(spec["Panorama"][0]["SlideY3PosUm"])
    )

    for j, acq in enumerate(spec["Acquisition"]):
        x, y, width, height = get_roi_coords(acq)
        # print(acq["ID"], x, y, width, height)

        ## if too large it's likely not a real ROI
        large = (width * height / slide_area) > 0.2
        if large and x == 0 and y == 0:
            print(f"ROI {acq['ID']} is empty. Skipping.")
            continue

        # Plot rectangle around ROI
        rect = Rectangle(
            (x, y), width, height, facecolor="none", edgecolor="black", linestyle="--"
        )
        ax.add_patch(rect)
        ax.text(
            x + width / 2, y - height, s=f"ROI '{acq['ID']}'", ha="center", color="black"
        )

        if not save_roi_arrays:
            continue

        # Export ROI in panorama image
        ## Find panorama containing ROI
        dists = pd.Series(dtype=float)
        for i, pos in pano_pos.items():
            d1 = pos[0] - x
            d2 = (x + width) - (pos[0] + pos[2])
            d3 = pos[1] - y
            d4 = (y + height) - (pos[0] + pos[3])
            dists[i] = abs(d1) + abs(d2) + abs(d3) + abs(d4)
        pano_img = pano_imgs[dists.idxmin()]
        px, py, pw, ph = pano_pos[dists.idxmin()]
        ry = abs(y - py - pano_img.shape[0])

        ## Plot ROI within panorama
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.imshow(pano_img)
        rect = Rectangle(
            (x - px, ry),
            width,
            -height,
            facecolor="none",
            edgecolor=colors[j],
            linestyle="--",
        )
        ax2.add_patch(rect)
        fig2.savefig(
            output_prefix + f"panorama_ROI_{j + 1}.png",
            dpi=1600,
            **fkws,
        )

        roi_img = pano_img[ry - height : ry, x - px : x - px + width, ...]

        ## Plot ROI on its own
        fig3, ax3 = plt.subplots(figsize=(4, 4))
        ax3.imshow(roi_img)
        fig3.savefig(output_prefix + f"ROI_{j + 1}.png", dpi=600, **fkws)

        # Save as array
        np.save(output_prefix + f"ROI_{j + 1}", roi_img, allow_pickle=False)

    ax.axis("off")
    fig.savefig(output_file, dpi=300, **fkws)


def get_mean_expression_per_cluster(a: AnnData) -> DataFrame:
    means = dict()
    for cluster in a.obs["cluster"].unique():
        means[cluster] = a[a.obs["cluster"] == cluster, :].X.mean(0)
    mean_expr = pd.DataFrame(means, index=a.var.index)
    mean_expr.columns.name = "cluster"
    return mean_expr


@tp.overload
def z_score(x: Array, axis: tp.Union[tp.Literal[0], tp.Literal[1]]) -> Array:
    ...


@tp.overload
def z_score(x: DataFrame, axis: tp.Union[tp.Literal[0], tp.Literal[1]]) -> DataFrame:
    ...


def z_score(
    x: tp.Union[Array, DataFrame], axis: tp.Union[tp.Literal[0], tp.Literal[1]] = 0
) -> tp.Union[Array, DataFrame]:
    """
    Standardize and center an array or dataframe.

    Parameters
    ----------
    x :
        A numpy array or pandas DataFrame.

    axis :
        Axis across which to compute - 0 == rows, 1 == columns.
        This effectively calculates a column-wise (0) or row-wise (1) Z-score.
    """
    return (x - x.mean(axis=axis)) / x.std(axis=axis)


def double_z_score(x, red_func: str = "mean"):
    # doubly Z-scored matrix
    mmz0 = (x - x.mean()) / x.std()
    _tmp = x.T
    mmz1 = (_tmp - _tmp.mean()) / _tmp.std()
    group = pd.concat([mmz0, mmz1.T]).groupby(level=0)
    return getattr(group, red_func)()


def filter_hot_pixels(img, n_bins=1000):
    from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm
    from jax import grad, vmap

    m = img.max()
    x = np.linspace(0, m, n_bins, dtype=float)
    y = np.asarray([(img > i).sum() for i in x]).astype(float)

    filter_ = y > 3
    x2 = x[filter_]
    y2 = y[filter_]

    mod = sm.GLM(y2, np.vstack([x2, np.ones(x2.shape)]).T, family=sm.families.Poisson())
    res = mod.fit()
    f1 = lambda x: np.e ** (res.params[0] * x + res.params[1])
    g1 = vmap(grad(f1))

    mod = LinearRegression()
    mod.fit(x2.reshape((-1, 1)), np.log2(y2))
    f2 = lambda x: 2 ** (mod.coef_[0] * x + mod.intercept_)
    g2 = vmap(grad(f2))

    f3 = scipy.interpolate.interp1d(x2, y2, fill_value="extrapolate")
    # g = vmap(grad(f3))

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(x, y)
    axes[0].plot(x2, y2)
    axes[0].plot(x2, f1(x2))
    axes[1].plot(x2, -g1(x2))
    axes[0].plot(x2, f2(x2))
    axes[1].plot(x2, -g2(x2))
    axes[0].plot(x2, f3(x2))
    # axes[1].plot(x2, -g(x2))
    axes[0].set_yscale("log")

    root = scipy.optimize.root(f3, m).x
    axes[0].axvline(root, linestyle="--", color="grey")

    img > root


# def segment(
#         arr: Array,
#         output_dir: str = None
# ) -> Array:
#     import h5py
#     import skimage.filters
#     from skimage.filters import threshold_local

#     nuclei, cyto, backgd = arr

#     # make cyto
#     nuc_cyto = eq(nuclei + cyto)

#     # smooth nuclei

#     # # typical diameter
#     5, 30

#     # # discard objects outside diameter range
#     False

#     # # discard touching border
#     True

#     # # local thresholding
#     from centrosome.threshold import get_threshold

#     nuc = eq(nuclei)
#     # # #
#     size_range = (5, 30)
#     # # # threshold smoothing scale 0
#     # # # threshold correction factor 1.2
#     # # # threshold bounds 0.0, 1.0
#     lt, gt = get_threshold(
#         "Otsu", "Adaptive", nuc,
#         threshold_range_min=0, threshold_range_max=1.0,
#         threshold_correction_factor=1.2, adaptive_window_size=50)

#     binary_image = (nuc >= lt) & (nuc >= gt)

#     # # # measure variance and entropy in foreground vs background

#     # fill holes inside foreground
#     def size_fn(size, is_foreground):
#         return size < size_range[1] * size_range[1]

#     binary_image = centrosome.cpmorphology.fill_labeled_holes(
#         binary_image, size_fn=size_fn
#     )


#     # label
#     labeled_image, object_count = scipy.ndimage.label(
#         binary_image, np.ones((3, 3), bool)
#     )


#     # alternatively with CellProfiler
#     from cellprofiler.workspace import Workspace
#     from cellprofiler.image import ImageSet, ImageSettp.List
#     from cellprofiler.modules.identifyprimaryobjects import IdentifyPrimaryObjects

#     w = Workspace(
#         pipeline="",
#         module=IdentifyPrimaryObjects,
#         image_set=ImageSet(0, 0, 0),
#         object_set=[],
#         measurements=None,
#         image_set_list=ImageSettp.List())

#     mod = IdentifyPrimaryObjects()
#     mod.create_settings()
#     mod.x_name.value = "filepath"


@tp.overload
def get_panorama_images(
    mcd_file: Path, output_file_prefix: Path, overwrite: bool
) -> None:
    ...


@tp.overload
def get_panorama_images(
    mcd_file: Path, output_file_prefix: None, overwrite: bool
) -> tp.List[Array]:
    ...


def get_panorama_images(
    mcd_file: Path, output_file_prefix: Path = None, overwrite: bool = False
) -> tp.Optional[tp.List[Array]]:
    import imageio

    byteoffset = 161

    mcd = McdParser(mcd_file)

    imgs = list()
    for slide in mcd.session.metadata["Panorama"]:
        start, end = (
            int(slide["ImageStartOffset"]),
            int(slide["ImageEndOffset"]),
        )
        img = mcd._get_buffer(start + byteoffset, end + byteoffset)
        if len(img) == 0:  # empty image
            continue
        if output_file_prefix is not None:
            output_file = output_file_prefix + f"_{slide['ID']}.png"
            if overwrite or (not output_file.exists()):
                with open(output_file, "wb") as f:
                    f.write(img)
        else:
            try:
                imgs.append(imageio.imread(img))
            except ValueError:
                continue
    mcd.close()
    if output_file_prefix is None:
        return imgs
    else:
        return None


# def draw_roi():
#     import matplotlib.patches as patches
#     fig, ax = plt.subplots(1, 1)

#     metas = dict()
#     for roi in session.acquisition_ids:
#         ac_meta = mcd.meta.get_acquisition_meta(roi)
#         metas[roi] = ac_meta
#         sx, sy = float(ac_meta['ROIStartXPosUm']), float(ac_meta['ROIStartYPosUm'])
#         ex, ey = float(ac_meta['ROIEndXPosUm']), float(ac_meta['ROIEndYPosUm'])

#         width = abs(ex - sx)
#         height = abs(ey - sy)

#         # ax.imshow(imageio.imread(img))
#         ax.add_patch(patches.Rectangle((sx, sy), width, height, color="black"))

#         plt.plot((sx, sy), (ex, ey))


def get_distance_to_lumen_border(roi):
    p = roi.probabilities
    back = p[-1] / 65535
    cells = p[:-1].sum(0) / 65535

    chs = roi.channel_labels
    nuclear_channels = ["DNA", "Histone"]
    cyto_channels = chs[~chs.str.contains("|".join(nuclear_channels), case=False)]
    roi._get_channels(cyto_channels.index.tolist())[1]


def polygon_to_mask(
    polygon_vertices: tp.Sequence[tp.Sequence[float]],
    shape: tp.Tuple[int, int],
    including_edges: bool = True,
) -> Array:
    """
    Convert a set of vertices to a binary array.

    Adapted and extended from: https://stackoverflow.com/a/36759414/1469535.
    """
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.geometry.collection import GeometryCollection

    if including_edges:
        # This makes sure edge pixels are also positive
        grid = Polygon([(0, 0), (shape[0], 0), (shape[0], shape[1]), (0, shape[1])])
        poly = Polygon(polygon_vertices)
        if not poly.is_valid:
            poly = poly.buffer(0)
        inter = grid.intersection(poly)
        if isinstance(inter, (MultiPolygon, GeometryCollection)):
            return np.asarray([polygon_to_mask(x, shape) for x in inter.geoms]).sum(0) > 0
        inter_verts = np.asarray(inter.exterior.coords.xy).T.tolist()
    else:
        inter_verts = polygon_vertices
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    path = matplotlib.path.Path(inter_verts)
    grid = path.contains_points(points, radius=-1)
    return grid.reshape((shape[1], shape[0]))


def mask_to_labelme(
    labeled_image: Array,
    filename: Path,
    overwrite: bool = False,
    simplify: bool = True,
    simplification_threshold: float = 5.0,
) -> None:
    import io
    import base64

    import imageio
    from imantics import Mask
    from shapely.geometry import Polygon

    output_file = filename.replace_(".tif", ".json")
    if overwrite or output_file.exists():
        return
    polygons = Mask(labeled_image).polygons()
    shapes = list()
    for point in polygons.points:

        if not simplify:
            poly = np.asarray(point).tolist()
        else:
            poly = np.asarray(
                Polygon(point).simplify(simplification_threshold).exterior.coords.xy
            ).T.tolist()
        shape = {
            "label": "A",
            "points": poly,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {},
        }
        shapes.append(shape)

    f = io.BytesIO()
    imageio.imwrite(f, tifffile.imread(filename), format="PNG")
    f.seek(0)
    encoded = base64.encodebytes(f.read())

    payload = {
        "version": "4.5.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": filename.name,
        "imageData": encoded.decode("ascii"),
        "imageHeight": labeled_image.shape[0],
        "imageWidth": labeled_image.shape[1],
    }
    with open(output_file.as_posix(), "w") as fp:
        json.dump(payload, fp, indent=2)
