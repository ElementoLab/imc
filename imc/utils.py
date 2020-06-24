#! /usr/bin/env python

"""
"""

import os
import re
import json
from typing import Union, List, Optional, Dict, Sequence, Callable

import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import fcluster
import scipy.ndimage as ndi
from scipy.stats import pearsonr

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx
import h5py
from anndata import AnnData
import scanpy as sc

import skimage
from skimage.exposure import equalize_hist as eq
from skimage.future import graph
import skimage.io
import skimage.measure
from skimage.transform import resize
from sklearn.linear_model import LinearRegression
import tifffile
from tqdm import tqdm

import imctools.io.mcdparser

from imc.types import DataFrame, Series, Array, Path, GenericType


matplotlib.rcParams["svg.fonttype"] = "none"
FIG_KWS = dict(bbox_inches="tight", dpi=300)


MAX_BETWEEN_CELL_DIST = 4
DEFAULT_COMMUNITY_RESOLUTION = 0.005
DEFAULT_SUPERCOMMUNITY_RESOLUTION = 0.5
# DEFAULT_SUPER_COMMUNITY_NUMBER = 12


def filter_kwargs_by_callable(kwargs: Dict, callabl: Callable, exclude: List[str] = None):
    from inspect import signature

    args = signature(callabl).parameters.keys()
    return {k: v for k, v in kwargs.items() if (k in args) and k not in (exclude or [])}


def cleanup_channel_names(series: Series) -> Series:
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
    for k, v in to_replace:
        series = series.str.replace(k, v)
    series = series.replace("", np.nan).fillna("<EMPTY>")
    series[series.str.contains(r"\(").values] = (
        series.str.extract(r"(.*)\(", expand=False).dropna().values
    )
    return series


def parse_acquisition_metadata(
    acquisition_csv: Path,
    acquisition_id: Optional[Union[str, int]] = None,
    reference: Optional[Union[Path, DataFrame]] = None,
    filter_full: bool = False,
):
    acquired = pd.read_csv(acquisition_csv)
    acquired = acquired.loc[~acquired["ChannelLabel"].isin(["X", "Y", "Z"]), :].drop_duplicates()
    if acquisition_id is not None:
        acquired = acquired.loc[
            acquired["AcquisitionID"].isin([acquisition_id, str(acquisition_id)])
        ]
        acquired = acquired[["ChannelLabel", "ChannelName"]]
    else:
        acquired = acquired[["AcquisitionID", "ChannelLabel", "ChannelName"]]

    # remove parenthesis from metal column
    acquired["ChannelName"] = acquired["ChannelName"].str.replace("(", "").str.replace(")", "")

    # clean up the channel name
    acquired["ChannelLabel"] = cleanup_channel_names(acquired["ChannelLabel"])
    acquired.index = acquired["ChannelLabel"] + "(" + acquired["ChannelName"] + ")"

    if reference is None:
        return acquired

    # Check matches, report missing
    if isinstance(reference, str):
        reference = pd.read_csv(reference, index_col=0)
    __c = acquired.index.isin(reference.index)
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
        reference.query("ilastik == True").index == joint_panel.query("ilastik == True").index
    )

    if filter_full:
        joint_panel = joint_panel.loc[joint_panel["full"].isin([1, "1", True, "TRUE"])]
    return joint_panel


def metal_order_to_channel_labels(
    metal_csv: Path, channel_metadata: Path, roi_number: Union[str, int]
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
                        res.loc[:, new_ch_name] = (
                            res.loc[:, original].T.stack().reset_index(level=1, drop=True)
                        )
                    # drop original rows
                    res = res.drop(original, axis=channel_axis)
    return res


def get_threshold_from_gaussian_mixture(
    x: Series, y: Optional[Series] = None, n_components: int = 2
) -> int:
    x = x.abs().sort_values()

    if y is None:
        from sklearn.mixture import GaussianMixture  # type: ignore

        mix = GaussianMixture(n_components=n_components)
        xx = x.values.reshape((-1, 1))
        mix.fit(xx)
        y = mix.predict(xx)
    else:
        y = y.reindex(x.index).values
    return x.loc[((y[:-1] < y[1::])).tolist() + [False]].squeeze()


def sorted_nicely(iterable: Sequence[GenericType]) -> Sequence[GenericType]:
    """
    Sort an iterable in the way that humans expect.

    Parameters
    ----------
    l : iterable
        Sequence to be sorted

    Returns
    -------
    iterable
        Sorted iterable
    """

    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(iterable, key=alphanum_key)


def read_image_from_file(file: Path, equalize: bool = False) -> Array:
    """
    Read images from a tiff or hdf5 file into a numpy array.
    Channels, if existing will be in first array dimension.
    If `equalize` is :obj:`True`, convert to float type bounded at [0, 1].
    """
    if not file.exists():
        raise FileNotFoundError(f"Could not find file: '{file}")
    # if str(file).endswith("_mask.tiff"):
    # arr = tifffile.imread(file) > 0
    if file.endswith(".ome.tiff"):
        arr = tifffile.imread(str(file), is_ome=True)
    elif file.endswith(".tiff"):
        arr = tifffile.imread(str(file))
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
    arr: Array, channel_labels: Sequence, output_prefix: Path, file_format: str = "png"
) -> None:
    if len(arr.shape) != 3:
        skimage.io.imsave(output_prefix + "." + "channel_mean" + "." + file_format, arr)
    else:
        __s = np.multiply(eq(arr.mean(axis=0)), 256).astype(np.uint8)
        skimage.io.imsave(output_prefix + "." + "channel_mean" + "." + file_format, __s)
        for channel, label in tqdm(enumerate(channel_labels), total=arr.shape[0]):
            skimage.io.imsave(
                output_prefix + "." + label + "." + file_format,
                np.multiply(arr[channel], 256).astype(np.uint8),
            )


def minmax_scale(x):
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


def get_canny_edge_image(image: Array, mask: Optional[Array], radius=30, sigma=0.5):
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
    ilastik_channels: List[str] = None,
    output_dir: Path = None,
    overwrite: bool = False,
    sample_name: str = None,
    partition_panels: bool = False,
    filter_full: bool = True,
    keep_original_roi_names: bool = False,
    allow_empty_rois: bool = True,
    only_crops: bool = False,
    n_crops: int = 5,
    crop_width: int = 500,
    crop_height: int = 500,
) -> None:
    def get_dataframe_from_channels(mcd):
        return pd.DataFrame(
            [mcd.get_acquisition_channels(x) for x in mcd.acquisition_ids],
            index=mcd.acquisition_ids,
        )

    def all_channels_equal(mcd):
        chs = get_dataframe_from_channels(mcd)
        return all([(chs[c].value_counts() == mcd.n_acquisitions).all() for c in chs.columns])

    def get_panel_partitions(mcd):
        chs = get_dataframe_from_channels(mcd)

        partitions = {k: set(k) for k in chs.drop_duplicates().index}
        for p in partitions:
            for _, row in chs.iterrows():
                print(p, row.name)
                if (row == chs.loc[list(partitions[p])[0]]).all():
                    partitions[p] = partitions[p].union(set([row.name]))
        return partitions.values()

    def build_channel_name(ac):
        return (
            cleanup_channel_names(pd.Series(ac.channel_labels))
            + "("
            + pd.Series(ac.channel_metals)
            + ")"
        ).rename("channel")

    def clip_hot_pixels(img, hp_filter_shape=(3, 3), hp_threshold=50):
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

    if pannel_csv is None and ilastik_channels is None:
        raise ValueError("One of `pannel_csv` or `ilastik_channels` must be given!")
    if ilastik_channels is None and pannel_csv is not None:
        panel = pd.read_csv(pannel_csv, index_col=0)
        ilastik_channels = panel.query("ilastik == 1").index.tolist()

    H5_YXC_AXISTAG = json.dumps(
        {
            "axes": [
                {"key": "y", "typeFlags": 2, "resolution": 0, "description": ""},
                {"key": "x", "typeFlags": 2, "resolution": 0, "description": ""},
                {"key": "c", "typeFlags": 1, "resolution": 0, "description": ""},
            ]
        }
    )

    # Parse MCD
    mcd = imctools.io.mcdparser.McdParser(mcd_file)

    if output_dir is None:
        output_dir = mcd_file.parent / "imc_dir"
    os.makedirs(output_dir, exist_ok=True)
    dirs = ["tiffs", "ilastik"]
    for d in dirs:
        os.makedirs(output_dir / d, exist_ok=True)

    if sample_name is None:
        sample_name = mcd.meta.metaname

    for i, ac_id in enumerate(mcd.acquisition_ids):
        print(ac_id)
        try:
            ac = mcd.get_imc_acquisition(ac_id)
        except imctools.io.abstractparserbase.AcquisitionError as e:
            if allow_empty_rois:
                print(e)
                continue
            raise e

        # Get output prefix
        if keep_original_roi_names:
            prefix = output_dir / "tiffs" / (ac.image_description.replace(" ", "_") + "_ac")
        else:
            prefix = output_dir / "tiffs" / (sample_name + "-" + str(i + 1).zfill(2))

        # Skip if not overwrite
        if (prefix + "_full.tiff").exists() and not overwrite:
            print("TIFF images exist and overwrite is set to `False`. Continuing.")
            continue

        # Filter channels
        channel_labels = build_channel_name(ac)
        to_exp = channel_labels[channel_labels.isin(ilastik_channels)]
        to_exp_ind = ac.get_metal_indices(
            list(map(lambda x: x[1].split(")")[0], to_exp.str.split("(")))
        )
        assert to_exp_ind == to_exp.index.tolist()

        if filter_full:
            # remove background and empty channels
            # TODO: find way to do this more systematically
            channel_labels = channel_labels[
                ~(channel_labels.str.contains(r"^\d") | channel_labels.str.contains("<EMPTY>"))
            ].reset_index(drop=True)

        # Filter hot pixels
        ac._data = np.asarray([clip_hot_pixels(x) for x in ac.data])

        # Make input for ilastik training
        to_exp = channel_labels[channel_labels.isin(ilastik_channels)]
        to_exp_ind = (
            np.array(
                ac.get_metal_indices(list(map(lambda x: x[1].split(")")[0], to_exp.str.split("("))))
            )
            + 3
        )
        # assert to_exp_ind == to_exp.index.tolist()

        # # zoom 2x
        s = tuple(x * 2 for x in ac.shape[:-1])
        full = np.moveaxis(np.asarray([resize(x, s) for x in ac.data[to_exp_ind]]), 0, -1)

        # Save input for ilastik prediction
        with h5py.File(prefix + "_ilastik_s2.h5", mode="w") as handle:
            d = handle.create_dataset("stacked_channels", data=full)
            d.attrs["axistags"] = H5_YXC_AXISTAG

        # # random crops
        iprefix = output_dir / "ilastik" / (ac.image_description.replace(" ", "_") + "_ac")
        for _ in range(n_crops):
            x = np.random.choice(range(s[0] - crop_width))
            y = np.random.choice(range(s[1] - crop_height))
            crop = full[x : (x + crop_width), y : (y + crop_height), :]
            assert crop.shape == (crop_width, crop_height, len(to_exp))
            with h5py.File(
                iprefix + f"_ilastik_x{x}_y{y}_w{crop_width}_h{crop_height}.h5", mode="w"
            ) as handle:
                d = handle.create_dataset("stacked_channels", data=crop)
                d.attrs["axistags"] = H5_YXC_AXISTAG
        if only_crops:
            continue

        # Save full image as TIFF
        p = prefix + "_full."
        ac.save_image(p + "tiff", metals=channel_labels.str.extract(r"\((.*)\)")[0])
        channel_labels.to_csv(p + "csv")

    mcd.close()

    # all_channels_equal(mcd)
    # partitions = get_panel_partitions(mcd)
    # for partition_id, partition in enumerate(partitions, start=1):
    #     for ac_id in partition:
    #         ac = mcd.get_imc_acquisition(ac_id)
    #         ac.save_image(pjoin(output_dir, f"partition_{partition_id}", ""))


def get_mean_expression_per_cluster(a: AnnData) -> DataFrame:
    means = dict()
    for cluster in a.obs["cluster"].unique():
        means[cluster] = a[a.obs["cluster"] == cluster, :].X.mean(0)
    mean_expr = pd.DataFrame(means, index=a.var.index)
    mean_expr.columns.name = "cluster"
    return mean_expr


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
#     from cellprofiler.image import ImageSet, ImageSetList
#     from cellprofiler.modules.identifyprimaryobjects import IdentifyPrimaryObjects

#     w = Workspace(
#         pipeline="",
#         module=IdentifyPrimaryObjects,
#         image_set=ImageSet(0, 0, 0),
#         object_set=[],
#         measurements=None,
#         image_set_list=ImageSetList())

#     mod = IdentifyPrimaryObjects()
#     mod.create_settings()
#     mod.x_name.value = "filepath"
