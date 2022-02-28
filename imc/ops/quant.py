"""
Operations of signal quantification.
"""

from __future__ import annotations
import typing as tp

import numpy as np
import pandas as pd
import parmap

import skimage.measure
from skimage.segmentation import clear_border

from imc.data_models import roi as _roi
from imc.types import DataFrame, Array, Path
from imc.utils import read_image_from_file, minmax_scale


def quantify_cell_intensity(
    stack: tp.Union[Array, Path],
    mask: tp.Union[Array, Path],
    red_func: str = "mean",
    border_objs: bool = False,
    equalize: bool = True,
    scale: bool = False,
    channel_include: Array = None,
    channel_exclude: Array = None,
) -> DataFrame:
    """
    Measure the intensity of each channel in each cell

    Parameters
    ----------
    stack: tp.Union[Array, Path]
        Image to quantify.
    mask: tp.Union[Array, Path]
        Mask to quantify.
    red_func: str
        Function to reduce pixels to object borders. Defaults to 'mean'.
    border_objs: bool
        Whether to quantify objects touching image border. Defaults to False.
    channel_include: :class:`~np.ndarray`
        Boolean array for channels to include.
    channel_exclude: :class:`~np.ndarray`
        Boolean array for channels to exclude.
    """
    from skimage.exposure import equalize_hist as eq

    if isinstance(stack, Path):
        stack = read_image_from_file(stack)
    if isinstance(mask, Path):
        mask = read_image_from_file(mask)
    if not border_objs:
        mask = clear_border(mask)

    if equalize:
        # stack = np.asarray([eq(x) for x in stack])
        _stack = list()
        for x in stack:
            p = np.percentile(x, 98)
            x[x > p] = p
            _stack.append(x)
        stack = np.asarray(_stack)
    if scale:
        stack = np.asarray([minmax_scale(x) for x in stack])

    cells = [c for c in np.unique(mask) if c != 0]
    n_channels = stack.shape[0]

    if channel_include is None:
        channel_include = np.asarray([True] * n_channels)
    if channel_exclude is None:
        channel_exclude = np.asarray([False] * n_channels)

    res = np.zeros((len(cells), n_channels), dtype=int if red_func == "sum" else float)
    for channel in np.arange(stack.shape[0])[channel_include & ~channel_exclude]:
        res[:, channel] = [
            getattr(x.intensity_image[x.image], red_func)()
            for x in skimage.measure.regionprops(mask, stack[channel])
        ]
    return pd.DataFrame(res, index=cells).rename_axis(index="obj_id")


def quantify_cell_morphology(
    mask: tp.Union[Array, Path],
    attributes: tp.Sequence[str] = [
        "area",
        "perimeter",
        "minor_axis_length",
        "major_axis_length",
        # In some images I get ValueError for 'minor_axis_length'
        # just like https://github.com/scikit-image/scikit-image/issues/2625
        # 'orientation', # should be random for non-optical imaging, so I'm not including it
        "eccentricity",
        "solidity",
        "centroid",
    ],
    border_objs: bool = False,
) -> DataFrame:
    if isinstance(mask, Path):
        mask = read_image_from_file(mask)
    if not border_objs:
        mask = clear_border(mask)

    return (
        pd.DataFrame(
            skimage.measure.regionprops_table(mask, properties=attributes),
            index=[c for c in np.unique(mask) if c != 0],
        )
        .rename_axis(index="obj_id")
        .rename(columns={"centroid-0": "X_centroid", "centroid-1": "Y_centroid"})
    )


def _quantify_cell_intensity__roi(roi: _roi.ROI, **kwargs) -> DataFrame:
    assignment = dict(roi=roi.name)
    if roi.sample is not None:
        assignment["sample"] = roi.sample.name
    return roi.quantify_cell_intensity(**kwargs).assign(**assignment)


def _quantify_cell_morphology__roi(roi: _roi.ROI, **kwargs) -> DataFrame:
    assignment = dict(roi=roi.name)
    if roi.sample is not None:
        assignment["sample"] = roi.sample.name
    return roi.quantify_cell_morphology(**kwargs).assign(**assignment)


def _correlate_channels__roi(roi: _roi.ROI, labels: str = "channel_names") -> DataFrame:
    xcorr = np.corrcoef(roi.stack.reshape((roi.channel_number, -1)))
    np.fill_diagonal(xcorr, 0)
    labs = getattr(roi, labels)
    return pd.DataFrame(xcorr, index=labs, columns=labs)


# def _get_adjacency_graph__roi(roi: _roi.ROI, **kwargs) -> DataFrame:
#     output_prefix = roi.sample.root_dir / "single_cell" / roi.name
#     return get_adjacency_graph(roi.stack, roi.mask, roi.clusters, output_prefix, **kwargs)


def quantify_cell_intensity_rois(
    rois: tp.Sequence[_roi.ROI],
    **kwargs,
) -> DataFrame:
    """
    Measure the intensity of each channel in each single cell.
    """
    return pd.concat(
        parmap.map(_quantify_cell_intensity__roi, rois, pm_pbar=True, **kwargs)
    ).rename_axis(index="obj_id")


def quantify_cell_morphology_rois(
    rois: tp.Sequence[_roi.ROI],
    **kwargs,
) -> DataFrame:
    """
    Measure the shape parameters of each single cell.
    """
    return pd.concat(
        parmap.map(_quantify_cell_morphology__roi, rois, pm_pbar=True, **kwargs)
    ).rename_axis(index="obj_id")


def quantify_cells_rois(
    rois: tp.Sequence[_roi.ROI],
    layers: tp.Sequence[str],
    intensity: bool = True,
    intensity_kwargs: tp.Dict[str, tp.Any] = {},
    morphology: bool = True,
    morphology_kwargs: tp.Dict[str, tp.Any] = {},
) -> DataFrame:
    """
    Measure the intensity of each channel in each single cell.
    """
    quants = list()
    if intensity:
        quants.append(
            quantify_cell_intensity_rois(rois=rois, layers=layers, **intensity_kwargs)
        )
    if morphology:
        quants.append(
            quantify_cell_morphology_rois(rois=rois, layers=layers, **morphology_kwargs)
        )

    return (
        # todo: this will fail if there's different layers in intensity and morphology
        pd.concat(
            # ignore because a ROI is not obliged to have a Sample
            [quants[0].drop(["sample", "roi"], axis=1, errors="ignore"), quants[1]],
            axis=1,
        )
        if len(quants) > 1
        else quants[0]
    ).rename_axis(index="obj_id")
