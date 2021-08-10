#! /usr/bin/env python

"""
Functions for compensation of imaging mass cytometry data.
"""

from functools import partial
import typing as tp

import numpy as np
import pandas as pd
from scipy.optimize import nnls
import parmap

from imc import ROI
from imc.types import Array, DataFrame


def stack_to_flat_array(stack: Array) -> Array:
    return stack.reshape((stack.shape[0], -1)).T


def _get_cytospill_spillover_matrix(
    array: DataFrame, subsample_frac: float = None, subsample_n: int = None
) -> Array:
    """
    The columns of array must be metal labels (e.g. Nd142Di)!

    Requires the Github version of CytoSpill installed from a local clone,
    not through devtools pointing to the Github repo - not sure why.

    $ git clone https://github.com/KChen-lab/CytoSpill.git
    $ R CMD INSTALL CytoSpill/
    """
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.packages import importr

    numpy2ri.activate()
    pandas2ri.activate()

    cytospill = importr("CytoSpill")

    if subsample_frac is not None:
        subsample_n = int(array.shape[0] * subsample_frac)

    kwargs = dict()
    if subsample_n is not None:
        kwargs["n"] = subsample_n

    spillover_matrix, thresholds = cytospill.GetSpillMat(
        data=array,
        cols=np.arange(array.shape[1]),
        threshold=0.1,
        flexrep=5,
        neighbor=2,
        **kwargs,
    )
    # spillover_matrix = pd.DataFrame(spillover_matrix, index=df.columns, columns=df.columns)
    return spillover_matrix


def _get_correlation_spillover_matrix(array: Array, k=60) -> Array:
    return k ** np.corrcoef(array.T) / k


def get_spillover_matrix(array: Array, method: str = "cytospill", **kwargs) -> Array:
    """"""
    if method == "cytospill":
        return _get_cytospill_spillover_matrix(array, **kwargs)
    if method == "correlation":
        return _get_correlation_spillover_matrix(array)
    raise ValueError("`method` must be one of 'cytospill' or 'correlation'.")


def compensate_array(
    flat_array: Array, spillover_matrix: Array, original_shape: tp.Tuple[int, int, int]
) -> Array:
    new_shape = original_shape[1:] + (original_shape[0],)
    _nnls = partial(nnls, spillover_matrix)
    res = parmap.map(_nnls, flat_array)
    comp = np.asarray([x[0] for x in res])
    return np.moveaxis(
        (comp).reshape(new_shape),
        -1,
        0,
    )


def compensate_image_stack(roi: ROI, normalize: bool = True) -> Array:
    from imc.segmentation import normalize as _normf

    stack = roi.stack
    if roi.channel_exclude is not None:
        if roi.channel_exclude.any():
            stack = stack[~roi.channel_exclude]
    if normalize:
        stack = _normf(stack)
    flat_array = stack_to_flat_array(stack)

    labels = roi.channel_labels[~roi.channel_exclude.values]
    metals = labels.str.extract(r".*\((.*)\)")[0] + "Di"
    df = pd.DataFrame(flat_array, columns=metals)  # .iloc[:, 4:-4]
    spill = get_spillover_matrix(df, subsample_n=2000)
    comp_stack = compensate_array(flat_array, spill, roi.stack.shape)
    return comp_stack
