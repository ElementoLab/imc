#! /usr/bin/env python

"""
Functions for compensation of imaging mass cytometry data.
"""

from __future__ import (
    annotations,
)  # fix the type annotatiton of not yet undefined classes
from functools import partial
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls
import parmap
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
    flat_array: Array, spillover_matrix: Array, original_shape: Tuple[int, int, int]
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


def compensate_image_stack(roi: "ROI", normalize: bool = True) -> Array:
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


# # cd ~/projects/archive/covid-imc
# # cd ~/projects/utuc-imc

# from imc import Project
# from csbdeep.utils import normalize

# prj = Project()
# roi = prj.rois[0]
# roi = prj.get_rois("20200804_NL1933A_RUN2-10")
# stack = roi.stack
# labels = roi.channel_labels
# metals = roi.channel_metals
# comp = pd.read_csv(
#     "/home/afr/projects/lung-dev/compensation_matrix.csv", index_col=0
# ).astype(float)
# assert (comp.index == comp.columns).all()

# fig, ax = plt.subplots()
# comp2 = comp.astype(float)
# np.fill_diagonal(comp2.values, np.nan)
# sns.heatmap(comp2, ax=ax)

# filt = metals.isin(comp)
# stack = stack[filt]
# # stack = np.log1p(stack[filt])
# # stack = np.asarray([normalize(np.log1p(c)) for c in stack[filt]])
# # stack = np.asarray([c / c.max() for c in stack])
# flat_array = stack_to_flat_array(stack)
# labels = labels[filt]
# metals = metals[filt]
# comp = comp.loc[metals, metals]

# # Compensate
# # stack_comp = compensate_array(flat_array, np.linalg.inv(c.T), stack.shape)
# c = (comp).clip(upper=1)
# stack_comp = compensate_array(flat_array, c.values, stack.shape)

# stack = np.log1p(stack)
# stack_comp = np.log1p(stack_comp)

# # Observe differences
# d = np.absolute(stack - stack_comp)
# diff_ch = pd.Series(d.sum((1, 2)) / np.multiply(*stack.shape[1:]), index=labels)
# diff_ch.sort_values()
# diff = d.sum(0) / stack.shape[0]

# fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
# axes[0].imshow(stack.mean(0))
# axes[1].imshow(diff)
# for ax in axes:
#     ax.axis("off")
# fig.show()

# # # Illustrate specific channels
# i = labels.tolist().index("Ki67(Er168)")
# i = labels.tolist().index("IL6(Gd160)")
# i = labels.tolist().index("SARSCoV2S1(Eu153)")
# i = labels.tolist().index("Keratin818(Yb174)")
# i = labels.tolist().index(diff_ch.idxmax())
# i = labels.tolist().index(diff_ch.sort_values().tail(2).head(1).index[0])
# i = labels.tolist().index(diff_ch.sort_values().tail(3).head(1).index[0])
# i = labels.tolist().index(diff_ch.sort_values().tail(4).head(1).index[0])
# i = labels.tolist().index(diff_ch.sort_values().tail(5).head(1).index[0])

# fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)
# n = labels.iloc[i - 1]
# m = None  # stack[i - 1].max()
# axes[0][0].imshow(stack[i - 1], vmax=m)
# axes[0][1].imshow(stack_comp[i - 1], vmax=m)
# d = (stack_comp[i - 1]) - (stack[i - 1])
# d[np.isnan(d)] = 0
# v = d.mean() + d.std() * 3
# axes[0][2].imshow(d, cmap="RdBu_r", vmin=-v, vmax=v)
# axes[0][1].set(title=n)

# n = labels.iloc[i]
# m = None  # stack[i].max()
# axes[1][0].imshow(((stack[i])), vmax=m)
# axes[1][1].imshow(((stack_comp[i])), vmax=m)
# d = (stack_comp[i]) - (stack[i])
# d[np.isnan(d)] = 0
# v = d.mean() + d.std() * 3
# axes[1][2].imshow(d, cmap="RdBu_r", vmin=-v, vmax=v)
# axes[1][1].set(title=n)

# n = labels.iloc[i + 1]
# m = None  # stack[i + 1].max()
# axes[2][0].imshow(((stack[i + 1])), vmax=m)
# axes[2][1].imshow(((stack_comp[i + 1])), vmax=m)
# d = (stack_comp[i + 1]) - (stack[i + 1])
# d[np.isnan(d)] = 0
# v = d.mean() + d.std() * 3
# axes[2][2].imshow(d, cmap="RdBu_r", vmin=-v, vmax=v)
# axes[2][1].set(title=n)

# for ax in axes.flat:
#     ax.axis("off")
# fig.set_tight_layout(True)
# fig.show()

# plt.close(fig)
