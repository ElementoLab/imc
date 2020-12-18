"""
Segmentation of image stacks using pretrained deep learning models
such as Stardist and DeepCell.
"""

from typing import Union, Literal, Dict, Tuple
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist as eq
from skimage.transform import resize
import tifffile

from imc.types import Array, Figure, Path, Series


def prepare_stack(
    stack: Array,
    channel_labels: Series,
    compartment: str = "nuclear",
    channel_exclude: Series = None,
) -> Array:
    assert compartment in ["nuclear", "cytoplasm", "both"]

    if channel_exclude is not None:
        stack = stack[~channel_exclude]
        channel_labels = channel_labels[~channel_exclude.values]

    # Get nuclear channels
    nuclear_chs = channel_labels.str.contains("DNA")
    nucl = np.asarray([eq(x) for x in stack[nuclear_chs]]).mean(0)
    if compartment == "nuclear":
        return nucl

    # Get cytoplasmatic channels
    cyto_chs = ~channel_labels.str.contains("DNA|Ki67|SMA")
    cyto = np.asarray([eq(x) for x in stack[~cyto_chs]]).mean(0)
    if compartment == "cytoplasm":
        return cyto

    # Combined together and expand to 4D
    return np.stack((nucl, cyto))


def stardist_segment_nuclei(
    image: Array, model_str: str = "2D_versatile_fluo"
) -> Array:
    from stardist.models import StarDist2D

    model = StarDist2D.from_pretrained(model_str)
    mask, _ = model.predict_instances(eq(image))
    return mask


def deepcell_resize_to_predict_shape(image: Array) -> Array:
    # new version requires fixed 256x256 size
    if len(image.shape) == 2:
        return resize(image, (256, 256))[np.newaxis, :, :, np.newaxis]
    if len(image.shape) == 3:
        return resize(np.moveaxis(image, 0, -1), (256, 256, image.shape[0]))[
            np.newaxis, ...
        ]
    raise ValueError("Image with unknown shape.")


def split_image_into_tiles(
    image, tile_shape=(256, 256)
) -> Dict[Tuple[int, int], Array]:
    shape = np.asarray(image.shape)
    x, y = np.asarray(tile_shape)

    n, m = shape // tile_shape + 2

    tiles = dict()
    for i in range(m):
        for j in range(n):
            cur_pos = (i * x, j * y)
            next_pos = ((i + 1) * x, (j + 1) * y)
            tile = image[cur_pos[0] : next_pos[0], cur_pos[1] : next_pos[1]]
            tiles[cur_pos] = tile
    return tiles


def join_tiles_to_image(tiles: Dict[Tuple[int, int], Array]):
    grid = np.asarray(list(tiles.keys()))

    image_shape = np.array([0, 0])
    first_row_value = grid[0][0]
    for pos, tile in tiles.items():
        if pos[0] != first_row_value:
            continue
        image_shape[1] += tile.shape[0]
    first_col_value = grid[0][1]
    for pos, tile in tiles.items():
        if pos[1] != first_col_value:
            continue
        image_shape[0] += tile.shape[1]

    rec = np.zeros(image_shape, dtype=tile.dtype)
    for (i, j), arr in tiles.items():
        x, y = arr.shape
        rec[i : i + x, j : j + y] = arr
        rec[i : i + x, j : j + y] = arr

    # clip
    rec = rec[:, rec.sum(0) > 0]
    rec = rec[rec.sum(1) > 0, :]

    return rec


def deepcell_segment(
    image: Array,
    compartment: str = None,
    image_mpp: float = None,
    use_tilling: bool = False,
) -> Array:
    from deepcell.applications import (
        MultiplexSegmentation,
        NuclearSegmentation,
        CytoplasmSegmentation,
    )

    if compartment is None:
        # If image has more than one channel i.e. CYX
        # assume segmentation of both compartments, otherwise nuclear
        compartment = "both" if image.shape[-1] > 1 else "nuclear"

    kwargs = dict()
    if compartment == "nuclear":
        app = NuclearSegmentation()
    if compartment == "cytoplasm":
        app = CytoplasmSegmentation()
    if compartment == "both":
        app = MultiplexSegmentation()
        kwargs = dict(compartment=compartment)

    if not use_tilling:
        pred = app.predict(image, image_mpp=image_mpp, **kwargs).squeeze()
    else:
        tiles = split_image_into_tiles(image.squeeze())
        preds = dict()
        for pos, tile in tiles.items():
            print([pos])
            if tile.sum() == 0:
                preds[pos] = tile
                continue
            preds[pos] = app.predict(
                tile[np.newaxis, ..., np.newaxis], image_mpp=1
            ).squeeze()
        pred = join_tiles_to_image(preds)

    if len(pred.shape) == 4:
        pred = resize(pred, image.shape[1:])
    return pred


def cellpose_segment(image, compartment="nuclear"):
    from cellpose import models

    assert compartment in ["nuclear", "cytoplasm"]

    comp = {"nuclear": "nuclei", "cytoplasm": "cyto"}
    if compartment == "nuclear":
        channels = [0, 0]
    elif compartment == "cytoplasm":
        raise NotImplementedError
        channels = [1, 2]
    model = models.Cellpose(gpu=False, model_type=comp[compartment])
    masks, flows, styles, diams = model.eval(
        [image],
        normalize=False,
        diameter=None,
        channels=channels,
    )
    return masks[0]


def plot_cellpose_output(image, masks, flows) -> Figure:
    n = len(masks)

    fig, axes = plt.subplots(
        n, 5, figsize=(5 * 4, n * 4), sharex=True, sharey=True, squeeze=False
    )
    for i in range(n):
        axes[i][0].imshow(image)
        m = np.ma.masked_array(masks[i], masks[i] == 0)
        axes[i][1].imshow(m)
        f = resize(flows[i][2], image.shape)
        axes[i][2].imshow(f)
        f = resize(flows[i][0], image.shape)
        axes[i][3].imshow(f)
        f = resize(flows[i][0].mean(-1), image.shape)
        axes[i][4].imshow(f)
    for ax in axes.flat:
        ax.axis("off")
    labs = [
        "Original image",
        "Predicted mask",
        "Flows1",
        "Flows2",
        "mean(Flows2)",
    ]
    for ax, lab in zip(axes[0], labs):
        ax.set(title=lab)
    return fig


def cellpose_postprocessing(image, mask, flow):
    from skimage import filters

    flo = flow[0].mean(-1)
    flo = flo / flo.max()

    image2 = resize(image, flo.shape)
    mask2 = resize(mask > 0, flo.shape)
    mask2 = np.ma.masked_array(mask2, mask2 == 0)

    algos = ["li", "mean", "minimum", "otsu", "triangle", "yen", "isodata"]

    segs = dict()
    _perf = dict()
    fig, axes = plt.subplots(3, 5, sharex=True, sharey=True)
    axes[0][0].imshow(image2)
    axes[1][0].imshow(mask2)
    axes[2][0].imshow(flo)  # np.ma.masked_array(mask, mask==0)
    for algo, ax in zip(algos, axes[:, 1:].flat):
        f = getattr(filters, f"threshold_{algo}")
        t = f(flo)
        segs[algo] = flo > t
        s = flo[segs[algo]].sum()
        r = s / segs[algo].sum()
        n = ndi.label(segs[algo])[1]
        _perf[algo] = (s, r, n)
        ax.imshow(segs[algo])
        ax.set(title=f"{algo}:\nsum = {s:.1f}, ratio = {r:.2f}, n = {n}")

    perf = pd.DataFrame(_perf, index=["sum", "ratio", "objs"]).T
    perf["objs"] = 5 + (2 ** np.log1p(perf["objs"]))
    perf["sum_norm"] = perf["sum"] / perf["sum"].sum()
    perf["ratio_norm"] = perf["ratio"] / perf["ratio"].sum()
    perf["weight"] = perf[["sum_norm", "ratio_norm"]].mean(1)

    seg = np.asarray(list(segs.values()))
    seg_t = np.average(seg, axis=0, weights=perf["weight"])  #  > 0.5
    axes[-1][-2].imshow(seg_t)
    axes[-1][-2].set(title="Mean of thresholding algorightms")
    axes[-1][-1].imshow(seg_t > 0.5)
    axes[-1][-1].set(title="Threshold of mean")
    for ax in axes.flat:
        ax.axis("off")

    fig, ax = plt.subplots()
    ax.scatter(*perf.T.values)
    for algo in perf.index:
        ax.text(perf.loc[algo, "sum"], perf.loc[algo, "ratio"], s=algo)

    return seg_t > 0.5


def inflection_point(curve):
    """Return the index of the inflection point of a curve"""
    from numpy.matlib import repmat

    n_points = len(curve)
    all_coord = np.vstack((range(n_points), curve)).T
    line_vec = all_coord[-1] - all_coord[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))
    vec_from_first = all_coord - all_coord[0]
    scalar_product = np.sum(
        vec_from_first * repmat(line_vec_norm, n_points, 1), axis=1
    )
    vec_to_line = vec_from_first - np.outer(scalar_product, line_vec_norm)
    return np.argmax(np.sqrt(np.sum(vec_to_line ** 2, axis=1)))


def plot_image_and_mask(image: Array, mask_dict: Dict[str, Array]) -> Figure:
    cols = 1 + len(mask_dict) * 2

    # Make RGB if multi channel
    if len(image.shape) == 3 and image.shape[0] == 2:
        image = np.stack([image[0], image[1], np.zeros(image.shape[1:])])
        image = np.moveaxis(image, 0, -1)

    for comp in mask_dict:
        mask_dict[comp] = np.ma.masked_array(
            mask_dict[comp], mask_dict[comp] == 0
        )

    fig, axes = plt.subplots(
        1, cols, figsize=(4 * cols, 4), sharex=True, sharey=True
    )
    axes[0].imshow(image, rasterized=True)
    for ax in axes:
        ax.axis("off")
    for i, comp in enumerate(mask_dict):
        ax = axes[1 + i]
        ax.imshow(mask_dict[comp])
        ax.set(title=f"{comp.capitalize()} mask")
    for i, comp in enumerate(mask_dict):
        ax = axes[1 + len(mask_dict) + i]
        ax.imshow(image, rasterized=True)
        cs = ax.contour(mask_dict[comp], cmap="Reds")
        # important to rasterize contours
        for c in cs.collections:
            try:
                c.set_rasterized(True)
            except Exception:
                pass
        ax.set(title=f"{comp.capitalize()} mask")
    return fig


def segment_roi(
    roi: "ROI",
    model: Union[Literal["deepcell", "stardist"]] = "deepcell",
    compartment: str = "nuclear",
    save: bool = True,
    overwrite: bool = True,
    plot_segmentation: bool = True,
) -> Array:
    """
    Segment the stack of an ROI.

    Parameters
    ----------
    roi: ROI
        ROI object to process.
    model: str
        One of ``deepcell`` or ``stardist``.
    compartment: str
        One of ``nuclear``, ``cytoplasm``, or ``both``.
    save: bool
        Whether to save the outputs to disk.
    overwrite: bool
        Whether to overwrite files when ``save`` is True.
    plot_segmentation: bool
        Whether to make a figure illustrating the segmentation.
    """
    assert model in ["deepcell", "stardist", "cellpose"]

    image = prepare_stack(
        roi.stack, roi.channel_labels, compartment, roi.channel_exclude
    )

    if model == "stardist":
        assert compartment == "nuclear"
        mask = stardist_segment_nuclei(image)

    elif model == "deepcell":
        input_image = deepcell_resize_to_predict_shape(image)
        image_mpp = (np.asarray(image.shape) / 256).mean()
        mask = deepcell_segment(
            input_image, compartment=compartment, image_mpp=image_mpp
        )
        if len(image.shape) == 2:
            mask = resize(mask, image.shape[:2])
        if len(image.shape) == 3:
            mask = resize(mask, image.shape[1:3] + (2,))
    elif model == "cellpose":
        mask = cellpose_segment(image)

    mask_dict = dict()
    if compartment == "both":
        for i, comp in enumerate(["cell", "nuclei"]):
            mask_dict[comp] = mask[..., i]
    elif compartment == "nuclear":
        mask_dict["nuclei"] = mask
    elif compartment == "cytoplasm":
        mask_dict["cell"] = mask

    if save:
        for comp in mask_dict:
            mask_file = roi._get_input_filename(comp + "_mask")
            if overwrite or (not overwrite and not mask_file.exists()):
                tifffile.imwrite(mask_file, mask_dict[comp])

    if plot_segmentation:
        fig = plot_image_and_mask(image, mask_dict)
        if save:
            fig_file = roi._get_input_filename("stack").replace_(
                ".tiff", "_segmentation.svg"
            )
            if overwrite or (not overwrite and not fig_file.exists()):
                fig.savefig(fig_file, dpi=300, bbox_inches="tight")

    return mask
