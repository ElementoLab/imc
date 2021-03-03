"""
Segmentation of image stacks using pretrained deep learning models
such as Stardist and DeepCell.
"""

from typing import Union, Literal, Dict, Tuple
from functools import partial

import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist as eq
from skimage.transform import resize
import tifffile

from imc.types import Array, Figure, Path, Series
from imc.graphics import random_label_cmap
from imc.utils import minmax_scale


def normalize(image, mode="cyx"):
    if len(image.shape) == 2:
        return minmax_scale(eq(image))
    elif len(image.shape) == 3:
        if mode == "cyx":
            return np.asarray([minmax_scale(eq(x)) for x in image])
        elif mode == "yxc":
            return np.stack(
                [
                    minmax_scale(eq(image[..., i]))
                    for i in range(image.shape[-1])
                ],
                axis=-1,
            )
    raise ValueError("Unknown image shape.")


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
    nucl = normalize(stack[nuclear_chs]).mean(0)
    if compartment == "nuclear":
        return nucl

    # Get cytoplasmatic channels
    cyto_chs = ~channel_labels.str.contains("DNA|Ki67|SMA")
    cyto = normalize(stack[~cyto_chs]).mean(0)
    if compartment == "cytoplasm":
        return cyto

    # Combined together and expand to 4D
    return np.stack((nucl, cyto))


def stardist_segment_nuclei(
    image: Array, model_str: str = "2D_versatile_fluo"
) -> Array:
    from stardist.models import StarDist2D

    model = StarDist2D.from_pretrained(model_str)
    mask, output = model.predict_instances(image)
    mask = model.predict(image)
    return mask
    # # visualize probs
    # mask2 = np.zeros(mask.shape, dtype=float)
    # for c, p in zip(np.unique(mask)[1:], output['prob']):
    #     mask2[mask == c] = p

    # # to change thresholds without training
    # from attmap import AttMap
    # model._thresholds = AttMap(dict(
    #     prob = model.thresholds.prob,
    #     nms  = model.thresholds.nms
    # ))

    # # to re-train thresholds
    # from imc import Project

    # # covidimc = Project(processed_dir="~/projects/covid-imc/processed")
    # covidimc.samples = [
    #     s for s in covidimc.samples if "20200708_COVID_21_LATE" in s.name
    # ]
    # X = [
    #     prepare_stack(roi.stack, roi.channel_labels, "nuclear")
    #     for roi in covidimc.rois
    # ]
    # Y = [roi.mask for roi in covidimc.rois]
    # Y = [(roi.mask > 0).astype(int) for roi in covidimc.rois]
    # # model.optimize_thresholds(X[:1], Y[:1])
    # model.optimize_thresholds(X, Y)
    # thres = {"prob": 0.5328677624109023, "nms": 0.5}  # 20 ROIs, labels
    # thres = {"prob": 0.4871256780834209, "nms": 0.5}  # 2 ROIs, labels
    # thres = {"prob": 0.46877386118214825, "nms": 0.3}  # 2 ROIs, > 0, int
    # thres = {"prob": 0.4742475427753275, "nms": 0.3}  # 20 ROIs, > 0, int
    # thres = {'prob': 0.46877386118214825, 'nms': 0.3}  #  2 ROIS, > 0, int from scratch

    # from attmap import AttMap

    # model._thresholds = AttMap(thres)
    # mask, output = model.predict(X[0])
    # plot_image_and_mask(X[0], {"nuclear": resize(Y[0], X[0].shape)})
    # plot_image_and_mask(X[0], {"nuclear": resize(mask, X[0].shape)})
    # plot_image_and_mask(X[0], {"nuclear": resize(normalize(mask), X[0].shape)})

    # lungdev = Project(processed_dir="~/projects/lung-dev/processed")
    # roi = lungdev.rois[0]
    # image = prepare_stack(roi.stack, roi.channel_labels, "nuclear")
    # mask, output = model.predict(image)
    # plot_image_and_mask(image, {"nuclear": resize(mask, image.shape)}).show()


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
    image, tile_shape=(256, 256), overlap=(0, 0), pad_to_tile_shape=False
) -> Dict[Tuple[int, int], Array]:
    shape = np.asarray(image.shape)
    x, y = np.asarray(tile_shape)

    r1, r2 = shape / tile_shape
    n, m = shape // tile_shape + 2
    n += int(r1)
    m += int(r2)

    tiles = dict()
    for i in range(m):
        for j in range(n):
            offsets = overlap[0] * i, overlap[1] * j
            start = (max(i * x - offsets[0], 0), max(j * y - offsets[1], 0))
            end = (
                ((i + 1) * x - offsets[0])
                if (i + 2) < n
                else shape[0] - start[0],
                ((j + 1) * y - offsets[1])
                if (j + 2) < m
                else shape[1] - start[1],
            )
            tile = image[start[0] : end[0], start[1] : end[1]]
            if pad_to_tile_shape:
                diff = np.asarray(tile.shape) - tile_shape
                if not (diff > 0).all():
                    tile = np.pad(tile, ((0, abs(diff[0])), (0, abs(diff[1]))))
            if (min(tile.shape) > 0) and tile.sum() > 0:
                tiles[start] = tile

    return tiles


def join_tiles_to_image(
    tiles: Dict[Tuple[int, int], Array], trim_all_zeros=True
):
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
    if trim_all_zeros:
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

        # together as batches
        a = np.asarray(list(tiles.values()))[..., np.newaxis]
        preds = app.predict(a, image_mpp=image_mpp, **kwargs).squeeze()
        pred = join_tiles_to_image(dict(zip(tiles.keys(), preds)))

        # # one by one
        # preds = dict()
        # for pos, tile in tiles.items():
        #     print([pos])
        #     if tile.sum() == 0:
        #         preds[pos] = tile
        #         continue
        #     preds[pos] = app.predict(
        #         tile[np.newaxis, ..., np.newaxis], image_mpp=1
        #     ).squeeze()
        # pred = join_tiles_to_image(preds)

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
    cols = 1 + len(mask_dict)

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
    axes[0].set(title=f"Original signal")
    for ax in axes:
        ax.axis("off")
    for i, comp in enumerate(mask_dict):
        ax = axes[1 + i]
        ax.imshow(
            mask_dict[comp], cmap=random_label_cmap(), interpolation="none"
        )
        ax.set(title=f"{comp.capitalize()} mask")
    # for i, comp in enumerate(mask_dict):
    #     ax = axes[1 + len(mask_dict) + i]
    #     ax.imshow(image, rasterized=True)
    #     cs = ax.contour(mask_dict[comp], cmap="Reds")
    #     # important to rasterize contours
    #     for c in cs.collections:
    #         try:
    #             c.set_rasterized(True)
    #         except Exception:
    #             pass
    #     ax.set(title=f"{comp.capitalize()} mask")
    return fig


def deepcell_segment_probabilities(cyx_image):
    """"""
    from deepcell.applications import MultiplexSegmentation

    app = MultiplexSegmentation()
    pred = app.predict(
        np.moveaxis(normalize(cyx_image[:-1, ...]), 0, -1)[np.newaxis, ...],
        image_mpp=1,
    ).squeeze()

    # from deepcell.applications import NuclearSegmentation
    # app = NuclearSegmentation()
    # pred2 = app.predict(
    #     normalize(cyx_image, mode="xyc")[np.newaxis, ..., 0][..., np.newaxis],
    #     image_mpp=1,
    # ).squeeze()
    return pred


def segment_roi(
    roi: "ROI",
    from_probabilities: bool = False,
    model: Union[Literal["deepcell", "stardist"]] = "deepcell",
    compartment: str = "nuclear",
    save: bool = True,
    overwrite: bool = True,
    plot_segmentation: bool = True,
) -> Array:
    """
    Segment the area of an ROI.

    Parameters
    ----------
    roi: ROI
        ROI object to process.
    from_probabilities: bool
        Whether to use the probability array instead of the channel stack.
        If true, only the "deepcell" ``model`` can be used and the ``compartment``
        must be "cytoplasm".
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
    image = prepare_stack(
        roi.stack, roi.channel_labels, compartment, roi.channel_exclude
    )

    if from_probabilities:
        assert model in ["deepcell"]
        assert compartment in ["cytoplasm"]
        mask = deepcell_segment_probabilities(
            resize(roi.probabilities, (3,) + image.shape[:2])
        )
        # mask = ndi.label((mask, image.shape) > 0)
    else:
        assert model in ["deepcell", "stardist", "cellpose"]
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


def segment_decomposition():
    """Some experiments with tensor decomposition"""
    import tensorly as tl
    from tensorly.decomposition import tucker, non_negative_tucker, parafac

    from imc.demo import generate_disk_masks, generate_stack

    # Synthetic data
    Y = np.asarray([generate_disk_masks()[np.newaxis, ...] for _ in range(100)])
    X = np.asarray([generate_stack(m.squeeze()) for m in Y])

    fig, axes = plt.subplots(1, 3)
    for ax, a in zip(axes, X[0]):
        ax.imshow(a)

    core, factors = tucker(X[0], [1, 128, 128])
    core, factors = non_negative_tucker(X[0], [1, 128, 128])
    reconstruction = tl.tucker_to_tensor((core, factors))

    fig, axes = plt.subplots(1, 5)
    for ax, a in zip(axes, X[0]):
        ax.imshow(a)
    q = factors[1] @ factors[2].T
    axes[-2].imshow(q)
    axes[-1].imshow(q > 1)

    factors[0]  # samples
    factors[1]  # channels

    # if decomposing in more than one factor:
    fig, axes = plt.subplots(1, 4)
    for ax, dim in zip(axes, factors):
        ax.scatter(*dim.T, s=2, alpha=0.5)

    fig, axes = plt.subplots(1, 4)
    for ax, dim in zip(axes, factors):
        ax.plot(dim)

    # Real data
    r = prj.rois[50]

    exc = r.channel_exclude.index.str.contains("EMPTY|80|129")
    stack = normalize(r.stack[~exc])

    core, factors = tucker(stack, [3] + list(r.shape[1:]))
    reconstruction = tl.tucker_to_tensor((core, factors))

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    axes[0][0].imshow(stack.mean(0))
    axes[0][1].imshow(reconstruction.mean(0))
    axes[1][0].imshow(stack[-1])
    v = np.absolute(reconstruction[-1]).max()
    axes[1][1].imshow(
        reconstruction[-1], cmap="RdBu_r", vmin=-v, vmax=v
    )  # .clip(min=0))

    channel_imp = pd.Series(factors[0].squeeze(), index=r.channel_labels[~exc])
    channel_mean = pd.Series(stack.mean((1, 2)), index=r.channel_labels[~exc])

    plt.scatter(channel_mean, channel_imp)

    fig, axes = plt.subplots(1, 8)
    for ax, a in zip(axes, X[0]):
        ax.imshow(a)
    q = factors[1] @ factors[2].T
    axes[-2].imshow(q)
    axes[-1].imshow(q > 1)

    probs = zoom(r.probabilities, (1, 0.5, 0.5))
    probs = probs / probs.max()
    core, factors = tucker(probs, [3] + list(r.shape[1:]))
    reconstruction = tl.tucker_to_tensor((core, factors))

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    axes[0].imshow(np.moveaxis(probs, 0, -1))
    axes[1].imshow(np.moveaxis(reconstruction, 0, -1))

    weights, factors = parafac(probs, rank=50, init="random", tol=10e-6)
    cp_reconstruction = tl.cp_to_tensor((weights, factors))

    plt.imshow(np.moveaxis(minmax_scale(cp_reconstruction), 0, -1))

    def PCA2D_2D(samples, row_top, col_top):
        """samples are 2d matrices"""
        size = samples[0].shape
        # m*n matrix
        mean = np.zeros(size)

        for s in samples:
            mean = mean + s

        # get the mean of all samples
        mean /= float(len(samples))

        # n*n matrix
        cov_row = np.zeros((size[1], size[1]))
        for s in samples:
            diff = s - mean
            cov_row = cov_row + np.dot(diff.T, diff)
        cov_row /= float(len(samples))
        row_eval, row_evec = np.linalg.eig(cov_row)
        # select the top t evals
        sorted_index = np.argsort(row_eval)
        # using slice operation to reverse
        X = row_evec[:, sorted_index[: -row_top - 1 : -1]]

        # m*m matrix
        cov_col = np.zeros((size[0], size[0]))
        for s in samples:
            diff = s - mean
            cov_col += np.dot(diff, diff.T)
        cov_col /= float(len(samples))
        col_eval, col_evec = np.linalg.eig(cov_col)
        sorted_index = np.argsort(col_eval)
        Z = col_evec[:, sorted_index[: -col_top - 1 : -1]]

        return X, Z


# def _segment_probabilities(
#     min_distance=10,
#     detection_threshold=0.1,
#     distance_threshold=0.01,
#     exclude_border=False,
#     small_objects_threshold=0
# ):
#     from skimage.feature import peak_local_max
#     from skimage.measure import label
#     from skimage.morphology import (
#         watershed,
#         remove_small_objects,
#         h_maxima,
#         disk,
#         square,
#         dilation,
#     )
#     from skimage.segmentation import relabel_sequential
#     from deepcell_toolbox.utils import erode_edges, fill_holes

#     probs = normalize(probs, "yxc")

#     inner_distance = probs[..., 0]
#     outer_distance = probs[..., 1]

#     coords = peak_local_max(
#         inner_distance,
#         min_distance=min_distance,
#         threshold_abs=detection_threshold,
#         exclude_border=exclude_border,
#     )

#     markers = np.zeros(inner_distance.shape)
#     markers[coords[:, 0], coords[:, 1]] = 1
#     markers = label(markers)
#     label_image = watershed(
#         -outer_distance, markers, mask=outer_distance > distance_threshold
#     )
#     label_image = erode_edges(label_image, 1)

#     # Remove small objects
#     label_image = remove_small_objects(
#         label_image, min_size=small_objects_threshold
#     )

#     # Relabel the label image
#     label_image, _, _ = relabel_sequential(label_image)

#     label_images.append(label_image)


#     interior_model_smooth = 1

#     interior_batch = probs[..., 0]
#     interior_batch = nd.gaussian_filter(interior_batch, interior_model_smooth)

#     if pixel_expansion is not None:
#         interior_batch = dilation(interior_batch, selem=square(pixel_expansion * 2 + 1))

#     maxima_batch = maxima_predictions[batch, ..., 0]
#     maxima_batch = nd.gaussian_filter(maxima_batch, maxima_model_smooth)

#     markers = h_maxima(image=maxima_batch,
#                        h=maxima_threshold,
#                        selem=disk(radius))

#     markers = label(markers)

#     label_image = watershed(-interior_batch,
#                             markers,
#                             mask=interior_batch > interior_threshold,
#                             watershed_line=0)

#     # Remove small objects
#     label_image = remove_small_objects(label_image, min_size=small_objects_threshold)

#     # fill in holes that lie completely within a segmentation label
#     if fill_holes_threshold > 0:
#         label_image = fill_holes(label_image, size=fill_holes_threshold)

#     # Relabel the label image
#     label_image, _, _ = relabel_sequential(label_image)
