"""
Segmentation of image stacks using pretrained deep learning models
such as Stardist and DeepCell.
"""


import typing as tp
from functools import partial

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist as eq
from skimage.transform import resize
import tifffile
import sklearn

from imc.types import Array, Figure, Path, Series
from imc.data_models import roi as _roi
from imc.graphics import random_label_cmap
from imc.utils import minmax_scale


def normalize(image: Array, mode: str = "cyx") -> Array:
    if len(image.shape) == 2:
        return minmax_scale(eq(image))
    elif len(image.shape) == 3:
        if mode == "cyx":
            return np.asarray([minmax_scale(eq(x)) for x in image])
        elif mode == "yxc":
            return np.stack(
                [minmax_scale(eq(image[..., i])) for i in range(image.shape[-1])],
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
    nuclear_chs = (
        channel_labels.str.contains(r"DNA|Iridium", regex=True)
        | channel_labels.str.contains("191Ir(Ir191)", regex=False)
        | channel_labels.str.contains("193Ir(Ir193)", regex=False)
    )
    if nuclear_chs.sum() == 0:
        raise ValueError(
            "Could not determine nuclear channels using 'DNA', 'Iridium', or Ir91/Ir193."
        )

    nucl = normalize(stack[nuclear_chs]).mean(0)
    if compartment == "nuclear":
        return nucl

    # Get cytoplasmatic channels
    cyto_chs = ~(nuclear_chs | channel_labels.str.contains(r"Ki67|SMA", regex=True))
    cyto = normalize(stack[~cyto_chs]).mean(0)
    if compartment == "cytoplasm":
        return cyto

    # Combined together and expand to 4D
    return np.stack((nucl, cyto))


def extract_features(
    arr: tp.Union[Array, tp.Sequence[Array]],
    model: str = "VGG19",
    pyramids: tp.Sequence[int] = (1, 2, 4),
) -> tp.List[Array]:
    """
    Extract features from images using pre-trained deep-learning model
    """
    import tensorflow as tf

    assert model in ["VGG16", "VGG19"]

    if isinstance(arr, np.ndarray):
        assert len(arr[0].shape) == 3
        arr = [arr]
    elif isinstance(arr, (list, tuple)):
        for a in arr:
            assert len(a.shape) == 3, "Input must be of shape YXC or list of YXC arrays!"
    channels = np.unique([a.shape[-1] for a in arr])
    assert len(channels) == 1, "All arrays must have the same number of channels!"
    n_channels = channels[0]
    if n_channels == 1:
        print("Array contained only one channel. Repeating to RGB channels.")
        arr = [np.concatenate([a, a, a], axis=2).squeeze() for a in arr]
    elif n_channels != 3:
        print("Array did not contain 3 channels. Using mean of all channels.")
        arr = [a.mean(-1) for a in arr]
        arr = [np.asarray([a, a, a]).squeeze() for a in arr]

    shapes = {
        j: [(arr[j].shape[0] // i, (arr[j].shape[1] // i)) for i in pyramids]
        for j in range(len(arr))
    }
    unique_shapes = tuple(
        map(tuple, np.unique(np.concatenate(list(shapes.values())), axis=0))
    )

    models = dict()
    for shape in unique_shapes:
        f = getattr(getattr(tf.keras.applications, model.lower()), model)
        m = f(weights="imagenet", include_top=False, input_shape=(shape[0], shape[1], 3))
        for layer in m.layers:
            layer.trainable = False
        models[shape] = tf.keras.models.Model(inputs=m.input, outputs=m.layers[2].output)

    features = list()
    for img_idx, shpx in shapes.items():
        orig_shape = arr[img_idx].shape[:-1]
        _feat = list()
        for sh in shpx:
            feat = (
                models[sh]
                .predict(resize(arr[img_idx], sh + (3,))[np.newaxis, ...])
                .squeeze()
            )
            _feat.append(resize(feat, orig_shape + (feat.shape[-1],)))
        features.append(np.concatenate(np.asarray(_feat), axis=-1))
    return features


def train_pixel_classifier(
    x: Array,
    y: Array,
    save_model: bool,
    output_file: Path,
    model: str = "RandomForestClassifier",
    subsample_frac: float = None,
    subsample_n: float = None,
) -> sklearn.base.BaseEstimator:
    """
    Train classifier to predict pixel classes from image stacks (x) and sparse labels (y).

    Parameters
    ----------
    x: np.ndarray, list
        Array with shape BCXY (smaple,channel,X,Y) of signal
    y: np.ndarray, list
        Array with shape BXY (smaple,X,Y) of cellular compartments (nucleus, cytoplasm, background).
    save_model: bool
        Whether to save the trained model.
    output_file: pathlib.Path
        File to save model to. Traditionally ends in '.sav'.
    model: str
        Type of model to use. Only 'RandomForestClassifier' available for the time being.
    subsample_frac: float
        Fraction of pixels to train on. Defaults to all.
    subsample_n: float
        Number of pixels to train on. Defaults to all.
    """
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    assert model == "RandomForestClassifier"

    if subsample_frac is not None:
        assert subsample_n is None

    if subsample_n is not None:
        assert subsample_frac is None

    # TODO: make sure unequally shaped samples are supported
    print("Extracting features.")
    xf = np.asarray(extract_features([xx for xx in x]))
    xf = xf.reshape((-1, xf.shape[-1]))
    yf = y.reshape((xf.shape[0],))

    # remove null pixels
    nan = pd.isnull(xf).any(1)
    xf = xf[~nan, :]
    yf = yf[~nan]

    # select only pixels with class
    sel = yf > 0
    xf = xf[sel, :]
    yf = yf[sel]

    # subsample if requested
    if subsample_frac is not None:
        subsample_n = int(yf.shape[0] * subsample_frac)

    if subsample_n is not None:
        c = np.random.choice(range(yf.shape[0]), subsample_n)
        xf = xf[c, :]
        yf = yf[c]

    print("Training model.")
    _model = RandomForestClassifier(n_jobs=-1)
    _model.fit(xf, yf)

    if save_model:
        print("Saving model.")
        joblib.dump(_model, output_file)
    return _model


def predict_pixel_classifier(x: Array, model_file: Path) -> Array:
    """
    Predict pixel classes from image stacks (x) using trained model.

    Parameters
    ----------
    x: np.ndarray, list
        Array with shape BCXY (smaple,channel,X,Y) of signal
    model_file: pathlib.Path
        File with trained model to load.
    """
    import joblib

    print("Loading model.")
    _model = joblib.load(model_file)

    print("Extracting features.")
    xf = np.asarray(extract_features([xx for xx in x]))
    shape = xf.shape[:-1] + (3,)
    xf = xf.reshape((-1, xf.shape[-1]))
    # fill null pixels with 0
    xf[pd.isnull(xf)] = 0
    print("Predicting.")
    probs = _model.predict_proba(xf)
    probs = probs.reshape(shape)
    return probs


def stardist_segment_nuclei(image: Array, model_str: str = "2D_versatile_fluo") -> Array:
    from stardist.models import StarDist2D

    model = StarDist2D.from_pretrained(model_str)
    mask, _ = model.predict_instances(image)
    return mask.astype("uint")


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
    image: Array,
    tile_shape: tp.Tuple[int, int] = (256, 256),
    overlap: tp.Tuple[int, int] = (0, 0),
    pad_to_tile_shape: bool = False,
) -> tp.Dict[tp.Tuple[int, int], Array]:
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
                ((i + 1) * x - offsets[0]) if (i + 2) < n else shape[0] - start[0],
                ((j + 1) * y - offsets[1]) if (j + 2) < m else shape[1] - start[1],
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
    tiles: tp.Dict[tp.Tuple[int, int], Array], trim_all_zeros: bool = True
) -> Array:
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


def cellpose_segment(
    image: Array, compartment: tp.Union[tp.Literal["nuclear"], tp.Literal["cytoplasm"]]
) -> Array:
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


def plot_cellpose_output(
    image: Array, masks: tp.Sequence[Array], flows: tp.Sequence[Array]
) -> Figure:
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


def cellpose_postprocessing(image: Array, mask: Array, flow: Array):
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


def inflection_point(curve: tp.Sequence[float]) -> int:
    """Return the index of the inflection point of a curve"""
    from numpy.matlib import repmat

    n_points = len(curve)
    all_coord = np.vstack((range(n_points), curve)).T
    line_vec = all_coord[-1] - all_coord[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))
    vec_from_first = all_coord - all_coord[0]
    scalar_product = np.sum(vec_from_first * repmat(line_vec_norm, n_points, 1), axis=1)
    vec_to_line = vec_from_first - np.outer(scalar_product, line_vec_norm)
    return np.argmax(np.sqrt(np.sum(vec_to_line ** 2, axis=1)))


def plot_image_and_mask(image: Array, mask_dict: tp.Dict[str, Array]) -> Figure:
    """

    image: np.ndarray

    mask_dict: dict[str, np.ndarray]
    """
    cols = 1 + len(mask_dict)

    # Make RGB if multi channel
    if len(image.shape) == 3:
        if image.shape[0] == 2:
            image = np.stack([image[0], image[1], np.zeros(image.shape[1:])])
        image = np.moveaxis(image, 0, -1)

    for comp in mask_dict:
        mask_dict[comp] = np.ma.masked_array(mask_dict[comp], mask_dict[comp] == 0)

    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4), sharex=True, sharey=True)
    axes[0].imshow(image, rasterized=True)
    axes[0].set(title=f"Original signal")
    cmap = random_label_cmap()
    for ax in axes:
        ax.axis("off")
    for i, comp in enumerate(mask_dict):
        ax = axes[1 + i]
        ax.imshow(mask_dict[comp], cmap=cmap, interpolation="none")
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


def deepcell_segment_probabilities(probabilities: Array, compartment: str) -> Array:
    """"""
    from deepcell.applications import MultiplexSegmentation

    # translate compartments
    imc2dc = {
        "cell": "whole-cell",
        "cytoplasm": "whole-cell",
        "nuclear": "nuclear",
        "both": "both",
    }

    app = MultiplexSegmentation()
    pred: Array = app.predict(
        probabilities[np.newaxis, ..., :-1], compartment=imc2dc[compartment]
    )
    return pred.squeeze()


def deepcell_postprocess_both_compartments(
    mask: Array,
    dominant_nuclei_threshold: float = 0.8,
    plot: bool = False,
    roi: _roi.ROI = None,
    fig_output_prefix: Path = None,
    verbose: bool = False,
) -> Array:
    """
    Function to perform post-processing of DeepCell masks when using
    the MultiplexSegmentation model with the 'both' compartment.

    The major goal is the alignment of the cellular and nuclear masks
    as much as possible.
    """
    import seaborn as sns
    from skimage.segmentation import find_boundaries

    def inspect_mask_pair(
        cell_mask: Array, nuclei_mask: Array
    ) -> tp.Tuple[int, list, dict]:
        match = 0
        unpaireds = list()
        multiplets = dict()
        for cell in np.unique(cell_mask)[1:]:
            o = np.unique(nuclei_mask[cell_mask == cell])
            o = [x for x in o if x != 0]
            if not o:
                # cells with no nucleus
                unpaireds.append(cell)
            elif len(o) == 1:
                # cells with exactly one nucleus
                match += 1
            else:
                # cells with more than one nucleus
                t = (cell_mask == cell).sum()
                fractions = dict()
                for nuclei in o:
                    m = (nuclei_mask == nuclei) & (cell_mask == cell)
                    fractions[nuclei] = len(cell_mask[m]) / t
                s = sum(fractions.values())
                multiplets[cell] = {k: v / s for k, v in fractions.items()}
        return match, unpaireds, multiplets

    def apply_postprocess(
        cell_mask: Array,
        nuclei_mask: Array,
        max_iter: int = 5,
        complement_singlets: bool = True,
        remove_border_from_complement: bool = False,
        remove_ambigous: bool = True,
    ) -> tp.Tuple[Array, Array]:
        it = 0
        match, unpaireds, multiplets = inspect_mask_pair(cell_mask, nuclei_mask)
        while not ((len(unpaireds) == 0) and (len(multiplets) == 0)):
            if it == max_iter:
                if verbose:
                    print("Reached maximum number of iterations.")
                break
            match, unpaireds, multiplets = inspect_mask_pair(cell_mask, nuclei_mask)
            # # 1. fix cells without nuclei
            if complement_singlets:
                # # # set nuclei to the whole mask minus one
                max_c = nuclei_mask.max()
                n_new = np.zeros(nuclei_mask.shape, dtype=int)
                for i, cell in enumerate(unpaireds, 1):
                    nuclei_mask[cell_mask == cell] = max_c + i
                    n_new[cell_mask == cell] = max_c + i
                # remove 1 pixel from border
                if remove_border_from_complement:
                    b = find_boundaries(n_new, 1, mode="inner", background=0)
                    nuclei_mask[b] = 0
                if verbose:
                    print(f"Added {len(unpaireds)} nuclei.")

            # # 2. fix cells overlaping more than one nuclei
            # # # If the matching is above threshold (i.e. 80%) then remove non-dominant nuclei
            # # # Else remove nuclei and cell
            if remove_ambigous:
                match, unpaireds, multiplets = inspect_mask_pair(cell_mask, nuclei_mask)
                n_removed = 0
                for cell, props in multiplets.items():
                    if max(props.values()) > dominant_nuclei_threshold:
                        dominant = pd.Series(props).idxmax()
                        nuclei_mask[
                            (cell_mask == cell)
                            & (nuclei_mask != dominant)
                            & (nuclei_mask > 0)
                        ] = dominant
                    else:
                        m = cell_mask == cell
                        cell_mask[cell_mask == cell] = 0
                        m = m | (np.isin(cell_mask, props.keys()))
                        rem = np.unique(nuclei_mask[m])
                        nuclei_mask[np.isin(nuclei_mask, rem)] = 0
                        n_removed += 1
                if verbose:
                    print(f"Removed {n_removed} cells with ambiguous nuclei assignments.")
            it += 1
        return cell_mask, nuclei_mask

    if (mask.ndim != 3) and (mask.shape[-1] != 2):
        raise ValueError(
            "Array must contain more than one segmentation layer of shape "
            "(X,Y,2), the last dimension assumed to be cell and nuclei."
        )

    cell_mask, nuclei_mask = mask[..., 0].copy(), mask[..., 1].copy()

    # Inspect cell-nuclei relationship
    lc = len(np.unique(cell_mask)) - 1
    ln = len(np.unique(nuclei_mask)) - 1
    pre_match, pre_unpaireds, pre_multiplets = inspect_mask_pair(cell_mask, nuclei_mask)

    if verbose:
        print("Statistics prior postprocessing:")
        print(f"\tSegmentation has {lc} cells and {ln} nuclei.")
        print(f"\t{pre_match} exact matches between cells and nuclei.")
        print(f"\t{len(pre_unpaireds)} cells without nuclei.")
        print(f"\t{len(pre_multiplets)} cells overlaping more than one nuclei.")
        fracts = np.asarray([max(v.values()) for k, v in pre_multiplets.items()])
        print(
            "Found mean overlap between cells and nuclei in "
            f"multiplets to be {fracts.mean():.3f}(+/-){fracts.std():.3f}."
        )

    # Postprocess
    print("Applying postprocessing to segmentation masks...")
    cell_mask, nuclei_mask = apply_postprocess(
        cell_mask, nuclei_mask, remove_border_from_complement=True
    )
    nuclei_mask, cell_mask = apply_postprocess(
        nuclei_mask,
        cell_mask,
        complement_singlets=False,
        remove_ambigous=True,
        remove_border_from_complement=False,
    )
    cell_mask, nuclei_mask = apply_postprocess(
        cell_mask, nuclei_mask, remove_border_from_complement=True
    )

    # Report postprocessed stats:
    post_match, post_unpaireds, post_multiplets = inspect_mask_pair(
        cell_mask, nuclei_mask
    )
    if verbose:
        lc = len(np.unique(cell_mask)) - 1
        ln = len(np.unique(nuclei_mask)) - 1
        print("After postprocessing stats:")
        print(f"\tSegmentation has {lc} cells and {ln} nuclei.")
        print(f"\t{post_match} exact matches between cells and nuclei.")
        print(f"\t{len(post_unpaireds)} cells without nuclei.")
        print(f"\t{len(post_multiplets)} cells overlaping more than one nucleus.")

    # Report postprocessed stats:
    post_match, post_unpaireds, post_multiplets = inspect_mask_pair(
        nuclei_mask, cell_mask
    )
    if verbose:
        print(f"\t{post_match} exact matches between nuclei and cells.")
        print(f"\t{len(post_unpaireds)} nuclei without cells.")
        print(f"\t{len(post_multiplets)} nuclei overlaping more than one cell.")

    # # 3. align nuclei and cell integer IDs
    new_cell = np.zeros(cell_mask.shape, dtype="uint64")
    # could probably downcast these
    new_nuclei = np.zeros(cell_mask.shape, dtype="uint64")
    for i, cell in enumerate(np.unique(cell_mask)[1:], 1):
        new_cell[cell_mask == cell] = i
        # # find out number of nuclei mask:
        sel = np.unique(nuclei_mask[cell_mask == cell])
        try:
            old = [x for x in sel if x != 0][0]
            new_nuclei[nuclei_mask == old] = i
        except IndexError:
            pass
    cell_mask = new_cell
    nuclei_mask = new_nuclei

    if plot:
        reason = "If `plot`, `fig_output_prefix` must be given!"
        assert fig_output_prefix is not None, reason
        rows = 3 if roi is None else 4
        fig, axes = plt.subplots(
            rows, 2, figsize=(2 * 4, rows * 4), sharex=True, sharey=True
        )
        cmap = random_label_cmap()
        for i, (tmpmask, layer) in enumerate(
            [(mask[..., 0], "Cell mask"), (mask[..., 1], "Nuclei mask")]
        ):
            tmpmask = np.ma.masked_array(tmpmask, mask=tmpmask == 0)
            axes[0][i].imshow(tmpmask, cmap=cmap, interpolation="none")
            axes[0][i].set(title=f"Prior: {layer}")

        c_unpaired = mask[..., 0].copy()
        c_unpaired[~np.isin(mask[..., 0], np.unique(np.asarray(pre_unpaireds)))] = 0
        c_multiplets = mask[..., 0].copy()
        c_multiplets[~np.isin(mask[..., 0], np.unique(list(pre_multiplets.keys())))] = 0
        axes[1][0].imshow(c_multiplets, cmap=cmap, interpolation="none")
        axes[1][0].set(title="Cells overlapping multiple nuclei")
        axes[1][1].imshow(c_unpaired, cmap=cmap, interpolation="none")
        axes[1][1].set(title="Cells without nuclei")

        for i, (tmpmask, layer) in enumerate(
            [(cell_mask, "Cell mask"), (nuclei_mask, "Nuclei mask")]
        ):
            tmpmask = np.ma.masked_array(tmpmask, mask=tmpmask == 0)
            axes[2][i].imshow(tmpmask, cmap=cmap, interpolation="none")
            axes[2][i].set(title=f"Post: {layer}")

        if roi is not None:
            try:
                for ax, ch in zip(axes[3], ["DNA", "mean"]):
                    ax.imshow(
                        np.log1p(roi._get_channel(ch, dont_warn=True)[1]),
                        interpolation="bilinear",
                    )
                    ax.set(title=ch)
            except ValueError:
                pass
        for ax in axes.flat:
            ax.axis("off")
        fig.savefig(
            fig_output_prefix + "deepcell.cell-nuclei_relationships.svg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # # Visualize fraction of cells with clear dominant vs ambiguous nuclei
    if plot:
        assert fig_output_prefix is not None, reason
        fig, ax = plt.subplots()
        sns.histplot(fracts, ax=ax)
        ax.axvline(dominant_nuclei_threshold, color="grey", linestyle="--")
        ax.set(
            title="Cells overlapping more than one nuclei",
            xlabel="Fraction of dominant nuclei in cell",
            ylabel="Cell count",
        )
        fig.savefig(
            fig_output_prefix
            + "deepcell.cell-nuclei.dominant_nuclei_area_distribution.svg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    return np.moveaxis(np.stack([cell_mask, nuclei_mask]), 0, -1)


def segment_roi(
    roi: _roi.ROI,
    from_probabilities: bool = False,
    model: tp.Union[tp.Literal["deepcell"], tp.Literal["stardist"]] = "deepcell",
    compartment: str = "nuclear",
    postprocessing: bool = True,
    save: bool = True,
    overwrite: bool = True,
    plot_segmentation: bool = True,
    verbose: bool = False,
) -> tp.Dict[str, Array]:
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
    if from_probabilities:
        assert model in ["deepcell"]
        image = roi.probabilities
        shape = tuple([int(x / 2) for x in image.shape[1:]])
        image = resize(image / image.max(), (3,) + shape)
        mask = deepcell_segment_probabilities(np.moveaxis(image, 0, -1), compartment)
        if postprocessing and compartment in ["both"]:
            mask = deepcell_postprocess_both_compartments(
                mask,
                plot=False,
                roi=roi,
                fig_output_prefix=roi.get_input_filename("stack").replace_(
                    ".tiff", "_segmentation."
                ),
                verbose=verbose,
            )
    else:
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
        for comp, mask in mask_dict.items():
            mask_file = roi.get_input_filename(comp + "_mask")
            if overwrite or (not overwrite and not mask_file.exists()):
                tifffile.imwrite(mask_file, mask)

    if plot_segmentation:
        fig = plot_image_and_mask(image, mask_dict)
        if save:
            fig_file = roi.get_input_filename("stack").replace_(
                ".tiff",
                f"_segmentation_{model}_{compartment}.svg",
            )
            if overwrite or (not overwrite and not fig_file.exists()):
                fig.savefig(fig_file, dpi=300, bbox_inches="tight")

    return mask_dict


# def segment_decomposition():
#     """Some experiments with tensor decomposition"""
#     import tensorly as tl
#     from tensorly.decomposition import tucker, non_negative_tucker, parafac

#     from imc.demo import generate_disk_masks, generate_stack

#     # Synthetic data
#     Y = np.asarray([generate_disk_masks()[np.newaxis, ...] for _ in range(100)])
#     X = np.asarray([generate_stack(m.squeeze()) for m in Y])

#     fig, axes = plt.subplots(1, 3)
#     for ax, a in zip(axes, X[0]):
#         ax.imshow(a)

#     core, factors = tucker(X[0], [1, 128, 128])
#     core, factors = non_negative_tucker(X[0], [1, 128, 128])
#     reconstruction = tl.tucker_to_tensor((core, factors))

#     fig, axes = plt.subplots(1, 5)
#     for ax, a in zip(axes, X[0]):
#         ax.imshow(a)
#     q = factors[1] @ factors[2].T
#     axes[-2].imshow(q)
#     axes[-1].imshow(q > 1)

#     factors[0]  # samples
#     factors[1]  # channels

#     # if decomposing in more than one factor:
#     fig, axes = plt.subplots(1, 4)
#     for ax, dim in zip(axes, factors):
#         ax.scatter(*dim.T, s=2, alpha=0.5)

#     fig, axes = plt.subplots(1, 4)
#     for ax, dim in zip(axes, factors):
#         ax.plot(dim)

#     # Real data
#     r = prj.rois[50]

#     exc = r.channel_exclude.index.str.contains("EMPTY|80|129")
#     stack = normalize(r.stack[~exc])

#     core, factors = tucker(stack, [3] + list(r.shape[1:]))
#     reconstruction = tl.tucker_to_tensor((core, factors))

#     fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
#     axes[0][0].imshow(stack.mean(0))
#     axes[0][1].imshow(reconstruction.mean(0))
#     axes[1][0].imshow(stack[-1])
#     v = np.absolute(reconstruction[-1]).max()
#     axes[1][1].imshow(reconstruction[-1], cmap="RdBu_r", vmin=-v, vmax=v)  # .clip(min=0))

#     channel_imp = pd.Series(factors[0].squeeze(), index=r.channel_labels[~exc])
#     channel_mean = pd.Series(stack.mean((1, 2)), index=r.channel_labels[~exc])

#     plt.scatter(channel_mean, channel_imp)

#     fig, axes = plt.subplots(1, 8)
#     for ax, a in zip(axes, X[0]):
#         ax.imshow(a)
#     q = factors[1] @ factors[2].T
#     axes[-2].imshow(q)
#     axes[-1].imshow(q > 1)

#     probs = zoom(r.probabilities, (1, 0.5, 0.5))
#     probs = probs / probs.max()
#     core, factors = tucker(probs, [3] + list(r.shape[1:]))
#     reconstruction = tl.tucker_to_tensor((core, factors))

#     fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
#     axes[0].imshow(np.moveaxis(probs, 0, -1))
#     axes[1].imshow(np.moveaxis(reconstruction, 0, -1))

#     weights, factors = parafac(probs, rank=50, init="random", tol=10e-6)
#     cp_reconstruction = tl.cp_to_tensor((weights, factors))

#     plt.imshow(np.moveaxis(minmax_scale(cp_reconstruction), 0, -1))

#     def PCA2D_2D(samples, row_top, col_top):
#         """samples are 2d matrices"""
#         size = samples[0].shape
#         # m*n matrix
#         mean = np.zeros(size)

#         for s in samples:
#             mean = mean + s

#         # get the mean of all samples
#         mean /= float(len(samples))

#         # n*n matrix
#         cov_row = np.zeros((size[1], size[1]))
#         for s in samples:
#             diff = s - mean
#             cov_row = cov_row + np.dot(diff.T, diff)
#         cov_row /= float(len(samples))
#         row_eval, row_evec = np.linalg.eig(cov_row)
#         # select the top t evals
#         sorted_index = np.argsort(row_eval)
#         # using slice operation to reverse
#         X = row_evec[:, sorted_index[: -row_top - 1 : -1]]

#         # m*m matrix
#         cov_col = np.zeros((size[0], size[0]))
#         for s in samples:
#             diff = s - mean
#             cov_col += np.dot(diff, diff.T)
#         cov_col /= float(len(samples))
#         col_eval, col_evec = np.linalg.eig(cov_col)
#         sorted_index = np.argsort(col_eval)
#         Z = col_evec[:, sorted_index[: -col_top - 1 : -1]]

#         return X, Z


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
