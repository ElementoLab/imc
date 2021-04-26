#!/usr/bin/env python

from typing import Tuple, List, Dict, Union
import tempfile

import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import tifffile
import pandas as pd
import skimage

from imc import Project
from imc.types import Array, Figure, Path
from imc.utils import filter_kwargs_by_callable as filter_kws


def generate_mask(
    shape: Tuple[int, int] = (8, 8),
    seeding_density: float = 0.1,
    # widths: int = None,
    # connectivity: float = None
) -> Array:
    mask = np.zeros(shape, dtype=bool)
    # Cells are placed in an effective mask area which is not touching borders
    eff_mask = mask[1:-1, 1:-1]
    centroids = np.random.choice(
        np.arange(eff_mask.size),
        int(np.ceil(eff_mask.size * seeding_density)),
        replace=False,
    )
    eff_mask.flat[centroids] = True  # type: ignore
    mask[1:-1, 1:-1] = eff_mask
    return ndi.label(mask, structure=np.zeros((3, 3)))[0]


def generate_disk_masks(
    shape: Tuple[int, int] = (128, 128),
    seeding_density: float = 0.1,
    disk_diameter: int = 10,
):
    mask = np.zeros(shape, dtype=bool)

    area = np.multiply(*mask.shape)
    n = int(np.ceil(mask.size * seeding_density) * (disk_diameter ** 2 / area))
    centroids = np.random.choice(np.arange(mask.size), n, replace=False)

    r = disk_diameter // 2
    disk = skimage.morphology.disk(r)
    x = centroids // shape[0]
    y = centroids % shape[1]
    for i in range(n):
        s = mask[x[i] - r : x[i] + r + 1, y[i] - r : y[i] + r + 1].shape
        mask[x[i] - r : x[i] + r + 1, y[i] - r : y[i] + r + 1] = disk[
            : s[0], : s[1]
        ]
    return ndi.label(mask)[0]


def generate_stack(
    mask: Array,
    n_channels: int = 3,
    channel_coeffs: Array = None,
    channel_std: Array = None,
    n_cell_types: int = 2,
    cell_type_coeffs: Array = None,
    cell_type_std: Array = None,
) -> Array:
    # partition cells into cell types
    n_cells = (mask > 0).sum()
    cells = np.arange(mask.size)[mask.flat > 0]
    assigned_cells = np.array([], dtype=int)
    ct_cells = dict()
    for i in range(n_cell_types):
        available_cells = [c for c in cells if c not in assigned_cells]
        ct_cells[i] = np.random.choice(
            available_cells,
            int(np.floor(n_cells / n_cell_types)),
            replace=False,
        )
        assigned_cells = np.append(assigned_cells, ct_cells[i])
    ct_cells[i] = np.append(ct_cells[i], cells[~np.isin(cells, assigned_cells)])
    assert sum([len(x) for x in ct_cells.values()]) == n_cells

    # assign intensity values
    stack = np.zeros((n_channels,) + mask.shape, dtype=float)
    std_sd = 0.1
    if channel_coeffs is None:
        channel_coeffs = np.random.choice(np.linspace(-5, 5), n_channels)
    if channel_std is None:
        channel_std = np.abs(channel_coeffs) * std_sd
    if cell_type_coeffs is None:
        cell_type_coeffs = np.random.choice(np.linspace(-5, 5), n_cell_types)
    if cell_type_std is None:
        cell_type_std = np.abs(cell_type_coeffs) * std_sd
    # means = intercept + np.dot(
    means = np.dot(
        channel_coeffs.reshape((-1, n_channels)).T,
        cell_type_coeffs.reshape((-1, n_cell_types)),
    )
    intercept = np.abs(means.min()) * 2
    means += intercept
    stds = channel_std.reshape((-1, n_channels)).T + cell_type_std.reshape(
        (-1, n_cell_types)
    )

    for cell_type in range(n_cell_types):
        n = ct_cells[i].size
        for channel in range(n_channels):
            stack[channel].flat[ct_cells[cell_type]] = np.random.normal(
                means[channel, cell_type], stds[channel, cell_type], n
            )

    # make sure array is non-negative
    if stack.min() < 0:
        stack[stack == 0] = stack.min()
        stack += abs(stack.min())
    return stack


def write_tiff(array: Array, output_file: Path) -> None:
    fr = tifffile.TiffWriter(output_file)
    fr.save(array)
    fr.close()


def write_roi_to_disk(mask: Array, stack: Array, output_prefix: Path) -> None:
    # mask
    write_tiff(mask, output_prefix + "_full_mask.tiff")
    # stack
    write_tiff(stack, output_prefix + "_full.tiff")
    # channel_labels
    labels = [str(c).zfill(2) for c in range(1, stack.shape[0] + 1)]
    channel_labels = pd.Series(
        [f"Ch{c}(Ch{c})" for c in labels], name="channel"
    )
    channel_labels.to_csv(output_prefix + "_full.csv")


def visualize_roi(mask: Array, stack: Array) -> Figure:
    fig, axes = plt.subplots(1, 5, figsize=(4 * 5, 4))
    axes[0].set_title("Mask")
    axes[0].imshow(mask, cmap="binary_r")
    axes[1].set_title("RGB signal")
    axes[1].imshow(np.moveaxis(stack, 0, -1) / stack.max())
    for i, (ax, cmap) in enumerate(zip(axes[2:], ["Reds", "Greens", "Blues"])):
        ax.set_title(f"Channel {i}")
        ax.imshow(stack[i] / stack.max(), cmap=cmap)
    return fig


def generate_project(
    name: str = None,
    n_samples: int = 3,
    rois_per_sample: int = 3,
    root_dir: Path = None,
    sample_names: List[str] = None,
    return_object: bool = True,
    visualize: bool = False,
    **kwargs,
) -> Union[Project, Path]:
    if name is None:
        name = "test_project"
    if root_dir is None:
        root_dir = Path(tempfile.mkdtemp())
    else:
        root_dir = Path(root_dir)
    root_dir.mkdir(exist_ok=True)
    meta_dir = root_dir / "metadata"
    meta_dir.mkdir(exist_ok=True)
    processed_dir = root_dir / "processed"
    processed_dir.mkdir(exist_ok=True)

    if sample_names is None:
        sample_names = [
            "test_sample_" + str(i).zfill(2) for i in range(1, n_samples + 1)
        ]
    _meta: Dict[str, Dict[str, Union[str, int]]] = dict()
    for sample in sample_names:
        tiffs_dir = processed_dir / sample / "tiffs"
        tiffs_dir.mkdir(exist_ok=True, parents=True)
        for roi in range(1, rois_per_sample + 1):
            roi_name = f"{sample}-{str(roi).zfill(2)}"
            output_prefix = tiffs_dir / roi_name
            mask = generate_mask(**filter_kws(kwargs, generate_mask))
            stack = generate_stack(mask, **filter_kws(kwargs, generate_stack))
            if visualize:
                visualize_roi(mask, stack)
            write_roi_to_disk(mask, stack, output_prefix)
            _meta[roi_name] = {"roi_number": roi, "sample_name": sample}

    # write metadata
    meta = pd.DataFrame(_meta).T
    meta.index.name = "roi_name"
    meta.to_csv(meta_dir / "samples.csv")
    return (
        Project(
            metadata=meta_dir / "samples.csv",
            processed_dir=processed_dir,
            results_dir=processed_dir.parent / "results",
        )
        if return_object
        else root_dir
    )
