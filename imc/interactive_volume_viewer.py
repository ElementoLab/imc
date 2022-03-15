#!/usr/bin/env python

"""
An example program to display a volumetric image from the command line.
"""

import sys
import typing as tp
from urlpath import URL
from functools import partial

import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from imc.types import Array, Axis, Figure, Path  # https://github.com/ElementoLab/imc


def multi_slice_viewer(
    volume: Array, up_key: str = "w", down_key: str = "s", **kwargs
) -> Figure:
    remove_keymap_conflicts({up_key, down_key})
    print(f"Press '{up_key}' and '{down_key}' for scrolling through image channels.")

    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index], **kwargs)
    fig.canvas.mpl_connect(
        "key_press_event", partial(process_key, up_key=up_key, down_key=down_key)
    )
    return fig


def remove_keymap_conflicts(new_keys_set: tp.Set) -> None:
    for prop in plt.rcParams:
        if prop.startswith("keymap."):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def process_key(event, up_key: str = "w", down_key: str = "s") -> None:
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == up_key:
        previous_slice(ax)
    elif event.key == down_key:
        next_slice(ax)
    fig.canvas.draw()


def previous_slice(ax: Axis) -> None:
    """Go to the previous slice."""
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])


def next_slice(ax: Axis) -> None:
    """Go to the next slice."""
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])


def get_volume() -> Array:
    base_url = URL("https://prod-images-static.radiopaedia.org/images/")
    start_n = 53734044
    length = 137

    imgs = list()
    for i in tqdm(range(length)):
        url = base_url / f"{start_n + i}/{i + 1}_gallery.jpeg"
        resp = url.get()
        c = resp.content
        imgs.append(imageio.read(c, format="jpeg").get_data(0))
    img = np.asarray(imgs)
    return img


def main() -> int:
    """
    Run
    """
    img_file = Path("/tmp/volumetric_image.npz")
    if not img_file.exists():
        print("Downloading volumetric image.")
        img = get_volume()
        np.savez_compressed(img_file, img)
    else:
        img = np.load(img_file)["arr_0"]

    _ = multi_slice_viewer(img)
    print("Displaying volume.")
    print("Press 'w' for up and 's' for down.")
    plt.show(block=True)
    print("Done.")
    return 0


if __name__ == "__main__" and "get_ipython" not in locals():
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
