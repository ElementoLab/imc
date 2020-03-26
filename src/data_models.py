import os
from typing import Dict, List, Iterable, Optional

from os.path import join as pjoin

import numpy as np
import pandas as pd


from utils import read_image_from_file, get_grid_dims, get_transparent_cmaps


FIG_KWS = dict(dpi=300, bbox_inches="tight")


class Project():
    """
    A class to model a IMC project.

    Parameters
    ----------
    name : :obj:`str`
        Project name. Defaults to "project".
    csv_metadata : :obj:`str`
        Path to CSV metadata sheet.

    Attributes
    ----------
    name : :obj:`str`
        Project name
    csv_metadata : :obj:`str`
        Path to CSV metadata sheet.
    metadata : :class:`pandas.DataFrame`
        Metadata dataframe
    samples : List[:class:`IMCSample`]
        List of IMC sample objects.
    """
    def __init__(
            self,
            name: str = "project",
            csv_metadata: str = None,
            sample_name_attribute: str = "acquisition_date"):
        # Initialize
        self.name = name
        self.csv_metadata = csv_metadata
        self.samples: List[IMCSample] = list()
        if self.csv_metadata is not None:
            self.initialize_project_from_annotation()

    def __repr__(self):
        return f"IMC Project '{self.name}' with '{len(self.samples)}' samples"

    def initialize_project_from_annotation(self):
        self.metadata = pd.read_csv(self.csv_file)

        self.samples = list()
        for row in self.metadata.iterrows():
            s = IMCSample(
                root_dir=pjoin("processed", str(row[self.sample_name_attribute])),
                **row.to_dict())
            self.samples.append(s)


class IMCSample():
    sample_name: str
    name: str  # sample_name shorthand
    root_dir: str
    panel_file: str
    panel: pd.DataFrame  # from read_panel
    rois: List[ROI]
    n_rois: int
    channel_labels: pd.Series  # from get_channel_labels
    xdim: Optional[int]
    ydim: Optional[int]
    # paths
    plots_dir: str

    def __init__(self, **kwargs):
        # Add kwargs as attributes
        self.__dict__.update(kwargs)

        # Check correct construction
        mandatory_attributes = [
            "sample_name",
            "root_dir",
            "panel_file"]
        missing = [attr for attr in mandatory_attributes if not hasattr(self, attr)]
        if missing:
            msg = "IMCSample must be initialized with the '%s' arguments!" % ", ".join(missing)
            raise ValueError(msg)

        # Set attributes
        if self.panel_file is not None:
            self.read_panel()
        self.name = self.sample_name

        # Make paths absolute
        for path in ['root_dir', 'panel_file']:
            setattr(self, path, os.path.abspath(getattr(self, path)))

        # Create ROIs
        if hasattr(self, "roi_names"):
            self.rois = list()
            if isinstance(getattr(self, "roi_names"), str):
                self.roi_names = self.roi_names.split(",")
            for roi in self.roi_names:
                self.rois.append(
                    ROI(**dict(
                        roi_name=roi,
                        root_dir=pjoin(self.root_dir, roi),
                        sample=self)))
        if not hasattr(self, "rois"):
            self.rois = list()

    def __repr__(self):
        return f"IMC Sample '{self.name}' with '{len(self.rois)}' ROIs"

    def read_panel(self, panel_file: Optional[str] = None) -> pd.DataFrame:
        """
        Read panel CSV file for sample.

        Parameters
        ----------
        panel_file : {str}, optional
            CSV file for panel.
            Defaults to the `panel_file` value.
        """
        self.panel = pd.read_csv(panel_file or self.panel_file)
        return self.panel

    def get_channel_labels(
            self,
            panel: Optional[pd.DataFrame] = None) -> pd.Series:
        """Get an order of channel labels from sample panel."""
        channel_labels = (
            (panel or self.panel)
            .query("full == 1")
            .index.to_series().rename("channels"))
        return channel_labels


class ROI:
    roi_name: str
    root_dir: str
    name: str  # roi_name shorthand
    sample: IMCSample  # from parent
    xdim: int
    ydim: int

    # paths
    plots_dir: str

    # data attributes
    matrix: np.ndarray
    features: np.ndarray
    probabilities: np.ndarray
    uncertainty: np.ndarray
    mask: np.ndarray

    def __init__(self, **kwargs):
        # Add kwargs as attributes
        self.__dict__.update(kwargs)

        # Check correct construction
        mandatory_attributes = [
            "roi_name",
            "root_dir"]
        missing = [attr for attr in mandatory_attributes if not hasattr(self, attr)]
        if missing:
            msg = "ROI must be initialized with the '%s' arguments!" % ", ".join(missing)
            raise ValueError(msg)

        self.name = self.roi_name

    def __repr__(self):
        return f"IMC Sample '{self.name}'"

    def read_all_inputs(
            self,
            only_these_keys: list = None,
            permissive: bool = True) -> None:
        """Reads in all sample-wise inputs:
            - raw matrix
            - extracted features
            - probabilities
            - uncertainty
            - segmentation mask.
        If `permissive` is :obj:`True`, skips non-existing inputs."""
        # the pattens is {"input_type": (directory, suffix, args)}
        to_read = {
            "matrix": ("tiffs", "_full.tiff", dict()),
            "features": ("tiffs", "_ilastik_s2_Features.h5", dict()),
            "probabilities": ("tiffs", "_ilastik_s2_Probabilities.tiff", dict(equalize=False)),
            "uncertainty": ("uncertainty", "_ilastik_s2_Probabilities_uncertainty.tiff", {}),
            "mask": ("cpout", "_ilastik_s2_Probabilities_mask.tiff", dict(equalize=False))}
        if only_these_keys is None:
            only_these_keys = to_read.keys()
        for ftype, (dir_, suffix, params) in to_read.items():
            if ftype not in only_these_keys:
                continue

            path = pjoin(self.sample.root_dir, dir_, self.sample.name + "_" + self.name + suffix)
            try:
                # logger.info()
                setattr(self, ftype, read_image_from_file(path, **params))
            except FileNotFoundError:
                if permissive:
                    continue
                else:
                    raise

    def get_distinct_marker_sets(self, n: int = 4, save_plot=False) -> Dict[int, Iterable[str]]:
        """Use cross-channel correlation to pick `n` clusters of distinct channels to overlay"""
        import seaborn as sns
        import scipy

        arr_flat = self.matrix.reshape((self.matrix.shape[0], -1))
        xcorr = pd.DataFrame(arr_flat, index=self.channel_labels).T.corr()
        np.fill_diagonal(xcorr.values, 0)

        grid = sns.clustermap(
            xcorr, cmap="RdBu_r", center=0, metric="correlation",
            cbar_kws=dict(label="Pearson correlation"))
        grid.ax_col_dendrogram.set_title("Pairwise channel correlation")
        if save_plot:
            grid.savefig(
                pjoin(self.root_dir, f"channel_pairwise_correlation.svg"), **FIG_KWS)

        c = pd.Series(
            scipy.cluster.hierarchy.fcluster(
                grid.dendrogram_col.linkage, n, criterion='maxclust'),
            index=xcorr.index)

        marker_sets = dict()
        for sp in range(1, n + 1):
            marker_sets[sp] = list()
            for i in np.random.choice(np.unique(c), 4, replace=False):
                marker_sets[sp].append(np.random.choice(c[c == i].index, 1, replace=False)[0])
        return marker_sets

    def plot_overlayied_channels_subplots(self, subplots):
        import matplotlib.pyplot as plt
        from matplotlib import mpatches
        marker_sets = self.get_distinct_marker_sets(self.matrix, n=subplots)

        n, m = get_grid_dims(self.matrix.shape[0])
        fig, axis = plt.subplots(n, m, figsize=(6 * m, 6 * n), sharex=True, sharey=True, squeeze=False)
        axis = axis.flatten()
        for i, (marker_set, mrks) in enumerate(marker_sets.items()):
            patches = list()
            cmaps = get_transparent_cmaps(len(mrks))
            for j, (m, c) in enumerate(zip(mrks, cmaps)):
                x = self.matrix[self.channel_labels == m, :, :].squeeze()
                v = x.mean() + x.std() * 2
                axis[i].imshow(x, cmap=c, vmin=0, vmax=v, label=m, interpolation="bilinear", rasterized=True)
                axis[i].axis('off')
                patches.append(mpatches.Patch(color=c(256), label=m))
            axis[i].legend(
                handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                title=marker_set)
