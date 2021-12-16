#!/usr/bin/env python

import numpy as np

from matplotlib.image import AxesImage
from matplotlib.legend import Legend


class TestCellTypePlotting:
    def test_clusters_labeled_with_numbers(self, project_with_clusters):
        p = project_with_clusters

        # # make pattern: "int (1-based) - str"
        c = (p.clusters + 1).astype(str) + " - " + (p.clusters + 1).astype(str)
        p.set_clusters(c)

        # Plot both clusters
        roi = p.rois[0]
        fig1 = roi.plot_cell_types()

        # Remove first cluster
        c2 = roi.clusters.copy()
        for e in c2.index:
            c2[e] = roi.clusters.max()
        roi.set_clusters(c2)
        fig2 = roi.plot_cell_types()

        # Get arrays back from images
        a1 = [i for i in fig1.axes[0].get_children() if isinstance(i, AxesImage)]
        a1 = [a for a in a1 if len(a.get_array().shape) == 3][0].get_array()
        a2 = [i for i in fig2.axes[0].get_children() if isinstance(i, AxesImage)]
        a2 = [a for a in a2 if len(a.get_array().shape) == 3][0].get_array()

        # Get legend of second image
        l2 = [i for i in fig2.axes[0].get_children() if isinstance(i, Legend)][0]

        # Get color of legend patch (RGBA)
        lc = l2.get_patches()[0].get_facecolor()[:-1]
        # Get color from array (should be only one besides black)
        _t = a2.reshape((8 * 8, 3))
        ac = _t[_t.sum(1) > 0][0]

        assert np.equal(ac, lc).all()
