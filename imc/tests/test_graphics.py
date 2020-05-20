#!/usr/bin/env python

import numpy as np
import pandas as pd
import seaborn as sns
import pytest


class Test_colorbar_decorator:
    def test_random_plot(self):
        x = pd.DataFrame(np.random.random((10, 5))).rename_axis(index="rows", columns="columns")
        g = sns.clustermap(x, row_colors=x.mean(1), col_colors=x.mean(0))
        assert len(g.fig.get_axes()) == 8
