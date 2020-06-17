#! /usr/bin/env python

from __future__ import annotations  # fix the type annotatiton of not yet undefined classes
import os
import sys
import logging
from functools import partialmethod
from pathlib import Path as _Path

from joblib import Memory
import matplotlib
import matplotlib.pyplot as plt
import seaborn as _sns

from imc.graphics import colorbar_decorator

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["text.usetex"] = False


def setup_logger(name: str = "imc", level: int = logging.INFO) -> logging.Logger:
    """Setup the logger for the package."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


LOGGER = setup_logger()

# Setup joblib memory
JOBLIB_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".imc")
MEMORY = Memory(location=JOBLIB_CACHE_DIR, verbose=0)

# Decorate seaborn clustermap
_sns.clustermap = colorbar_decorator(_sns.clustermap)


_Path.mkdir = partialmethod(_Path.mkdir, exist_ok=True, parents=True)


from imc.data_models.project import Project
from imc.data_models.sample import IMCSample
from imc.data_models.roi import ROI
