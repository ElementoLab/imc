#! /usr/bin/env python

import os
import sys
import logging

from joblib import Memory
import matplotlib.pyplot as plt
import seaborn as _sns

from imc.graphics import colorbar_decorator

plt.rcParams["svg.fonttype"] = "none"


def setup_logger(level=logging.INFO):
    logger = logging.getLogger("imcpipeline")
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


LOGGER = setup_logger()

# Setup joblib memory
JOBLIB_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".imcpipeline")
MEMORY = Memory(location=JOBLIB_CACHE_DIR, verbose=0)

# Decorate seaborn clustermap
_sns.clustermap = colorbar_decorator(_sns.clustermap)


from imc.data_models.project import Project
from imc.data_models.sample import IMCSample
from imc.data_models.roi import ROI
