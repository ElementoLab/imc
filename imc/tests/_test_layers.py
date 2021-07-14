import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile
import anndata
import scipy.ndimage as ndi

from imc import Project
from imc.graphics import random_label_cmap

layer_names = ["cell", "nuclei", "cytoplasm", "membrane", "extracellular"]

prj = Project()

roi = prj.rois[25]
fig, axes = plt.subplots(1, 5, figsize=(5 * 4, 4), sharex=True, sharey=True)
cmap = random_label_cmap()
for i, layer in enumerate(layer_names):
    mask = getattr(roi, layer + "_mask")
    mask = np.ma.masked_array(mask, mask=mask == 0)
    axes[i].imshow(mask, cmap=cmap)
    axes[i].set(title=layer)
    axes[i].axis("off")


prj.rois = prj.rois[25:27]
quant = prj.quantify_cells(layers=layer_names, set_attribute=False)


quant = quant.reset_index().melt(
    id_vars=["roi", "obj_id", "layer"], var_name="channel"
)
quant = quant.pivot_table(
    index=["roi", "obj_id"], columns=["layer", "channel"], values="value"
)
quant = quant.reset_index()

X = quant.loc[:, layer_names[0]]
obs = quant[["roi", "obj_id"]]
obs["in_tissue"] = 1
obs["array_row"] = ...
obs["array_col"] = ...
obs.columns = ["roi", "obj_id"]
layers = quant.loc[:, layer_names[1:]]

a = anndata.AnnData(
    X=X.reset_index(drop=True),
    obs=obs,
    layers={l: layers[l] for l in layer_names[1:]},
)

a = anndata.AnnData(X=quant.drop(["roi", "obj_id"], 1), obs=obs)

for roi in prj.rois:
    a.uns["spatial"][roi.name] = {
        "images": {"hires": roi.stack},
        "metadata": {},
        "scalefactors": {
            "spot_diameter_fullres": 89.56665687930325,
            "tissue_hires_scalef": 0.150015,
            "fiducial_diameter_fullres": 144.6845995742591,
            "tissue_lowres_scalef": 0.045004502,
        },
    }


sc.pp.log1p(a)
sc.pp.scale(a)
sc.pp.pca(a)
