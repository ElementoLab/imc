---
title: Hyperion CyTOF@EIPM
created: '2020-03-13'
modified: '2020-05-27'
---

# `imc` package


1. Method to map clusters between samples/groups of ROIs
1. Adapt `roi._get_file_names` to support subfolder_per_sample=False
2. Fix Project.plot_channels

1. Implement pharmacoscopy method for interactions
1. Decorator to enforce type str -> Path in functions
3. Reveamp ROI.plot_cell_types
3. Revisit `panel_metadata` attribute, probably retire.
2. For GMM fit several `k` choose optimal e.g. https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html
3. Use GMM for thresholding cluster means during cell type labeling


4. Clustermap colorbars: try to add new axes to the figure; add kwargs for the colorbar_kws; fix NAN

Real test data:
 - change channel names to unique or support repeated


## Tasks

1. Pipeline
  - Replace second CP call with custom numpy code


1. Data storage in object
  - xarray, zarr

1. Extract manual labels from ilastik models
  - extract positions
  - relate to original large image, not crop
  - use for validation of VAE or whatever  

0. Quantification
  - Filter small cells?
  - Add shape/eccentricity to quantification step by default?
1. Cell type assignment:
  - Use shape/eccentricity for clustering? try
  - Be more stringent with which channels are used for clustering?
2. Cell neighbor network
  - revisit the supercommunity reduction by making it more specific (not overal stringent!)
  - find a way to plot colors consistently across samples/roi for the same label with new RGB plotting method

0. pipeline improvements:
  - replace second cellprofiler call with own custom code
  - image/roi/channel filtering
    - this could lead to automated channel selection for pipeline input (panel.csv)
  - drift correction?
  - channel imputation?
  - plot ROI coordinate origin of top of slide image

0. VAE or similar for unsupervised object detection/tracking

### Bonus

1. Unify data from various regions of interest (ROI) on same slide
