---
title: Hyperion CyTOF@EIPM
created: '2020-03-13'
modified: '2020-05-27'
---

# `imc` package


1. Tests:
 - make sure every object is serializable e.g. parmap compatible
1. Add method to plot channels overlayed in ROI.
2. Implement Sample.plot_channels
1. Decorator to enforce type str -> Path in functions
3. Reveamp ROI.plot_cell_types
2. For GMM fit several `k` choose optimal e.g. https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html


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
