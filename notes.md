---
title: Hyperion CyTOF@EIPM
created: '2020-03-13'
modified: '2020-05-27'
---

# Hyperion CyTOF@EIPM


## Tasks

1. Pipeline
  - Replace second CP call with custom numpy code
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
