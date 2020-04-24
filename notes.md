---
title: Hyperion CyTOF@EIPM
created: '2020-03-13T20:49:04.079Z'
modified: '2020-04-24T22:50:25.007Z'
---

# Hyperion CyTOF@EIPM

## Tasks



1. Pipeline
  Replace second CP call with custom numpy code
0. Quantification
  Filter small cells?
  Add shape/eccentricity to quantification step by default?
1. Cell type assignment:
  Use shape/eccentricity for clustering? try
  Be more stringent with which channels are used for clustering?
2. Cell neighbor network
  revisit the supercommunity reduction by making it more specific (not overal stringent!)
  find a way to plot colors consistently across samples/roi for the same label with new RGB plotting method

0. pipeline improvements:
  replace second cellprofiler call with own custom code
  image/roi/channel filtering
     this could lead to automated channel selection for pipeline input (panel.csv)
  drift correction?
  channel imputation?
  plot ROI coordinate origin of top of slide image
  
0. VAE or similar for unsupervised object detection/tracking

### Bonus
1. Unify data from various regions of interest (ROI) on same slide

## Ask
 - Panorama?
 - Is CD45 just that or CD45RA? Ie. sometimes there's no signal in clearly immune cells
 - ROI 9: uneven staining, what's the practical thing to do?

## Backburner:
 - get more data
 - re-train model:
    - 20191212: add crops to train data, make sure uncertainty is uniform
    - 20191025: ROI 3: very small dense nuclei; remaining ROIS; retrain to make sure nuclei are not too big/clumped

## Info

## Requirements
