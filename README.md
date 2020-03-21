# Imaging mass cytometry

This is so far an umbrela repo for the development/implementation of a imaging
mass cytometry (IMC) processing pipeline and metadata on IMC data acquired with
EIPM's Hyperion CyTOF instrument.

The immediate goals are to provide a streamlined pipeline for the analysis of IMC
data that is robust enough for pan-cancer analysis.

This involves image- and channel-wise quality control, image preprocessing and
filtering, feature selection and semi-supervised pixel classification,
image segmentation into cell masks and cell quantification.

The blueprint is for now largely based on
[Vito Zanotelli's IMC pipeline](https://github.com/BodenmillerGroup/ImcSegmentationPipeline).

On the long-term I aim to develop a fully unsupervised pipeline using
variational autoencoders (VAE).

## Organization

- Sample and technical (CyTOF panels) metadata are present in the
[metadata](metadata/) directory.
- Pipeline are analysis source code are avaible in the [src](src/) directory.
