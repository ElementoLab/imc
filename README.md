# Imaging mass cytometry

This is an umbrela repository for the development of imaging mass cytometry
(IMC) software.

The immediate goals are to provide a streamlined pipeline for the analysis of IMC
data that is robust enough for pan-cancer analysis.

This involves image- and channel-wise quality control, image preprocessing and
filtering, feature selection and semi-supervised pixel classification,
image segmentation into cell masks and cell quantification.

The blueprint is for now largely based on
[Vito Zanotelli's IMC pipeline]\
(https://github.com/BodenmillerGroup/ImcSegmentationPipeline).

On the long-term I aim to develop a fully unsupervised pipeline using
variational autoencoders (VAE).

## Organization

This repo is for now hosting a [pipeline](imcpipeline/pipeline.py), a
[cross-environment job submitter](imcpipeline/runner.py) for the pipeline,
[OOP models for IMC data](imc/data_models) and other
[various utilities](imc/utils.py).
All is pip installable. The pipeline scripts `imcrunner` and `imcpipeline` will
be installed.

Sample and technical (CyTOF panels) metadata are present in the
[metadata](metadata/) directory (work in progress).

The specific commands to reproduce the processed data and analysis and their
order is in the [Makefile](Makefile).

Scripts supporting exploratory analysis or other functions that are still work
in progress are present in the [scripts directory](scripts).


## Requirements and installation

Requires:

- Python >= 3.7
- The requirements specified in [requirements.txt](requirements.txt) (will be
installed automatically by `pip`).

For the image processing pipeline:

- One of: `docker`, `singularity` or `cellprofiler` in a local installation.

In due time it will be released to [Pypi](), but while the repository is private
you can install with:

```bash
pip install git+ssh://git@github.com/elementolab/imc.git
```

The `git+ssh` protocol requires proper git configuration.

## Testing

Tests are still very limited, but you can run tests this way:

```bash
python -m pytest --pyargs imc
```


## Documentation

Documentation is for now mostly a skeleton but will be enlarged soon:

```bash
make docs
```
