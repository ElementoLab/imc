# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- 
### Changed
- 
### Removed
-

## [0.0.12] - 2021-07-19
### Added
- functions to handle multi-cell masks (topological domains)
- napari + napari_imc to view MCD files
### Changed
- fix support of OSX in ilastik segmentation
- centralized package data under `.imc`

## [0.0.11] - 2021-07-01
### Added
- Command `imc process`.

## [0.0.10] - 2021-07-01
### Added
- CI on Github actions
- add more CLI commands
### Changed
- centralized package data under `.imc`
- fix packaging

## [0.0.8] - 2021-06-01
### Added
- add `.pyproject.toml`
- support subcellular mask quantification
### Changed
- rasterized linecollection plots by default

## [0.0.7] - 2021-04-26
### Added
- initial support subcellular mask quantification
- DeepCell postprocessing to match nuclear and cellular masks
- function to plot and extract panorama images matching ROIs
- Cellpose as segmentation method
- add CLI command for segmentation
### Changed
- rasterized linecollection plots by default

## [0.0.6] - 2020-12-16
### Added
- segmentation module
- mask layers to support alternative segmentations
### Changed
- rasterized linecollection plots by default
### Removed
-
- graphics code that was abstracted to `seaborn_extensions` module

## [0.0.5] - 2020-12-07
### Added
- segmentation module
- mask layers to support alternative segmentations
### Changed
- export panoramas by default
- support ome-tiff
- upgrade to `imctools==2.1.0`

## [0.0.4] - 2020-10-07


## [0.0.3] - 2020-06-17
### Changed
- Patch `pathlib.Path` to support path building with `+` (operator overload)

## [0.0.2] - 2020-06-15
### Added
- Many features


## [0.0.1] - 2020-04-14
### Added
- Project, Sample and ROI modules/objects

[Unreleased]: https://github.com/ElementoLab/imc/compare/0.0.2...HEAD
[0.0.11]: https://github.com/ElementoLab/imc/compare/0.0.10...v0.0.11
[0.0.10]: https://github.com/ElementoLab/imc/compare/0.0.9...v0.0.10
[0.0.9]: https://github.com/ElementoLab/imc/compare/0.0.8...v0.0.9
[0.0.8]: https://github.com/ElementoLab/imc/compare/0.0.7...v0.0.8
[0.0.7]: https://github.com/ElementoLab/imc/compare/0.0.6...v0.0.7
[0.0.6]: https://github.com/ElementoLab/imc/compare/0.0.5...v0.0.6
[0.0.5]: https://github.com/ElementoLab/imc/compare/0.0.4...v0.0.5
[0.0.4]: https://github.com/ElementoLab/imc/compare/0.0.3...v0.0.4
[0.0.3]: https://github.com/ElementoLab/imc/compare/0.0.2...v0.0.3
[0.0.2]: https://github.com/ElementoLab/imc/compare/0.0.1...v0.0.2
[0.0.1]: https://github.com/ElementoLab/imc/releases/tag/v0.0.1
