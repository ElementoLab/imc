# Imaging mass cytometry

This is a package for the analysis of imaging mass cytometry (IMC) data.

It implements image- and channel-wise quality control, quantification of cell
intenstity and morphology, cell type discovery through clustering, automated
cell type labeling, community and super-community finding and differential
comparisons between sample groups, in addition to many handy visualization tools.

Above all, it is a tool for the use of IMC data at scale. To do that, it
implements out-of-memory handling of image stacks and masks.


## Requirements and installation

Requires `Python >= 3.7`.

Install with `pip`:
```bash
pip install git+ssh://git@github.com/elementolab/imc.git
```
While the repository is private, the `git+ssh` protocol requires proper git
configuration.


## Testing

Tests are still very limited, but you can run tests this way:

```bash
python -m pytest --pyargs imc
```

## Quick start

### Demo data
```python
>>> from imc.demo import generate_project
>>> prj = generate_project(n_samples=2, n_rois_per_sample=3, dims=(8, 8))
>>> prj
Project 'project' with 2 samples and 6 ROIs in total.

>>> prj.samples  # type: List[IMCSample]
[Sample 'test_sample_01' with 3 ROIs,
 Sample 'test_sample_02' with 3 ROIs]

>>> prj.rois  # type: List[ROI]
[Region1 of sample 'test_sample_01',
 Region2 of sample 'test_sample_01',
 Region3 of sample 'test_sample_01',
 Region1 of sample 'test_sample_02',
 Region2 of sample 'test_sample_02',
 Region3 of sample 'test_sample_02']

>>> prj.samples[0].rois  # type: List[ROI]
[Region1 of sample 'test_sample_01',
 Region2 of sample 'test_sample_01',
 Region3 of sample 'test_sample_01']

>>> roi = prj.rois[0]  # Let's assign one ROI to explore it
>>> roi.channel_labels  # type: pandas.Series; `channel_names`, `channel_metals` also available
0    Ch01(Ch01)
1    Ch02(Ch02)
2    Ch03(Ch03)
Name: channel, dtype: object

>>> roi.mask  # type: numpy.ndarray
array([[0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 2, 0, 0, 0, 3, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 4, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)

>>> roi.stack.shape  # roi.stack -> type: numpy.ndarray
(3, 8, 8)

>>> # QC
>>> prj.channel_correlation()
>>> prj.channel_summary()

>>> # Cell type discovery
>>> prj.cluster_cells()
>>> prj.find_communities()

```

### Your own data

You'd only need to have a CSV file with one row per sample, or one row per ROI
and pass that into the `Project` constructor:
```python
from imc import Project

prj = Project("path/to/sample/annotation.csv")
```

## Documentation

Documentation is for now mostly a skeleton but will be enlarged soon:

```bash
make docs
```
