import pickle
from typing import Any

import parmap
import pandas as pd

from imc import Project, IMCSample, ROI
from imc.ops.quant import _quantify_cell_intensity__roi
from imc.types import Path


def roundtrip(obj: Any, _dir: Path) -> Any:
    pickle.dump(obj, open(_dir / "file.pkl", "wb"))
    return pickle.load(open(_dir / "file.pkl", "rb"))


class TestSimpleSerialization:
    def test_empty_project(self, tmp_path):
        p = Project(name="test_empty_project")
        q = roundtrip(p, tmp_path)
        assert q.name == "test_empty_project"
        # assert p is q

    def test_empty_sample(self, tmp_path):
        s = IMCSample(sample_name="test_empty_sample", root_dir=".")
        r = roundtrip(s, tmp_path)
        assert r.name == "test_empty_sample"
        # assert s is r

    def test_empty_roi(self, tmp_path):
        r = ROI(name="test_empty_roi", roi_number=1)
        s = roundtrip(r, tmp_path)
        assert s.name == "test_empty_roi"
        # assert r is s


def func(roi: ROI) -> int:
    return len(roi.shape)


class TestParmapSerialization:
    def test_simple_parmap(self, project):

        res = parmap.map(func, project.rois)
        assert all(x == 3 for x in res)

    def test_quant_parmap_lowlevel(self, project):

        _res = parmap.map(_quantify_cell_intensity__roi, project.rois)
        res = pd.concat(_res)
        assert not res.empty
        assert all(
            res.columns == project.rois[0].channel_labels.tolist() + ["roi", "sample"]
        )

    def test_quant_parmap_highlevel(self, project):
        res = project.quantify_cell_intensity()
        assert not res.empty
        assert all(
            res.columns == project.rois[0].channel_labels.tolist() + ["roi", "sample"]
        )
