import pickle
import tempfile

import parmap
import pandas as pd

from imc import Project, IMCSample, ROI
from imc.demo import generate_project
from imc.operations import _quantify_cell_intensity__roi


class TestSimpleSerialization:
    def test_empty_project(self):
        p = Project(name="test_empty_project")
        f = tempfile.NamedTemporaryFile()
        pickle.dump(p, open(f.name, "wb"))
        q = pickle.load(open(f.name, "rb"))

        assert q.name == "test_empty_project"
        # assert p is q

    def test_empty_sample(self):
        s = IMCSample(sample_name="test_empty_sample", root_dir=".")
        f = tempfile.NamedTemporaryFile()
        pickle.dump(s, open(f.name, "wb"))
        r = pickle.load(open(f.name, "rb"))

        assert r.name == "test_empty_sample"
        # assert s is r

    def test_empty_roi(self):
        r = ROI(name="test_empty_roi", roi_number=1)
        f = tempfile.NamedTemporaryFile()
        pickle.dump(r, open(f.name, "wb"))
        s = pickle.load(open(f.name, "rb"))

        assert s.name == "test_empty_roi"
        # assert r is s


def func(roi: ROI):
    return len(roi.shape)


class TestParmapSerialization:
    def test_simple_parmap(self, project):

        res = parmap.map(func, project.rois)
        assert all([x == 3 for x in res])

    def test_quant_parmap_lowlevel(self, project):

        _res = parmap.map(_quantify_cell_intensity__roi, project.rois)
        res = pd.concat(_res)
        assert not res.empty
        assert all(res.columns == project.rois[0].channel_labels.tolist() + ["roi", "sample"])

    def test_quant_parmap_highlevel(self, project):
        res = project.quantify_cell_intensity()
        res.to_csv("/home/afr/test.csv")
        assert not res.empty
        assert all(res.columns == project.rois[0].channel_labels.tolist() + ["roi", "sample"])
