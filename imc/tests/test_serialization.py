import pickle
import tempfile

from imc import Project, IMCSample, ROI


class TestProjectSerialization:
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
