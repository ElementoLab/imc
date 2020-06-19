#!/usr/bin/env python


import pickle
import tempfile

import pytest

from imc import Project, IMCSample, ROI
from imc.demo import generate_project
from imc.data_models.project import DEFAULT_PROJECT_NAME
from imc.data_models.sample import DEFAULT_SAMPLE_NAME
from imc.data_models.roi import DEFAULT_ROI_NAME


class TestProjectInitialization:
    def test_empty_project(self):
        p = Project()
        assert p.name == DEFAULT_PROJECT_NAME
        assert isinstance(p.samples, list)
        assert isinstance(p.rois, list)
        assert not p.samples
        assert not p.rois

    def test_empty_sample(self):
        s = IMCSample()
        assert s.name == DEFAULT_SAMPLE_NAME
        assert isinstance(s.rois, list)
        assert not s.rois

    def test_empty_roi(self):
        r = ROI()
        assert r.name == DEFAULT_ROI_NAME

    def test_creation_without_rois(self, tmp_path):
        p = generate_project(root_dir=tmp_path)
        p2 = Project(p.metadata[["sample_name"]].drop_duplicates(), processed_dir=p.processed_dir)
        assert len(p2.samples) == 3
        assert len(p2.rois) == 9
