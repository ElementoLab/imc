import pytest

from imc.demo import generate_project


# # To run manually:
# import tempfile
# tmp_path = tempfile.TemporaryDirectory().name


@pytest.fixture
def project(tmp_path):
    return generate_project(root_dir=tmp_path)


@pytest.fixture
def metadata(project):
    return project.sample_metadata


@pytest.fixture
def project_with_clusters(tmp_path):
    p = generate_project(root_dir=tmp_path)
    p.quantify_cells()
    c = (
        p.quantification.set_index(["sample", "roi"], append=True)
        .rename_axis(["obj_id", "sample", "roi"])
        .reorder_levels([1, 2, 0])
        .assign(cluster=(p.quantification.index % 2))["cluster"]
    )
    p.set_clusters(c, write_to_disk=True)
    return p
