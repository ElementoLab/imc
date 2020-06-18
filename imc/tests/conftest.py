import pytest

from imc.demo import generate_project


@pytest.fixture
def project():
    return generate_project()


@pytest.fixture
def project_with_clusters():
    p = generate_project()
    p.quantify_cells()
    c = (
        p.quantification.set_index(["sample", "roi"], append=True)
        .rename_axis(["obj_id", "sample", "roi"])
        .reorder_levels([1, 2, 0])
        .assign(cluster=(p.quantification.index % 2))["cluster"]
    )
    p.set_clusters(c, write_to_disk=True)
    return p
