import pytest


class TestHighOrderFunctions:
    @pytest.mark.slow
    @pytest.mark.xfail
    def test_cluster_cells(self, project):
        project.cluster_cells()

    @pytest.mark.slow
    def test_measure_adjacency(self, project_with_clusters):
        files = [
            "cluster_adjacency_graph.frequencies.csv",
            "cluster_adjacency_graph.norm_over_random.clustermap.svg",
            "cluster_adjacency_graph.norm_over_random.csv",
            "cluster_adjacency_graph.norm_over_random.heatmap.svg",
            "cluster_adjacency_graph.random_frequencies.all_iterations_100.csv",
            "cluster_adjacency_graph.random_frequencies.csv",
            "neighbor_graph.gpickle",
            "neighbor_graph.svg",
        ]

        with project_with_clusters as prj:
            prj.measure_adjacency()
            assert (prj.results_dir / "single_cell" / "project.adjacency.all_rois.pdf").exists()

        for roi in prj.rois:
            prefix = roi.sample.root_dir / "single_cell" / roi.name + "."
            for file in files:
                assert (prefix + file).exists()
