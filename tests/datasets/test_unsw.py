import tempfile

import pytest

from torch_geometric_edge.datasets import UNSWNB15


@pytest.mark.skip(reason="GitHub Actions runner has not enough RAM")
def test_unswnb15_sizes():
    expected_sizes = {
        "ip/binary": {
            "num_classes": 2,
            "num_nodes": 49,
            "num_edges": 2_539_739,
        },
        "ip/multi": {
            "num_classes": 14,
            "num_nodes": 49,
            "num_edges": 2_539_739,
        },
        "ip_port/binary": {
            "num_classes": 2,
            "num_nodes": 1_112_275,
            "num_edges": 2_539_739,
        },
        "ip_port/multi": {
            "num_classes": 14,
            "num_nodes": 1_112_275,
            "num_edges": 2_539_739,
        },

    }
    with tempfile.TemporaryDirectory() as root:
        for version, expected in expected_sizes.items():
            dataset = UNSWNB15(version=version, root=root)

            assert len(dataset) == 1
            assert dataset.num_classes == expected["num_classes"]

            data = dataset[0]

            assert data.num_nodes == expected["num_nodes"]
            assert data.num_edges == expected["num_edges"]
