import tempfile

from torch_geometric_edge.datasets import Cuneiform


def test_cuneiform_sizes():
    with tempfile.TemporaryDirectory() as root:
        dataset = Cuneiform(root=root)

        assert len(dataset) == 1
        assert dataset.num_classes == 2

        data = dataset[0]

        assert data.num_nodes == 5_680
        assert data.num_edges == 23_922
