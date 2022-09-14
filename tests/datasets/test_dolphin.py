import tempfile

from torch_geometric_edge.datasets import DolphinSocialNetwork


def test_dolphin_sizes():
    with tempfile.TemporaryDirectory() as root:
        dataset = DolphinSocialNetwork(root=root)

        assert len(dataset) == 1
        assert dataset.num_classes == 4

        data = dataset[0]

        assert data.num_nodes == 62
        # NOTE(pbielak): The Line2vec paper uses undirected graphs.
        # The reported number of edges was taken from the NetworkX
        # implementation. In PyG, for undirected graphs, we add reciprocal
        # edges. Hence, we end up with twice the number of reported edges.
        assert data.num_edges == 159 * 2
