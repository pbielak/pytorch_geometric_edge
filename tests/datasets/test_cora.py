import tempfile

from torch_geometric_edge.datasets import Cora


def test_cora_sizes():
    with tempfile.TemporaryDirectory() as root:
        dataset = Cora(root=root)

        assert len(dataset) == 1
        assert dataset.num_classes == 7

        data = dataset[0]

        assert data.num_nodes == 2_708
        # NOTE(pbielak): The Line2vec paper uses undirected graphs.
        # The reported number of edges was taken from the NetworkX
        # implementation. In PyG, for undirected graphs, we add reciprocal
        # edges. Hence, we end up with twice the number of reported edges.
        assert data.num_edges == 5_278 * 2
