import tempfile

from pytorch_geometric_edge.datasets import PubMed


def test_pubmed_sizes():
    with tempfile.TemporaryDirectory() as root:
        dataset = PubMed(root=root)

        assert len(dataset) == 1
        assert dataset.num_classes == 3

        data = dataset[0]

        assert data.num_nodes == 19_717
        # NOTE(pbielak): The Line2vec paper uses undirected graphs. The reported
        # number of edges was taken from the NetworkX implementation.
        # In PyG, for undirected graphs, we add reciprocal edges. Hence, we end
        # up with twice the number of reported edges.

        # NOTE(pbielak): In the Line2vec paper there are 44_327 edges, but in
        # this implementation there are 44_324 edges!
        assert data.num_edges == 44_324 * 2

