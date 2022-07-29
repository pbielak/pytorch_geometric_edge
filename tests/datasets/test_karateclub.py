from pytorch_geometric_edge.datasets import KarateClub


def test_karateclub_sizes():
    dataset = KarateClub()

    assert len(dataset) == 1
    assert dataset.num_classes == 3

    data = dataset[0]

    assert data.num_nodes == 34
    # NOTE(pbielak): The Line2vec paper uses undirected graphs. The reported
    # number of edges was taken from the NetworkX implementation. In PyG, for
    # undirected graphs, we add reciprocal edges. Hence, we end up with twice
    # the number of reported edges.
    assert data.num_edges == 78 * 2

