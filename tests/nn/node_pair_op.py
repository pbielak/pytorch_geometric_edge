import pytest
import torch
from torch_geometric.data import Data

from pytorch_geometric_edge.nn import (
    AvgNodePairOp,
    HadamardNodePairOp,
    L1NodePairOp,
    L2NodePairOp,
)


@pytest.fixture()
def sample_data() -> Data:
    return Data(
        x=torch.tensor([
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4],
        ]),
        edge_index=torch.tensor([
            [0, 0, 1, 1],
            [1, 2, 2, 3]
        ]),
    )


def test_avg_node_pair_op(sample_data):
    op = AvgNodePairOp()

    z = op(x=sample_data.x, edge_index=sample_data.edge_index)

    assert z.shape == (sample_data.num_edges, sample_data.x.shape[1])

    expected = torch.tensor([
        [1.5, 1.5, 1.5, 1.5, 1.5],
        [2, 2, 2, 2, 2],
        [2.5, 2.5, 2.5, 2.5, 2.5],
        [3, 3, 3, 3, 3],
    ])

    assert (z == expected).all()


def test_hadamard_node_pair_op(sample_data):
    op = HadamardNodePairOp()

    z = op(x=sample_data.x, edge_index=sample_data.edge_index)

    assert z.shape == (sample_data.num_edges, sample_data.x.shape[1])

    expected = torch.tensor([
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [6, 6, 6, 6, 6],
        [8, 8, 8, 8, 8],
    ])

    assert (z == expected).all()


def test_l1_node_pair_op(sample_data):
    op = L1NodePairOp()

    z = op(x=sample_data.x, edge_index=sample_data.edge_index)

    assert z.shape == (sample_data.num_edges, sample_data.x.shape[1])

    expected = torch.tensor([
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
    ])

    assert (z == expected).all()


def test_l2_node_pair_op(sample_data):
    op = L2NodePairOp()

    z = op(x=sample_data.x, edge_index=sample_data.edge_index)

    assert z.shape == (sample_data.num_edges, sample_data.x.shape[1])

    expected = torch.tensor([
        [1, 1, 1, 1, 1],
        [4, 4, 4, 4, 4],
        [1, 1, 1, 1, 1],
        [4, 4, 4, 4, 4],
    ])

    assert (z == expected).all()
