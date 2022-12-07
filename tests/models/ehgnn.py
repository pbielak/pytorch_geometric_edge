import torch
from torch_geometric.data import Data

from torch_geometric_edge.models import apply_DHT


def test_apply_DHT():
    data = Data(
        x=torch.randn(50, 128),
        edge_index=torch.randint(0, 50, size=(2, 300)),
        edge_attr=torch.randn(300, 64),
        num_nodes=50,
    )

    hyper_data = apply_DHT(data=data, add_loops=False)

    assert (hyper_data.x == data.edge_attr).all()
    assert (hyper_data.edge_attr == data.x).all()
    assert hyper_data.edge_index.shape == (2, 2 * data.num_edges)
