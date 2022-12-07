import torch
from torch_geometric.data import Data


def apply_DHT(
    data: Data,
    add_loops: bool = False,
) -> Data:
    """Dual Hypergraph Transformation (DHT) as introduced in the EHGNN paper:
    `Edge Representation Learning with Hypergraphs`
    (`https://arxiv.org/pdf/2106.15845.pdf`)

    Implementation based on:
    `https://github.com/harryjo97/EHGNN/blob/main/models/models.py`
    """
    num_edges = data.num_edges
    device = data.edge_index.device

    # Create hyperedge list of the Dual Hypergraph
    edge_to_node_index = (
        torch.arange(0, num_edges, step=1, device=device)
        .repeat_interleave(2)
        .view(1, -1)
    )
    hyperedge_index = data.edge_index.T.reshape(1, -1)
    hyperedge_index = torch.cat([
        edge_to_node_index,
        hyperedge_index,
    ], dim=0).long()

    # Add self-loops to each node in the dual hypergraph
    if add_loops:
        bincount = hyperedge_index[1].bincount()
        mask = bincount[hyperedge_index[1]] != 1
        max_edge = hyperedge_index[1].max()
        loops = torch.cat([
            torch.arange(0, num_edges, step=1, device=device).view(1, -1),
            torch.arange(
                start=max_edge + 1,
                end=max_edge + num_edges + 1,
                step=1,
                device=device,
            ).view(1, -1),
        ], dim=0)

        hyperedge_index = torch.cat([hyperedge_index[:, mask], loops], dim=1)

    out = Data(
        x=data.edge_attr,
        edge_index=hyperedge_index,
        edge_attr=data.x,
    )

    # Transform batch of nodes to batch of edges
    if data.batch is not None:
        edge_batch = hyperedge_index[1, :].reshape(-1, 2)[:, 0]
        edge_batch = torch.index_select(data.batch, 0, edge_batch)

        out.batch = edge_batch

    return out
