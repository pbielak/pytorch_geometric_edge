"""Implementation of trainable Node2Edge and Edge2Node operators/layers."""
from typing import Optional

import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing  # pylint: disable=E0611


class Node2Edge(nn.Module):
    r"""Layer for computing edge embeddings as a function of node embeddings
    and optionally edge features, as introduced in `Neural Relational Inference
    for Interacting Systems` by Kipf et al.

    \mathbf{h}_{ij} = f_\theta([\mathbf{h}_i, \mathbf{h}_j, \mathbf{x}_{ij}])

    where:
    - \mathbf{h}_{ij} is the computed representation of the edge (i, j)
    - \mathbf{h}_i, \mathbf{h}_j are node representations (from the previous
      layer or initial node features)
    - \mathbf{x}_{ij} are the features of the edge (i, j)
    - f_\theta is a small trainable neural network (like an MLP),
    - [., .] denotes concatenation

    If no edge features are present, this layer will simplify to the following:

    \mathbf{h}_{ij} = f_\theta([\mathbf{h}_i, \mathbf{h}_j])

    If the neural network f_\theta (`net` in argument list) is not provided,
    we use a single linear layer without any activations.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        out_dim: int,
        net: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.out_dim = out_dim

        self.net = net

        if self.net is None:
            self.net = nn.Linear(
                in_features=2 * node_dim + edge_dim,
                out_features=self.out_dim,
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h_i = x[edge_index[0]]
        h_j = x[edge_index[1]]

        if edge_attr is None:
            inp = torch.cat([h_i, h_j], dim=-1)
        else:
            inp = torch.cat([h_i, h_j, edge_attr], dim=-1)

        return self.net(inp)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.out_dim})"


class Edge2Node(MessagePassing):  # pylint: disable=W0223
    r"""Layer for computing node embeddings as a function of edge embeddings
    and optionally node features, as introduced in `Neural Relational Inference
    for Interacting Systems` by Kipf et al.

    \mathbf{h}_i = f_\theta([
        \sum_{j \in \mathcal{N}_i} \mathbf{h}_{ij}, \mathbf{x}_i
    ])

    where:
    - \mathbf{h}_i is the computed node representation
    - \mathbf{h}_{ij} is the computed representation of the edge (i, j)
      (from the previous layer or initial edge features)
    - \mathbf{x}_i are the features of the i-th node
    - f_\theta is a small trainable neural network (like an MLP),
    - \mathcal{N}_i is the i-th node's set of neighbors
    - [., .] denotes concatenation

    If no node features are present, this layer will simplify to the following:

    \mathbf{h}_i = f_\theta(\sum_{j in \mathcal{N}_i} \mathbf{h}_{ij})

    If the neural network f_\theta (`net` in argument list) is not provided,
    we use a single linear layer without any activations.
    """

    def __init__(
        self,
        num_nodes: int,
        node_dim: int,
        edge_dim: int,
        out_dim: int,
        net: Optional[nn.Module] = None,
    ):
        super().__init__(aggr="sum")

        self.num_nodes = num_nodes

        self.out_dim = out_dim

        self.net = net

        if self.net is None:
            self.net = nn.Linear(
                in_features=edge_dim + node_dim,
                out_features=self.out_dim,
            )

    def forward(
        self,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.propagate(
            edge_index=edge_index,
            edge_attr=edge_attr,
            size=(self.num_nodes, self.num_nodes),
        )

        if x is not None:
            inp = torch.cat([out, x], dim=-1)
        else:
            inp = out

        return self.net(inp)

    def message(  # pylint: disable=W0237
        self,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        return edge_attr

    def __repr__(self):
        return f"{self.__class__.__name__}({self.out_dim})"


class Node2Edge2NodeBlock(nn.Module):
    """A combination of a single `Node2Edge` and a single `Edge2Node` layer."""

    def __init__(
        self,
        num_nodes: int,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        out_dim: int,
        node2edge_net: Optional[nn.Module] = None,
        edge2node_net: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.node2edge = Node2Edge(
            node_dim=node_dim,
            edge_dim=edge_dim,
            out_dim=hidden_dim,
            net=node2edge_net,
        )
        self.edge2node = Edge2Node(
            num_nodes=num_nodes,
            node_dim=node_dim,
            edge_dim=hidden_dim,
            out_dim=out_dim,
            net=edge2node_net,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h_edge = self.node2edge(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        h_node = self.edge2node(
            edge_attr=h_edge,
            edge_index=edge_index,
            x=x,
        )

        return h_node
