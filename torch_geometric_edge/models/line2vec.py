from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn.models import Node2Vec  # pylint: disable=E0611

from torch_geometric_edge.transforms import Line2vecTransform


@dataclass(frozen=True)
class Node2vecParams:
    p: float
    q: float
    walk_length: int
    context_size: int
    walks_per_node: int


class Line2vec(nn.Module):
    """Implementation of the Line2vec model."""

    def __init__(
        self,
        data: Data,
        emb_dim: int,
        node2vec_params: Node2vecParams,
        alpha: float,
        lambda_: float,
        gamma: float,
    ):
        super().__init__()

        self.emb_dim = emb_dim

        self.alpha = alpha
        self.lambda_ = lambda_
        self.gamma = gamma

        self.gammas = torch.tensor([gamma] * data.num_nodes)

        self.original_graph = data
        self.line_graph = Line2vecTransform()(data)

        self.node2vec = Node2Vec(
            edge_index=self.line_graph.edge_index,
            embedding_dim=self.emb_dim,
            p=node2vec_params.p,
            q=node2vec_params.q,
            walk_length=node2vec_params.walk_length,
            context_size=node2vec_params.context_size,
            walks_per_node=node2vec_params.walks_per_node,
            num_nodes=self.line_graph.num_nodes,
        )

        def modified_sample_fn(batch):
            if not isinstance(batch, torch.Tensor):
                batch = torch.tensor(batch)
            return (
                batch,
                self.node2vec.pos_sample(batch),
                self.node2vec.neg_sample(batch),
            )

        self.node2vec.sample = modified_sample_fn

        self.node2vec.reset_parameters()

        self.centers = nn.Parameter(data=self._init_centers())
        self.radii = nn.Parameter(data=self._init_radii())

    def _init_centers(self) -> torch.Tensor:
        G = self.original_graph

        centers = torch.empty((G.num_nodes, self.emb_dim))

        # For each node `u` in the original graph `G`, collect the indices of
        # the line graph nodes defined by the neighbors of `u` in `G`
        node2neighbors_line_idxs = defaultdict(list)

        base_edges = G.edge_index[:, G.edge_index[0] < G.edge_index[1]]

        for idx, (u, v) in enumerate(base_edges.t().tolist()):
            node2neighbors_line_idxs[u].append(idx)
            node2neighbors_line_idxs[v].append(idx)

        # Compute centers as averages of the neighbors embeddings
        emb = self.node2vec.embedding.weight

        for u in range(G.num_nodes):
            centers[u] = emb[node2neighbors_line_idxs[u]].mean(dim=0)

        return centers

    def _init_radii(self) -> torch.Tensor:
        G = self.original_graph

        neighbors = defaultdict(list)
        for u, v in G.edge_index.t().tolist():
            neighbors[u].append(v)

        radii = torch.empty(G.num_nodes)

        for u in range(G.num_nodes):
            with torch.no_grad():
                radii[u] = (
                    (self.centers[u] - self.centers[neighbors[u]])
                    .norm(p=2, dim=-1)
                    .max()
                )

        return radii

    def forward(self, batch=None):
        return self.node2vec(batch=batch)

    def loader(self, **kwargs):
        return self.node2vec.loader(**kwargs)

    def loss(
        self,
        batch: torch.Tensor,
        edge_emb: torch.Tensor,
        pos_rw: torch.Tensor,
        neg_rw: torch.Tensor,
    ) -> torch.Tensor:
        # Node2vec loss
        n2v_loss = self.node2vec.loss(pos_rw=pos_rw, neg_rw=neg_rw)

        # Collective homophily
        G = self.original_graph
        ei = G.edge_index[:, G.edge_index[0] < G.edge_index[1]]
        ei = ei[:, batch]

        dist_from_centers = (
            (edge_emb - self.centers[ei[0]])
            .norm(p=2, dim=-1)
        )

        radii_size_objective = self.radii[ei[0]].pow(2).sum()
        dist_constraint = (
            dist_from_centers.pow(2) - self.radii[ei[0]].pow(2)
        ).relu().sum()
        negative_radii_constraint = (-self.radii[ei[0]]).relu()

        loss = (
            n2v_loss
            + self.alpha * radii_size_objective
            + self.lambda_ * dist_constraint
            + (self.gammas[ei[0]] * negative_radii_constraint).sum()
        )

        return loss
