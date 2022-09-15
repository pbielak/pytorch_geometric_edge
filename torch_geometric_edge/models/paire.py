from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.conv import MessagePassing  # pylint: disable=E0611


class PairEDefaultEncoder(nn.Module):
    """Default implementation of the PairE edge encoder."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.lin1_self = nn.Linear(2 * in_dim, 4 * out_dim)
        self.lin2_self = nn.Linear(4 * out_dim, out_dim)

        self.lin1_aggr = nn.Linear(2 * in_dim, 4 * out_dim)
        self.lin2_aggr = nn.Linear(4 * out_dim, out_dim)

        h_dim = (
            self.lin1_self.out_features
            + self.lin2_self.out_features
            + self.lin1_aggr.out_features
            + self.lin2_aggr.out_features
        )

        self.bn = nn.BatchNorm1d(h_dim)
        self.lin_emb = nn.Linear(h_dim, out_dim)

    def forward(
        self,
        x_self: torch.Tensor,
        x_aggr: torch.Tensor,
    ) -> torch.Tensor:
        h1_self = self.lin1_self(x_self)
        h2_self = self.lin2_self(h1_self)

        h1_aggr = self.lin1_aggr(x_aggr)
        h2_aggr = self.lin2_aggr(h1_aggr)

        h = torch.cat([h1_self, h2_self, h1_aggr, h2_aggr], dim=-1)
        h = self.bn(h)
        h = self.lin_emb(h)

        return h


class PairEDefaultDecoder(nn.Module):
    """Default implementation of the PairE decoder."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.dec_self = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LogSoftmax(dim=-1),
        )
        self.dec_aggr = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rec_self = self.dec_self(h)
        rec_aggr = self.dec_aggr(h)

        return rec_self, rec_aggr


class PairE(MessagePassing):  # pylint: disable=W0223
    """Implementation of the PairE model introduced in `Graph Representation
    Learning Beyond Node and Homophily` <https://arxiv.org/pdf/2203.01564.pdf>.
    """

    def __init__(
        self,
        num_nodes: int,
        node_feature_dim: int,
        emb_dim: int,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
    ):
        super().__init__(aggr="mean", flow="target_to_source")
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.emb_dim = emb_dim

        self.encoder = encoder
        self.decoder = decoder

        if self.encoder is None:
            self.encoder = PairEDefaultEncoder(
                in_dim=node_feature_dim,
                out_dim=emb_dim,
            )

        if self.decoder is None:
            self.decoder = PairEDefaultDecoder(
                in_dim=emb_dim,
                out_dim=2 * node_feature_dim,
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x_self, x_aggr = self.extract_self_aggr(x=x, edge_index=edge_index)
        h_edge = self.encoder(x_self=x_self, x_aggr=x_aggr)

        return h_edge

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j

    def extract_self_aggr(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_self = torch.cat([
            x[edge_index[0]], x[edge_index[1]]
        ], dim=-1)

        aggr = self.propagate(
            x=x,
            edge_index=edge_index,
            size=(self.num_nodes, self.num_nodes),
        )

        x_aggr = torch.cat([
            aggr[edge_index[0]], aggr[edge_index[1]]
        ], dim=-1)

        return x_self, x_aggr

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self(x=x, edge_index=edge_index)

    def decode(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(h)

    @staticmethod
    def loss(
        x_self: torch.Tensor,
        x_aggr: torch.Tensor,
        x_self_rec: torch.Tensor,
        x_aggr_rec: torch.Tensor,
    ) -> torch.Tensor:
        p_self = x_self / x_self.sum(dim=1, keepdim=True)
        p_aggr = x_aggr / x_aggr.sum(dim=1, keepdim=True)

        loss = (
            F.kl_div(input=x_self_rec, target=p_self, reduction="batchmean")
            + F.kl_div(input=x_aggr_rec, target=p_aggr, reduction="batchmean")
        )

        return loss
