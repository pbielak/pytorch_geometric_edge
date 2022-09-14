"""Edge embeddings as simple functions of pairs of node embeddings."""
from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseNodePairOp(nn.Module, ABC):
    r"""Compute edge embedding as function applied on pairs of node embeddings.

    \mathbf{z}_{uv} = op(\mathbf{z}_u, \mathbf{z}_v)
    """

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        z_u = x[edge_index[0]]
        z_v = x[edge_index[1]]

        return self.op(z_u=z_u, z_v=z_v)

    @abstractmethod
    def op(self, z_u: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        pass


class AvgNodePairOp(BaseNodePairOp):
    r"""Compute edge embedding as average of node embeddings.

    \mathbf{z}_{uv} = \frac{\mathbf{z}_u + \mathbf{z}_v}{2}
    """

    def op(self, z_u: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        return (z_u + z_v) / 2


class HadamardNodePairOp(BaseNodePairOp):
    r"""Compute edge embedding as Hadamard product of node embeddings.

    \mathbf{z}_{uv} = \mathbf{z}_u * \mathbf{z}_v
    """

    def op(self, z_u: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        return z_u * z_v


class L1NodePairOp(BaseNodePairOp):
    r"""Compute edge embedding as L1 distance of node embeddings.

    \mathbf{z}_{uv} = ||\mathbf{z}_u - \mathbf{z}_v||_1
    """

    def op(self, z_u: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        return (z_u - z_v).abs()


class L2NodePairOp(BaseNodePairOp):
    r"""Compute edge embedding as L2 distance of node embeddings.

    \mathbf{z}_{uv} = ||\mathbf{z}_u - \mathbf{z}_v||_2
    """

    def op(self, z_u: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        return (z_u - z_v).pow(2)
