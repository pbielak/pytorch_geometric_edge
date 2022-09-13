from collections import defaultdict
from typing import List, NamedTuple, Tuple

import torch


class RandomWalker:
    """Implements uniform random walks with forbidden paths."""

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_walks: int,
        walk_length: int,
    ):
        self.neighbors = defaultdict(list)

        for idx, (u, v) in enumerate(edge_index.t().tolist()):
            self.neighbors[u].append((v, idx))

        self.num_walks = num_walks
        self.walk_length = walk_length

    def walk(self, start_node: int, forbidden: Tuple[int, int]):
        # Forbid also reverse edge
        forbidden = (forbidden, (forbidden[1], forbidden[0]))

        return torch.stack([
            self._make_walk(start_node, forbidden)
            for _ in range(self.num_walks)
        ], dim=0)

    def _make_walk(
        self,
        start_node: int,
        forbidden: Tuple[Tuple[int, int], Tuple[int, int]],
    ) -> torch.Tensor:
        edge_walk = []

        node = start_node
        while len(edge_walk) != self.walk_length:
            neighbors = [
                (v, idx)
                for v, idx in self.neighbors[node]
                if (node, v) not in forbidden and (v, node) not in forbidden
            ]

            if len(neighbors) == 0:
                edge_walk.extend([-1] * (self.walk_length - len(edge_walk)))
                return torch.tensor(edge_walk)

            i = torch.randint(0, len(neighbors), (1,))
            node = neighbors[i][0]
            edge_walk.append(neighbors[i][1])

        return torch.tensor(edge_walk)

    def empty_walk(self):
        return ((-1) * torch.ones(self.num_walks, self.walk_length)).long()


class AttrE2vecBatch(NamedTuple):
    batch: torch.Tensor  # shape: [batch_size, N]
    walks_u: torch.Tensor  # shape: [batch_size, N, num_walks, walk_length]
    walks_v: torch.Tensor  # shape: [batch_size, N, num_walks, walk_length]

    def __repr__(self):
        name = self.__class__.__name__
        b_shape = self.batch.shape
        u_shape = self.walks_u.shape
        v_shape = self.walks_v.shape
        return f"{name}(batch={b_shape}, walks_u={u_shape}, walks_v={v_shape})"


class AttrE2vecSampler:
    """Implementation of sampler as proposed in AttrE2vec."""

    def __init__(
        self,
        edge_index: torch.Tensor,
        sample_pos_neg: bool,
        num_walks: int,
        walk_length: int,
        num_pos: int,
        num_neg: int,
    ):
        self.edge_index = edge_index
        self.sample_pos_neg = sample_pos_neg

        self.num_pos = num_pos
        self.num_neg = num_neg

        self.rw = RandomWalker(
            edge_index=self.edge_index,
            num_walks=num_walks,
            walk_length=walk_length,
        )

    def __call__(self, batch: List[int]):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        # Sample random walks for edges in batch
        walks_u, walks_v = self.sample_walks(batch)

        base = AttrE2vecBatch(
            batch=batch.unsqueeze(dim=-1),
            walks_u=walks_u.unsqueeze(dim=1),
            walks_v=walks_v.unsqueeze(dim=1),
        )

        if not self.sample_pos_neg:
            return base

        # Sample random walks for positive and negative samples of edges
        pos_batch, pos_walks_u, pos_walks_v = [], [], []
        neg_batch, neg_walks_u, neg_walks_v = [], [], []

        for idx, wu, wv in zip(batch, walks_u, walks_v):
            neighborhood = self.compute_neighborhood(wu, wv)

            # Positive samples
            pb = self.sample_pos(neighborhood)
            pu, pv = self.sample_walks(pb)

            pos_batch.append(pb)
            pos_walks_u.append(pu)
            pos_walks_v.append(pv)

            # Negative samples
            nb = self.sample_neg(idx, neighborhood)
            nu, nv = self.sample_walks(nb)

            neg_batch.append(nb)
            neg_walks_u.append(nu)
            neg_walks_v.append(nv)

        pos = AttrE2vecBatch(
            batch=torch.stack(pos_batch, dim=0),
            walks_u=torch.stack(pos_walks_u, dim=0),
            walks_v=torch.stack(pos_walks_v, dim=0),
        )

        neg = AttrE2vecBatch(
            batch=torch.stack(neg_batch, dim=0),
            walks_u=torch.stack(neg_walks_u, dim=0),
            walks_v=torch.stack(neg_walks_v, dim=0),
        )

        return base, pos, neg

    def sample_walks(self, batch: torch.Tensor):
        walks_u, walks_v = [], []

        for idx in batch.tolist():
            if idx == -1:
                w = self.rw.empty_walk()
                walks_u.append(w)
                walks_v.append(w)
            else:
                u, v = self.edge_index[:, idx].tolist()

                walks_u.append(self.rw.walk(start_node=u, forbidden=(u, v)))
                walks_v.append(self.rw.walk(start_node=v, forbidden=(u, v)))

        return torch.stack(walks_u, dim=0), torch.stack(walks_v, dim=0)

    @staticmethod
    def compute_neighborhood(
        walks_u: torch.Tensor,
        walks_v: torch.Tensor,
    ):
        walks = torch.cat([walks_u, walks_v], dim=0)
        neighborhood = walks.flatten().unique()
        return neighborhood

    def sample_pos(self, neighborhood: torch.Tensor):
        indices = torch.randint(
            high=neighborhood.shape[0],
            size=(self.num_pos,),
        )
        return neighborhood[indices]

    def sample_neg(self, current_idx: int, neighborhood: torch.Tensor):
        neighborhood = neighborhood[neighborhood != -1]

        mask = torch.ones(self.edge_index.shape[1]).bool()
        mask[neighborhood] = False
        mask[current_idx] = False

        candidates = torch.arange(self.edge_index.shape[1])[mask]
        indices = torch.randint(
            high=candidates.shape[0],
            size=(self.num_neg,),
        )

        return candidates[indices]
