from collections import defaultdict
from typing import List, NamedTuple, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data


class RandomWalker:
    """Uniform random walk with forbidden paths."""

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
    batch: Optional[torch.Tensor]  # shape: [batch_size]
    walks_u: torch.Tensor  # shape: [batch_size, N, num_walks, walk_length]
    walks_v: torch.Tensor  # shape: [batch_size, N, num_walks, walk_length]

    def __repr__(self):
        name = self.__class__.__name__
        u_shape = self.walks_u.shape
        v_shape = self.walks_v.shape
        return f"{name}(walks_u={u_shape}, walks_v={v_shape})"


class AttrE2vecSampler:

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_walks: int,
        walk_length: int,
        num_pos: int,
        num_neg: int,
    ):
        self.edge_index = edge_index

        self.num_pos = num_pos
        self.num_neg = num_neg

        self.random_walker = RandomWalker(
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

        # Sample random walks for positive and negative samples of edges
        pos_batch, pos_walks_u, pos_walks_v = [], [], []
        neg_batch, neg_walks_u, neg_walks_v = [], [], []
        for idx, wu, wv in zip(batch, walks_u, walks_v):
            neighborhood = self.compute_neighborhood(wu, wv)

            pb = self.sample_pos(neighborhood)
            pu, pv = self.sample_walks(pb)

            pos_batch.append(pb)
            pos_walks_u.append(pu)
            pos_walks_v.append(pv)

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
                w = self.random_walker.empty_walk()
                walks_u.append(w)
                walks_v.append(w)
            else:
                u, v = self.edge_index[:, idx].tolist()

                walks_u.append(
                    self.random_walker.walk(start_node=u, forbidden=(u, v))
                )
                walks_v.append(
                    self.random_walker.walk(start_node=v, forbidden=(u, v))
                )

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


class PaddedFeatures:

    def __init__(self, features: torch.Tensor):
        self._x = torch.cat([
            torch.zeros(1, features.shape[-1]),
            features
        ])

    def __call__(self, indices: torch.Tensor):
        return self._x[indices + 1]


class AvgAggregator(nn.Module):

    def forward(self, rw: torch.Tensor) -> torch.Tensor:
        # rw -> shape: [batch_size, num_examples, num_walks, walk_length, dim]
        return rw.mean(dim=3)


class ExponentialAvgAggregator(nn.Module):

    def forward(self, rw: torch.Tensor) -> torch.Tensor:
        walk_length = rw.size(3)

        weights = ((-1) * torch.arange(walk_length)).exp()
        weights = weights.reshape(1, 1, 1, -1, 1)

        return (rw * weights).mean(dim=3)


class GRUAggregator(nn.Module):

    def __init__(self, edge_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(input_size=edge_dim, hidden_size=edge_dim)

    def forward(self, rw: torch.Tensor) -> torch.Tensor:
        batch_size, N, num_walks, walk_length, dim = rw.size()

        h = None

        for idx in reversed(range(walk_length)):
            h = self.gru(input=rw[:, :, :, idx, :].view(-1, dim), hx=h)

        return h.view(batch_size, N, num_walks, dim)


class EdgeEncoder(nn.Module):

    def __init__(self, edge_dim: int, emb_dim: int):
        super().__init__()
        self._layers = nn.Sequential(
            nn.Linear(3 * edge_dim, 2 * edge_dim),
            nn.ReLU(),
            nn.Linear(2 * edge_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, emb_dim),
            nn.Tanh(),
        )
        self._heads = nn.ModuleList([
            nn.Sequential(nn.Linear(edge_dim, 1), nn.Tanh())
            for _ in range(3)
        ])

    def forward(
        self,
        features: torch.Tensor,
        S_u: torch.Tensor,
        S_v: torch.Tensor,
    ) -> torch.Tensor:
        alpha_f, alpha_u, alpha_v = torch.cat([
            self._heads[0](features),
            self._heads[1](S_u),
            self._heads[2](S_v),
        ], dim=-1).softmax(dim=-1).split(1, dim=-1)

        inp = torch.cat([
            alpha_f * features,
            alpha_u * S_u,
            alpha_v * S_v,
        ], dim=-1)

        return self._layers(inp)


class FeatureDecoder(nn.Module):

    def __init__(self, edge_dim: int, emb_dim: int):
        super().__init__()
        hidden_dim = (emb_dim + edge_dim) // 2

        self.layers = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, edge_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.layers(h)


class AttrE2vec(nn.Module):

    def __init__(
        self,
        graph: Data,
        num_walks: int,
        walk_length: int,
        num_pos: int,
        num_neg: int,
        mixing: float,
        aggregator: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()

        self.graph = graph

        self.num_walks = num_walks
        self.walk_length = walk_length

        self.num_pos = num_pos
        self.num_neg = num_neg

        self.mixing = mixing

        # Edge attributes are padded, so that non existing edges (`-1` index)
        # will get a zero vector assigned
        self.edge_features = PaddedFeatures(self.graph.edge_attr)

        self.aggregator = aggregator
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch: AttrE2vecBatch) -> torch.Tensor:
        features = self.edge_features(batch.batch)
        rw_u = self.edge_features(batch.walks_u)
        rw_v = self.edge_features(batch.walks_v)

        S_u = self.aggregator(rw_u).mean(dim=2)  # average over walks
        S_v = self.aggregator(rw_v).mean(dim=2)  # average over walks

        h = self.encoder(features, S_u, S_v)

        return h

    def loss(
        self,
        base: AttrE2vecBatch,
        pos: AttrE2vecBatch,
        neg: AttrE2vecBatch,
    ):
        h_base = self(base)
        h_pos = self(pos)
        h_neg = self(neg)

        batch_size = h_base.size(0)
        dim = h_base.size(-1)

        # Cosine embedding loss
        emb_loss = F.cosine_embedding_loss(
            input1=torch.cat([
                h_base.repeat_interleave(self.num_pos, dim=1),
                h_base.repeat_interleave(self.num_neg, dim=1),
            ], dim=1).reshape(-1, dim),
            input2=torch.cat([h_pos, h_neg], dim=1).reshape(-1, dim),
            target=torch.cat([
                torch.ones(batch_size, self.num_pos),
                (-1) * torch.ones(batch_size, self.num_neg),
            ], dim=1).reshape(-1),
        )

        # Edge feature reconstruction loss
        rec_loss = F.mse_loss(
            input=self.decoder(h_base),
            target=self.edge_features(base.batch),
        )

        loss = self.mixing * emb_loss + (1 - self.mixing) * rec_loss

        return loss

    def loader(self, **kwargs):
        sampler = AttrE2vecSampler(
            edge_index=self.graph.edge_index,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_pos=self.num_pos,
            num_neg=self.num_neg,
        )

        return DataLoader(
            dataset=range(self.graph.num_edges),
            collate_fn=sampler,
            **kwargs,
        )


def main():
    # TODO: train_mask, *_mask for edges
    from torch_geometric.transforms import Compose
    from pytorch_geometric_edge.datasets import Cora
    from pytorch_geometric_edge.transforms import (
        Doc2vecTransform,
        EdgeFeatureExtractorTransform,
        MatchingNodeLabelsTransform,
    )

    data = Cora(root="/tmp/pyge/")[0]

    transform = Compose([
        Doc2vecTransform(num_epochs=5, emb_dim=128),
        EdgeFeatureExtractorTransform(),
        MatchingNodeLabelsTransform(),
    ])

    out = transform(data)

    model = AttrE2vec(
        graph=out,
        num_walks=16,
        walk_length=8,
        num_pos=5,
        num_neg=10,
        mixing=0.5,
        aggregator=AvgAggregator(),
        #aggregator=ExponentialAvgAggregator(),
        #aggregator=GRUAggregator(edge_dim=out.num_edge_features),
        encoder=EdgeEncoder(edge_dim=out.num_edge_features, emb_dim=64),
        decoder=FeatureDecoder(edge_dim=out.num_edge_features, emb_dim=64),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=5e-4,
    )

    batch_size = 12 #64
    loader = model.loader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    from tqdm import tqdm

    num_epochs = 5

    for epoch in range(num_epochs):
        total_loss = 0

        for base, pos, neg in tqdm(iterable=loader, total=len(loader)):
            optimizer.zero_grad()

            loss = model.loss(base, pos, neg)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch: {epoch} -> loss: {avg_loss:.3f}")

    breakpoint()


if __name__ == "__main__":
    main()
