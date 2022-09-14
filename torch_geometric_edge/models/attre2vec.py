import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torch_geometric_edge.samplers import AttrE2vecBatch, AttrE2vecSampler


class AvgAggregator(nn.Module):

    def forward(self, rw: torch.Tensor) -> torch.Tensor:
        # rw -> shape: [batch_size, num_examples, num_walks, walk_length, dim]
        return rw.mean(dim=3)


class ExponentialAvgAggregator(nn.Module):

    def forward(self, rw: torch.Tensor) -> torch.Tensor:
        # rw -> shape: [batch_size, num_examples, num_walks, walk_length, dim]
        walk_length = rw.size(3)

        weights = ((-1) * torch.arange(walk_length)).exp()
        weights = weights.reshape(1, 1, 1, -1, 1)

        return (rw * weights).mean(dim=3)


class GRUAggregator(nn.Module):

    def __init__(self, edge_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(input_size=edge_dim, hidden_size=edge_dim)

    def forward(self, rw: torch.Tensor) -> torch.Tensor:
        batch_size, num_examples, num_walks, walk_length, dim = rw.size()

        h = None

        for idx in reversed(range(walk_length)):
            h = self.gru(input=rw[:, :, :, idx, :].view(-1, dim), hx=h)

        return h.view(batch_size, num_examples, num_walks, dim)


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

        self.num_walks = num_walks
        self.walk_length = walk_length

        self.num_pos = num_pos
        self.num_neg = num_neg

        self.mixing = mixing

        self.aggregator = aggregator
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        x: torch.Tensor,
        batch: AttrE2vecBatch,
    ) -> torch.Tensor:
        features = self.index_features(x, index=batch.batch)
        rw_u = self.index_features(x, index=batch.walks_u)
        rw_v = self.index_features(x, batch.walks_v)

        S_u = self.aggregator(rw_u).mean(dim=2)  # average over walks
        S_v = self.aggregator(rw_v).mean(dim=2)  # average over walks

        h = self.encoder(features, S_u, S_v)

        return h

    def loss(
        self,
        base_features: torch.Tensor,
        h_base: torch.Tensor,
        h_pos: torch.Tensor,
        h_neg: torch.Tensor,
    ):
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
            target=base_features,
        )

        loss = self.mixing * emb_loss + (1 - self.mixing) * rec_loss

        return loss

    def loader(
        self,
        edge_index: torch.Tensor,
        indices: torch.Tensor,
        sample_pos_neg: bool,
        **kwargs,
    ):
        sampler = AttrE2vecSampler(
            edge_index=edge_index,
            sample_pos_neg=sample_pos_neg,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            num_pos=self.num_pos,
            num_neg=self.num_neg,
        )

        return DataLoader(
            dataset=indices,
            collate_fn=sampler,
            **kwargs,
        )

    @staticmethod
    def index_features(
        x: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        """Indexes features by given index.

        The `index` parameter is a multidimensional index, but with values of
        `-1` denoting missing edges. Such edges should receive zero vectors.
        If we append a zero vector at the end of the feature tensor, we obtain
        exactly such behaviour (using `-1` while indexing retrieves the last
        element of a given sequence).
        """
        x_padded = torch.cat([
            x,
            torch.zeros(1, x.shape[-1]),
        ], dim=0)

        return x_padded[index]
