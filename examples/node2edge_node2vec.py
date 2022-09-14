from typing import Optional

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch_geometric.nn.models import Node2Vec  # pylint: disable=E0611
from tqdm import tqdm

from pytorch_geometric_edge.datasets import Cora
from pytorch_geometric_edge.evaluation import LogisticRegressionEvaluator
from pytorch_geometric_edge.nn import Node2Edge2NodeBlock
from pytorch_geometric_edge.transforms import MatchingNodeLabelsTransform


class EnhancedEmbeddingLookup(nn.Module):

    def __init__(self, num_nodes: int, emb_dim: int, edge_index: torch.Tensor):
        super().__init__()

        self.edge_index = edge_index

        self.emb = nn.Embedding(num_nodes, emb_dim)

        self.block1 = Node2Edge2NodeBlock(
            num_nodes=num_nodes,
            node_dim=emb_dim,
            edge_dim=0,
            hidden_dim=128,
            out_dim=emb_dim,
        )

    def forward(
        self,
        batch: torch.Tensor,
        return_edge_embeddings: bool = False,
    ):
        x = self.emb.weight

        if not return_edge_embeddings:
            z = self.block1(x=x, edge_index=self.edge_index)
        else:
            z = self.block1.node2edge(x=x, edge_index=self.edge_index)

        return z if batch is None else z.index_select(0, batch)


class Model(Node2Vec):

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
    ):
        super().__init__(
            edge_index=edge_index,
            embedding_dim=512,
            walk_length=20,
            context_size=10,
            num_nodes=num_nodes,
        )

        self.embedding = EnhancedEmbeddingLookup(
            num_nodes=num_nodes,
            emb_dim=512,
            edge_index=edge_index,
        )

    def forward(self, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.embedding(batch=batch, return_edge_embeddings=True)


def evaluate(
    model: Model,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
):
    with torch.no_grad():
        Z = model()

        mtr = LogisticRegressionEvaluator(["auc"]).evaluate(
            Z=Z,
            Y=labels,
            train_mask=train_mask,
            test_mask=test_mask,
        )

    metrics = {k: f"{v * 100.:.2f} [%]" for k, v in mtr.items()}

    return metrics


def main():
    data = Cora("/tmp/pyge/")[0]
    out = MatchingNodeLabelsTransform()(data)

    train_mask, test_mask = train_test_split(
        torch.arange(out.num_edges),
        stratify=out.y,
        test_size=0.8,
    )

    model = Model(
        edge_index=out.edge_index,
        num_nodes=out.num_nodes,
    )

    lr = 1e-3
    batch_size = 128
    num_epochs = 30
    evaluation_interval = 5

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = model.loader(batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        epoch_loss = 0

        for pos_rw, neg_rw in tqdm(
            iterable=loader,
            desc="Train (batches)",
            leave=False,
        ):
            optimizer.zero_grad()

            loss = model.loss(pos_rw, neg_rw)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch == 0 or (epoch + 1) % evaluation_interval == 0:
            avg_loss = epoch_loss / len(loader)
            metrics = evaluate(model, out.y, train_mask, test_mask)

            print(f"--- Epoch {epoch:02d} ---")
            print(f"Avg loss: {avg_loss:.3f}")
            print("Metrics:", metrics)
            print("------------------")


if __name__ == "__main__":
    main()
