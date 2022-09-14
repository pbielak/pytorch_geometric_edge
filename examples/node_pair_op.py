from pprint import pprint

import torch
from sklearn.model_selection import train_test_split
from torch_geometric import seed_everything
from torch_geometric.nn.models import Node2Vec  # pylint: disable=E0611
from tqdm import tqdm

from torch_geometric_edge.datasets import Cora
from torch_geometric_edge.evaluation import LogisticRegressionEvaluator
from torch_geometric_edge.nn import (
    AvgNodePairOp,
    HadamardNodePairOp,
    L1NodePairOp,
    L2NodePairOp,
)
from torch_geometric_edge.transforms import MatchingNodeLabelsTransform


def evaluate(
    model: Node2Vec,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
):
    with torch.no_grad():
        Z = model()

        metrics = {}

        for name, op in (
            ("avg", AvgNodePairOp()),
            ("hadamard", HadamardNodePairOp()),
            ("L1", L1NodePairOp()),
            ("L2", L2NodePairOp()),
        ):
            mtr = LogisticRegressionEvaluator(
                metric_names=["auc"],
                downstream_model_kwargs=dict(max_iter=1000),
            ).evaluate(
                Z=op(Z, edge_index),
                Y=labels,
                train_mask=train_mask,
                test_mask=test_mask,
            )
            metrics[name] = {k: f"{v * 100.:.2f} [%]" for k, v in mtr.items()}

    return metrics


def main():
    seed_everything(42)

    data = Cora("/tmp/pyge/")[0]
    out = MatchingNodeLabelsTransform()(data)

    model = Node2Vec(
        edge_index=out.edge_index,
        embedding_dim=128,
        walk_length=20,
        context_size=10,
    )

    mask = out.edge_index[0] <= out.edge_index[1]

    edges = out.edge_index[:, mask]
    labels = out.y[mask]

    train_mask, test_mask = train_test_split(
        torch.arange(edges.shape[1]),
        stratify=labels,
        test_size=0.8,
    )

    lr = 0.1
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

            loss = model.loss(
                pos_rw=pos_rw,
                neg_rw=neg_rw,
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch == 0 or (epoch + 1) % evaluation_interval == 0:
            avg_loss = epoch_loss / len(loader)
            metrics = evaluate(model, edges, labels, train_mask, test_mask)

            print(f"--- Epoch {epoch:02d} ---")
            print(f"Avg loss: {avg_loss:.3f}")
            print("Metrics:")
            pprint(metrics)
            print("------------------")


if __name__ == "__main__":
    main()
