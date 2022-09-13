import torch
from sklearn.model_selection import train_test_split
from torch_geometric import seed_everything

from pytorch_geometric_edge.datasets import KarateClub
from pytorch_geometric_edge.evaluation import LogisticRegressionEvaluator
from pytorch_geometric_edge.models import Line2vec, Node2vecParams


def evaluate(
    model: Line2vec,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
):
    with torch.no_grad():
        Z = model()

        mtr = LogisticRegressionEvaluator(["auc"]).evaluate(
            Z=Z,
            Y=model.line_graph.y,
            train_mask=train_mask,
            test_mask=test_mask,
        )

    C = model.centers
    R = model.radii

    metrics = {
        "value_ranges": {
            "edge_emb": f"[{Z.min().item():.3f}, {Z.max().item():.3f}]",
            "centers": f"[{C.min().item():.3f}, {C.max().item():.3f}]",
            "radii": f"[{R.min().item():.3f}, {R.max().item():.3f}]",
        },
        "classification": {k: f"{v * 100.:.2f} [%]" for k, v in mtr.items()},
    }

    return metrics


def main():
    seed_everything(42)

    data = KarateClub()[0]

    model = Line2vec(
        data=data,
        emb_dim=16,
        node2vec_params=Node2vecParams(
            p=1.0,
            q=1.0,
            walk_length=10,
            context_size=5,
            walks_per_node=1,
        ),
        alpha=1,
        lambda_=0.1,
        gamma=1,
    )

    train_mask, test_mask = train_test_split(
        torch.arange(model.line_graph.num_nodes),
        stratify=model.line_graph.y,
        test_size=0.8,
    )

    lr = 0.1
    batch_size = 4
    num_epochs = 30
    evaluation_interval = 5

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    loader = model.loader(batch_size=batch_size)

    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch, pos_rw, neg_rw in loader:
            optimizer.zero_grad()

            loss = model.loss(
                batch=batch,
                edge_emb=model(batch),
                pos_rw=pos_rw,
                neg_rw=neg_rw,
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch == 0 or (epoch + 1) % evaluation_interval == 0:
            avg_loss = epoch_loss / len(loader)
            metrics = evaluate(model, train_mask, test_mask)

            print(f"--- Epoch {epoch:02d} ---")
            print(f"Avg loss: {avg_loss:.3f}")
            print("Metrics:", metrics)
            print("------------------")


if __name__ == "__main__":
    main()
