import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from torch_geometric_edge.datasets import Cora
from torch_geometric_edge.evaluation import LogisticRegressionEvaluator
from torch_geometric_edge.models import PairE
from torch_geometric_edge.transforms import MatchingNodeLabelsTransform


def evaluate(
    model: PairE,
    data: Data,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
):
    model.eval()
    with torch.no_grad():
        Z = model(x=data.x, edge_index=data.edge_index)

        mtr = LogisticRegressionEvaluator(["auc"]).evaluate(
            Z=Z,
            Y=data.y,
            train_mask=train_mask,
            test_mask=test_mask,
        )

    metrics = {k: f"{v * 100.:.2f} [%]" for k, v in mtr.items()}

    return metrics


def main():
    data = Cora("/tmp/pyge/")[0]
    out = MatchingNodeLabelsTransform()(data)

    model = PairE(
        num_nodes=out.num_nodes,
        node_feature_dim=out.num_node_features,
        emb_dim=128,
    )

    train_mask, test_mask = train_test_split(
        torch.arange(out.num_edges),
        stratify=out.y,
        test_size=0.8,
    )

    lr = 1e-3
    num_epochs = 30
    evaluation_interval = 5

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        x_self, x_aggr = model.extract_self_aggr(
            x=out.x,
            edge_index=out.edge_index,
        )
        h_edge = model.encode(x=out.x, edge_index=out.edge_index)
        x_self_rec, x_aggr_rec = model.decode(h=h_edge)

        loss = model.loss(
            x_self=x_self,
            x_aggr=x_aggr,
            x_self_rec=x_self_rec,
            x_aggr_rec=x_aggr_rec,
        )

        loss.backward()
        optimizer.step()

        if epoch == 0 or (epoch + 1) % evaluation_interval == 0:
            metrics = evaluate(model, out, train_mask, test_mask)

            print(f"--- Epoch {epoch:02d} ---")
            print(f"Avg loss: {loss:.3f}")
            print("Metrics:", metrics)
            print("------------------")


if __name__ == "__main__":
    main()
