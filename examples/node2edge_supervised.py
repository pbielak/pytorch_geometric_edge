import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch_geometric.transforms import RandomLinkSplit

from torch_geometric_edge.datasets import Cora
from torch_geometric_edge.nn import Node2Edge, Node2Edge2NodeBlock
from torch_geometric_edge.transforms import MatchingNodeLabelsTransform


class Model(nn.Module):

    def __init__(
        self,
        num_nodes: int,
        node_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.block1 = Node2Edge2NodeBlock(
            num_nodes=num_nodes,
            node_dim=node_dim,
            edge_dim=0,
            hidden_dim=256,
            out_dim=128,
        )
        self.n2e = Node2Edge(
            node_dim=128,
            edge_dim=0,
            out_dim=num_classes,
            net=nn.Sequential(
                nn.Linear(2 * 128, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
                nn.LogSoftmax(dim=-1),
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        h = self.block1(x=x, edge_index=edge_index)
        y_pred = self.n2e(x=h, edge_index=edge_index)

        return y_pred


def main():
    data = Cora("/tmp/pyge/")[0]
    out = MatchingNodeLabelsTransform()(data)

    train_data, val_data, test_data = RandomLinkSplit(
        num_val=0.1,
        num_test=0.8,
        is_undirected=True,
        key='y',
        add_negative_train_samples=False,
    )(out)

    model = Model(
        num_nodes=out.num_nodes,
        node_dim=out.num_node_features,
        num_classes=out.y.unique().shape[0],
    )

    lr = 1e-3
    num_epochs = 30
    evaluation_interval = 5

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        y_pred = model(train_data.x, train_data.y_index)
        y_true = train_data.y

        loss = F.nll_loss(input=y_pred, target=y_true)

        loss.backward()
        optimizer.step()

        if epoch == 0 or (epoch + 1) % evaluation_interval == 0:
            with torch.no_grad():
                metrics = {
                    "train/auc": roc_auc_score(
                        y_true=train_data.y,
                        y_score=model(train_data.x, train_data.y_index).exp(),
                        multi_class="ovr",
                    ),
                    "val/auc": roc_auc_score(
                        y_true=val_data.y,
                        y_score=model(val_data.x, val_data.y_index).exp(),
                        multi_class="ovr",
                    ),
                    "test/auc": roc_auc_score(
                        y_true=test_data.y,
                        y_score=model(test_data.x, test_data.y_index).exp(),
                        multi_class="ovr",
                    ),
                }

                metrics = {k: f"{v * 100:.2f} [%]" for k, v in metrics.items()}

            print(f"--- Epoch {epoch:02d} ---")
            print(f"Avg loss: {loss:.3f}")
            print("Metrics:", metrics)
            print("------------------")


if __name__ == "__main__":
    main()
