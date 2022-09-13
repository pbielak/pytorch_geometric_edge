import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch_geometric import seed_everything
from torch_geometric.transforms import Compose
from tqdm import tqdm

from pytorch_geometric_edge.datasets import Cora
from pytorch_geometric_edge.evaluation import LogisticRegressionEvaluator
from pytorch_geometric_edge.models import (
    AttrE2vec,
    AvgAggregator,
    EdgeEncoder,
    FeatureDecoder,
)
from pytorch_geometric_edge.transforms import (
    Doc2vecTransform,
    EdgeFeatureExtractorTransform,
    MatchingNodeLabelsTransform,
)


def prepare_inductive_split(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    labels: torch.Tensor,
    train_size: float = 0.2,
):
    mask = edge_index[0] <= edge_index[1]

    train_mask, test_mask = train_test_split(
        torch.arange(edge_index.shape[1])[mask],
        train_size=train_size,
        stratify=labels[mask],
    )

    # Train split
    train_edge_index = torch.cat([
        edge_index[:, train_mask],
        edge_index[:, train_mask].flip(0),
    ], dim=-1)
    train_features = torch.cat([
        edge_attr[train_mask],
        edge_attr[train_mask],
    ], dim=0)
    train_indices = torch.arange(train_mask.shape[0])
    train_labels = labels[train_mask]

    # Test split
    test_edge_index = edge_index  # Use all edges
    test_features = edge_attr
    test_indices = test_mask
    test_labels = labels[test_mask]

    return (
        train_indices,
        train_edge_index,
        train_features,
        train_labels,
        test_indices,
        test_edge_index,
        test_features,
        test_labels,
    )


def evaluate(
    model: AttrE2vec,
    train_loader: DataLoader,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_loader: DataLoader,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
):
    model.eval()

    mask = torch.arange(train_labels.shape[0] + test_labels.shape[0])
    train_mask = mask[:train_labels.shape[0]]
    test_mask = mask[train_labels.shape[0]:]

    Z_train, Z_test = [], []
    with torch.no_grad():
        for base in tqdm(train_loader, desc="Infer (train)", leave=False):
            Z_train.append(model(train_features, base).squeeze(dim=1))

        for base in tqdm(test_loader, desc="Infer (test)", leave=False):
            Z_test.append(model(test_features, base).squeeze(dim=1))

        Z = torch.cat([*Z_train, *Z_test], dim=0)
        Y = torch.cat([train_labels, test_labels], dim=0)

        mtr = LogisticRegressionEvaluator(["auc"]).evaluate(
            Z=Z,
            Y=Y,
            train_mask=train_mask,
            test_mask=test_mask,
        )

    metrics = {k: f"{v * 100.:.2f} [%]" for k, v in mtr.items()}

    return metrics


def main():
    seed_everything(42)

    data = Cora(root="/tmp/pyge/")[0]

    transform = Compose([
        Doc2vecTransform(num_epochs=200, emb_dim=128),
        EdgeFeatureExtractorTransform(),
        MatchingNodeLabelsTransform(),
    ])

    out = transform(data)

    model = AttrE2vec(
        num_walks=16,
        walk_length=8,
        num_pos=5,
        num_neg=10,
        mixing=0.5,
        aggregator=AvgAggregator(),
        encoder=EdgeEncoder(edge_dim=out.num_edge_features, emb_dim=64),
        decoder=FeatureDecoder(edge_dim=out.num_edge_features, emb_dim=64),
    )

    (
        train_indices,
        train_edge_index,
        train_features,
        train_labels,
        test_indices,
        test_edge_index,
        test_features,
        test_labels
    ) = prepare_inductive_split(
        edge_index=out.edge_index,
        edge_attr=out.edge_attr,
        labels=out.y,
        train_size=0.2,
    )

    lr = 1e-3
    batch_size = 64
    num_epochs = 30
    evaluation_interval = 5

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_loader = model.loader(
        edge_index=train_edge_index,
        sample_pos_neg=True,
        indices=train_indices,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    sorted_train_loader = model.loader(
        edge_index=train_edge_index,
        sample_pos_neg=False,
        indices=train_indices,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    test_loader = model.loader(
        edge_index=test_edge_index,
        sample_pos_neg=False,
        indices=test_indices,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for base, pos, neg in tqdm(
            iterable=train_loader,
            total=len(train_loader),
            leave=False,
        ):
            optimizer.zero_grad()

            loss = model.loss(
                base_features=model.index_features(train_features, base.batch),
                h_base=model(train_features, base),
                h_pos=model(train_features, pos),
                h_neg=model(train_features, neg),
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch == 0 or (epoch + 1) % evaluation_interval == 0:
            avg_loss = epoch_loss / len(train_loader)
            metrics = evaluate(
                model=model,
                train_loader=sorted_train_loader,
                train_features=train_features,
                train_labels=train_labels,
                test_loader=test_loader,
                test_features=test_features,
                test_labels=test_labels,
            )

            print(f"--- Epoch: {epoch:02d} ---")
            print(f"Avg loss: {avg_loss:.3f}")
            print("Metrics:", metrics)
            print("------------------")


if __name__ == "__main__":
    main()
