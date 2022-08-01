import tempfile

import pytest

from pytorch_geometric_edge import datasets as ds
from pytorch_geometric_edge.transforms import Line2vecTransform
from pytorch_geometric_edge.transforms.line2vec import _order_indices


def test_order_indices():
    assert _order_indices(((0, 1), (0, 2))) == (1, 0, 2)
    assert _order_indices(((0, 1), (2, 0))) == (1, 0, 2)
    assert _order_indices(((1, 0), (0, 2))) == (1, 0, 2)
    assert _order_indices(((1, 0), (2, 0))) == (1, 0, 2)

    with pytest.raises(RuntimeError) as exc:
        _order_indices(((0, 1), (2, 3)))

        assert "Line-Edge has no common node: '((0, 1), (2, 3))'" == exc.value


def test_karate_transformation():
    data = ds.KarateClub(transform=Line2vecTransform())[0]

    assert data.num_nodes == 78
    assert data.num_edges == 528 * 2
    assert data.y.unique().shape[0] == 3 + 1
    assert data.edge_weight.isfinite().all()


def test_dolphin_transformation():
    with tempfile.TemporaryDirectory() as root:
        data = ds.DolphinSocialNetwork(
            root=root,
            transform=Line2vecTransform(),
        )[0]

        assert data.num_nodes == 159
        assert data.num_edges == 923 * 2
        assert data.y.unique().shape[0] == 4 + 1
        assert data.edge_weight.isfinite().all()


def test_cora_transformation():
    with tempfile.TemporaryDirectory() as root:
        data = ds.Cora(
            root=root,
            transform=Line2vecTransform(),
        )[0]

        assert data.num_nodes == 5_278
        assert data.num_edges == 52_301 * 2
        assert data.y.unique().shape[0] == 7 + 1
        assert data.edge_weight.isfinite().all()


def test_pubmed_transformation():
    with tempfile.TemporaryDirectory() as root:
        data = ds.PubMed(
            root=root,
            transform=Line2vecTransform(),
        )[0]

        assert data.num_nodes == 44_324
        assert data.num_edges == 699_342 * 2
        assert data.y.unique().shape[0] == 3 + 1
        assert data.edge_weight.isfinite().all()
