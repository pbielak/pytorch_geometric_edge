from collections import defaultdict
from typing import Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, LineGraph
from torch_geometric.utils import degree


class Line2vecTransform(BaseTransform):
    """Line graph transformation as described in the Line2vec paper."""

    def __call__(self, data: Data) -> Data:
        original_graph = data.clone()

        if "edge_weight" not in original_graph:
            original_graph.edge_weight = torch.ones(original_graph.num_edges)

        # The Line2vec paper is using undirected graphs
        line_graph = LineGraph(force_directed=False)(original_graph.clone())

        line_graph.y = _compute_edge_labels(graph=original_graph)

        line_graph.edge_weight = _compute_edge_weights(
            graph=original_graph,
            line_graph=line_graph,
        )

        return line_graph


def _compute_edge_labels(graph: Data) -> torch.Tensor:
    """Compute edge labels (in terms of the original graph)

    If the two nodes of a given edge have the same label, the edge receives
    the same label. However, if they are different, this edge will receive a
    special class (-1). In the Line2vec paper, the authors omit such edges
    (class = -1) during evaluation.
    """
    node_labels = graph.y

    edges = _remove_reciprocal_edges(graph.edge_index)
    edge_y = torch.ones(edges.shape[1]).long() * (-1)

    mask = node_labels[edges[0]] == node_labels[edges[1]]
    edge_y[mask] = node_labels[edges[0][mask]]

    return edge_y


def _compute_edge_weights(graph: Data, line_graph: Data) -> torch.Tensor:
    """Compute edge weight in line graph according to the Line2vec paper."""
    # Original graph
    degrees = degree(graph.edge_index[0])
    edges = graph.edge_index
    edges_without_reciprocal = _remove_reciprocal_edges(edges)
    weights = graph.edge_weight

    edge2idx = {tuple(e): idx for idx, e in enumerate(edges.t().tolist())}
    neighbors = defaultdict(list)
    for src, dst in edges.t().tolist():
        neighbors[src].append(dst)

    # Line graph
    line_edge_weights = []

    for u, v in line_graph.edge_index.t().tolist():
        e1 = tuple(edges_without_reciprocal[:, u].tolist())
        e2 = tuple(edges_without_reciprocal[:, v].tolist())

        i, j, k = _order_indices((e1, e2))

        d_i = degrees[i]
        d_j = degrees[j]

        w_jk = weights[edge2idx[(j, k)]]
        w_ij = weights[edge2idx[(i, j)]]
        w_jr = weights[[edge2idx[(j, r)] for r in neighbors[j]]].sum()

        line_edge_weights.append(
            (d_i / (d_i + d_j)) * (w_jk / (w_jr - w_ij))
        )

    return torch.tensor(line_edge_weights)


def _order_indices(
    edge: Tuple[Tuple[int, int], Tuple[int, int]],
) -> Tuple[int, int, int]:
    """Each edge in the line graph is composed of two edges `e_ij` and `e_jk`
    from the original graph. The indices might be mixed up, but we want to know
    the common node index (i.e., `j` in the example above)."""
    ((a, b), (c, d)) = edge

    if a == c:
        return b, a, d

    if a == d:
        return b, a, c

    if b == c:
        return a, b, d

    if b == d:
        return a, b, c

    raise RuntimeError(f"Line-Edge has no common node: '{edge}'")


def _remove_reciprocal_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Remove reciprocal edges (similarly to the `LineGraph` transform)"""
    return edge_index[:, edge_index[0] < edge_index[1]]
