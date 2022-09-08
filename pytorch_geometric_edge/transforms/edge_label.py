import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class MatchingNodeLabelsTransform(BaseTransform):
    """Extract edge labels based on the given node labels.

    The label `y_{(u, v)}` a given edge `(u, v)` is defined as follows:
    - `y_{(u, v)} = y_k`, if `u` and `v` have the same (matching) label `y_k`,
    - `y_{(u, v)} = -1`, if the node labels are different
    """

    def __call__(self, data: Data) -> Data:
        out = data.clone()

        y = data.y

        edge_y = torch.ones(out.num_edges).long() * (-1)

        mask = y[out.edge_index[0]] == y[out.edge_index[1]]
        edge_y[mask] = y[out.edge_index[0, mask]]

        out.y = edge_y

        return out
