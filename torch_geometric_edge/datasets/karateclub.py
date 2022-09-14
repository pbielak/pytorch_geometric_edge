from typing import Callable, Optional

import networkx as nx
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx


class KarateClub(InMemoryDataset):  # pylint: disable=abstract-method
    """The Karate Club dataset as used in the Line2vec paper"""

    def __init__(self, transform: Optional[Callable] = None):
        super().__init__(transform=transform)

        graph = nx.karate_club_graph()

        # Compute communities based on modularity
        communities = nx.algorithms.community.greedy_modularity_communities(
            G=graph,
            best_n=3,
        )
        node_labels = {
            node: community_index
            for community_index, community_members in enumerate(communities)
            for node in community_members
        }
        nx.set_node_attributes(G=graph, values=node_labels, name="y")

        # Convert to PyG `Data` object
        data = from_networkx(G=graph)
        data.edge_weight = data.weight
        del data.weight
        del data.club

        self.data, self.slices = self.collate([data])
