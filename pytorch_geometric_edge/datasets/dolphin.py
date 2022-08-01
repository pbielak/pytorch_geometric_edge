import os
from typing import Callable, Optional

import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import from_networkx


class DolphinSocialNetwork(InMemoryDataset):
    """Dolphin social network as used in the Line2vec paper."""
    URL = "http://www-personal.umich.edu/~mejn/netdata/dolphins.zip"

    def __init__(self, root: str, transform: Optional[Callable] = None):
        super().__init__(root=root, transform=transform)

        self.data, self.slices = torch.load(f=self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["dolphins.gml", "dolphins.txt"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        path = download_url(url=self.URL, folder=self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        fname = self.raw_paths[0]

        graph = nx.read_gml(fname, label=None)
        nx.set_edge_attributes(G=graph, values=1, name="edge_weight")

        # Compute communities based on modularity
        communities = nx.algorithms.community.greedy_modularity_communities(
            G=graph,
            best_n=4,
        )
        node_labels = {
            node: community_index
            for community_index, community_members in enumerate(communities)
            for node in community_members
        }
        nx.set_node_attributes(G=graph, values=node_labels, name="y")

        # Convert to PyG `Data` object
        data = from_networkx(G=graph)

        torch.save(obj=self.collate([data]), f=self.processed_paths[0])
