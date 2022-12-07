import os
from typing import Callable, Optional

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url


class Cuneiform(InMemoryDataset):
    """The Cuneiform for handwriting Hittite cuneiform signs as used
    in the PairE paper."""
    NAME = "Cuneiform"
    URL = (
        "https://raw.githubusercontent.com/fseiffarth/AppOfTukeyDepth/"
        "master/Graphs/Cuneiform/"
    )

    def __init__(self, root: str, transform: Optional[Callable] = None):
        super().__init__(root=root, transform=transform)

        self.data, self.slices = torch.load(f=self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.NAME, "raw")

    @property
    def raw_file_names(self):
        return [
            "Cuneiform_A.txt",
            "Cuneiform_node_attributes.txt",
            "Cuneiform_edge_attributes.txt",
            "Cuneiform_edge_labels.txt",
        ]

    def download(self):
        for fname in self.raw_file_names:
            download_url(
                url=os.path.join(self.URL, fname),
                folder=self.raw_dir,
            )

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.NAME, "processed")

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        edge_index = self.read_tensor_from_csv(self.raw_paths[0]).t()
        x = self.read_tensor_from_csv(self.raw_paths[1]).float()
        edge_attr = self.read_tensor_from_csv(self.raw_paths[2]).float()
        y = self.read_tensor_from_csv(self.raw_paths[3]).squeeze(dim=-1)

        data = Data(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr,
            y=y,
            num_nodes=x.shape[0],
        )

        torch.save(obj=self.collate([data]), f=self.processed_paths[0])

    @staticmethod
    def read_tensor_from_csv(path: str) -> torch.Tensor:
        return torch.from_numpy(pd.read_csv(path, header=None).values)
