import os
import ssl
import sys
import urllib
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch_geometric.data import Data, InMemoryDataset


class UNSWNB15(InMemoryDataset):
    """The UNSW-NB15 cybersecurity dataset."""
    URL = (
        "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download"
        "?path=%2FUNSW-NB15%20-%20CSV%20Files&files={filename}"
    )
    NAME = "UNSW_NB15"
    VERSIONS = (
        "ip/binary",
        "ip/multi",
        "ip_port/binary",
        "ip_port/multi",
    )

    def __init__(
        self,
        version: str,
        root: str,
        transform: Optional[Callable] = None,
    ):
        if version not in self.VERSIONS:
            raise ValueError(
                f"Unknown dataset version - choose from: {self.VERSIONS}"
            )
        self.version = version
        self.node_type, self.label_type = version.split("/")
        super().__init__(root=root, transform=transform)

        self.data, self.slices = torch.load(f=self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.NAME, "raw")

    @property
    def raw_file_names(self) -> List[str]:
        return [
            "UNSW-NB15_1.csv",
            "UNSW-NB15_2.csv",
            "UNSW-NB15_3.csv",
            "UNSW-NB15_4.csv",
            "NUSW-NB15_features.csv",
        ]

    @property
    def processed_dir(self) -> str:
        return os.path.join(
            self.root,
            self.NAME,
            self.version.replace("/", "_"),
        )

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self):
        for filename in self.raw_file_names:
            download_url(
                url=self.URL.format(filename=filename),
                folder=self.raw_dir,
                filename=filename,
            )

    def process(self):
        connections = self._read_raw_connections()

        edge_index = self._build_edge_index(connections=connections)
        edge_attr = self._build_edge_attr(connections=connections)
        labels = self._build_labels(connections=connections)

        data = Data(
            num_nodes=edge_index.flatten().unique().shape[0],
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=labels,
        )
        torch.save(obj=self.collate([data]), f=self.processed_paths[0])

    def _read_raw_connections(self) -> pd.DataFrame:
        col_names = pd.read_csv(
            self.raw_paths[-1],
            encoding_errors="replace",
        )["Name"].tolist()

        parts = []
        for filename in self.raw_paths[:-1]:
            print(f"Reading file: {filename}")
            parts.append(
                pd.read_csv(
                    filename,
                    index_col=None,
                    header=None,
                    low_memory=False,
                ),
            )

        df = pd.concat(parts, axis=0, ignore_index=True)
        df.columns = col_names

        df["attack_cat"] = df["attack_cat"].fillna("")
        df = df.fillna(0)

        df["sport"] = df["sport"].apply(to_integer)
        df["dsport"] = df["dsport"].apply(to_integer)

        df = df[~df["sport"].isna() & ~df["dsport"].isna()]

        df["sport"] = df["sport"].astype(int)
        df["dsport"] = df["dsport"].astype(int)

        return df

    def _build_edge_index(self, connections: pd.DataFrame) -> torch.Tensor:
        if self.node_type == "ip":
            src = connections["srcip"]
            dst = connections["dstip"]
        else:
            assert self.node_type == "ip_port"
            src = connections["srcip"] + ":" + connections["sport"].apply(str)
            dst = connections["dstip"] + ":" + connections["dsport"].apply(str)

        nodes = set(src).union(set(dst))
        nodes_mapping = dict(zip(nodes, range(len(nodes))))

        return torch.tensor([
            src.apply(lambda node: nodes_mapping[node]).tolist(),
            dst.apply(lambda node: nodes_mapping[node]).tolist(),
        ])

    def _build_edge_attr(self, connections: pd.DataFrame) -> torch.Tensor:
        # Categorical columns
        proto = OneHotEncoder(sparse=False).fit_transform(
            connections["proto"].values.reshape(-1, 1)
        )
        state = OneHotEncoder(sparse=False).fit_transform(
            connections["state"].values.reshape(-1, 1)
        )
        service = OneHotEncoder(sparse=False).fit_transform(
            connections["service"].values.reshape(-1, 1)
        )

        # Numerical columns
        numerical_columns = connections.columns.difference([
            "srcip",
            "dstip",
            "sport",
            "dsport",
            "proto",
            "state",
            "service",
            "attack_cat",
            "Label",
            "ct_ftp_cmd",
        ]).tolist()

        if self.node_type == "ip":
            numerical_columns.append("sport")
            numerical_columns.append("dsport")

        numerical = connections[numerical_columns].values

        return torch.cat([
            torch.from_numpy(proto).float(),
            torch.from_numpy(state).float(),
            torch.from_numpy(service).float(),
            torch.from_numpy(numerical).float(),
        ], dim=-1)

    def _build_labels(self, connections: pd.DataFrame) -> torch.Tensor:
        if self.label_type == "binary":
            return torch.from_numpy(connections["Label"].values)

        assert self.label_type == "multi"
        attack_cat = LabelEncoder().fit_transform(connections["attack_cat"])
        return torch.from_numpy(attack_cat)


def to_integer(x):
    try:
        return int(x)
    except ValueError:
        return np.nan


def download_url(url: str, filename: str, folder: str, log: bool = True):
    """Downloads an artifact from the given URL.

    This function is based on the `download_url` function from PyG.
    The original implementation handles the filename incorrectly
    for this dataset.
    """
    path = os.path.join(folder, filename)

    if os.path.exists(path):
        if log:
            print(f"Using existing file {filename}", file=sys.stderr)
        return path

    if log:
        print(f"Downloading {url}", file=sys.stderr)

    os.makedirs(folder, exist_ok=True)

    context = ssl._create_unverified_context()  # pylint: disable=W0212
    data = urllib.request.urlopen(  # pylint: disable=R1732
        url,
        context=context,
    )

    with open(path, 'wb') as f:
        f.write(data.read())

    return path
