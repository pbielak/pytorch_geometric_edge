from typing import Callable, Optional

from torch_geometric.datasets import Planetoid


class PubMed(Planetoid):
    """Wrapper for PubMed dataset so it can be used directly

    The `download()` and `process()` methods must be redeclared here, otherwise
    the check `if 'download' in self.__class__.__dict__` in the super class will
    return False and the dataset will not get downloaded and processed.
    """

    def __init__(self, root: str, transform: Optional[Callable] = None):
        super().__init__(
            root=root,
            name="PubMed",
            split="public",
            transform=transform,
        )

    def download(self):
        super().download()

    def process(self):
        super().process()
