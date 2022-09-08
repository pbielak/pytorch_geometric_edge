import torch
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class Doc2vecTransform(BaseTransform):
    """Transforms binary Bag of Words features into dense Doc2vec features."""

    def __init__(self, num_epochs: int, emb_dim: int):
        self.num_epochs = num_epochs
        self.emb_dim = emb_dim

    def __call__(self, data: Data) -> Data:
        out = data.clone()

        documents = [
            TaggedDocument(
                words=[
                    str(idx)
                    for idx in torch.nonzero(out.x[i]).flatten().tolist()
                ],
                tags=[f"Doc_{i}"]
            )
            for i in range(out.num_nodes)
        ]

        doc2vec = Doc2Vec(vector_size=self.emb_dim, dm=0)
        doc2vec.build_vocab(corpus_iterable=documents)
        doc2vec.train(
            corpus_iterable=documents,
            total_examples=len(documents),
            epochs=self.num_epochs,
        )

        out.x = torch.from_numpy(doc2vec.dv.vectors)

        return out


class EdgeFeatureExtractorTransform(BaseTransform):

    def __call__(self, data: Data) -> Data:
        out = data.clone()

        x = out.x
        del out.x

        x_u = x[out.edge_index[0]]
        x_v = x[out.edge_index[1]]

        out.edge_attr = torch.cat([
            F.cosine_similarity(x1=x_u, x2=x_v).unsqueeze(dim=-1),
            torch.cat([x_u, x_v], dim=-1),
        ], dim=-1)

        return out
