"""Evaluators for edge classification"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> float:
    return accuracy_score(y_true=y_true, y_pred=y_pred)


def f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> float:
    return f1_score(y_true=y_true, y_pred=y_pred)


def auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> float:
    if len(np.unique(y_true)) > 2:
        kwargs = dict(multi_class="ovr")
    else:
        kwargs = {}

    return roc_auc_score(y_true=y_true, y_score=y_score, **kwargs)


class BaseEdgeClassificationEvaluator(ABC):
    r"""Base class for all edge classification evaluators.

    We expect an edge embedding matrix:

    `\mathbf{Z} \in \mathbb{R}^{|E| \times d}`

    and edge labels:

     `\mathbf{Y} \in \{0, 1, ..., C\}^{|E|},`

     where `C` is the number of classes
    """

    METRIC_FUNCTIONS = {
        "accuracy": accuracy,
        "f1": f1,
        "auc": auc,
    }

    def __init__(
        self,
        metric_names: List[str],
        downstream_model_kwargs: Optional[dict] = None,
    ):
        self._metric_names = metric_names
        self._downstream_model_kwargs = downstream_model_kwargs or {}

    def compute_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_score: torch.Tensor,
    ) -> Dict[str, float]:
        return {
            metric_name: self.METRIC_FUNCTIONS[metric_name](
                y_true=y_true,
                y_pred=y_pred,
                y_score=y_score,
            )
            for metric_name in self._metric_names
        }

    @property
    def available_metrics(self) -> List[str]:
        return list(self.METRIC_FUNCTIONS.keys())

    @abstractmethod
    def evaluate(
        self,
        Z: torch.Tensor,
        Y: torch.Tensor,
        train_mask: torch.Tensor,
        test_mask: torch.Tensor,
        val_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        pass


class LogisticRegressionEvaluator(BaseEdgeClassificationEvaluator):
    """Evaluate edge embedding in classification using logistic regression."""

    def evaluate(
        self,
        Z: torch.Tensor,
        Y: torch.Tensor,
        train_mask: torch.Tensor,
        test_mask: torch.Tensor,
        val_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        lr = LogisticRegression(**self._downstream_model_kwargs)
        lr.fit(X=Z[train_mask], y=Y[train_mask])

        splits = [
            ("train", train_mask),
            ("test", test_mask),
        ]

        if val_mask is not None:
            splits.append(("val", val_mask))

        metrics = {}

        for split_prefix, mask in splits:
            mtr = self.compute_metrics(
                y_true=Y[mask].numpy(),
                y_pred=lr.predict(Z[mask]),
                y_score=lr.predict_proba(Z[mask]),
            )

            metrics.update({f"{split_prefix}/{k}": v for k, v in mtr.items()})

        return metrics
