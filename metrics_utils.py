"""Metrics and prediction table helpers."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def compute_summary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "negative_precision": 0.0,
            "negative_recall": 0.0,
            "negative_f1": 0.0,
        }

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "negative_precision": float(precision[0]),
        "negative_recall": float(recall[0]),
        "negative_f1": float(f1[0]),
    }
