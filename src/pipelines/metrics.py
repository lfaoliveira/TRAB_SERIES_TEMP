from collections.abc import Sequence
from typing import TypeAlias, TypedDict

from darts import TimeSeries
from torch import Tensor
import torch
from torchmetrics import (
    AUROC,
    AveragePrecision,
    ConfusionMatrix,
    FBetaScore,
    MetricCollection,
    Precision,
    Recall,
)


class ValidationMetricLog(TypedDict, total=False):
    val_auroc: float | Tensor
    val_f1: float | Tensor


class TestMetricLog(TypedDict, total=False):
    auroc: float | Tensor
    ap: float | Tensor
    f1: float | Tensor
    precision: float | Tensor
    recall: float | Tensor
    cm: Tensor


class DetectionMetricSummary(TestMetricLog):
    name: str


ValidationMetrics: TypeAlias = list[ValidationMetricLog]
TestMetrics: TypeAlias = list[TestMetricLog]
ScoreSeriesMap: TypeAlias = dict[str, list[TimeSeries]]
DetectionSummaryMap: TypeAlias = dict[str, DetectionMetricSummary]


def build_validation_metrics() -> MetricCollection:
    return MetricCollection(
        {
            "val_auroc": AUROC(task="binary"),
            "val_f1": FBetaScore(task="binary", beta=1.0),
        }
    )


def build_test_metrics() -> MetricCollection:
    return MetricCollection(
        {
            "auroc": AUROC(task="binary"),
            "ap": AveragePrecision(task="binary"),
            "f1": FBetaScore(task="binary", beta=1.0),
            "precision": Precision(task="binary"),
            "recall": Recall(task="binary"),
            "cm": ConfusionMatrix(task="binary"),
        }
    )


def anomaly_detect_mse(mse_tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    # mse_tensor: (batch, window)
    max = mse_tensor.max(dim=0).values
    limiar = threshold * max
    mask = mse_tensor > limiar
    anomaly: torch.Tensor = mse_tensor[mask]

    return anomaly


def calculate_detection_summary(
    test_labels: Sequence[TimeSeries], scores: ScoreSeriesMap, device: torch.device
) -> DetectionSummaryMap:
    result: DetectionSummaryMap = {}

    for name, model_scores in scores.items():
        test_metrics = build_test_metrics().to(device)

        for label_ts, score_ts in zip(test_labels, model_scores):
            labels = torch.as_tensor(label_ts.values(copy=False).flatten(), device=device)
            scores_tensor = torch.as_tensor(score_ts.values(copy=False).flatten(), device=device)
            scores_tensor = torch.nan_to_num(scores_tensor, nan=0.0)
            labels = labels.to(dtype=torch.int)
            test_metrics.update(scores_tensor, labels)

        test_result: dict[str, float] = test_metrics.compute()

        result[name] = test_result
        result[name]["name"] = name

    return result
