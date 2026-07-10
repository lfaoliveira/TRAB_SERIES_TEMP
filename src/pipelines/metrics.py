from collections.abc import Sequence
from typing import TypeAlias, TypedDict

from darts import TimeSeries
from torch import Tensor
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torchmetrics import (
    ConfusionMatrix,
    FBetaScore,
    MetricCollection,
    MeanAbsoluteError,
    MeanSquaredError,
    Precision,
    Recall,
    SymmetricMeanAbsolutePercentageError,
)


class ValidationMetricLog(TypedDict, total=False):
    val_mse: float | Tensor
    val_smape: float | Tensor
    val_mae: float | Tensor


class TestMetricLog(TypedDict, total=False):
    val_mse: float | Tensor
    val_smape: float | Tensor
    val_mae: float | Tensor
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


class CentralMetricsStore:
    _data: dict[str, dict[str, list[dict[str, float]]]] = {}

    @classmethod
    def clear(cls) -> None:
        cls._data = {}

    @classmethod
    def add(cls, model_name: str, split: str, metrics: dict[str, Tensor | float]) -> None:
        split_store = cls._data.setdefault(model_name, {}).setdefault(split, [])
        split_store.append(
            {
                name: float(value.detach().cpu()) if isinstance(value, Tensor) else float(value)
                for name, value in metrics.items()
            }
        )

    @classmethod
    def as_dict(cls) -> dict[str, dict[str, list[dict[str, float]]]]:
        return cls._data

    @classmethod
    def plot_metric(cls, metric_name: str, split: str = "validation") -> go.Figure:
        fig = go.Figure()

        for model_name, model_splits in cls._data.items():
            epoch_metrics = model_splits.get(split, [])
            epochs = list(range(1, len(epoch_metrics) + 1))
            values = [
                epoch_metric.get(metric_name) for epoch_metric in epoch_metrics if metric_name in epoch_metric
            ]
            if not values:
                continue
            fig.add_trace(
                go.Scatter(x=epochs[: len(values)], y=values, mode="lines+markers", name=model_name)
            )

        fig.update_layout(
            title=f"{metric_name} por modelo ({split})",
            xaxis_title="Epoch",
            yaxis_title=metric_name,
            template="plotly_white",
        )
        return fig

    @classmethod
    def plot_all_metrics(cls, split: str = "validation") -> dict[str, go.Figure]:
        metric_names: set[str] = set()
        for model_splits in cls._data.values():
            for metric_block in model_splits.get(split, []):
                metric_names.update(metric_block.keys())

        return {
            metric_name: cls.plot_metric(metric_name, split=split) for metric_name in sorted(metric_names)
        }


CentralMetricsPlotter = CentralMetricsStore


def build_validation_metrics() -> MetricCollection:
    return MetricCollection(
        {
            "val_mse": MeanSquaredError(),
            "val_smape": SymmetricMeanAbsolutePercentageError(),
            "val_mae": MeanAbsoluteError(),
        }
    )


def build_test_metrics() -> MetricCollection:
    return MetricCollection(
        {
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


if __name__ == "__main__":
    CentralMetricsStore.clear()
    CentralMetricsStore.add("TCN", "validation", {"val_mse": 1.0, "val_smape": 2.0, "val_mae": 3.0})
    CentralMetricsStore.add("VAE", "validation", {"val_mse": 2.0, "val_smape": 3.0, "val_mae": 4.0})
    CentralMetricsStore.add("TCN", "test", {"val_mse": 1.5, "val_smape": 2.5, "val_mae": 3.5})
    CentralMetricsStore.add("VAE", "test", {"val_mse": 2.5, "val_smape": 3.5, "val_mae": 4.5})
    validation_metrics = build_validation_metrics()
    test_metrics = build_test_metrics()
    assert set(validation_metrics.keys()) == {"val_mse", "val_smape", "val_mae"}
    assert {"val_mse", "val_smape", "val_mae", "f1", "precision", "recall", "cm"} <= set(test_metrics.keys())
    figures = CentralMetricsStore.plot_all_metrics()
    assert set(figures.keys()) == {"val_mae", "val_mse", "val_smape"}
