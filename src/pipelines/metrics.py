from collections.abc import Sequence
import logging
from typing import TypeAlias, TypedDict, cast

from darts import TimeSeries
import numpy as np
from torch import Tensor
import torch
import plotly.graph_objects as go
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
    _confusion_matrices: dict[str, dict[str, list[list[list[int]]]]] = {}

    @classmethod
    def clear(cls) -> None:
        cls._data = {}
        cls._confusion_matrices = {}

    @classmethod
    def add(cls, model_name: str, split: str, metrics: dict[str, Tensor | float]) -> None:
        split_store = cls._data.setdefault(model_name, {}).setdefault(split, [])

        scalar_metrics: dict[str, float] = {}
        for name, value in metrics.items():
            if isinstance(value, Tensor):
                if value.numel() == 1:
                    scalar_metrics[name] = float(value.detach().cpu())
                else:
                    # tensor multi-elemento (ex: matriz de confusão) -> loga separadamente
                    logging.debug(f"CentralMetricsStore: tensor multi-elemento {value}")
                    cls.add_confusion_matrix(model_name, split, value)
            elif isinstance(value, (int, float)):
                scalar_metrics[name] = float(value)
            # strings (ex: "name") e outros tipos não-numéricos são ignorados aqui

        split_store.append(scalar_metrics)

    @classmethod
    def add_confusion_matrix(cls, model_name: str, split: str, cm: Tensor | np.ndarray) -> None:
        """
        Loga uma matriz de confusão (2D) para plotar posteriormente.
        Aceita Tensor ou ndarray, converte para lista aninhada de ints.
        """
        if isinstance(cm, Tensor):
            cm_list = cm.detach().cpu().to(torch.int64).tolist()
        else:
            cm_list = np.asarray(cm).astype(int).tolist()

        store = cls._confusion_matrices.setdefault(model_name, {}).setdefault(split, [])
        store.append(cm_list)

    @classmethod
    def get_confusion_matrices(cls, model_name: str, split: str = "test") -> list[list[list[int]]]:
        return cls._confusion_matrices.get(model_name, {}).get(split, [])

    @classmethod
    def as_dict(cls) -> dict[str, dict[str, list[dict[str, float]]]]:
        return cls._data

    @classmethod
    def plot_metric(cls, metric_name: str, split: str = "validation") -> go.Figure:
        # precision, recall e f1 são interdependentes — plota como tabela única
        if metric_name in {"precision", "recall", "f1"}:
            return cls._plot_detection_table(split=split)

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
    def _plot_detection_table(cls, split: str = "test") -> go.Figure:
        headers = ["Modelo", "Precision", "Recall", "F1"]
        col_model: list[str] = []
        col_precision: list[str] = []
        col_recall: list[str] = []
        col_f1: list[str] = []

        for model_name, model_splits in cls._data.items():
            epoch_metrics = model_splits.get(split, [])
            if not epoch_metrics:
                continue
            # última época que contém as 3 métricas
            last = epoch_metrics[-1]
            p = last.get("precision")
            r = last.get("recall")
            f = last.get("f1")
            if p is not None and r is not None and f is not None:
                col_model.append(model_name)
                col_precision.append(f"{p:.4f}")
                col_recall.append(f"{r:.4f}")
                col_f1.append(f"{f:.4f}")

        fig = go.Figure(
            data=go.Table(
                header=dict(values=headers, align="left"),
                cells=dict(
                    values=[col_model, col_precision, col_recall, col_f1],
                    align="left",
                    format=[None, ".4f", ".4f", ".4f"],
                ),
            )
        )
        fig.update_layout(title=f"Métricas de Detecção ({split})", template="plotly_white")
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

    @classmethod
    def plot_confusion_matrix(cls, model_name: str, split: str = "test", index: int = -1) -> go.Figure:
        """
        Plota a matriz de confusão de um modelo/split específico como heatmap.
        `index` seleciona qual matriz logada (-1 = última, útil se houver várias épocas/runs).
        """
        matrices = cls.get_confusion_matrices(model_name, split)
        if not matrices:
            raise ValueError(f"Nenhuma matriz de confusão encontrada para '{model_name}' / '{split}'.")

        cm = np.array(matrices[index])
        labels = ["Normal", "Outlier"] if cm.shape == (2, 2) else [str(i) for i in range(cm.shape[0])]

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=[f"Predito: {label}" for label in labels],
                y=[f"Real: {label}" for label in labels],
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues",
            )
        )
        fig.update_layout(
            title=f"Matriz de Confusão — {model_name} ({split})",
            template="plotly_white",
            yaxis_autorange="reversed",  # mantém a origem no canto superior esquerdo
        )
        return fig

    @classmethod
    def plot_all_confusion_matrices(cls, split: str = "test") -> dict[str, go.Figure]:
        return {
            model_name: cls.plot_confusion_matrix(model_name, split=split)
            for model_name in cls._confusion_matrices
            if cls._confusion_matrices[model_name].get(split)
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
    test_labels: Sequence[TimeSeries],
    scores: ScoreSeriesMap,
    device: torch.device,
    model_test_metrics: dict[str, MetricCollection] | None = None,
) -> DetectionSummaryMap:
    result: DetectionSummaryMap = {}

    for name, model_scores in scores.items():
        if model_test_metrics is not None and name in model_test_metrics:
            test_metrics = model_test_metrics[name].to(device)
            test_metrics.reset()
        else:
            test_metrics = build_test_metrics().to(device)

        for label_ts, score_ts in zip(test_labels, model_scores):
            labels = torch.as_tensor(label_ts.values(copy=False).flatten(), device=device)
            scores_tensor = torch.as_tensor(score_ts.values(copy=False).flatten(), device=device)
            scores_tensor = torch.nan_to_num(scores_tensor, nan=0.0)
            labels = labels.to(dtype=torch.int)
            test_metrics.update(scores_tensor, labels)

        test_result = test_metrics.compute()
        test_result["name"] = name  # type: ignore[arg-type]
        result[name] = cast(DetectionMetricSummary, test_result)

    return result
