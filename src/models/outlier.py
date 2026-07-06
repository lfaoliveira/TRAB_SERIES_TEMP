import logging
from collections.abc import Sequence
from typing import Optional, TypeAlias, TypedDict

from abc import ABC, abstractmethod
from darts import TimeSeries
from lightning import LightningModule
from torch import Tensor
from torchmetrics import (
    AveragePrecision,
    ConfusionMatrix,
    FBetaScore,
    MetricCollection,
    Precision,
    Recall,
    AUROC,
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


class DetectionMetricSummary(TypedDict):
    name: str
    auc_roc: float
    auc_pr: float


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


class OutlierDetector(ABC):
    model_dict: Optional[dict[str, LightningModule]] = None

    def apply(
        self, train: list[TimeSeries], test: list[TimeSeries], test_labels: Sequence[TimeSeries]
    ) -> DetectionSummaryMap:
        logging.info(f"MODELO: {self.__class__.__name__}")
        logging.info("TREINANDO ...")
        self.fit(train, test)
        logging.info("TESTANDO ...")
        scores = self.test_scorer(test)
        logging.info("METRIFICANDO ...")
        metrics = self.metrics(test_labels, scores)
        logging.info("FINALIZADO!\n")
        return metrics

    @abstractmethod
    def fit(self, train: list[TimeSeries], test: list[TimeSeries]):
        pass

    @abstractmethod
    def test_scorer(self, test: list[TimeSeries]) -> ScoreSeriesMap:
        pass

    @abstractmethod
    def metrics(self, test_labels: Sequence[TimeSeries], scores: ScoreSeriesMap) -> DetectionSummaryMap:
        pass
