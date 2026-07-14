import logging
from collections.abc import Sequence
from typing import Optional

from abc import ABC, abstractmethod
from darts import TimeSeries
from lightning import LightningModule

from src.models.nn import TCN_train
from src.models.nn import VAE
from src.pipelines.metrics import (
    DetectionSummaryMap,
    ScoreSeriesMap,
)


class OutlierDetector(ABC):
    model_dict: Optional[dict[str, TCN_train | VAE]] = None

    def apply(
        self,
        train: list[TimeSeries],
        train_labels: list[TimeSeries],
        test: list[TimeSeries],
        test_labels: list[TimeSeries],
    ) -> DetectionSummaryMap:
        logging.info(f"MODELO: {self.__class__.__name__}")
        logging.info("TREINANDO ...")
        self.fit(train, train_labels, test, test_labels)
        logging.info("TESTANDO ...")
        scores = self.test_scorer(test)
        logging.info("METRIFICANDO ...")
        metrics = self.metrics(test_labels, scores)
        logging.info("FINALIZADO!\n")
        return metrics

    @abstractmethod
    def fit(
        self,
        train: list[TimeSeries],
        train_labels: list[TimeSeries],
        test: list[TimeSeries],
        test_labels: list[TimeSeries],
    ):
        pass

    @abstractmethod
    def test_scorer(self, test: list[TimeSeries]) -> ScoreSeriesMap:
        pass

    @abstractmethod
    def metrics(self, test_labels: Sequence[TimeSeries], scores: ScoreSeriesMap) -> DetectionSummaryMap:
        pass
