import logging
from collections.abc import Sequence
from typing import Any, Optional

from lightning import LightningModule
import numpy as np
from abc import ABC, abstractmethod
from darts import TimeSeries


class OutlierDetector(ABC):
    model_dict: Optional[dict[str, LightningModule]] = None

    def apply(
        self, train: list[TimeSeries], test: list[TimeSeries], test_labels: Sequence[TimeSeries]
    ) -> dict[str, dict[str, Any]]:
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
    def test_scorer(self, test: list[TimeSeries]) -> dict[str, list[TimeSeries]]:
        pass

    @abstractmethod
    def metrics(
        self, test_labels: Sequence[TimeSeries], scores: dict[str, list[TimeSeries]]
    ) -> dict[str, dict[str, Any]]:
        pass
