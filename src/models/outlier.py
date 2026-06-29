import logging
from typing import Any

import numpy as np
from abc import ABC, abstractmethod
from darts import TimeSeries


class OutlierDetector(ABC):
    # NOTE: COMENTADO POIS NEM TODOS MODELOS USAM JANELAMENTO!
    # def __init__(
    #     self, group_id: str = "series_id", target_id: str = "target", window_size=7
    # ) -> None:
    #     super().__init__()
    #     self.window_size = window_size
    #     self.group_id = group_id
    #     self.target_id = target_id

    def apply(
        self, train: list[TimeSeries], test: list[TimeSeries], test_labels: np.ndarray
    ) -> list[Any]:
        logging.info(f"MODELO: {self.__class__.__name__}")
        logging.info("TREINANDO ...")
        self.train(train)
        logging.info("TESTANDO ...")
        scores = self.test_scorer(test)
        logging.info("METRIFICANDO ...")
        metrics = self.metrics(test_labels, scores)
        logging.info("FINALIZADO!\n")
        return metrics

    @abstractmethod
    def train(self, train: list[TimeSeries]):
        pass

    @abstractmethod
    def test_scorer(self, test: list[TimeSeries]) -> list[TimeSeries]:
        pass

    @abstractmethod
    def metrics(self, test_labels: np.ndarray, scores: list[TimeSeries]) -> list[Any]:
        pass
