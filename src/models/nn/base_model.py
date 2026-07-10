import gc
import logging
from collections.abc import Sequence
from typing import Optional, cast

import numpy as np
import torch
import torch.nn.functional as F
from darts import TimeSeries

from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from src.models.outlier import OutlierDetector
from src.pipelines.metrics import (
    DetectionMetricSummary,
    ScoreSeriesMap,
    ValidationMetrics,
    calculate_detection_summary,
)
from src.data.dataset import SlidingWindowDataset


class OutlierModelWrapper(OutlierDetector):
    """
    Base class for outlier detection models using PyTorch Lightning.

    Adere à interface ``OutlierDetector`` (train → test_scorer → metrics),
    permitindo o uso no pipeline ``apply()``.

    Subclasses devem popular ``self.model_dict`` (um ``dict[str, LightningModule]``
    onde cada entrada é um sub-modelo nomeado).

    Nota
    ----
    O treinamento é não supervisionado: janelas deslizantes do treino são
    tratadas como classe "normal" (rótulo 0). O modelo aprende a reconhecer
    o padrão normal; pontos que fogem desse padrão recebem escore alto.
    """

    def __init__(
        self,
        input_dim: int,
        dev=False,
        model_dict: Optional[dict[str, LightningModule]] = None,
        window_size: int = 20,
        lr: float = 1e-3,
        threshold: float = 0.99,
        batch_size: int = 32,
        max_epochs: int = 10,
        accelerator: str = "cuda",
    ) -> None:
        OutlierDetector.__init__(self)

        self.lr = lr
        self.window_size = window_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self._accelerator = accelerator
        self.model_dict = model_dict
        self.dev = dev
        self.threshold = threshold

    def pipeline(
        self,
        train: list[TimeSeries],
        train_labels: list[TimeSeries],
        test: list[TimeSeries],
        test_labels: list[TimeSeries],
    ) -> dict[str, DetectionMetricSummary]:
        return OutlierDetector.apply(self, train, train_labels, test, test_labels)

    def fit(
        self,
        train: list[TimeSeries],
        train_labels: list[TimeSeries],
        test: list[TimeSeries],
        test_labels: list[TimeSeries],
    ) -> None:
        """
        Converte as séries de treino em janelas deslizantes (rótulo 0 —
        normal) e as séries de teste em janelas para validação.

        O dataset de teste é usado como validação para que o modelo veja
        padrões anômalos durante o treinamento, permitindo early stopping
        baseado em desempenho real de detecção.
        """

        ws = self.window_size
        if not train:
            raise ValueError(
                f"Nenhuma série de treino longa o suficiente para window_size={ws}. "
                f"Reduza window_size ou forneça séries mais longas."
            )
        if not test:
            raise ValueError(f"Nenhuma série de teste longa o suficiente para window_size={ws}.")

        # --- Instanciação usando o novo Dataset Nativo ---
        self.train_dataset = SlidingWindowDataset(train, labels_list=train_labels, window_size=ws)
        logging.debug(f"SHAPE BATCH 0: {self.train_dataset[0][0].shape}")
        self.test_dataset = SlidingWindowDataset(test, labels_list=test_labels, window_size=ws)

        # --- Train loader: janelas do treino (normais) ---
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        # --- Val loader: janelas do teste (contêm anomalias reais) ---
        self.val_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.trainer = Trainer(
            fast_dev_run=self.dev,
            max_epochs=self.max_epochs,
            accelerator=self._accelerator,
            enable_checkpointing=False,
            logger=True,
            enable_progress_bar=True,
        )

        assert self.model_dict is not None
        for name, model in self.model_dict.items():
            if model is None:
                logging.info(f"PL MODEL NULO — chave '{name}' — pulando fit")
                continue

            logging.info(f"Treinando modelo '{name}' …")

            self.trainer.fit(model, train_dataloaders=self.train_loader, val_dataloaders=self.val_loader)
            gc.collect()
            torch.cuda.empty_cache()

    def test_scorer(self, test: list[TimeSeries]) -> ScoreSeriesMap:
        """
        Para cada modelo em ``model_dict``, calcula o score de anomalia via
        erro de reconstrução (MSE) em janelas deslizantes.

        Usa o ``test_dataset`` (SlidingWindowDataset) para inferência em lote
        e depois converte o MSE por janela de volta para score por ponto
        via ``windows_to_point_scores``.

        Retorna um dicionário ``{nome_do_modelo: list[TimeSeries]}``.
        """
        result: ScoreSeriesMap = {}

        assert self.model_dict is not None
        for name, model in self.model_dict.items():
            if model is None:
                logging.info(f"PL MODEL NULO — chave '{name}' — pulando")
                continue

            validation_metrics = cast(
                ValidationMetrics,
                self.trainer.validate(model, dataloaders=self.val_loader),
            )
            logging.debug("VALIDATION METRICS %s: %s", name, validation_metrics)

            model.eval()
            all_mse: list[np.ndarray] = []
            logging.info(f"TESTANDO MODELO {name}!")
            with torch.no_grad():
                for batch in self.val_loader:
                    x, _ = batch
                    recon = model(x)
                    mse = F.mse_loss(recon, x, reduction="none").mean(dim=1)
                    all_mse.append(mse.cpu().numpy())

            if not all_mse:
                result[name] = [TimeSeries.from_values(np.array([0.0])) for _ in test]
                continue

            point_mse = np.concatenate(all_mse)
            # MSE de reconstrucao -> classificacao point-wise com base em threshold
            point_scores = self.test_dataset.windows_to_point_scores(point_mse, threshold=self.threshold)
            logging.debug(f"POINT: {point_scores}\n")

            model_scores: list[TimeSeries] = [
                TimeSeries.from_values(scores_arr) for scores_arr in point_scores if (scores_arr is not None)
            ]

            result[name] = model_scores

        return result

    def metrics(
        self, test_labels: Sequence[TimeSeries], scores: ScoreSeriesMap
    ) -> dict[str, DetectionMetricSummary]:
        if self._accelerator != "auto":
            device = torch.device(self._accelerator)
        elif self._accelerator == "cuda":
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("")
        else:
            device = torch.device("cpu")

        return calculate_detection_summary(test_labels, scores, device)
