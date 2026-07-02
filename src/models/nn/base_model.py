import gc
import logging
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from darts import TimeSeries
from darts.utils.data.torch_datasets.training_dataset import SequentialTorchTrainingDataset
from darts.utils.data.torch_datasets.inference_dataset import SequentialTorchInferenceDataset
from lightning import LightningModule, Trainer
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import (
    AveragePrecision,
    ConfusionMatrix,
    FBetaScore,
    MetricCollection,
    Precision,
    Recall,
    ROC,
)
from src.data.utils import extract_windows
from src.models.outlier import OutlierDetector


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
        model_dict: Optional[dict[str, LightningModule]] = None,
        window_size: int = 20,
        lr: float = 1e-3,
        pos_weight: float = 1.0,
        batch_size: int = 32,
        max_epochs: int = 10,
        accelerator: str = "auto",
    ) -> None:
        OutlierDetector.__init__(self)

        self.model: nn.Module = nn.Identity()
        self.lr = lr
        self.window_size = window_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self._accelerator = accelerator
        self.model_dict = model_dict

        self.val_metrics = MetricCollection(
            {
                "val_auroc": ROC(task="binary"),
                "val_ap": AveragePrecision(task="binary"),
            }
        )

        self.test_metrics = MetricCollection(
            {
                "auroc": ROC(task="binary"),
                "ap": AveragePrecision(task="binary"),
                "f1": FBetaScore(task="binary", beta=1.0),
                "precision": Precision(task="binary"),
                "recall": Recall(task="binary"),
                "cm": ConfusionMatrix(task="binary"),
            }
        )

    def pipeline(
        self, train: list[TimeSeries], test: list[TimeSeries], test_labels: Sequence[TimeSeries]
    ) -> dict[str, dict[str, Any]]:
        return OutlierDetector.apply(self, train, test, test_labels)

    def fit(
        self,
        train: list[TimeSeries],
        test: list[TimeSeries],
    ) -> None:
        """
        Converte as séries de treino em janelas deslizantes (rótulo 0 —
        normal) e as séries de teste em janelas para validação.

        O dataset de teste é usado como validação para que o modelo veja
        padrões anômalos durante o treinamento, permitindo early stopping
        baseado em desempenho real de detecção.
        """

        ws = self.window_size

        # --- Train loader: janelas do treino (normais) ---

        train_dataset = SequentialTorchTrainingDataset(test, input_chunk_length=ws, output_chunk_length=1)
        print(f"TRAIN 0: {train_dataset[0]}")
        test_dataset = SequentialTorchInferenceDataset(test, input_chunk_length=ws, output_chunk_length=1)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        # --- Val loader: janelas do teste (contêm anomalias reais) ---
        val_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        assert self.model_dict is not None
        for name, model in self.model_dict.items():
            if model is None:
                logging.info(f"PL MODEL NULO — chave '{name}' — pulando fit")
                continue

            logging.info(f"Treinando modelo '{name}' …")
            trainer = Trainer(
                max_epochs=self.max_epochs,
                accelerator=self._accelerator,
                enable_checkpointing=False,
                logger=True,
                enable_progress_bar=True,
            )
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            gc.collect()

    def test_scorer(self, test: list[TimeSeries]) -> dict[str, list[TimeSeries]]:
        """
        Para cada modelo em ``model_dict``, calcula o score de anomalia via
        erro de reconstrução (MSE) em janelas deslizantes centradas.

        Retorna um dicionário ``{nome_do_modelo: list[TimeSeries]}``.
        """
        result: dict[str, list[TimeSeries]] = {}

        assert self.model_dict is not None
        for name, model in self.model_dict.items():
            if model is None:
                logging.info(f"PL MODEL NULO — chave '{name}' — pulando")
                continue

            ws = self.window_size
            scores: list[TimeSeries] = []
            model.eval()

            with torch.no_grad():
                for ts in test:
                    vals = ts.values(copy=False).flatten()
                    n = len(vals)
                    score_vals = np.full(n, 0.0, dtype=float)

                    if n < ws:
                        scores.append(TimeSeries.from_values(score_vals))
                        continue

                    # Padding reflexivo nas bordas para janelas centradas
                    half = ws // 2
                    padded = np.pad(vals, (half, ws - half - 1), mode="edge")

                    for i in range(n):
                        window = padded[i : i + ws]
                        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                        recon = model(x)  # (1, input_dim)
                        # AVISO: ver se esse MSE faz sentido para todos os modelos!
                        mse = F.mse_loss(recon, x, reduction="none").mean().item()
                        score_vals[i] = mse

                    scores.append(TimeSeries.from_values(score_vals))

            result[name] = scores

        return result

    def metrics(
        self, test_labels: Sequence[TimeSeries], scores: dict[str, list[TimeSeries]]
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}

        for name, model_scores in scores.items():
            y_true: list[int] = []
            y_score: list[float] = []

            for label_ts, score_ts in zip(test_labels, model_scores):
                labels = label_ts.values(copy=False).flatten()
                score_vals = np.nan_to_num(score_ts.values(copy=False).flatten(), nan=0.0)
                y_true.extend(labels.astype(int).tolist())
                y_score.extend(score_vals.tolist())

            auc_roc = float(roc_auc_score(y_true, y_score))
            auc_pr = float(average_precision_score(y_true, y_score))

            result[name] = {"name": name, "auc_roc": auc_roc, "auc_pr": auc_pr}

        return result

    # ------------------------------------------------------------------
    # LightningModule
    # ------------------------------------------------------------------

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     return self.model(x)

    # def training_step(
    #     self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    # ) -> torch.Tensor:
    #     data, labels = batch
    #     logits = self.forward(data).flatten()
    #     loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.class_weight)
    #     self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
    #     return loss

    # def configure_optimizers(self) -> torch.optim.Optimizer:
    #     return optim.AdamW(self.parameters(), lr=self.lr)
