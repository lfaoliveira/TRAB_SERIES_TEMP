import logging
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from darts import TimeSeries
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

from src.models.outlier import OutlierDetector


class OutlierModelWrapper(OutlierDetector):
    """
    Base class for outlier detection models using PyTorch Lightning.

    Adere à interface ``OutlierDetector`` (train → test_scorer → metrics),
    permitindo o uso no pipeline ``apply()``.

    Subclasses devem definir ``self.model`` (um ``nn.Module`` que retorna
    um único logit por amostra).

    Nota
    ----
    O treinamento é não supervisionado: janelas deslizantes do treino são
    tratadas como classe "normal" (rótulo 0). O modelo aprende a reconhecer
    o padrão normal; pontos que fogem desse padrão recebem escore alto.
    """

    def __init__(
        self,
        input_dim: int,
        pl_model: Optional[LightningModule] = None,
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
        self.pl_model = pl_model

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
        self, train: list[TimeSeries], test: list[TimeSeries], test_labels: np.ndarray
    ) -> dict[str, Any]:
        return OutlierDetector.apply(self, train, test, test_labels)

    def fit(
        self,
        train: list[TimeSeries],
        pl_model: Optional[LightningModule | None] = None,
    ) -> None:
        """
        Converte as séries de treino em janelas deslizantes (rótulo 0 —
        normal) e executa o Trainer do Lightning.

        O parâmetro ``mode`` é herdado de ``nn.Module.train(mode)`` e
        deve ser passado como keyword-only. Quando chamado pelo pipeline
        ``apply()``, apenas a série é fornecida.
        """
        model = pl_model or self.pl_model
        if model is None:
            logging.info("PL MODEL NULO — pulando fit")
            raise RuntimeError("PL MODEL NULO")

        ws = self.window_size
        X_list: list[np.ndarray] = []

        for ts in train:
            vals = ts.values(copy=False).flatten()
            if len(vals) < ws:
                continue
            for i in range(len(vals) - ws + 1):
                X_list.append(vals[i : i + ws])

        if not X_list:
            return

        X = torch.tensor(np.stack(X_list), dtype=torch.float32)
        # Todas as janelas de treino são consideradas "normais"
        y = torch.zeros(len(X), dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=self.batch_size,
            shuffle=False,
        )

        trainer = Trainer(
            max_epochs=self.max_epochs,
            accelerator=self._accelerator,
            enable_checkpointing=False,
            logger=True,
            enable_progress_bar=True,
        )
        trainer.fit(model, loader)

    def test_scorer(
        self, test: list[TimeSeries], pl_model: Optional[LightningModule | None] = None
    ) -> list[TimeSeries]:
        """
        Para cada série, calcula o score de anomalia via erro de reconstrução (MSE)
        em janelas deslizantes centradas. Retorna uma TimeSeries de scores por série.
        """
        model = pl_model or self.pl_model
        if model is None:
            logging.info("PL MODEL NULO!")
            return []

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

        return scores

    def metrics(self, test_labels: np.ndarray, scores: list[TimeSeries]) -> dict[str, Any]:
        y_true: list[int] = []
        y_score: list[float] = []

        for labels, score in zip(test_labels, scores):
            score_vals = np.nan_to_num(score.values(copy=False).flatten(), nan=0.0)
            y_true.extend(labels.tolist())
            y_score.extend(score_vals.tolist())

        auc_roc = float(roc_auc_score(y_true, y_score))
        auc_pr = float(average_precision_score(y_true, y_score))

        return {"name": self.__class__.__name__, "auc_roc": auc_roc, "auc_pr": auc_pr}

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
