from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from darts import TimeSeries
from lightning import LightningModule, Trainer
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import optim
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


class OutlierNetwork(OutlierDetector, LightningModule):
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

    class_weight: torch.Tensor

    def __init__(
        self,
        input_dim: int,
        window_size: int = 20,
        lr: float = 1e-3,
        pos_weight: float = 1.0,
        batch_size: int = 32,
        max_epochs: int = 10,
        accelerator: str = "auto",
    ) -> None:
        OutlierDetector.__init__(self)
        LightningModule.__init__(self)
        self.save_hyperparameters()

        self.model: nn.Module = nn.Identity()
        self.lr = lr
        self.window_size = window_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self._accelerator = accelerator

        # Peso para a classe positiva (anomalia) no BCEWithLogitsLoss
        self.register_buffer(
            "class_weight",
            torch.tensor(pos_weight, dtype=torch.float32),
        )

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

    def apply(  # type: ignore[override]
        self, train: list[TimeSeries], test: list[TimeSeries], test_labels: np.ndarray
    ) -> list[Any]:
        return OutlierDetector.apply(self, train, test, test_labels)

    def train(self, train: list[TimeSeries], *, mode: bool = True) -> None:  # type: ignore[override]
        """
        Converte as séries de treino em janelas deslizantes (rótulo 0 —
        normal) e executa o Trainer do Lightning.

        O parâmetro ``mode`` é herdado de ``nn.Module.train(mode)`` e
        deve ser passado como keyword-only. Quando chamado pelo pipeline
        ``apply()``, apenas a série é fornecida.
        """
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
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(self, loader)

    def test_scorer(self, test: list[TimeSeries]) -> list[TimeSeries]:
        """
        Para cada série, calcula P(anomalia) = sigmoid(logit) via janela
        deslizante centrada. Retorna uma TimeSeries de scores por série.
        """
        ws = self.window_size
        scores: list[TimeSeries] = []
        self.eval()

        with torch.no_grad():
            for ts in test:
                vals = ts.values(copy=False).flatten()
                n = len(vals)
                score_vals = np.full(n, 0.0, dtype=float)

                if n < ws:
                    scores.append(TimeSeries.from_values(score_vals))
                    continue

                # Padding reflexivo nas bordas
                half = ws // 2
                padded = np.pad(vals, (half, ws - half - 1), mode="edge")

                for i in range(n):
                    window = padded[i : i + ws]
                    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                    logit = self.forward(x).item()
                    score_vals[i] = torch.sigmoid(torch.tensor(logit)).item()

                scores.append(TimeSeries.from_values(score_vals))

        return scores

    def metrics(self, test_labels: np.ndarray, scores: list[TimeSeries]) -> list[Any]:
        y_true: list[int] = []
        y_score: list[float] = []

        for labels, score in zip(test_labels, scores):
            score_vals = score.values(copy=False).flatten()
            y_true.extend(labels.tolist())
            y_score.extend(score_vals.tolist())

        auc_roc = float(roc_auc_score(y_true, y_score))
        auc_pr = float(average_precision_score(y_true, y_score))

        return [auc_roc, auc_pr]

    # ------------------------------------------------------------------
    # LightningModule
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        data, labels = batch
        logits = self.forward(data).flatten()
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.class_weight)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return optim.AdamW(self.parameters(), lr=self.lr)
