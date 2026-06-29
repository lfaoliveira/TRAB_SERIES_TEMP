import logging

import numpy as np
import torch
from darts import TimeSeries
from darts.models import TCNModel
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.outlier import OutlierDetector


class TCN(OutlierDetector):
    """
    Outlier detection using a TCN forecasting model.
    A TCN is trained on the train series, then anomaly scores are computed as the
    absolute residual |observation - forecast| from historical_forecasts.
    """

    def __init__(
        self,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        kernel_size: int = 3,
        num_filters: int = 6,
        num_layers: int | None = None,
        dropout: float = 0.0,
        n_epochs: int = 20,
        batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.model: TCNModel | None = None

    def train(self, train: list[TimeSeries]):
        if not train:
            return
        # ponytail: usa 1 device só — devices=-1 spawna multiprocessing que toma
        # SIGTERM em ambientes como Kaggle. devices=1 usa uma GPU sem subprocessos.
        accel = "gpu" if torch.cuda.is_available() else "cpu"
        devices = 1

        self.model = TCNModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            kernel_size=self.kernel_size,
            num_filters=self.num_filters,
            num_layers=self.num_layers,
            dropout=self.dropout,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            pl_trainer_kwargs={
                "accelerator": accel,
                "devices": devices,
            },
            **self.kwargs,
        )
        self.model.fit(train, verbose=False)

    def test_scorer(self, test: list[TimeSeries]) -> list[TimeSeries]:
        if self.model is None:
            raise RuntimeError("TCNModel must be trained before scoring.")

        scores: list[TimeSeries] = []
        for ts in test:
            # historical_forecasts com retrain=False usa o modelo pré-treinado
            pred = self.model.historical_forecasts(
                ts,
                forecast_horizon=self.output_chunk_length,
                stride=1,
                retrain=False,
                last_points_only=True,
                verbose=False,
            )
            obs = ts.values(copy=False).flatten()
            est = pred.values(copy=False).flatten()
            # ponytail: alinha pelo fim se houver warmup do input_chunk_length
            if len(est) < len(obs):
                obs = obs[-len(est) :]
            residual = np.abs(obs - est)
            scores.append(TimeSeries.from_values(residual))

        return scores

    def metrics(self, test_labels: np.ndarray, scores: list[TimeSeries]):
        y_true = []
        y_score = []
        for labels, score in zip(test_labels, scores):
            score_vals = score.values(copy=False).flatten()
            n_warmup = len(labels) - len(score_vals)
            labels_aligned = labels[n_warmup:] if n_warmup > 0 else labels
            y_true.extend(labels_aligned)
            y_score.extend(score_vals)

        auc_roc = roc_auc_score(y_true, y_score)
        auc_pr = average_precision_score(y_true, y_score)

        logging.info(f"AUC-ROC: {auc_roc:.4f}")
        logging.info(f"AUC-PR : {auc_pr:.4f}")
        return [auc_roc, auc_pr]
