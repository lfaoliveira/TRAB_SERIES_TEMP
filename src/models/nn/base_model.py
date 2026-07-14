import gc
import logging
from collections.abc import Sequence
import traceback
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F

from torchmetrics import MetricCollection
from darts import TimeSeries

from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from src.models.outlier import OutlierDetector
from src.pipelines.metrics import (
    CentralMetricsStore,
    DetectionMetricSummary,
    ScoreSeriesMap,
    ValidationMetrics,
    calculate_detection_summary,
)
from src.data.dataset import SlidingWindowDataset
from lightning.pytorch.callbacks import Callback


def validation_step_reconstruction(
    model: LightningModule,
    batch: tuple[torch.Tensor, torch.Tensor],
    threshold: float = 0.99,
) -> torch.Tensor:
    """Utility validation_step para modelos de detecção de anomalias baseados em reconstrução.

    Computa:
    - Reconstruction loss (MSE) entre entrada e reconstrução
    - Métricas de reconstrução (MSE, SMAPE, MAE) em ``model.val_metrics``
    - Métricas de classificação binária (F1, Precision, Recall, CM) em
      ``model.val_class_metrics``, derivadas do erro por janela versus
      ``threshold * max(MSE)``

    O ``y`` do batch (rótulo do último ponto de cada janela) é usado como
    ground truth para classificação.

    Parâmetros
    ----------
    model : LightningModule
        Modelo que implementa ``forward(x) -> recon``.
    batch : tuple[torch.Tensor, torch.Tensor]
        Tupla ``(x, y)`` onde ``x`` é a janela de entrada e ``y`` o rótulo.
    threshold : float
        Fração do maior MSE usada como limiar de anomalia.

    Retorna
    -------
    torch.Tensor
        Loss escalar de reconstrução (MSE).
    """
    x, y = batch
    recon = model(x)
    loss = F.mse_loss(recon, x)

    # Erro por amostra (janela) → predição binária de anomalia
    mse_per_sample = F.mse_loss(recon, x, reduction="none").mean(dim=1)
    max_mse = mse_per_sample.max()
    limiar = threshold * max_mse if max_mse > 0 else 0.0
    preds = (mse_per_sample > limiar).to(torch.int)

    model.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    # Métricas de reconstrução
    val_metrics = getattr(model, "val_metrics", None)
    if val_metrics is not None:
        val_metrics.update(recon, x)

    # Métricas de classificação (F1, Precision, Recall, CM)
    val_class_metrics = getattr(model, "val_class_metrics", None)
    if val_class_metrics is not None:
        val_class_metrics.update(preds, y)

    return loss


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
        model_dict: dict[str, LightningModule],
        dev=False,
        window_size: int = 20,
        lr: float = 1e-3,
        threshold: float = 0.99,
        batch_size: int = 32,
        max_epochs: int = 10,
        accelerator: str = "cuda",
        trainer_callbacks: list[Callback] | None = None,
        enable_progress_bar: bool = True,
        enable_model_summary: bool = True,
        hyper_optim: bool = False,
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
        self.trainer_callbacks = trainer_callbacks or []
        self._enable_progress_bar = enable_progress_bar
        self._enable_model_summary = enable_model_summary
        self.hyper_optim = hyper_optim

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
            num_workers=0,
            batch_size=self.batch_size,
            shuffle=False,
        )

        # --- Val loader: janelas do teste (contêm anomalias reais) ---
        self.val_loader = DataLoader(
            self.test_dataset,
            num_workers=0,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.trainer = Trainer(
            devices=1,
            fast_dev_run=self.dev,
            max_epochs=self.max_epochs,
            accelerator=self._accelerator,
            enable_checkpointing=False,
            logger=True,
            enable_progress_bar=self._enable_progress_bar,
            enable_model_summary=self._enable_model_summary,
            callbacks=self.trainer_callbacks,
        )
        try:
            assert self.model_dict is not None
            for name, model in self.model_dict.items():
                if model is None:
                    logging.info(f"PL MODEL NULO — chave '{name}' — pulando fit")
                    continue

                logging.info(f"Treinando modelo '{name}' …")

                self.trainer.fit(model, train_dataloaders=self.train_loader, val_dataloaders=self.val_loader)
                gc.collect()
                torch.cuda.empty_cache()

                if self.hyper_optim:
                    break
        except:
            traceback.print_exc()
            raise

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

        model_test_metrics = {
            name: cast(MetricCollection, model.test_metrics)
            for name, model in (self.model_dict or {}).items()
            if model is not None and hasattr(model, "test_metrics")
        }
        detect_metrics = calculate_detection_summary(
            test_labels, scores, device, model_test_metrics=model_test_metrics
        )
        for name, met_collection in model_test_metrics.items():
            valores = detect_metrics[name]
            print(f"METRICAS: {valores}")
            CentralMetricsStore.add(name, "test", cast(dict, valores))

        return detect_metrics
