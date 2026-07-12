"""Otimização de hiperparâmetros com Optuna para modelos OutlierModelWrapper."""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from lightning.pytorch.callbacks import EarlyStopping

from src.models.nn.base_model import OutlierModelWrapper


# ---------------------------------------------------------------------------
# Tipos auxiliares
# ---------------------------------------------------------------------------
ParamSuggester = Callable[[optuna.Trial], dict[str, Any]]
"""Recebe um ``optuna.Trial`` e retorna um dicionário de hiperparâmetros."""

ModelFactory = Callable[[dict[str, Any]], dict[str, Any]]
"""Recebe o dicionário de hiperparâmetros e retorna ``model_dict`` (dict[str, LightningModule])."""


class HyperparamOptim:
    """Otimização de hiperparâmetros para modelos que usam ``OutlierModelWrapper``.

    Parameters
    ----------
    model_factory : ModelFactory
        Função que recebe ``params`` (dict saído de ``param_suggester``) e
        retorna um ``model_dict`` compatível com ``OutlierModelWrapper``.
    param_suggester : ParamSuggester
        Função que recebe um ``optuna.Trial`` e retorna um dict com os
        hiperparâmetros a serem amostrados.
    train_data : tuple
        ``(train_values, train_labels)`` — listas de ``TimeSeries``.
    test_data : tuple
        ``(test_values, test_labels)`` — listas de ``TimeSeries``.
    fixed_params : dict | None
        Parâmetros fixos passados para ``OutlierModelWrapper`` (ex.:
        ``window_size``, ``threshold``, ``accelerator``). Se não fornecidos,
        usa os defaults da classe.
    max_epochs : int
        Máximo de épocas por trial.
    patience : int
        Paciência do ``EarlyStopping``.
    n_trials : int
        Número de trials do Optuna.
    direction : str
        Direção da otimização (``"minimize"`` ou ``"maximize"``).
    study_name : str | None
        Nome opcional para o estudo Optuna.
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        param_suggester: ParamSuggester,
        train_data: tuple,
        test_data: tuple,
        fixed_params: dict[str, Any] | None = None,
        max_epochs: int = 50,
        patience: int = 10,
        n_trials: int = 30,
        direction: str = "maximize",
        study_name: str | None = None,
    ) -> None:
        self.model_factory = model_factory
        self.param_suggester = param_suggester
        self.train_values, self.train_labels = train_data
        self.test_values, self.test_labels = test_data
        self.fixed_params = fixed_params or {}
        self.max_epochs = max_epochs
        self.patience = patience
        self.n_trials = n_trials
        self.direction = direction
        self.study_name = study_name

        self.model_count = 0

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def optimize(self) -> optuna.Study:
        """Roda o estudo Optuna e printa o melhor trial."""
        study: optuna.Study = optuna.create_study(
            direction=self.direction,
            pruner=optuna.pruners.PercentilePruner(25.0),
            study_name=self.study_name,
        )
        study.optimize(self.objective, n_trials=self.n_trials)

        logging.info("\n" + "=" * 50)
        logging.info("  MELHOR TRIAL")
        logging.info("=" * 50)
        logging.info(
            f"  Valor ({'f1' if self.direction == 'maximize' else 'loss'}): {study.best_trial.value:.4f}"
        )
        logging.info("  Hiperparâmetros:")
        for k, v in study.best_trial.params.items():
            logging.info(f"    {k}: {v}")
        logging.info("=" * 50)

        return study

    # ------------------------------------------------------------------
    # Função objetivo do Optuna
    # ------------------------------------------------------------------

    def objective(self, trial: optuna.Trial) -> float:
        # 1. Amostra hiperparâmetros
        params = self.param_suggester(trial)

        # 2. Separa params do wrapper vs params do modelo
        wrapper_keys = {"batch_size", "window_size", "threshold"}
        copia = params.copy()
        wrapper_params = {k: copia.pop(k) for k in wrapper_keys if k in copia.copy()}
        # o que sobra em params vai para model_factory

        # 3. Constrói os modelos via factory com os params restantes
        model_dict = self.model_factory(params)

        # 4. Prepara callbacks: EarlyStopping + pruning
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
        early_stop = EarlyStopping(monitor="val_loss", patience=self.patience)

        # 5. Monta o wrapper
        wrapper = OutlierModelWrapper(
            model_dict=model_dict,
            max_epochs=self.max_epochs,
            trainer_callbacks=[pruning_callback, early_stop],
            enable_progress_bar=True,
            enable_model_summary=False,
            **self.fixed_params,
            **wrapper_params,
        )

        # 6. Roda o pipeline completo: fit → test_scorer → metrics
        metrics = wrapper.apply(
            self.train_values,
            self.train_labels,
            self.test_values,
            self.test_labels,
        )

        # 7. Extrai F1 do primeiro modelo
        first_model_name = list(model_dict.keys())[0]
        result = metrics[first_model_name]
        f1 = float(result["f1"])

        # 8. Pruning
        pruning_callback.check_pruned()

        return f1
