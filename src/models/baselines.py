import logging
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from darts import TimeSeries
from darts.ad.scorers import KMeansScorer
import sklearn.ensemble
import sklearn.neighbors

from src.data.utils import extract_windows
from src.models.outlier import OutlierDetector




# TODO: adaptar funcoes para que retornem array de booleanos contendo se elemento da janela eh oulier ou nao
# def threshold_mean(threshold: float) -> WindowPredicate:
#     return lambda w: w.mean() > threshold


# def any_above(series: pd.Series | None = None, threshold: float = 0) -> pd.Series:
#     if series is not None:
#         return series > threshold
#     else:
#         return pd.Series(np.array([False], dtype=bool))


# def is_monotone_increasing() -> WindowPredicate:
#     return lambda w: w.is_monotonic_increasing


# def std_below(max_std: float) -> WindowPredicate:
#     return lambda w: w.std() < max_std


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------


class KMeans(OutlierDetector):
    """
    Outlier detection using the Z-Score method.
    Identifies points that are more than 'threshold' standard deviations away from the mean.
    """

    def __init__(
        self,
        group_id="series_id",
        target_id: str = "target",
        window_size=7,
        threshold: tuple[float, float] = (0.1, 0.1),
    ):
        super().__init__()
        self.scorer = KMeansScorer(
            k=10,
            window=600,
            component_wise=False,
        )
        self.threshold = threshold

    def fit(self, train: list[TimeSeries], test: list[TimeSeries]) -> None:
        self.scorer.fit(train)

    def test_scorer(self, test: list[TimeSeries]) -> dict[str, list[TimeSeries]]:
        if self.scorer is None:
            raise RuntimeError("KMeans must be trained before scoring.")

        scores: list[TimeSeries] = [self.scorer.score(ts) for ts in test]  # type: ignore
        return {self.__class__.__name__: scores}

    def metrics(
        self, test_labels: Sequence[TimeSeries], scores: dict[str, list[TimeSeries]]
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for name, model_scores in scores.items():
            y_true = []
            y_score = []
            for label_ts, score in zip(test_labels, model_scores):
                labels = label_ts.values(copy=False).flatten().astype(int)
                score_vals = score.values(copy=False).flatten()
                # O score tem window-1 valores a menos que os labels no início
                n_warmup = len(labels) - len(score_vals)
                labels_aligned = labels[n_warmup:]
                assert len(labels_aligned) == len(score_vals), (
                    f"labels {len(labels_aligned)} != scores {len(score_vals)} (warmup={n_warmup})"
                )
                y_true.extend(labels_aligned.tolist())
                y_score.extend(score_vals.tolist())

            auc_roc = roc_auc_score(y_true, y_score)
            auc_pr = average_precision_score(y_true, y_score)

            logging.info(f"[{name}] AUC-ROC: {auc_roc:.4f} | AUC-PR: {auc_pr:.4f}")
            result[name] = {"name": name, "auc_roc": auc_roc, "auc_pr": auc_pr}

        return result


class Hampel(OutlierDetector):
    """
    Outlier detection using the Hampel Filter, adaptado para a interface OutlierDetector.
    Usa uma janela deslizante para computar escores de anomalia baseados no desvio
    absoluto em relação à mediana local, escalado pelo MAD (Median Absolute Deviation).
    """

    def __init__(self, window_size=10, n_sigmas=3):
        super().__init__()
        self.window_size = window_size
        self.n_sigmas = n_sigmas

    def fit(self, train: list[TimeSeries], test: list[TimeSeries]):
        # Filtro de Hampel é não-supervisionado e sem estado — nada a treinar
        pass

    def test_scorer(self, test: list[TimeSeries]) -> dict[str, list[TimeSeries]]:
        """
        Para cada TimeSeries, computa um escore contínuo de anomalia:
        |x_i - mediana_local| / sigma_local.
        Quanto maior o escore, mais anômalo o ponto.
        """
        scores: list[TimeSeries] = []

        for ts in test:
            data = np.asarray(ts.univariate_values(), dtype=float)
            windows = extract_windows(data, self.window_size, centered=True)

            median = np.median(windows, axis=1)
            mad = np.median(np.abs(windows - median[:, None]), axis=1)
            sigma = 1.4826 * mad

            # ponytail: onde sigma == 0 (janela constante), score é 0
            score_vals = np.where(sigma > 0, np.abs(data - median) / sigma, 0.0)

            scores.append(TimeSeries.from_values(score_vals))

        return {self.__class__.__name__: scores}

    def metrics(
        self, test_labels: Sequence[TimeSeries], scores: dict[str, list[TimeSeries]]
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for name, model_scores in scores.items():
            y_true = []
            y_score = []
            for label_ts, score in zip(test_labels, model_scores):
                labels = label_ts.values(copy=False).flatten().astype(int)
                score_vals = score.values(copy=False).flatten()
                # Hampel produz score para todo ponto (mesmo comprimento dos labels)
                y_true.extend(labels.tolist())
                y_score.extend(score_vals.tolist())

            auc_roc = roc_auc_score(y_true, y_score)
            auc_pr = average_precision_score(y_true, y_score)

            logging.info(f"[{name}] AUC-ROC: {auc_roc:.4f} | AUC-PR: {auc_pr:.4f}")
            result[name] = {"name": name, "auc_roc": auc_roc, "auc_pr": auc_pr}

        return result


class LocalOutlierFactor(OutlierDetector):
    """
    Outlier detection using sklearn's LocalOutlierFactor with sliding window features.
    Each window of `window_size` consecutive points becomes a feature vector;
    the anomaly score for the center point of the window is the negated LOF score
    (higher = more anomalous). LOF is used in a novelty detection mode: it is fitted
    on the training windows, then scores test windows without refitting.
    """

    def __init__(
        self,
        window_size: int = 20,
        n_neighbors: int = 20,
        contamination: float | str = "auto",
        **kwargs,
    ):
        super().__init__()
        self.window_size = window_size
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.kwargs = kwargs
        self.model: sklearn.neighbors.LocalOutlierFactor | None = None
        self._n_features: int = 0

    def fit(self, train: list[TimeSeries], test: list[TimeSeries]):
        ws = self.window_size
        X_train: list[np.ndarray] = []
        for ts in train:
            vals = ts.values(copy=False).flatten()
            w = extract_windows(vals, ws, centered=False)
            if w.shape[0] > 0:
                X_train.append(w)

        if not X_train:
            raise ValueError(f"Nenhuma janela de tamanho {ws} pôde ser extraída do treino.")

        X = np.concatenate(X_train, axis=0)
        self._n_features = X.shape[1]

        # novelty=True permite fit + predict separados (como IsolationForest)
        self.model = sklearn.neighbors.LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,
            **self.kwargs,
        )
        self.model.fit(X)

    def test_scorer(self, test: list[TimeSeries]) -> dict[str, list[TimeSeries]]:
        if self.model is None:
            raise RuntimeError("LocalOutlierFactor must be trained before scoring.")

        ws = self.window_size
        scores: list[TimeSeries] = []

        for ts in test:
            vals = ts.values(copy=False).flatten()
            windows = extract_windows(vals, ws, centered=True)
            # score_samples retorna valores negativos, mais negativo = mais anômalo
            # invertemos: positivo grande = mais anômalo
            score_vals = -self.model.score_samples(windows).ravel()
            scores.append(TimeSeries.from_values(score_vals))

        return {self.__class__.__name__: scores}

    def metrics(
        self, test_labels: Sequence[TimeSeries], scores: dict[str, list[TimeSeries]]
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for name, model_scores in scores.items():
            y_true = []
            y_score = []
            for label_ts, score in zip(test_labels, model_scores):
                labels = label_ts.values(copy=False).flatten().astype(int)
                score_vals = score.values(copy=False).flatten()
                y_true.extend(labels.tolist())
                y_score.extend(score_vals.tolist())

            auc_roc = roc_auc_score(y_true, y_score)
            auc_pr = average_precision_score(y_true, y_score)

            logging.info(f"[{name}] AUC-ROC: {auc_roc:.4f} | AUC-PR: {auc_pr:.4f}")
            result[name] = {"name": name, "auc_roc": auc_roc, "auc_pr": auc_pr}

        return result


class IsolationForest(OutlierDetector):
    """
    Outlier detection using sklearn's IsolationForest with sliding window features.
    Each window of `window_size` consecutive points becomes a feature vector;
    the anomaly score for the center point of the window is the negated decision
    function output (so higher = more anomalous).
    """

    def __init__(
        self,
        window_size: int = 20,
        contamination: float | str = "auto",
        n_estimators: int = 100,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.window_size = window_size
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.kwargs = kwargs
        self.model: sklearn.ensemble.IsolationForest | None = None
        self._n_features: int = 0

    def fit(self, train: list[TimeSeries], test: list[TimeSeries]):
        ws = self.window_size
        X_train: list[np.ndarray] = []
        for ts in train:
            vals = ts.values(copy=False).flatten()
            w = extract_windows(vals, ws, centered=False)
            if w.shape[0] > 0:
                X_train.append(w)

        if not X_train:
            raise ValueError(f"Nenhuma janela de tamanho {ws} pôde ser extraída do treino.")

        X = np.concatenate(X_train, axis=0)
        self._n_features = X.shape[1]

        self.model = sklearn.ensemble.IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            **self.kwargs,
        )
        self.model.fit(X)

    def test_scorer(self, test: list[TimeSeries]) -> dict[str, list[TimeSeries]]:
        if self.model is None:
            raise RuntimeError("IsolationForest must be trained before scoring.")

        ws = self.window_size
        scores: list[TimeSeries] = []

        for ts in test:
            vals = ts.values(copy=False).flatten()
            windows = extract_windows(vals, ws, centered=True)
            score_vals = -self.model.decision_function(windows).ravel()
            scores.append(TimeSeries.from_values(score_vals))

        return {self.__class__.__name__: scores}

    def metrics(
        self, test_labels: Sequence[TimeSeries], scores: dict[str, list[TimeSeries]]
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for name, model_scores in scores.items():
            y_true = []
            y_score = []
            for label_ts, score in zip(test_labels, scores):
                labels = label_ts.values(copy=False).flatten().astype(int)
                score_vals = score.values(copy=False).flatten()
                y_true.extend(labels.tolist())
                y_score.extend(score_vals.tolist())

            auc_roc = roc_auc_score(y_true, y_score)
            auc_pr = average_precision_score(y_true, y_score)

            logging.info(f"AUC-ROC: {auc_roc:.4f}")
            logging.info(f"AUC-PR : {auc_pr:.4f}")
            result[name] = {"name": name, "auc_roc": auc_roc, "auc_pr": auc_pr}
        return result
