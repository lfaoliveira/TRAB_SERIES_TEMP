import logging
from typing import Any, Callable, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from darts import TimeSeries
from darts.ad.scorers import KMeansScorer
from darts.models import ARIMA as DartsSARIMA
import sklearn.ensemble

from src.models.outlier import OutlierDetector


type WindowPredicate = Callable[[np.ndarray, Any], pd.Series[bool]]


def rolling_window_apply(
    series_list: List[TimeSeries],
    window_size: int,
    predicate: WindowPredicate,
    overlap=False,
) -> List[np.ndarray]:
    """
    Aplica um rolling window de tamanho fixo sobre cada TimeSeries e avalia
    *predicate* em cada janela, retornando um array booleano por série.

    Parameters
    ----------
    series_list : List[TimeSeries]
        Lista de sequências numéricas 1-D.
    window_size : int
        Número de observações em cada janela.
    predicate : Callable[[pd.Series], bool]
        Função que recebe uma pd.Series (uma janela) e retorna bool.
    min_periods : int | None
        Mínimo de observações válidas para calcular o resultado.
        None → igual a window_size (apenas janelas completas).
        Use 1 para obter resultado desde o primeiro elemento (como pad_value=0).

    Returns
    -------
    List[np.ndarray]
        Um np.ndarray de bool por série de entrada.
    """
    if window_size < 1:
        raise ValueError(f"window_size deve ser >= 1, recebeu {window_size}")

    results: List[np.ndarray] = []

    # NOTE: EVITAR USAR OVERLAP! (eficiência)
    step = window_size if not overlap else None

    for series in series_list:
        s = pd.Series(series.univariate_values(), dtype=float)

        rolling = s.rolling(window=window_size, step=step, min_periods=window_size, center=True)
        for rol in rolling:
            logging.info(rol)
        # raw=False → pd.Series  # pd.NA para janelas incompletas
        bool_series = rolling.apply(lambda w: predicate(w), raw=False).astype("boolean")

        results.append(bool_series.to_numpy(dtype=bool, na_value=False))

    return results


# TODO: adaptar funcoes para que retornem array de booleanos contendo se elemento da janela eh oulier ou nao
def threshold_mean(threshold: float) -> WindowPredicate:
    return lambda w: w.mean() > threshold


def any_above(series: pd.Series | None = None, threshold: float = 0) -> pd.Series:
    if series is not None:
        return series > threshold
    else:
        return pd.Series(np.array([False], dtype=bool))


def is_monotone_increasing() -> WindowPredicate:
    return lambda w: w.is_monotonic_increasing


def std_below(max_std: float) -> WindowPredicate:
    return lambda w: w.std() < max_std


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

    def fit(self, train: list[TimeSeries]):
        self.scorer.fit(train)

    def test_scorer(self, test: list[TimeSeries]):
        if self.scorer is None:
            raise RuntimeError("KMeans must be trained before scoring.")

        scores: list[TimeSeries] = [self.scorer.score(ts) for ts in test]
        return scores

    def metrics(self, test_labels: np.ndarray, scores: list[TimeSeries]):
        y_true = []
        y_score = []
        for labels, score in zip(test_labels, scores):
            score_vals = score.values(copy=False).flatten()
            # O score tem window-1 valores a menos que os labels no início
            # (as primeiras window-1 posições não formam uma janela completa)
            n_warmup = len(labels) - len(score_vals)
            labels_aligned = labels[n_warmup:]
            assert len(labels_aligned) == len(score_vals), (
                f"labels {len(labels_aligned)} != scores {len(score_vals)} (warmup={n_warmup})"
            )
            y_true.extend(labels_aligned)
            y_score.extend(score_vals)

        logging.info(
            "SHAPES — y_true: {np.array(y_true).shape}, y_score: {np.array(y_score).shape}"
        )
        auc_roc = roc_auc_score(y_true, y_score)
        auc_pr = average_precision_score(y_true, y_score)

        logging.info(f"AUC-ROC: {auc_roc:.4f}")
        logging.info(f"AUC-PR : {auc_pr:.4f}")
        return [auc_roc, auc_pr]


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

    def fit(self, train: list[TimeSeries]):
        # Filtro de Hampel é não-supervisionado e sem estado — nada a treinar
        pass

    def test_scorer(self, test: list[TimeSeries]) -> list[TimeSeries]:
        """
        Para cada TimeSeries, computa um escore contínuo de anomalia:
        |x_i - mediana_local| / sigma_local.
        Quanto maior o escore, mais anômalo o ponto.
        """

        scores: list[TimeSeries] = []
        k = self.window_size // 2

        for ts in test:
            data = np.asarray(ts.univariate_values(), dtype=float)
            n = len(data)
            score_vals = np.zeros(n, dtype=float)

            for i in range(n):
                start = max(0, i - k)
                end = min(n, i + k + 1)
                window = data[start:end]

                median = np.median(window)
                mad = np.median(np.abs(window - median))
                sigma = 1.4826 * mad

                # ponytail: se sigma == 0 (janela constante), score é 0
                score_vals[i] = np.abs(data[i] - median) / sigma if sigma > 0 else 0.0

            scores.append(TimeSeries.from_values(score_vals))

        return scores

    def metrics(self, test_labels: np.ndarray, scores: list[TimeSeries]):
        y_true = []
        y_score = []
        for labels, score in zip(test_labels, scores):
            score_vals = score.values(copy=False).flatten()
            # Hampel produz score para todo ponto (mesmo comprimento dos labels)
            y_true.extend(labels)
            y_score.extend(score_vals)

        auc_roc = roc_auc_score(y_true, y_score)
        auc_pr = average_precision_score(y_true, y_score)

        logging.info(f"AUC-ROC: {auc_roc:.4f}")
        logging.info(f"AUC-PR : {auc_pr:.4f}")
        return [auc_roc, auc_pr]


class SARIMA(OutlierDetector):
    """
    Outlier detection using a SARIMA forecasting model via darts.
    A SARIMA model is trained per series, then anomaly scores are computed as the
    absolute residual |observation - forecast| from historical_forecasts.
    """

    def __init__(
        self,
        seasonal_order: tuple[int, int, int, int] = (1, 0, 0, 12),
        order: tuple[int, int, int] = (1, 0, 0),
        n_epochs: int = 10,
        **kwargs,
    ):
        super().__init__()
        self.seasonal_order = seasonal_order
        self.order = order
        self.n_epochs = n_epochs
        self.kwargs = kwargs
        self.model: DartsSARIMA | None = None

    def fit(self, train: list[TimeSeries]):
        if not train:
            return
        self.model = DartsSARIMA(
            order=self.order,
            seasonal_order=self.seasonal_order,
            n_epochs=self.n_epochs,
            **self.kwargs,
        )
        self.model.fit(train, verbose=False)

    def test_scorer(self, test: list[TimeSeries]) -> list[TimeSeries]:
        if self.model is None:
            raise RuntimeError("SARIMA must be trained before scoring.")

        scores: list[TimeSeries] = []
        for ts in test:
            pred = self.model.historical_forecasts(
                ts,
                forecast_horizon=1,
                stride=1,
                retrain=False,
                last_points_only=True,
                verbose=False,
            )
            obs = ts.values(copy=False).flatten()
            est = pred.values(copy=False).flatten()
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

    def fit(self, train: list[TimeSeries]):
        ws = self.window_size
        X_train: list[np.ndarray] = []
        for ts in train:
            vals = ts.values(copy=False).flatten()
            if len(vals) < ws:
                continue
            for i in range(len(vals) - ws + 1):
                X_train.append(vals[i : i + ws])

        if not X_train:
            raise ValueError(f"Nenhuma janela de tamanho {ws} pôde ser extraída do treino.")

        X = np.stack(X_train)
        self._n_features = X.shape[1]

        self.model = sklearn.ensemble.IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            **self.kwargs,
        )
        self.model.fit(X)

    def test_scorer(self, test: list[TimeSeries]) -> list[TimeSeries]:
        if self.model is None:
            raise RuntimeError("IsolationForest must be trained before scoring.")

        ws = self.window_size
        scores: list[TimeSeries] = []

        for ts in test:
            vals = ts.values(copy=False).flatten()
            n = len(vals)
            score_vals = np.full(n, 0.0, dtype=float)

            if n < ws:
                scores.append(TimeSeries.from_values(score_vals))
                continue

            half = ws // 2
            for i in range(n):
                start = max(0, i - half)
                end = min(n, i + half + 1)
                window = vals[start:end]
                if len(window) < ws:
                    pad_before = max(0, half - i)
                    pad_after = max(0, (i + half + 1) - n)
                    window = np.pad(window, (pad_before, pad_after), mode="edge")
                score_vals[i] = -self.model.decision_function(window.reshape(1, -1)).item()

            scores.append(TimeSeries.from_values(score_vals))

        return scores

    def metrics(self, test_labels: np.ndarray, scores: list[TimeSeries]):
        y_true = []
        y_score = []
        for labels, score in zip(test_labels, scores):
            score_vals = score.values(copy=False).flatten()
            y_true.extend(labels)
            y_score.extend(score_vals)

        auc_roc = roc_auc_score(y_true, y_score)
        auc_pr = average_precision_score(y_true, y_score)

        logging.info(f"AUC-ROC: {auc_roc:.4f}")
        logging.info(f"AUC-PR : {auc_pr:.4f}")
        return [auc_roc, auc_pr]
