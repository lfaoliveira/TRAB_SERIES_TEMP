import logging
from typing import Any, Callable, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from statsmodels.tsa.seasonal import STL
from darts.ad.detectors import QuantileDetector
from darts import TimeSeries
from abc import ABC, abstractmethod
from darts.ad.scorers import KMeansScorer


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
            print(rol)
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
        self.train(train)
        scores = self.test_scorer(test)
        metrics = self.metrics(test_labels, scores)
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


# ---------------------------------------------------------------------------
# Predicate factories (mesmo contrato da versão anterior)
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
            k=40,
            window=600,
            component_wise=False,
        )
        self.threshold = threshold

    def train(self, train: list[TimeSeries]):
        self.scorer.fit(train)

    def test_scorer(self, test: list[TimeSeries]):

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

        logging.info("AUC-ROC: {auc_roc:.4f}")
        logging.info("AUC-PR : {auc_pr:.4f}")
        return [auc_roc, auc_pr]

    # #AVISO: LEGACY
    # def detect(self, data: list[TimeSeries]) -> pd.DataFrame:
    #     """
    #     Detect outliers in the data.
    #     Returns a boolean array where True indicates an outlier.

    #     """
    #     threshold = 0.2
    #     low, high = 0 + threshold, 1 - threshold

    #     outlier_list = []
    #     for idx, ts in enumerate(data):
    #         detector = QuantileDetector(low, high)
    #         outliers = detector.fit_detect(ts)
    #         outlier_pd = outliers.values()
    #         outlier_list.append({"id": idx, "series": outlier_pd})

    #     is_outlier = pd.DataFrame(outlier_list)
    #     is_outlier = is_outlier.fillna(0)

    #     outlier_df = pd.DataFrame(outlier_list)
    #     is_outlier = df_target.abs() > self.threshold
    #     is_outlier = is_outlier.fillna(False)

    #     return pd.DataFrame(is_outlier)


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

    def train(self, train: list[TimeSeries]):
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

        logging.info("AUC-ROC: {auc_roc:.4f}")
        logging.info("AUC-PR : {auc_pr:.4f}")
        return [auc_roc, auc_pr]



class STLHampelOutlierDetector(OutlierDetector):
    """
    Outlier detection using STL Decomposition followed by a Hampel Filter on the residuals.
    Useful for seasonal data where outliers are defined as anomalies in the residual component.
    """

    def __init__(self, period=None, window_size=10, n_sigmas=3):
        self.period = period
        self.hampel = HampelFilterOutlierDetector(window_size=window_size, n_sigmas=n_sigmas)

    def detect(self, data):
        """
        Detect outliers in the data.
        Returns a boolean array where True indicates an outlier.
        """
        # Ensure data is a pandas Series for STL
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        # STL decomposition: data = trend + seasonal + resid
        stl = STL(data, period=self.period).fit()
        resid = stl.resid

        # Apply Hampel filter to the residual component
        return self.hampel.detect(resid.values)
