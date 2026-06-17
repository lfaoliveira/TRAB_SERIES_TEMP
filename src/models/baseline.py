from typing import Any, Callable, List


import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from darts.ad.detectors import QuantileDetector
from darts import TimeSeries
from abc import ABC, abstractmethod


type WindowPredicate = Callable[[np.ndarray, Any], pd.Series[bool]]


class OutlierDetector(ABC):
    def __init__(
        self, group_id: str = "series_id", target_id: str = "target", window_size=7
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.group_id = group_id
        self.target_id = target_id

    @abstractmethod
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    """
rolling_window_pandas.py
------------------------
Versão com Pandas do rolling_window_apply.
Muito mais simples: pd.Series.rolling() cuida de janelas, padding e alinhamento.
"""


# input: Pd.Series, output: bool


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Predicate factories (mesmo contrato da versão anterior)
# ---------------------------------------------------------------------------


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
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    series_list: List[TimeSeries] = [
        [1, 3, 5, 7, 6, 4, 2],
        [10, 2, 2, 2, 2, 2, 10],
        [1, 2, 3, 4, 5, 6, 7],
    ]

    WINDOW = 3

    # print("=== mean > 4 (apenas janelas completas) ===")
    # pprint.pprint(
    #     [r.tolist() for r in rolling_window_apply(series_list, WINDOW, threshold_mean(4))]
    # )

    # print("\n=== monotone increasing ===")
    # pprint.pprint(
    #     [
    #         r.tolist()
    #         for r in rolling_window_apply(series_list, WINDOW, is_monotone_increasing())
    #     ]
    # )

    print("\n=== std < 2 (sem padding manual) ===")
    pprint.pprint(
        [r.tolist() for r in rolling_window_apply(series_list, WINDOW, any_above, overlap=False)]
    )

    # print("\n=== lambda inline: último > primeiro ===")
    # pprint.pprint(
    #     [
    #         r.tolist()
    #         for r in rolling_window_apply(
    #             series_list, WINDOW, lambda w: w.iloc[-1] > w.iloc[0]
    #         )
    #     ]
    # )


class Quantile(OutlierDetector):
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
        super().__init__(
            target_id=target_id,
            group_id=group_id,
            window_size=window_size,
        )
        self.threshold = threshold

    def previsao(data: list[TimeSeries]) -> pd.DataFrame:
        threshold = 0.2
        low, high = 0 + threshold, 1 - threshold

        outlier_list = []
        for idx, ts in enumerate(data):
            detector = QuantileDetector(low, high)
            outliers = detector.fit_detect(ts)
            outlier_pd = outliers.values()
            outlier_list.append({"id": idx, "series": outlier_pd})

        is_outlier = pd.DataFrame(outlier_list)
        is_outlier = is_outlier.fillna(0)
        bool_outlier = is_outlier[is_outlier == 1]

        return bool_outlier

    def detect(self, data: list[TimeSeries]) -> pd.DataFrame:
        """
        Detect outliers in the data.
        Returns a boolean array where True indicates an outlier.

        """
        threshold = 0.2
        low, high = 0 + threshold, 1 - threshold

        outlier_list = []
        for idx, ts in enumerate(data):
            detector = QuantileDetector(low, high)
            outliers = detector.fit_detect(ts)
            outlier_pd = outliers.values()
            outlier_list.append({"id": idx, "series": outlier_pd})

        is_outlier = pd.DataFrame(outlier_list)
        is_outlier = is_outlier.fillna(0)

        outlier_df = pd.DataFrame(outlier_list)
        is_outlier = df_target.abs() > self.threshold
        is_outlier = is_outlier.fillna(False)

        return pd.DataFrame(is_outlier)


class HampelFilterOutlierDetector(OutlierDetector):
    """
    Outlier detection using the Hampel Filter.
    Uses a sliding window to identify points that differ significantly from the local median.
    """

    def __init__(self, window_size=10, n_sigmas=3):
        self.window_size = window_size
        self.n_sigmas = n_sigmas

    def detect(self, data: pd.DataFrame):
        """
        Detect outliers in the data.
        Returns a boolean array where True indicates an outlier.
        """
        data = np.array(data)
        n = len(data)
        outliers = np.zeros(n, dtype=bool)

        # Half window size
        k = self.window_size // 2

        for i in range(n):
            # Define window boundaries
            start = max(0, i - k)
            end = min(n, i + k + 1)
            window = data[start:end]

            median = np.median(window)
            mad = np.median(np.abs(window - median))

            # Scale factor for MAD to approximate standard deviation for normal distribution
            sigma = 1.4826 * mad

            if np.abs(data[i] - median) > self.n_sigmas * sigma:
                outliers[i] = True

        return outliers


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
