import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from darts.ad.detectors import QuantileDetector


class OutlierDetector:
    def __init__(
        self, group_id: str = "series_id", target_id: str = "target", window_size=7
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.group_id = group_id
        self.target_id = target_id

    # def detect(
    #     self, data: pd.DataFrame, detect_fn: Callable | None = None
    # ) -> pd.DataFrame:
    #


class ZScoreOutlierDetector(OutlierDetector):
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

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect outliers in the data.
        Returns a boolean array where True indicates an outlier.

        """
        # opera ao longo das colunas!!!
        df = data.copy()
        df = df.T

        df_groups = df.groupby(self.group_id)
        low, high = self.threshold
        detector = QuantileDetector(low, high)
        outlier_list = []
        for group_name, grupo_df in df_groups:
            outliers = detector.detect(pd.Series(grupo_df["target"]))
            outlier_list.append(outliers)
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
