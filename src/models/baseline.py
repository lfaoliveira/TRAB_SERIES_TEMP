import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


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
        threshold=3,
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
        rolling = df_groups.loc[self.target_id, :].rolling(self.window_size)

        medias_moveis = rolling.mean().dropna().reset_index(level=0, drop=True)
        stds_moveis = (
            rolling.std(ddof=0)
            .dropna()
            .reset_index(level=0, drop=True)  # dropna para evitar divisao por zero!
        )
        logging.info("DROPANDO NAN NO ZSCORE PRA EVITAR DIVISAO POR ZERO!")

        # 2. formula do Z-Score
        # transforma 0 em nan pra evitar divisao por zero
        stds_moveis = stds_moveis.apply(lambda x: x if x != 0 else np.nan)

        df_target = df - medias_moveis
        df_target = df_target / stds_moveis

        # 3. Identifica os Outliers (Z-Score absoluto maior que 3)
        # Preenche os NaNs iniciais da janela como False para não quebrar a lógica
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
        self.hampel = HampelFilterOutlierDetector(
            window_size=window_size, n_sigmas=n_sigmas
        )

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
