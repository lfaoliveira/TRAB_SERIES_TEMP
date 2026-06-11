from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config.config import ProjectSettings
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from src.data.torch_dataset import Horizons


class NasaDataset:
    """Loads telemetry .npy time series and labeled anomalies for outlier detection.

    Exposes the same minimal interface used by the existing datamodule:
      - build_train_dataset()
      - build_validation_dataset(train_dataset)
      - build_test_dataset(train_dataset)

    Internally this class provides `df_train_wide` and `df_test_wide` in a
    wide-format DataFrame (index=series_id, columns as integer time steps) and
    converts to the long format required by the torch forecasting pipeline.
    """

    def __init__(self, base_path: Path | str = "data/m4", normalize=False) -> None:
        self.base_path = Path(base_path)
        self.train_dir = self.base_path / "data" / "train"
        self.test_dir = self.base_path / "data" / "test"
        self.labels_file = self.base_path / "labeled_anomalies.csv"

        self.df_train_wide, self.df_test_wide = self._load_npy_dataset()

        # Convert to long format used by torch dataset builders in the repo
        self.labels_df = self._read_labels()

        self.df_train = self._wide_to_long(self.df_train_wide, include_labels=False)
        self.df_test = self._wide_to_long(self.df_test_wide, include_labels=True)

        # dataset hyperparams
        self.frequency = ProjectSettings.dataset.dataset_frequency
        self.input_width = Horizons.input_width(self.frequency)
        self.output_width = Horizons.output_width(self.frequency)
        self.series_normalizer = GroupNormalizer(groups=["series_id"])

        # build TimeSeriesDataSet objects lazily
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

    def _read_labels(self) -> pd.DataFrame:
        """Reads labeled_anomalies.csv if present and returns a DataFrame.

        Expected CSV format with columns: ``series_id,start,end`` where start/end
        are integer positions (0-based or 1-based). This function will attempt to
        coerce to integers. If the file is missing, returns empty DataFrame.
        """
        if not self.labels_file.exists():
            return pd.DataFrame(columns=["series_id", "start", "end"])

        df = pd.read_csv(self.labels_file)
        # normalize column names
        df = df.rename(columns={c: c.strip() for c in df.columns})
        expected = [c for c in ["series_id", "start", "end"] if c in df.columns]
        if not set(["series_id", "start", "end"]).issubset(df.columns):
            return pd.DataFrame(columns=["series_id", "start", "end"])

        df = df[["series_id", "start", "end"]].copy()
        df["start"] = pd.to_numeric(df["start"], errors="coerce").astype("Int64")
        df["end"] = pd.to_numeric(df["end"], errors="coerce").astype("Int64")
        return df.dropna()

    def _series_to_wide(self, series_dict: Dict[str, np.ndarray] | None) -> pd.DataFrame:
        """Constructs a wide DataFrame from a dict of arrays.

        Shorter series are right-padded with NaN to match the longest length.
        Columns are integer time indices starting at 0.
        """
        if not series_dict:
            return pd.DataFrame()
        #
        max_len = max(len(v) for v in series_dict.values())
        data = {}
        for series_id, arr in series_dict.items():
            data[series_id] = pd.Series(arr)

        # DataFrame with columns as series_ids and rows as time steps; we want
        # index=series_id and columns=time steps -> transpose
        df = pd.DataFrame(data)
        df = df.T
        # Set ordered integer column names
        df.columns = [f"t{idx + 1}" for idx in range(df.shape[1])]
        df.index.name = "series_id"
        return df

    def _load_npy_directory(self, directory: Path) -> Dict[str, np.ndarray]:
        """Reads all .npy files in a directory and returns a dict series_id->array."""
        series: Dict[str, np.ndarray] = {}
        if not directory.exists():
            return series

        for p in sorted(directory.glob("*.npy")):
            series_id = p.stem
            try:
                arr = np.load(p)
                # flatten 1D arrays
                arr = np.asarray(arr).reshape(-1)
                series[series_id] = arr
            except Exception:
                # skip unreadable files
                raise
        return series

    def _load_npy_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_series = self._load_npy_directory(self.train_dir)
        test_series = self._load_npy_directory(self.test_dir)

        df_train_wide = self._series_to_wide(train_series)
        df_test_wide = self._series_to_wide(test_series)

        return df_train_wide, df_test_wide

    def _wide_to_long(
        self, df: pd.DataFrame, include_labels: bool = False
    ) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["series_id", "time_idx", "target"])

        frame = df.copy()
        frame.index = frame.index.astype(str)
        frame.index.name = "series_id"

        long_df = frame.reset_index().melt(
            id_vars="series_id", var_name="time_step", value_name="target"
        )
        long_df["target"] = pd.to_numeric(long_df["target"], errors="coerce")
        # extract integer index from column name like t1 -> 1
        long_df["time_idx"] = (
            long_df["time_step"].str.extract(r"(\d+)")[0].astype(int) - 1
        )

        long_df = (
            long_df.dropna(subset=["target"])
            .sort_values(["series_id", "time_idx"])
            .reset_index(drop=True)
        )

        if include_labels and not self.labels_df.empty:
            # initialize flag
            long_df["anomaly"] = False
            # mark anomalies per series
            for _, row in self.labels_df.iterrows():
                sid = str(row["series_id"]).strip()
                try:
                    start = int(row["start"])
                    end = int(row["end"])
                except Exception:
                    continue
                # assume labeled file may be 1-based -> convert to 0-based
                start_idx = max(0, start - 1)
                end_idx = max(0, end - 1)
                mask = (long_df["series_id"] == sid) & (
                    long_df["time_idx"].between(start_idx, end_idx)
                )
                long_df.loc[mask, "anomaly"] = True

        cols = ["series_id", "time_idx", "target"]
        if include_labels:
            cols.append("anomaly")

        return long_df[cols]

    # Minimal compatibility wrappers used by the datamodule
    def _build_dataset(self, data: pd.DataFrame) -> TimeSeriesDataSet:
        if data.isnull().values.any():
            raise Exception("TEM NAN! ")

        return TimeSeriesDataSet(
            data,
            time_idx="time_idx",
            target="target",
            group_ids=["series_id"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["target"],
            max_encoder_length=self.input_width,
            min_encoder_length=self.input_width,
            max_prediction_length=self.output_width,
            min_prediction_length=self.output_width,
            target_normalizer=self.series_normalizer,
            add_relative_time_idx=False,
            add_target_scales=False,
            add_encoder_length=True,
            allow_missing_timesteps=False,
            randomize_length=None,
        )

    def build_train_dataset(self) -> TimeSeriesDataSet:
        self.train_dataset = self._build_dataset(self.df_train)
        return self.train_dataset

    def build_validation_dataset(
        self, train_dataset: TimeSeriesDataSet
    ) -> TimeSeriesDataSet:
        return TimeSeriesDataSet.from_dataset(
            train_dataset,
            self.df_train,
            predict=True,
            stop_randomization=True,
        )

    def build_test_dataset(self, train_dataset: TimeSeriesDataSet) -> TimeSeriesDataSet:
        # create evaluation frame by concatenating train+test for matching series
        frames = []
        for series_id, train_group in self.df_train.groupby("series_id", sort=False):
            test_group = self.df_test[self.df_test["series_id"] == series_id]
            if test_group.empty:
                continue

            test_group = test_group.copy()
            start_idx = int(train_group["time_idx"].max()) + 1
            test_group["time_idx"] = range(start_idx, start_idx + len(test_group))
            frames.append(pd.concat([train_group, test_group], ignore_index=True))

        if not frames:
            raise ValueError("Nenhuma série encontrada para avaliação.")

        evaluation_frame = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["series_id", "time_idx"])
            .reset_index(drop=True)
        )

        return TimeSeriesDataSet.from_dataset(
            train_dataset,
            evaluation_frame,
            predict=True,
            stop_randomization=True,
        )
