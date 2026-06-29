from __future__ import annotations

import logging
import os
from pathlib import Path
import traceback
from typing import Tuple

import numpy as np
import pandas as pd

from src.config.config import ProjectSettings
from darts import TimeSeries
from src.data.torch_dataset import Horizons
import ast
from pandas import DataFrame


class NasaDataset:
    """Loads multivariate telemetry .npy time series and labeled anomalies for the
    NASA SMAP/MSL anomaly-detection dataset
    (https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl).

    Dataset layout expected on disk (after extracting the Kaggle zip):
        <base_path>/
            data/
                data/
                    train/   <- one .npy per channel, shape (n_timesteps, n_features)
                    test/    <- idem
            labeled_anomalies.csv

    The first column of each .npy array is the univariate target telemetry value;
    the remaining columns are one-hot encoded command / context features.

    labeled_anomalies.csv columns used:
        chan_id            – matches the .npy filename stem (e.g. "P-1")
        spacecraft         – "SMAP" or "MSL"
        anomaly_sequences  – list of [start, end] index pairs (1-based, inclusive)
        class              – anomaly type label (kept for reference, not used in training)

    Public interface (compatible with the existing datamodule):
        build_train_dataset()
        build_validation_dataset(train_dataset)
        build_test_dataset(train_dataset)

    Useful attributes after __init__:
    df_train  – long-format DataFrame with columns [series_id, time_idx, target, feat_0]
    df_test   – idem, plus an "anomaly" boolean column
    labels_df – parsed anomaly labels with columns
        [chan_id, spacecraft, class, seq_start, seq_end] (one row per sequence)
    """

    def __init__(
        self,
        base_path: Path,
        normalize: bool = False,
        export_path: Path = Path(".", "exports"),
        verbose=False,
    ) -> None:
        self.verbose = verbose
        export_path.mkdir(parents=True, exist_ok=True)

        self.telemetry_column = "feat_0"

        prototype = ProjectSettings.run_mode == "prototype"

        # Load raw numpy arrays ------------------------------------------------
        train_wide, test_wide = self.load_dataset(base_path, prototype=prototype, skip_load=True)
        if prototype:
            test_wide[:10_000].to_csv(export_path / "test_wide.csv")

        # Parse labels ---------------------------------------------------------
        self.labels_df = self._read_labels()
        if prototype:
            self.labels_df[:10_000].to_csv(export_path / "./labels.csv")

        # Convert to long format -----------------------------------------------
        self.df_test, self.tam_dataset = self.multi_to_df(test_wide, include_labels=True)
        if prototype:
            self.df_test[:10_000].to_csv(export_path / "./test.csv")

        # Get feature columns == feat_0 column ---
        self.feature_cols = [
            col for col in self.df_test.columns if col not in ["series_id", "time_idx", "target"]
        ]

        # Dataset hyperparams --------------------------------------------------
        self.frequency = ProjectSettings.dataset.dataset_frequency
        self.input_width = Horizons.input_width(self.frequency)
        self.output_width = Horizons.output_width(self.frequency)
        # self.series_normalizer = GroupNormalizer(groups=["series_id"])

        # Lazily built TimeSeriesDataSet objects
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_dataset(
        self, base_path: Path, prototype: bool = False, skip_load=True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carrega os arquivos M4 em wide format e cachea em Parquet."""
        base_path.mkdir(parents=True, exist_ok=True)
        self.labels_file = base_path / "labeled_anomalies.csv"

        download_path = base_path
        base_path = base_path / "data" / "data"
        train_dir = base_path / "train"
        test_dir = base_path / "test"
        pqt_train: Path = download_path / "nasa_train.parquet"
        pqt_test: Path = download_path / "nasa_test.parquet"

        # Initialize variables to avoid unbound variable errors
        df_train: DataFrame
        df_test: DataFrame

        try:
            if pqt_train.exists() and pqt_test.exists():
                # se existe, tenta ler parquet
                if self.verbose:
                    logging.info(f"Carregando cache: {pqt_train}")
                df_train = pd.read_parquet(pqt_train, engine="auto")
                df_test = pd.read_parquet(pqt_test, engine="auto")

            # DOWNLOAD DO DATASET
            elif not train_dir.exists() or not test_dir.exists():
                if self.verbose:
                    logging.info(f"Baixando M4 via KaggleHub para {download_path}")
                import kagglehub

                dataset_path = kagglehub.dataset_download(
                    "patrickfleith/nasa-anomaly-detection-dataset-smap-msl",
                    output_dir=str(download_path),
                )
                logging.info(f"DATASET PATH: {dataset_path}")
                if os.environ["AMBIENTE"] == "KAGGLE":
                    base_path = Path(dataset_path)
                    base_path = base_path / "data" / "data"
                    train_dir = base_path / "train"
                    test_dir = base_path / "test"
                    self.labels_file = Path(dataset_path) / "labeled_anomalies.csv"

                if not train_dir.exists() or not test_dir.exists():
                    raise FileNotFoundError(f".npy DA NASA não encontrados em {base_path}")

                logging.info("Loading train split …")
                df_train = self._load_npy_directory(train_dir, prototype=prototype)
                logging.info(f"DF TRAIN:\n {df_train.head(5)}")
                logging.info("Loading test split …")
                df_test = self._load_npy_directory(test_dir, prototype=prototype)

                if df_train is None or df_test is None:
                    raise RuntimeError("Failed to load dataset: df_train or df_test is None")

                df_train.to_parquet(pqt_train, compression="gzip", engine="auto", index=True)
                df_test.to_parquet(pqt_test, compression="gzip", engine="auto", index=True)
                if self.verbose:
                    logging.info(f"Cache criado: {pqt_train} e {pqt_test}")
            else:
                raise Exception("SEM PARQUET NA PASTA DATASET!")

        except Exception as e:
            if self.verbose:
                logging.info(f"ERRO AO CARREGAR DATASET: {e}")
                traceback.print_exc()
            raise

        if self.verbose:
            logging.info(f"Dataset: treino {df_train.shape}, teste {df_test.shape}")

        return df_train, df_test

    def _load_npy_directory(self, directory: Path, prototype: bool = False) -> pd.DataFrame:
        """Read every .npy file in *directory*.

        Each file must contain an array of shape (n_timesteps,) or
        (1, n_timesteps).
        Returns a DataFrame with MultiIndex (series_id, feature_id, time_idx ).
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        records = []

        for path in sorted(directory.glob("*.npy")):
            chan_id = path.stem
            arr: np.ndarray = np.load(path)
            n_timesteps, n_features = arr.shape

            # Build records with MultiIndex structure
            f = 0
            # AVISO: NO DATASET, APENAS O PRIMEIRO CANAL É TELEMETRIA!
            for t in range(n_timesteps):
                records.append(
                    {
                        "series_id": chan_id,
                        "feature_id": f"feat_{f}",
                        "time_idx": t,
                        "value": arr[t, f],
                    }
                )

            if prototype:
                logging.info(f"  Loaded (prototype): {chan_id}  shape={arr.shape}")
                # one channel is enough for prototyping

        # Create DataFrame with MultiIndex
        df = pd.DataFrame(records)

        # Set MultiIndex
        df = df.set_index(["series_id", "feature_id", "time_idx"])
        df.sort_index(level=0)

        logging.info(
            f"Loaded {len(df.index.get_level_values('series_id').unique())} channels from {directory}"
        )
        return df

    # ------------------------------------------------------------------
    # Label parsing
    # ------------------------------------------------------------------

    def _read_labels(self) -> pd.DataFrame:
        """Parse labeled_anomalies.csv into a tidy DataFrame.

        Returns a DataFrame with one row per anomalous sequence:
            series_id      – channel identifier matching .npy filename stem
            spacecraft   – "SMAP" or "MSL"
            class        – anomaly type string
            seq_start    – start index (0-based, inclusive)
            seq_end      – end index   (0-based, inclusive)
        """

        df = pd.read_csv(
            self.labels_file,
            usecols=["chan_id", "anomaly_sequences"],
        )

        records = []
        for _, row in df.iterrows():
            chan_id = str(row["chan_id"]).strip()
            # anomaly_sequences is a string like "[[2149, 2349], [4536, 4844]]"
            sequences = ast.literal_eval(str(row["anomaly_sequences"]))
            for start, end in sequences:
                records.append(
                    {
                        "series_id": chan_id,
                        # The original labels are 1-based; convert to 0-based.
                        "seq_start": max(0, int(start) - 1),
                        "seq_end": max(0, int(end) - 1),
                    }
                )

        labels = pd.DataFrame(records)
        logging.info(
            f"Loaded {len(labels)} anomaly sequences across {labels['series_id'].nunique()} channels."
        )
        return labels

    # ------------------------------------------------------------------
    # Wide dict → long DataFrame
    # ------------------------------------------------------------------

    def multi_to_df(
        self, df_multi: pd.DataFrame, include_labels: bool = False, drop=True
    ) -> tuple[pd.DataFrame, tuple[int, int]]:
        """Converte o DataFrame MultiIndex para o formato final desejado:

        time_idx | target | feat_0
        """
        if drop:
            row_all_nan = (
                df_multi["value"].groupby(level="time_idx").transform(lambda x: x.isnull().all())
            )
            logging.info(df_multi[row_all_nan].head(5))
            # deixa apenas os time_idx que NÃO são totalmente compostos por NaN
            df_multi = df_multi[~row_all_nan]

        tam_dataset = df_multi.shape

        # 1. Faz o unstack para trazer o 'feat_0' para as colunas
        df_wide = df_multi["value"].unstack(level="feature_id")

        # 3. Reseta o índice para mover 'series_id' e 'time_idx' de volta como colunas comuns
        df_wide = df_wide.reset_index().sort_values(by=["series_id", "time_idx"])

        # 5. Se include_labels for True, cria a coluna de target
        if include_labels and not self.labels_df.empty:
            # Cria uma cópia do target como base para a coluna de anomalias
            df_wide["target"] = 0

            for series_id in self.labels_df["series_id"].unique():
                # Filtra as anomalias para esta série
                series_labels: pd.Series = self.labels_df[self.labels_df["series_id"] == series_id]

                # Filtra o df_wide para esta série
                series_mask = df_wide["series_id"] == series_id
                series_data = df_wide[series_mask]

                if series_data.empty:
                    raise ValueError(f"ERA PRA TER VALOR NA SERIE {series_id}")

                max_time_idx = series_data["time_idx"].max()

                # Cria um array booleano de zeros com o tamanho do maior time_idx + 1
                anomaly_array = np.zeros(max_time_idx + 1, dtype=int)

                # Para cada sequência de anomalia nesta série
                for _, row in series_labels.iterrows():
                    seq_start = int(row["seq_start"])
                    seq_end = int(row["seq_end"])
                    # Garante que os índices estão dentro dos limites
                    seq_start = max(0, seq_start)
                    seq_end = min(max_time_idx, seq_end)
                    # Marca os índices entre seq_start e seq_end como 1
                    anomaly_array[seq_start : seq_end + 1] = 1

                # Atualiza a coluna target no df_wide para esta série
                df_wide.loc[series_mask, "target"] = df_wide.loc[series_mask, "time_idx"].apply(
                    lambda x: anomaly_array[x] if x <= max_time_idx else np.nan
                )

        return df_wide, tam_dataset

    # ------------------------------------------------------------------
    # TimeSeriesDataSet builders
    # ------------------------------------------------------------------

    def build_train_dataset(self) -> list[TimeSeries]:
        logging.info(f"DF TEST: {self.df_test}")
        logging.info(f"COLUNAS: {self.df_test.columns}")
        self.train_dataset = TimeSeries.from_group_dataframe(
            self.df_test,
            group_cols="series_id",
            time_col="time_idx",
            value_cols=self.feature_cols,
        )

        return self.train_dataset

    def build_test_dataset(
        self,
    ) -> list[TimeSeries]:
        """Build the test TimeSeries using the same telemetry features as train."""
        self.test_dataset = TimeSeries.from_group_dataframe(
            self.df_test,
            group_cols="series_id",
            time_col="time_idx",
            value_cols=self.feature_cols,
        )
        return self.test_dataset

    def build_test_labels(self) -> list[np.ndarray]:
        """Return the ground-truth anomaly labels (0/1) per series as a list of 1-D arrays.

        Each array has length equal to the number of time steps of that series
        in df_test, perfectly aligned with the series returned by
        ``build_test_dataset``.
        """
        labels_list = []
        for series_id, group in self.df_test.groupby("series_id", sort=False):
            group = group.sort_values("time_idx")
            labels_list.append(group["target"].values.astype(np.float64))
        return labels_list
