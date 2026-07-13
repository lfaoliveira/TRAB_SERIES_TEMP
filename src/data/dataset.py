from __future__ import annotations

import logging
import os
from pathlib import Path
import traceback
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config.config import ProjectSettings
from darts import TimeSeries
import ast
from pandas import DataFrame
from darts.utils.model_selection import train_test_split


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
        train_split: float = 0.8,
        export_path: Path = Path(".", "exports"),
        verbose=False,
    ) -> None:
        self.verbose = verbose
        export_path.mkdir(parents=True, exist_ok=True)

        self.telemetry_column = "feat_0"

        prototype = ProjectSettings.run_mode == "prototype"

        # Load raw numpy arrays ------------------------------------------------
        train_wide, test_wide = self.load_dataset(base_path, prototype=prototype)
        if prototype:
            test_wide[:10_000].to_csv(export_path / "test_wide.csv")

        # Parse labels ---------------------------------------------------------
        self.labels_df = self._read_labels()
        if prototype:
            self.labels_df[:10_000].to_csv(export_path / "./labels.csv")

        # Convert to long format -----------------------------------------------
        self.df_test, self.tam_dataset = self.multi_to_df(test_wide, include_labels=True, drop=True)
        self.df_train, _ = self.multi_to_df(train_wide, include_labels=False, drop=True)
        if prototype:
            self.df_test[:10_000].to_csv(export_path / "./test.csv")

        # Get feature columns == feat_0 column ---
        self.feature_cols = [
            col for col in self.df_test.columns if col not in ["series_id", "time_idx", "target"]
        ]

        splitados = self.splitar(train_split)
        self.train_values, self.test_values, self.train_labels, self.test_labels = splitados

        # Dataset hyperparams --------------------------------------------------
        self.frequency = ProjectSettings.dataset.dataset_frequency
        # self.input_width = Horizons.input_width(self.frequency)
        # self.output_width = Horizons.output_width(self.frequency)
        # self.series_normalizer = GroupNormalizer(groups=["series_id"])

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_dataset(self, base_path: Path, prototype: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carrega os arquivos M4 em wide format e cachea em Parquet."""

        self.labels_file = base_path / "labeled_anomalies.csv"
        base_path = base_path / "data" / "data"
        train_dir = base_path / "train"
        test_dir = base_path / "test"

        if os.environ.get("AMBIENTE") != "KAGGLE":
            # NOTE No Kaggle, mkdir casusa erro
            base_path.mkdir(parents=True, exist_ok=True)

        download_path = base_path

        # Initialize variables to avoid unbound variable errors
        df_train: DataFrame
        df_test: DataFrame

        try:
            if train_dir.exists() and test_dir.exists():
                # Carrega direto dos .npy
                logging.info("Loading train split …")
                df_train = self._load_npy_directory(train_dir, prototype=prototype)
                logging.info("Loading test split …")
                df_test = self._load_npy_directory(test_dir, prototype=prototype)

                if df_train is None or df_test is None:
                    raise RuntimeError("Failed to load dataset: df_train or df_test is None")

            else:
                # DOWNLOAD DO DATASET
                if self.verbose:
                    logging.info(f"Baixando NASA via KaggleHub para {download_path}")
                import kagglehub

                dataset_path = kagglehub.dataset_download(
                    "patrickfleith/nasa-anomaly-detection-dataset-smap-msl",
                    output_dir=str(download_path),
                )
                logging.info(f"DATASET PATH: {dataset_path}")
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
                logging.debug(f"  Loaded (prototype): {chan_id}  shape={arr.shape}")
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
            row_all_nan = df_multi["value"].groupby(level="time_idx").transform(lambda x: x.isnull().all())
            logging.info(df_multi[row_all_nan].head(5))
            # deixa apenas os time_idx que NÃO são totalmente compostos por NaN
            df_multi = df_multi[~row_all_nan]

        tam_dataset = df_multi.shape

        # 1. Faz o unstack para trazer o 'feat_0' para as colunas
        df_wide = df_multi["value"].unstack(level="feature_id")

        # 3. Reseta o índice para mover 'series_id' e 'time_idx' de volta como colunas comuns e ORDENA
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

    def splitar(
        self, train_split: float
    ) -> tuple[Sequence[TimeSeries], Sequence[TimeSeries], Sequence[TimeSeries], Sequence[TimeSeries]]:

        # Valores de treino vêm de self.df_train (já é o conjunto de treino)
        train_values = TimeSeries.from_group_dataframe(
            self.df_train,
            group_cols="series_id",
            time_col="time_idx",
            value_cols=self.feature_cols,
        )

        # Valores de teste vêm de self.df_test (já é o conjunto de teste)
        test_values = TimeSeries.from_group_dataframe(
            self.df_test,
            group_cols="series_id",
            time_col="time_idx",
            value_cols=self.feature_cols,
        )

        # Labels de treino: df_train não tem coluna target (dados não rotulados),
        # então criamos uma dummy com zeros (assunção: treino é todo normal)
        df_train_with_target = self.df_train.copy()
        df_train_with_target["target"] = 0
        train_labels = TimeSeries.from_group_dataframe(
            df_train_with_target,
            group_cols="series_id",
            time_col="time_idx",
            value_cols="target",
        )

        # Labels de teste vêm do df_test (que tem a coluna target)
        test_labels = TimeSeries.from_group_dataframe(
            self.df_test,
            group_cols="series_id",
            time_col="time_idx",
            value_cols="target",
        )

        return (train_values, test_values, train_labels, test_labels)  # pyright: ignore[reportReturnType]

    # ------------------------------------------------------------------
    # TimeSeriesDataSet builders
    # ------------------------------------------------------------------

    def build_train_dataset(
        self, labels=True
    ) -> Sequence[TimeSeries] | tuple[Sequence[TimeSeries], Sequence[TimeSeries]]:
        if labels:
            return self.train_values, self.train_labels
        else:
            return self.train_values

    def build_test_dataset(
        self, labels=True
    ) -> Sequence[TimeSeries] | tuple[Sequence[TimeSeries], Sequence[TimeSeries]]:
        if labels:
            return self.test_values, self.test_labels
        else:
            return self.test_values


class SlidingWindowDataset(Dataset):
    """
    Dataset PyTorch que transforma uma lista de séries temporais em
    janelas deslizantes de tamanho fixo de forma eficiente.

    NOTE: USADO APENAS PARA TAREFAS DE RECONSTRUCAO.
    """

    def __init__(self, series_list: list, window_size: int, labels_list: list | None = None):
        self.window_size = window_size
        self.windows = []
        self.targets = []
        self._series_window_counts: list[int] = []
        self._series_original_lengths: list[int] = []

        if labels_list is not None and len(labels_list) != len(series_list):
            raise ValueError("labels_list must have the same length as series_list")

        for i, ts in enumerate(series_list):
            # Garante que estamos extraindo os valores brutos como array 1D
            if hasattr(ts, "values"):
                arr = ts.values(copy=False).flatten()
            else:
                arr = np.asarray(ts).flatten()

            if labels_list is None:
                raise ValueError("SEM LABELS!")
            else:
                label_ts = labels_list[i]
                if hasattr(label_ts, "values"):
                    labels = label_ts.values(copy=False).flatten().astype(np.int64)
                else:
                    labels = np.asarray(label_ts).flatten().astype(np.int64)

            # Filtra séries mais curtas que a janela desejada
            if len(arr) < window_size:
                self._series_window_counts.append(0)
                self._series_original_lengths.append(len(arr))
                continue

            """ Cria janelas deslizantes sem duplicar os dados na memória (O(1) memory)
             Para um array de tamanho N, gera (N - window_size + 1) janelas. 
             O número de janelas possíveis é sempre (N - window_size + 1) """
            shape_v = (arr.size - window_size + 1, window_size)
            shape_tg = (labels.size - window_size + 1, window_size)
            # quantidade de bytes para pular para a próxima linha / coluna
            strides = (arr.strides[0], arr.strides[0])
            # cria matriz linearizada em memória (para cada série temporal)
            ts_windows = np.lib.stride_tricks.as_strided(arr, shape=shape_v, strides=strides)
            target_windows = np.lib.stride_tricks.as_strided(labels, shape=shape_tg, strides=strides)

            self.windows.append(ts_windows)
            self.targets.append(target_windows[:, -1])
            # linhas = janelas
            self._series_window_counts.append(len(ts_windows))
            self._series_original_lengths.append(len(ts))
            """
            Ex:
            Linha 0 (Janela 1): [10, 20, 30]
            Linha 1 (Janela 2): [20, 30, 40]
            Linha 2 (Janela 3): [30, 40, 50]
            """

        # Concatena todas as janelas de todas as séries em uma única matriz
        if self.windows:
            """ O np.concatenate(..., axis=0) pega todas essas matrizes e as 
            "empilha" verticalmente (uma embaixo da outra). """
            self.windows = np.concatenate(self.windows, axis=0)
            self.targets = np.concatenate(self.targets, axis=0)
        else:
            raise ValueError("SEM JANELAS!")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y

    def windows_to_point_scores(
        self, mse_per_window: np.ndarray, threshold: float | None = None
    ) -> list[np.ndarray | None]:
        """Converte erro de reconstrução por janela para score binário por ponto.

        Cada ponto da série original pertence a até ``window_size`` janelas.
        Primeiro, cada janela é classificada como anômala (1) ou normal (0)
        comparando seu MSE com ``threshold * max(MSE)``. Em seguida, um ponto
        só é considerado anômalo se MAIS DE 50% das janelas que o contêm
        forem anômalas (voto majoritário).

        Parâmetros
        ----------
        mse_per_window : np.ndarray
            Array 1D com o MSE de reconstrução de cada janela, na mesma ordem
            do dataset (``self.windows``).
        threshold : float
            Fração do maior MSE usada como limiar para classificar uma janela
            como anômala.

        Retorna
        -------
        list[np.ndarray | None]
            Lista com um array binário por série original (1 = ponto anômalo).
            ``None`` para séries que eram curtas demais e não geraram janelas.
        """
        window_scores = np.asarray(mse_per_window).ravel()
        if len(window_scores) != len(self.windows):
            raise ValueError(
                f"mse_per_window deve ter {len(self.windows)} valores, recebeu {len(window_scores)}"
            )

        # 1. Classifica cada janela como anômala (1) ou normal (0) via threshold
        max_score = window_scores.max()
        limiar = threshold * max_score if max_score > 0 else 0.0
        is_window_anomalous = (window_scores > limiar).astype(int)

        scores: list[np.ndarray | None] = []
        idx = 0
        window_size = self.window_size

        for n_windows, orig_len in zip(self._series_window_counts, self._series_original_lengths):
            if n_windows == 0 or orig_len == 0:
                scores.append(None)
                continue

            series_windows = is_window_anomalous[idx : idx + n_windows]
            idx += n_windows

            # 2. Para cada ponto, conta em quantas janelas ele aparece
            #    e em quantas dessas janelas foi marcado como anômalo.
            anomalous_votes = np.zeros(orig_len, dtype=int)
            total_votes = np.zeros(orig_len, dtype=int)

            for window_idx, is_anomalous in enumerate(series_windows):
                start = window_idx
                end = min(window_idx + window_size, orig_len)
                total_votes[start:end] += 1
                anomalous_votes[start:end] += is_anomalous

            # 3. Ponto é anômalo somente se mais de 50% das janelas
            #    que o cobrem forem anômalas.
            vote_ratio = np.divide(
                anomalous_votes,
                total_votes,
                out=np.zeros(orig_len, dtype=float),
                where=total_votes > 0,
            )
            scores.append((vote_ratio > 0.5).astype(int))

        return scores
