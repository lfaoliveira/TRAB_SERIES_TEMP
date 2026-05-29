import os
from pathlib import Path
from typing import Literal
import numpy as np

import kagglehub
import pandas as pd
from pandera.typing import DataFrame
from sklearn.preprocessing import StandardScaler
from torch import from_numpy
from torch.types import Tensor
from torch.utils.data import Dataset

from src.config.config import CentralConfig


class Horizons:
    HORIZON = {
        # Output width of model (specified by m4 competition)
        "Yearly": 6,
        "Quarterly": 8,
        "Monthly": 18,
        "Daily": 14,
        "Hourly": 48,
    }
    INPUT_WIDTH = {
        # Input width of model (adjust as needed. Taking seasonality into
        # account could be a good approach.)
        "Yearly": 8,
        "Quarterly": 10,
        "Monthly": 28,
        "Daily": 18,
        "Hourly": 16,
    }


class StrokeDataset(Dataset):
    original_df: DataFrame
    dataframe: DataFrame
    data: Tensor
    labels: Tensor
    LABELS_COLUMN: str

    def __init__(
        self,
        PATH_FONTE_DADOS: Path = Path("data/m4"),
    ) -> None:
        super().__init__()

        self.PATH_FONTE_DADOS = PATH_FONTE_DADOS
        self.PATH_FONTE_DADOS.mkdir(exist_ok=True)
        self.frequency = CentralConfig.dataset_frequency
        self.df_train, self.df_test = self.load_dataset()
        self.data_prep()

    def __getitem__(self, index: Tensor | list[int] | int):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    def load_dataset(self, verbose=False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Carrega dataset do Kaggle, salva em Parquet.

        Args:
            path_dados: Caminho para armazenar dados
            usar_parquet: Se True, tenta ler de Parquet primeiro

        Returns:
            DataFrame pré-processado
        """

        dataset_train = self.PATH_FONTE_DADOS / f"{self.frequency}-train.parquet"
        dataset_test = self.PATH_FONTE_DADOS / f"{self.frequency}-test.parquet"

        # Carregar dados se existir
        if dataset_train.exists():
            print(
                f"Lendo dataset a partir de Parquet: {dataset_train}"
            ) if verbose else None

            df_train = pd.read_parquet(dataset_train, engine="auto")
            df_test = pd.read_parquet(dataset_test, engine="auto")
        else:
            # baixar do Kaggle
            print(
                f"Baixando dataset  M4 todo via Kagglehub em {self.PATH_FONTE_DADOS}"
            ) if verbose else None
            kagglehub.dataset_download(
                "yogesh94/m4-forecasting-competition-dataset",
                output_dir=str(self.PATH_FONTE_DADOS),
            )

            file_csv = str(dataset_train).replace("parquet", "csv")
            df_train = pd.read_csv(file_csv, header="infer", index_col=0)
            df_test = pd.read_csv(file_csv, header="infer", index_col=0)

            try:
                df_train.to_parquet(
                    dataset_train, compression="gzip", engine="auto", index=True
                )
                df_test.to_parquet(
                    dataset_test, compression="gzip", engine="auto", index=True
                )

                print(
                    f"Dataset salvo com sucesso em: {dataset_train} e {dataset_test}"
                ) if verbose else None

            except Exception as e:
                print(f"Falha ao salvar Parquet: {e}")

        # Pré-processamento: remover NaN
        df_train = df_train.dropna()
        df_test = df_test.dropna()

        print(f"Dataset final: {df_train.shape}") if verbose else None
        print(f"Dataset final: {df_test.shape}") if verbose else None

        return df_train, df_test

    def data_prep(
        self,
        df: DataFrame | None = None,
        imputation: Literal["mean", "repeat", "interpolate"] | None = None,
    ) -> None:
        """
        function for data normalization

        :param self: imputa dados faltantes (ou apenas dropa eles), normaliza dados

        """

        # usar df fornecido ou o de treino como padrão
        if df is None:
            df = self.df_train

        # garantir numérico (força NaN se não for conversível)
        df = df.apply(pd.to_numeric, errors="coerce")

        # imputação por série (coluna)
        if imputation == "mean":
            df = df.apply(lambda col: col.fillna(col.mean()), axis=0)
        elif imputation == "repeat":
            df = df.apply(
                lambda col: col.fillna(method="ffill").fillna(method="bfill"), axis=0
            )
        elif imputation == "interpolate":
            df = df.apply(
                lambda row: (
                    row.interpolate().fillna(method="bfill").fillna(method="ffill")
                ),
                axis=0,
            )
        elif imputation is None:
            df = df.dropna(axis=1, how="any")

        # converter para numpy array (n_series, timesteps)
        values = df.to_numpy(dtype=float)

        # normalizar por série (zero mean, unit std) usando StandardScaler
        # aplicamos StandardScaler em cada série (linha) individualmente
        # aplicar StandardScaler em cada série (linha) individualmente
        scaled_rows = [
            StandardScaler().fit_transform(row.reshape(-1, 1)).ravel() for row in values
        ]
        scaled = np.vstack(scaled_rows)

        # definir horizon (últimos valores como labels) baseado na frequência

        freq_key = str(self.frequency)
        horizon = Horizons.HORIZON.get(freq_key, 1)

        labels = values[:, -horizon:]

        # converter para tensores do PyTorch
        self.data = from_numpy(scaled).float()
        self.labels = from_numpy(np.asarray(labels, dtype=np.float32)).float()
