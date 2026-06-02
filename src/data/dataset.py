import os
from pathlib import Path
from typing import Literal
import numpy as np

import kagglehub
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from torch import from_numpy
from torch.types import Tensor
from torch.utils.data import Dataset

from src.config.config import ProjectSettings


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
    data_train: Tensor
    labels_train: Tensor

    data_test: Tensor
    labels_test: Tensor
    LABELS_COLUMN: str

    def __init__(
        self,
        PATH_FONTE_DADOS: Path = Path("/data/m4"),
    ) -> None:
        super().__init__()

        self.PATH_FONTE_DADOS = PATH_FONTE_DADOS
        self.PATH_FONTE_DADOS.mkdir(exist_ok=True)
        self.frequency = ProjectSettings.dataset_frequency
        df_train, df_test = self.load_dataset()

        # Preparar tensores para treino e teste
        data_train, label_train = self.data_prep(df_train, imputation=None)
        data_test, label_test = self.data_prep(df_test, imputation=None)

        # Armazenar tensores como atributos da instância
        self.data_train = data_train
        self.labels_train = label_train
        # Manter também versões de teste separadas
        self.data_test = data_test
        self.labels_test = label_test

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
            file_csv_test = str(dataset_test).replace("parquet", "csv")
            df_test = pd.read_csv(file_csv_test, header="infer", index_col=0)

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

        print(f"Dataset final: {df_train.shape}") if verbose else None
        print(f"Dataset final: {df_test.shape}") if verbose else None

        return df_train, df_test

    def data_prep(
        self,
        df: DataFrame,
        imputation: Literal["mean", "repeat", "interpolate"] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        function for data normalization

        :param self: imputa dados faltantes (ou apenas dropa eles), normaliza dados

        """

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
            # TODO ADICIONAR THRESHOLD PARA DROPNA
            df = df.dropna(axis=1, how="all")
        else:
            raise ValueError("")

        # converter para numpy array (timesteps, n_series)
        values = df.to_numpy(dtype=np.float64)

        # normalizar por série (zero mean, unit std) usando StandardScaler
        # aplicamos StandardScaler em cada série (coluna) individualmente
        scaled_cols = [
            StandardScaler().fit_transform(col.reshape(-1, 1)).ravel()
            for col in values.T
        ]
        scaled = np.column_stack(scaled_cols)

        # definir horizon (últimos valores como labels) baseado na frequência
        freq_key = str(self.frequency)
        horizon = Horizons.HORIZON.get(freq_key)
        assert horizon is not None

        # slice deve ser por linhas
        labels = values[-horizon:, :]

        # converter para tensores do PyTorch
        data = from_numpy(scaled).float()
        labels = from_numpy(np.asarray(labels, dtype=np.float32)).float()

        return data, labels
