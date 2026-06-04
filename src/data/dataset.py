import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from darts import TimeSeries
from darts.utils.data import (
    SequentialTorchTrainingDataset,
    SequentialTorchInferenceDataset,
)
from src.config.config import ProjectSettings


class Horizons:
    HORIZON: dict[str, int] = {
        "Yearly": 6,
        "Quarterly": 8,
        "Monthly": 18,
        "Daily": 14,
        "Hourly": 48,
    }
    INPUT_WIDTH: dict[str, int] = {
        "Yearly": 8,
        "Quarterly": 10,
        "Monthly": 28,
        "Daily": 7,
        "Hourly": 16,
    }

    @classmethod
    def input_width(cls, frequency: str) -> int:
        return cls.INPUT_WIDTH[frequency]

    @classmethod
    def output_width(cls, frequency: str) -> int:
        return cls.HORIZON[frequency]


class StrokeDataset:
    df_train_wide: DataFrame
    df_test_wide: DataFrame

    def __init__(self, PATH_FONTE_DADOS: Path = Path("data/m4"), verbose=False) -> None:
        self.PATH_FONTE_DADOS: Path = PATH_FONTE_DADOS
        self.PATH_FONTE_DADOS.mkdir(parents=True, exist_ok=True)
        self.frequency = ProjectSettings.dataset_frequency
        self.input_width: int = Horizons.input_width(self.frequency)
        self.output_width: int = Horizons.output_width(self.frequency)

        # Carrega dados originais (formato Wide)
        self.df_train_wide, self.df_test_wide = self.load_dataset(verbose=verbose)
        if verbose:
            logging.info("DADOS BAIXADOS E LIDOS!!")

        if ProjectSettings.run_mode == "prototype":
            # Mantém poucas séries que têm um tamanho mínimo aceitável
            self.df_train_wide = self.df_train_wide.dropna(thresh=20)
            # Garante o alinhamento das séries de teste com o protótipo
            self.df_test_wide = self.df_test_wide.loc[self.df_train_wide.index]

        self.build_dataset()

        if verbose:
            logging.info("SERIES CRIADAS!")

    def load_dataset(self, verbose=False) -> Tuple[DataFrame, DataFrame]:
        """Carrega os arquivos M4 em wide format e cachea em Parquet."""
        pqt_train: Path = self.PATH_FONTE_DADOS / f"{self.frequency}-train.parquet"
        pqt_test: Path = self.PATH_FONTE_DADOS / f"{self.frequency}-test.parquet"
        csv_train: Path = self.PATH_FONTE_DADOS / f"{self.frequency}-train.csv"
        csv_test: Path = self.PATH_FONTE_DADOS / f"{self.frequency}-test.csv"

        if pqt_train.exists() and pqt_test.exists():
            if verbose:
                logging.info(f"Carregando cache: {pqt_train}")
            df_train: DataFrame = pd.read_parquet(pqt_train, engine="auto")
            df_test: DataFrame = pd.read_parquet(pqt_test, engine="auto")
        else:
            if not csv_train.exists() or not csv_test.exists():
                if verbose:
                    logging.info(
                        f"Baixando M4 via KaggleHub para {self.PATH_FONTE_DADOS}"
                    )
                import kagglehub

                kagglehub.dataset_download(
                    "yogesh94/m4-forecasting-competition-dataset",
                    output_dir=str(self.PATH_FONTE_DADOS),
                )

            if not csv_train.exists() or not csv_test.exists():
                raise FileNotFoundError(
                    f"CSVs do M4 não encontrados em {self.PATH_FONTE_DADOS}"
                )

            df_train: DataFrame = pd.read_csv(csv_train, header="infer", index_col=0)
            df_test: DataFrame = pd.read_csv(csv_test, header="infer", index_col=0)

            try:
                df_train.to_parquet(
                    pqt_train, compression="gzip", engine="auto", index=True
                )
                df_test.to_parquet(
                    pqt_test, compression="gzip", engine="auto", index=True
                )
                if verbose:
                    logging.info(f"Cache criado: {pqt_train} e {pqt_test}")
            except Exception as e:
                if verbose:
                    logging.info(f"Aviso: não foi possível cachear Parquet: {e}")

        if verbose:
            logging.info(f"Dataset: treino {df_train.shape}, teste {df_test.shape}")

        return df_train, df_test

    def _wide_to_darts_series(self, df: DataFrame) -> List[TimeSeries]:
        """Converte diretamente o DataFrame Wide do M4 em uma lista de objetos Darts TimeSeries."""
        series_list = []

        # Iterando sobre cada linha (cada série temporal única do M4)
        for series_id, row in df.iterrows():
            # Remove valores nulos (o M4 wide preenche séries curtas com NaN no final)
            clean_values = row.dropna().values

            if len(clean_values) == 0:
                continue

            # Como o M4 original usa passos de tempo inteiros abstratos (V1, V2...),
            # criamos um range numérico simples para o índice.
            time_axis = np.arange(len(clean_values))

            # Cria o Dataframe individual esperado pelo Darts
            single_ts_df = pd.DataFrame({"target": clean_values}, index=time_axis)

            # Instancia o objeto TimeSeries do Darts
            ts = TimeSeries.from_dataframe(
                single_ts_df,
                value_cols="target",
                time_col=None,  # Quando None, assume o índice do dataframe
            )

            # Adiciona metadados estáticos (opcional, útil para alguns modelos do Darts)
            # ts = ts.with_static_covariates(pd.Series([series_id], index=["series_id"]))

            series_list.append(ts)

        return series_list

    def build_dataset(self):

        # No Darts, trabalhamos com List[TimeSeries] para múltiplas séries (Global Models)
        self.train_series: List[TimeSeries] = self._wide_to_darts_series(
            self.df_train_wide
        )
        logging.info("SERIES DE TREINO CRIADAS!")

        # Validação: No Darts, a validação geralmente são os últimos 'output_width' pontos do treino
        self.val_series: List[TimeSeries] = [
            ts[-self.output_width :] for ts in self.train_series
        ]

        logging.info("SERIES DE VALIDAÇÃO CRIADAS!")

        # Teste: O dataset de teste do M4 são os passos futuros reais
        self.test_series: List[TimeSeries] = self._wide_to_darts_series(
            self.df_test_wide
        )

        # ---#

        self.train_dataset = SequentialTorchTrainingDataset(
            series=self.train_series,
            input_chunk_length=self.input_width,
            output_chunk_length=self.output_width,
        )

        self.val_dataset = SequentialTorchInferenceDataset(
            series=self.val_series,
            input_chunk_length=self.input_width,
            output_chunk_length=self.output_width,
        )

        self.test_dataset = SequentialTorchInferenceDataset(
            series=self.test_series,
            input_chunk_length=self.input_width,
            output_chunk_length=self.output_width,
        )

    # Métodos para manter compatibilidade de assinatura externa se necessário
    def build_train_dataset(self) -> SequentialTorchTrainingDataset:

        return self.train_dataset

    def build_validation_dataset(self) -> SequentialTorchInferenceDataset:
        return self.val_dataset

    def build_test_dataset(self) -> SequentialTorchInferenceDataset:
        return self.test_dataset
