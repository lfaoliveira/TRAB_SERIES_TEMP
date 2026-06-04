import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from src.config.config import ProjectSettings


class Horizons:
    HORIZON: dict[str, int] = {
        # Output width of model (specified by m4 competition)
        "Yearly": 6,
        "Quarterly": 8,
        "Monthly": 18,
        "Daily": 14,
        "Hourly": 48,
    }
    INPUT_WIDTH: dict[str, int] = {
        # Input width of model (adjust as needed. Taking seasonality into
        # account could be a good approach.)
        "Yearly": 8,
        "Quarterly": 10,
        "Monthly": 28,
        "Daily": 7,  # 18,
        "Hourly": 16,
    }

    @classmethod
    def input_width(cls, frequency: str) -> int:
        return cls.INPUT_WIDTH[frequency]

    @classmethod
    def output_width(cls, frequency: str) -> int:
        return cls.HORIZON[frequency]


class TorchStrokeDataset:
    df_train_wide: DataFrame
    df_test_wide: DataFrame

    def __init__(self, PATH_FONTE_DADOS: Path = Path("data/m4"), verbose=False) -> None:
        self.PATH_FONTE_DADOS: Path = PATH_FONTE_DADOS
        self.PATH_FONTE_DADOS.mkdir(parents=True, exist_ok=True)
        self.frequency = ProjectSettings.dataset_frequency
        self.input_width: int = Horizons.input_width(self.frequency)
        self.output_width: int = Horizons.output_width(self.frequency)
        self.series_normalizer = GroupNormalizer(groups=["series_id"])

        self.df_train_wide, self.df_test_wide = self.load_dataset()
        if ProjectSettings.run_mode == "prototype":
            # Fica com poucas series que nao sao nulas
            self.df_train_wide = self.df_train_wide.dropna(thresh=200)

        if verbose:
            logging.info("DADOS BAIXADOS E LIDOS!!")
        self.df_train: DataFrame = self._wide_to_long(self.df_train_wide)
        self.df_test: DataFrame = self._wide_to_long(self.df_test_wide)

        self.train_dataset: TimeSeriesDataSet = self._build_dataset(self.df_train)
        if verbose:
            logging.info("DATASET DE TREINO CRIADO!")
        self.test_dataset: TimeSeriesDataSet = self.build_test_dataset(
            self.train_dataset
        )
        if verbose:
            logging.info("DATASET DE TESTE CRIADO!")
        self.val_dataset: TimeSeriesDataSet = self.build_validation_dataset(
            self.train_dataset
        )
        if verbose:
            logging.info("DATASET DE VALIDAÇÂO CRIADO!")

    def load_dataset(self, verbose=False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Carrega os arquivos M4 em wide format e cachea em Parquet.
        """
        pqt_train: Path = self.PATH_FONTE_DADOS / f"{self.frequency}-train.parquet"
        pqt_test: Path = self.PATH_FONTE_DADOS / f"{self.frequency}-test.parquet"
        csv_train: Path = self.PATH_FONTE_DADOS / f"{self.frequency}-train.csv"
        csv_test: Path = self.PATH_FONTE_DADOS / f"{self.frequency}-test.csv"

        # Tentar carregar do cache Parquet
        if pqt_train.exists() and pqt_test.exists():
            if verbose:
                logging.info(f"Carregando cache: {pqt_train}")
            df_train: DataFrame = pd.read_parquet(pqt_train, engine="auto")
            df_test: DataFrame = pd.read_parquet(pqt_test, engine="auto")
        else:
            # Download remoto se necessário
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

            # Carregue dos CSVs
            df_train: DataFrame = pd.read_csv(csv_train, header="infer", index_col=0)
            df_test: DataFrame = pd.read_csv(csv_test, header="infer", index_col=0)

            # Cachear em Parquet
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

    def _wide_to_long(self, df: DataFrame) -> DataFrame:
        """Converte formato wide do M4 para formato longo do TimeSeriesDataSet."""
        frame: DataFrame = df.copy()
        frame.index = frame.index.astype(str)
        frame.index.name = "series_id"

        long_df: DataFrame = frame.reset_index().melt(
            id_vars="series_id", var_name="time_step", value_name="target"
        )
        long_df["target"] = pd.to_numeric(long_df["target"], errors="raise")
        long_df["time_idx"] = long_df["time_step"].str.extract(r"(\d+)")
        long_df["time_idx"] = long_df["time_idx"].astype(int) - 1

        return (
            long_df.dropna(subset=["target"])
            .sort_values(["series_id", "time_idx"])
            .reset_index(drop=True)[["series_id", "time_idx", "target"]]
        )

    def _build_dataset(self, data: DataFrame) -> TimeSeriesDataSet:

        # gera exeption se tem NaN

        if data.isnull().values.any():
            raise Exception(f"TEM NAN! ")

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
            add_relative_time_idx=False,  # nao adiciona time_idx como feature
            add_target_scales=False,  # nao adiciona mediana e escala como features
            add_encoder_length=True,  #
            allow_missing_timesteps=False,  # nao permite lacunas no indice temporal
            randomize_length=None,  # sem random
        )

    def build_train_dataset(self) -> TimeSeriesDataSet:
        return self._build_dataset(self.df_train)

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
        return TimeSeriesDataSet.from_dataset(
            train_dataset,
            self._evaluation_frame(),
            predict=True,
            stop_randomization=True,
        )

    def _evaluation_frame(self) -> DataFrame:
        """Concatena treino e teste em sequência temporal contínua."""
        frames: list[DataFrame] = []

        for series_id, train_group in self.df_train.groupby("series_id", sort=False):
            test_group: DataFrame = self.df_test[self.df_test["series_id"] == series_id]
            if test_group.empty:
                continue

            test_group: DataFrame = test_group.copy()
            start_idx: int = int(train_group["time_idx"].max()) + 1
            test_group["time_idx"] = range(start_idx, start_idx + len(test_group))
            frames.append(pd.concat([train_group, test_group], ignore_index=True))

        if not frames:
            raise ValueError("Nenhuma série encontrada para avaliação.")

        return (
            pd.concat(frames, ignore_index=True)
            .sort_values(["series_id", "time_idx"])
            .reset_index(drop=True)
        )
