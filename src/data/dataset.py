from enum import StrEnum
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pandera.pandas as pa
from pandera.typing import DataFrame, Series, Index
from sklearn.preprocessing import StandardScaler
from torch.types import Tensor
from kagglehub import KaggleDatasetAdapter, dataset_download, dataset_load
from torch import from_numpy


class CATEGORICAL_COLUMNS(StrEnum):
    GENDER = "gender"
    MARRIED = "ever_married"
    WORK = "work_type"
    RESIDENCE = "Residence_type"
    SMOKE_STATUS = "smoking_status"


class MySchema(pa.DataFrameModel):
    id: Index[int]
    age: Series[float]
    gender: Series[str]
    ever_married: Series[str]
    work_type: Series[str]
    Residence_type: Series[str]
    smoking_status: Series[str]
    hypertension: Series[int]
    heart_disease: Series[int]
    avg_glucose_level: Series[float]
    bmi: Series[float]
    stroke: Series[int]
    # pred: Optional[Series[int]]
    # error: Optional[Series[str]]


class StrokeDataset(Dataset):
    original_df: DataFrame[MySchema]
    dataframe: DataFrame[MySchema]
    data: Tensor
    labels: Tensor
    LABELS_COLUMN: str

    def __init__(self) -> None:
        super().__init__()

        self.read_df()
        STR_COL = list(CATEGORICAL_COLUMNS)
        self.LABELS_COLUMN = "stroke"
        self.data_prep(STR_COL)

    def __getitem__(self, index: Tensor | list[int] | int):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    def read_df(self):
        local_filename = "stroke.csv"
        if not os.path.exists(local_filename):
            dataset_name = "fedesoriano/stroke-prediction-dataset"
            dataset_download(dataset_name)
            # Set the path to the file you'd like to load
            dataset_path = "healthcare-dataset-stroke-data.csv"

            df = dataset_load(
                KaggleDatasetAdapter.PANDAS,
                dataset_name,
                dataset_path,
            )
            df.to_csv(local_filename, sep=",")
        else:
            df = pd.read_csv(local_filename)

        # remove null values to avoid problems
        df = (
            df.dropna()
            .set_index("id")
            .drop(columns=["Unnamed: 0"], errors="ignore")
            .sort_index()
        )
        # validate schema
        self.dataframe = MySchema.validate(df)
        self.original_df = MySchema.validate(df)

    def data_prep(self, bad_columns: list[CATEGORICAL_COLUMNS]) -> None:
        """
        function for data normalization

        :param self: Description
        :param bad_columns: columns to be normalized (transforms categorical columns into normalized numeric values)
        :type bad_columns: list[CATEGORICAL_COLUMNS]
        """
        STR_COL = bad_columns

        # iterate over the column set and assign categorical number for each categorical column
        for col in STR_COL:
            self.dataframe[col] = self.dataframe[f"{col}"].astype("category")
            self.dataframe[f"{col}_code"] = self.dataframe[f"{col}"].cat.codes

        self.dataframe = self.dataframe.drop(columns=STR_COL)
        # labels before normalization (doesnt need to be normalized)
        self.labels = from_numpy(
            self.dataframe.loc[:, self.LABELS_COLUMN].values
        ).float()
        self.dataframe = self.dataframe.drop(columns=self.LABELS_COLUMN)

        # standard scaler to normalize dataframe to mean 0 and standard deviation 1
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(self.dataframe)
        self.dataframe = DataFrame(
            scaled_values, columns=self.dataframe.columns, index=self.dataframe.index
        )
        self.data = from_numpy(self.dataframe.values).float()

        print("\n")
        print(f"DATASET:\n{self.dataframe.head()}\n")


class ErrorModelDataset(Dataset):
    original_df: DataFrame[MySchema]
    dataframe: DataFrame[MySchema]
    data: np.typing.NDArray
    labels: np.typing.NDArray

    def __init__(self, predictions_df: pd.DataFrame) -> None:
        super().__init__()

        self.OBJECTIVE_COL = "error"
        self.LABELS_COLUMN = "stroke"
        self.clean_df(predictions_df)
        STR_COL = list(CATEGORICAL_COLUMNS)
        self.data_prep(STR_COL)

    def __getitem__(self, index: Tensor | list[int] | int):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    def clean_df(self, df: pd.DataFrame):

        # validate schema
        self.dataframe = MySchema.validate(df)
        self.original_df = MySchema.validate(df)

        if self.OBJECTIVE_COL not in self.original_df.columns:
            raise ValueError("SEM COLUNA OBJETIVO")

    def data_prep(self, bad_columns: list) -> None:
        """
        Optimized for Random Forest: Handles categorical encoding
        without unnecessary scaling.
        """
        # 1. Convert categorical columns to numeric codes
        # We use OrdinalEncoder or .cat.codes as RF handles integers well
        for col in bad_columns:
            self.dataframe[col] = self.dataframe[col].astype("category")
            self.dataframe[f"{col}_code"] = self.dataframe[col].cat.codes

        # 2. Extract Labels before dropping columns
        # Ensure labels are numeric (LabelEncode them if they are strings)
        self.labels = np.asarray(
            self.dataframe[self.OBJECTIVE_COL].values, dtype=np.dtype("U2")
        )

        # 3. Clean up the dataframe
        # Drop original string columns and the objective column
        drop_list = bad_columns + [self.OBJECTIVE_COL]
        # Note: Adding a check to avoid errors if LABELS_COLUMN is already in drop_list
        if (
            hasattr(self, "LABELS_COLUMN")
            and self.LABELS_COLUMN in self.dataframe.columns
        ):
            # drop_list.append(self.LABELS_COLUMN)
            pass

        self.dataframe = self.dataframe.drop(columns=drop_list)

        # 4. Prepare final data
        # For RF, we usually keep it as a NumPy array or Pandas DataFrame
        self.data = self.dataframe.values

        print("\n")
        print(f"RANDOM FOREST READY DATASET:\n{self.dataframe.head()}\n")
