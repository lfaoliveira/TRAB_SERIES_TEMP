from lightning.pytorch.core.datamodule import LightningDataModule
from src.data.dataset import StrokeDataset
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np


class StrokeDataModule(LightningDataModule):
    def __init__(self, BATCH_SIZE: int | None = None, WORKERS: int = 4):
        super().__init__()

        self.BATCH_SIZE = BATCH_SIZE
        self.WORKERS = WORKERS
        self.input_dims = None

    # preparacao dos dados
    def prepare_data(self):
        # NOTE: cuidado para nao carregar dados pesados demais na memoria (estoura memoria da GPU!!!)
        self.dataset = StrokeDataset()

    # setup for transformation and augmentation
    def setup(self, stage=None):
        # Obter dimensões de entrada
        data, label = self.dataset[0]
        self.input_dims = data.shape[0]

        # Criar TensorDatasets para treino e teste
        dataset_train = TensorDataset(
            self.dataset.data_train, self.dataset.labels_train
        )
        dataset_test = TensorDataset(self.dataset.data_test, self.dataset.labels_test)

        # Split treino em train (80%) e validação (20%)
        train_size = int(0.8 * len(dataset_train))
        val_size = len(dataset_train) - train_size
        self.m4_train, self.m4_val = random_split(dataset_train, [train_size, val_size])
        self.m4_test = dataset_test

    def train_dataloader(self, BATCH_SIZE: int | None = None):
        BATCH_SIZE = BATCH_SIZE if BATCH_SIZE else self.BATCH_SIZE

        train_loader = DataLoader(
            self.m4_train,
            batch_size=BATCH_SIZE,
            num_workers=self.WORKERS,
            persistent_workers=True,
        )
        return train_loader

    def val_dataloader(self, BATCH_SIZE: int | None = None):
        BATCH_SIZE = BATCH_SIZE if BATCH_SIZE else self.BATCH_SIZE
        val_loader = DataLoader(
            self.m4_val,
            batch_size=BATCH_SIZE,
            num_workers=self.WORKERS,
            persistent_workers=True,
        )
        return val_loader

    def test_dataloader(self, BATCH_SIZE: int | None = None):
        """Dataloader de teste"""
        BATCH_SIZE = BATCH_SIZE if BATCH_SIZE else self.BATCH_SIZE
        test_loader = DataLoader(
            self.m4_test,
            batch_size=BATCH_SIZE,
            num_workers=self.WORKERS,
            persistent_workers=True,
        )
        return test_loader
