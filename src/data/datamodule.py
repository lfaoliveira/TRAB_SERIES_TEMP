from lightning.pytorch.core.datamodule import LightningDataModule
from DataProcesser.dataset import StrokeDataset
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import numpy as np


class StrokeDataModule(LightningDataModule):
    def __init__(self, BATCH_SIZE: int, WORKERS: int):
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
        DATA_SPLIT = [0.7, 0.3]

        data, label = self.dataset[0]
        self.input_dims = data.shape[0]
        self.stroke_train, self.stroke_val = random_split(self.dataset, DATA_SPLIT)

        train_labels = np.array([self.dataset[i][1] for i in self.stroke_train.indices])
        class_counts = np.bincount(train_labels.astype(int))
        n_classes = len(class_counts)
        total_samples = len(train_labels)
        class_weights = total_samples / (n_classes * class_counts)
        self.class_weights = (
            class_weights.tolist()
        )  # shape [n_classes] – for loss weighting
        # self.sample_weights = class_weights[
        #     train_labels.astype(int)
        # ].tolist()  # shape [n_samples] – for sampler

    def train_dataloader(self, BATCH_SIZE: int | None = None):
        BATCH_SIZE = BATCH_SIZE if BATCH_SIZE else self.BATCH_SIZE

        # -------disabling class weights on sampling to test in the loss
        self.sample_weights = [1.0, 1.0]
        train_sampler = WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            self.stroke_train,
            sampler=train_sampler,
            batch_size=BATCH_SIZE,
            num_workers=self.WORKERS,
            persistent_workers=True,
        )
        return train_loader

    def val_dataloader(self, BATCH_SIZE: int | None = None):
        BATCH_SIZE = BATCH_SIZE if BATCH_SIZE else self.BATCH_SIZE
        val_loader = DataLoader(
            self.stroke_val,
            batch_size=BATCH_SIZE,
            num_workers=self.WORKERS,
            persistent_workers=True,
        )
        return val_loader

    def test_dataloader(self, BATCH_SIZE: int | None = None):
        """Dataloader de teste"""
        # use passed argument, else use class value
        BATCH_SIZE = BATCH_SIZE if BATCH_SIZE else self.BATCH_SIZE
        test_loader = DataLoader(
            self.stroke_val,
            batch_size=BATCH_SIZE,
            num_workers=self.WORKERS,
            persistent_workers=True,
        )
        return test_loader, self.stroke_val
