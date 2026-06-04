from lightning.pytorch.core.datamodule import LightningDataModule
from src.data.dataset import StrokeDataset
from torch.utils.data import DataLoader


class StrokeDataModule(LightningDataModule):
    def __init__(self, BATCH_SIZE: int | None = None, WORKERS: int = 4):
        super().__init__()

        self.BATCH_SIZE = BATCH_SIZE
        self.WORKERS = WORKERS

    # preparacao dos dados
    def prepare_data(self):
        # Carrega os dados em memoria uma vez por processo.
        self.dataset = StrokeDataset()

    # setup for transformation and augmentation
    def setup(self, stage=None):
        self.m4_train = self.dataset.build_train_dataset()
        self.m4_val = self.dataset.build_validation_dataset()
        self.m4_test = self.dataset.build_test_dataset()

    def _resolve_batch_size(self, batch_size: int | None) -> int:
        resolved_batch_size = batch_size if batch_size is not None else self.BATCH_SIZE
        if resolved_batch_size is None:
            raise ValueError("BATCH_SIZE precisa ser informado no datamodule.")
        return resolved_batch_size

    def train_dataloader(self, BATCH_SIZE: int | None = None):
        BATCH_SIZE = self._resolve_batch_size(BATCH_SIZE)

        train_loader = DataLoader(
            self.m4_train,
            batch_size=BATCH_SIZE,
            num_workers=self.WORKERS,
            persistent_workers=True,
        )
        return train_loader

    def val_dataloader(self, BATCH_SIZE: int | None = None):
        BATCH_SIZE = self._resolve_batch_size(BATCH_SIZE)
        val_loader = DataLoader(
            self.m4_val,
            batch_size=BATCH_SIZE,
            num_workers=self.WORKERS,
            persistent_workers=True,
        )
        return val_loader

    def test_dataloader(self, BATCH_SIZE: int | None = None):
        """Dataloader de teste"""
        BATCH_SIZE = self._resolve_batch_size(BATCH_SIZE)
        test_loader = DataLoader(
            self.m4_test,
            batch_size=BATCH_SIZE,
            num_workers=self.WORKERS,
            persistent_workers=True,
        )
        return test_loader
