import torch
from torch import nn
from lightning import LightningModule


class TCN(LightningModule):
    """
    TCN forecasting model wrapped as a LightningModule.

    The underlying ``TCNModel`` (darts) is trained externally via its own
    ``.fit()`` method. This wrapper stores it so it can be composed inside
    a Lightning pipeline if needed.

    Subclasses may override ``forward``, ``training_step``, etc.
    """

    def __init__(
        self,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        kernel_size: int = 3,
        num_filters: int = 6,
        num_layers: int | None = None,
        dropout: float = 0.0,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr

        # The actual model is constructed and trained externally
        self.model: nn.Module = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        raise NotImplementedError(
            "TCN is trained via darts' TCNModel.fit(), not Lightning. "
            "Override training_step to use Lightning."
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
