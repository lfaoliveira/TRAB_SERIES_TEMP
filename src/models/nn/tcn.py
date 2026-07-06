from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F
from lightning import LightningModule
from pytorch_tcn import TCN


class TCN_train(LightningModule):
    """
    1D Convolutional Autoencoder for anomaly detection.
    Reconstructs input windows of size ``input_dim``;
    reconstruction error (MSE) = anomaly score.
    """

    def __init__(
        self,
        input_dim: int = 20,
        num_channels: list[int] = [],
        use_norm: Literal["weight_norm", "batch_norm"] = "weight_norm",
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        if not num_channels:
            num_channels = [1, 64, 64, 128, 256, 512]

        self.model = TCN(
            num_inputs=input_dim,
            num_channels=num_channels,
            kernel_size=5,
            dilations=None,  # DILATACAO PADRAO exponencial
            dilation_reset=32,  # reseta dilatacao
            dropout=0.0,
            causal=True,  # ignora informacoes futuras
            use_norm=use_norm,
            kernel_initializer="xavier_uniform",
            use_skip_connections=True,
            input_shape="NCL",  # exige input (batch size, number of input channels, sequence length)
            embedding_shapes=None,  # SEM EMBEDDINGS, PRA SIMPLIFICAR
            output_projection=None,  # PROJETA DE VOLTA PRA DIM DO INPUT
            output_activation=None,  # APENAS RECONSTRUCAO, SEM CLASSIFICACAO DIRETA
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        recon = self.model(x)  # (batch, latent_dim)
        return recon.squeeze(1)  # (batch, input_dim)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, _ = batch
        recon = self(x)
        loss = F.mse_loss(recon, x)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, _ = batch
        recon = self(x)
        loss = F.mse_loss(recon, x)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
