import logging
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F
from lightning import LightningModule
from pytorch_tcn import TCN

from src.pipelines.metrics import CentralMetricsStore, build_test_metrics, build_validation_metrics


class TCN_train(LightningModule):
    """
    1D Convolutional Autoencoder for anomaly detection.
    Reconstructs input windows of size ``input_dim``;
    reconstruction error (MSE) = anomaly score.
    """

    def __init__(
        self,
        window_size: int,
        num_channels: list[int] = [],
        use_norm: Literal["weight_norm", "batch_norm"] = "weight_norm",
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        if not num_channels:
            num_channels = [1, 64, 64, 128, 256, 512]
        HORIZON = window_size

        # embed_dim = window_size
        out_channels_tcn = num_channels[-1]

        self.tcn = TCN(
            num_inputs=1,  # recebe apenas 1 canal
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
            output_projection=out_channels_tcn,  # COM ATTENTION, NÃO PRECISA REPROJETAR PRO INPUT
            output_activation=None,  # APENAS RECONSTRUCAO, SEM CLASSIFICACAO DIRETA
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=out_channels_tcn, num_heads=4, batch_first=True
        )  # formato esperado: (batch, seq_len ou window_size, embed_dim)

        self.attn_norm = nn.LayerNorm(out_channels_tcn)

        # --- Cabeça de previsão ---
        self.head = nn.Linear(out_channels_tcn, HORIZON)

        self.val_metrics = build_validation_metrics()
        self.test_metrics = build_test_metrics()
        self.test_recon_metrics = build_validation_metrics()

        self.heartbeat = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window)
        x = x.unsqueeze(1)  # (batch, 1, window)
        tcn_out: torch.Tensor = self.tcn(x)  # retorno: (batch, num_channels[-1], window)
        tcn_out = tcn_out.transpose(1, 2)  # (batch, window, channel)
        seq_len = tcn_out.size(1)
        # Criar na CPU primeiro e mover usando .to() evita deadlocks de sincronização em containers instáveis
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1).to(x.device)

        attn_out: torch.Tensor | None
        attn_weights: torch.Tensor | None

        attn_out, attn_weights = self.attn(
            query=tcn_out,
            key=tcn_out,
            value=tcn_out,
            attn_mask=causal_mask,
            need_weights=True,  # útil pra visualizar depois quais timesteps importaram
        )
        assert attn_out is not None and attn_weights is not None

        # Conexão residual + normalização (padrão tipo Transformer)
        out = self.attn_norm(tcn_out + attn_out)

        # Usa o último timestep para prever o horizonte futuro
        last_step = out[:, -1, :]  # (batch, d_model)
        recon = self.head(last_step)  # (batch, forecast_horizon)

        return recon  # (batch, window)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        recon = self(x)
        loss = F.mse_loss(recon, x)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        recon = self(x)
        loss = F.mse_loss(recon, x)

        # tenta reconstruir 'x'
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.val_metrics.update(recon, x)

        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        recon = self(x)
        loss = F.mse_loss(recon, x)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.test_recon_metrics.update(recon, x)
        return loss

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)
        CentralMetricsStore.add(self.__class__.__name__, "validation", metrics)
        self.val_metrics.reset()

        self.heartbeat += 1

    def on_test_epoch_end(self):
        metrics = self.test_recon_metrics.compute()
        self.log_dict(metrics)
        CentralMetricsStore.add(self.__class__.__name__, "test", metrics)
        self.test_recon_metrics.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
