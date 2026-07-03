import torch
from torch import nn
import torch.nn.functional as F
from lightning import LightningModule


class TCN(LightningModule):
    """
    1D Convolutional Autoencoder for anomaly detection.
    Reconstructs input windows of size ``input_dim``;
    reconstruction error (MSE) = anomaly score.
    """

    def __init__(
        self,
        input_dim: int = 20,
        latent_dim: int = 8,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        # AVISO: SUbstituir por modelo do Darts ou modelo da lib https://github.com/paul-krug/pytorch-tcn
        # Encoder:  1D convs + MLP bottleneck
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * input_dim, latent_dim),
        )

        # Decoder: MLP + transposed convs
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8 * input_dim),
            nn.ReLU(),
            nn.Unflatten(1, (8, input_dim)),
            nn.ConvTranspose1d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        z = self.encoder(x)  # (batch, latent_dim)
        recon = self.decoder(z)  # (batch, 1, input_dim)
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
