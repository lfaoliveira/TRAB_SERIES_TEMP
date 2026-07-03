import logging

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import KLDivLoss
from torch.distributions import Normal, kl_divergence
from lightning import LightningModule


class VAE(LightningModule):
    """
    Variational Autoencoder trained as a LightningModule.

    The model learns to reconstruct normal patterns; anomaly detection is done
    by thresholding the reconstruction error (MSE) at inference time.
    """
    #AVISO: VAE PRECISA APENAS DOS DADOS NORMAIS PARA APRENDER

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Latent space projections
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def kl_formula(self, logvar: torch.Tensor, mu: torch.Tensor):
        # $$\text{KL} = -0.5 \sum (1 + \log(\sigma^2) - \mu^2 - \sigma^2 )$$
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return kl

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, _ = batch
        # IGUAL FORWARD!
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)

        recon_loss = F.mse_loss(x_recon, x, reduction="sum")
        # logging.info(f"RECON LOSS: {recon_loss}\n")
        # logging.info(f"MU: {mu}\n")
        # logging.info(f"LOGVAR: {logvar.mul(0.5).exp()}\n")

        # AVISO calculo direto da Divergencia KL PRA EVITAR INSTABILIDADE NUMERICA!
        kl = self.kl_formula(logvar, mu)
        loss = recon_loss + kl

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_recon_loss", recon_loss, prog_bar=False)
        self.log("train_kl_loss", kl, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)

        recon_loss = F.mse_loss(x_recon, x, reduction="sum")
        kl = self.kl_formula(logvar, mu)
        loss = recon_loss + kl

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
