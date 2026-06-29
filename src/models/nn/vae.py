import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import pytorch_lightning as pl


class VAE(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        lr: float = 1e-3,
    ):
        # TODO: refazer tudo
        super().__init__()
        self.save_hyperparameters()

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

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def training_step(self, batch, batch_idx):
        x, _ = batch  # Assuming batch is (x, y)
        x_recon, mu, logvar = self.forward(x)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction="sum")

        # KL Divergence: D_KL(q(z|x) || p(z)) onde p(z) = N(0, I)
        q = Normal(mu, logvar.mul(0.5).exp())
        p = Normal(0, 1)
        kl_loss = kl_divergence(q, p).sum()

        loss = recon_loss + kl_loss

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_recon_loss", recon_loss, prog_bar=False)
        self.log("train_kl_loss", kl_loss, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, mu, logvar = self.forward(x)

        recon_loss = F.mse_loss(x_recon, x, reduction="sum")
        q = Normal(mu, logvar.mul(0.5).exp())
        p = Normal(0, 1)
        kl_loss = kl_divergence(q, p).sum()
        loss = recon_loss + kl_loss

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
