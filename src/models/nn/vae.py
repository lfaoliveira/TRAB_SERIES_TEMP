import torch
from torch import nn
import torch.nn.functional as F
from lightning import LightningModule
from src.models.outlier import build_test_metrics, build_validation_metrics


class VAE(LightningModule):
    """
    Variational Autoencoder trained as a LightningModule.

    The model learns to reconstruct normal patterns; anomaly detection is done
    by thresholding the reconstruction error (MSE) at inference time.
    """

    # AVISO: VAE PRECISA APENAS DOS DADOS NORMAIS PARA APRENDER
    # NOTE: usado para comparar com TCN mais simples

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr

        # Encoder

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LazyLinear(hidden_dim // 2),
            nn.ReLU(),
        )

        # Latent space projections
        self.fc_mu = nn.Sequential(nn.LazyLinear(latent_dim), nn.ReLU(), nn.LazyLinear(latent_dim // 2))
        self.fc_logvar = nn.Sequential(nn.LazyLinear(latent_dim), nn.ReLU(), nn.LazyLinear(latent_dim // 2))

        # Decoder
        self.decoder = nn.Sequential(
            nn.LazyLinear(hidden_dim // 2),
            nn.ReLU(),
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.LazyLinear(input_dim),
        )

        self.val_metrics = build_validation_metrics()
        self.test_metrics = build_test_metrics()

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
        # VAE tenta estimar log da variancia, que é usado diretamente no formula,
        # Calculo normal da divergencia KL seria instavel aqui
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return kl

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, _ = batch
        # IGUAL FORWARD!
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)

        recon_loss = F.mse_loss(x_recon, x, reduction="sum")

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
        self.log("val_recon_loss", recon_loss, prog_bar=False)
        self.log("val_kl_loss", kl, prog_bar=False)

        self.val_metrics.update(x_recon, x)
        return loss

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)
        self.val_metrics.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
