import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class CVAE(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 784,
        num_classes: int = 10,
        latent_dim: int = 20,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder_fc1 = nn.Linear(input_dim + num_classes, 512)
        self.encoder_fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder_fc1 = nn.Linear(latent_dim + num_classes, 256)
        self.decoder_fc2 = nn.Linear(256, 512)
        self.decoder_out = nn.Linear(512, input_dim)

        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

    def encode(self, x: torch.Tensor, labels_one_hot: torch.Tensor):
        x = torch.cat([x, labels_one_hot], dim=1)
        h = self.activation(self.encoder_fc1(x))
        h = self.activation(self.encoder_fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, labels_one_hot: torch.Tensor):
        z = torch.cat([z, labels_one_hot], dim=1)
        h = self.activation(self.decoder_fc1(z))
        h = self.activation(self.decoder_fc2(h))
        return self.output_activation(self.decoder_out(h))

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        x = x.view(x.size(0), -1)
        labels_one_hot = F.one_hot(
            labels, num_classes=self.hparams.num_classes
        ).float()

        mu, logvar = self.encode(x, labels_one_hot)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, labels_one_hot)

        return x_recon, mu, logvar

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ):
        x = x.view(-1, self.hparams.input_dim)
        bce = F.binary_cross_entropy(recon_x, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + kld

    def training_step(self, batch, batch_idx):
        x, y = batch
        recon_x, mu, logvar = self(x, y)
        loss = self.loss_function(recon_x, x, mu, logvar)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        recon_x, mu, logvar = self(x, y)
        loss = self.loss_function(recon_x, x, mu, logvar)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)