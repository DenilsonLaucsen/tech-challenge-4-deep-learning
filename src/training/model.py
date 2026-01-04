# app/training/lightning_module.py
import pytorch_lightning as pl
import torch
import torch.nn.functional as F


class TimeSeriesLit(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        # example metric placeholders; você pode usar torchmetrics se quiser
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # ensure shape alignment
        if y_hat.dim() == 1 and y.dim() > 1:
            y_hat = y_hat.unsqueeze(-1)
        loss = F.mse_loss(y_hat, y)
        self.log("train/mse", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if y_hat.dim() == 1 and y.dim() > 1:
            y_hat = y_hat.unsqueeze(-1)
        val_loss = F.mse_loss(y_hat, y)
        self.log("val/mse", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
