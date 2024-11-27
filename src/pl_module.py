import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from transformers import SegformerForSemanticSegmentation


class SegformerModule(LightningModule):
    def __init__(
        self,
        ckpt_dir: str = "./checkpoints/mit-b0/",
        lr: float = 1e-3,
    ):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(ckpt_dir)
        self.model.decode_head.train()
        self.save_hyperparameters()

    def forward(self, pixel_values, labels):
        output = self.model.forward(pixel_values, labels)
        return output

    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        output = self(imgs, masks)
        loss = output.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        output = self(imgs, masks)
        loss = output.loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
