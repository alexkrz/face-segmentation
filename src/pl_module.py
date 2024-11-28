import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from transformers import SegformerConfig, SegformerForSemanticSegmentation


class SegformerModule(LightningModule):
    def __init__(
        self,
        config: SegformerConfig = SegformerConfig(),
        lr: float = 1e-3,
    ):
        super().__init__()
        self.model = SegformerForSemanticSegmentation(config)
        self.config = self.model.config
        self.save_hyperparameters(ignore=["config"])
        self.__configure_mode()

    @classmethod
    def from_pretrained(cls, ckpt_dir: str):
        pretrained_model = SegformerForSemanticSegmentation.from_pretrained(ckpt_dir)
        # Overwrite config
        config = pretrained_model.config
        pl_module = cls(config)
        pl_module.model = pretrained_model
        pl_module.__configure_mode()
        return pl_module

    def __configure_mode(self):
        self.model.decode_head.train()  # Set decode_head to train mode

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
