import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics.segmentation import MeanIoU
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
        self.num_labels = self.model.config.num_labels
        self.mean_iou = MeanIoU(num_classes=self.num_labels)
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
        self.log("val_loss", loss, prog_bar=False)

        logits = output.logits
        # Upscale logits to mask size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=masks.shape[-2::],
            mode="bilinear",
            align_corners=False,  # H x W
        )
        preds = upsampled_logits.argmax(dim=1)
        mean_iou = self.mean_iou(preds, masks)
        self.log("val_mean_iou", mean_iou, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
