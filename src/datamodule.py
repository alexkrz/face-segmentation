import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]


class SegmentationDataset(Dataset):
    segformer_img_tfms = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )

    def __init__(
        self,
        root_dir: Path,
        img_tfm: Optional[transforms.Compose] = segformer_img_tfms,
        mask_tfm: Optional[str] = "custom",
    ):
        assert root_dir.exists()
        self.img_dir = root_dir / "images"
        self.mask_dir = root_dir / "masks"
        img_files = sorted(list(self.img_dir.glob("*.jpg")), key=lambda x: int(x.stem))
        img_files = [fp.name for fp in img_files]
        mask_files = sorted(list(self.mask_dir.glob("*.png")), key=lambda x: int(x.stem))
        mask_files = [fp.name for fp in mask_files]
        self.df = pd.DataFrame({"img_fp": img_files, "mask_fp": mask_files})
        with open(root_dir / "id2label.json") as f:
            id2label = json.load(f)
        self.id2label = id2label
        self.img_tfm = img_tfm
        self.mask_tfm = mask_tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        entry = self.df.loc[idx]
        img = Image.open(self.img_dir / entry["img_fp"])
        if self.img_tfm is not None:
            img = self.img_tfm(img)
        mask = Image.open(self.mask_dir / entry["mask_fp"])
        if self.mask_tfm is not None:
            mask = np.array(mask, dtype=np.int64)
            mask = torch.tensor(mask)
        return img, mask


class SegmentationDatamodule(LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 8,
        num_workers: int = 8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.save_hyperparameters()

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        dataset = SegmentationDataset(root_dir=Path(self.data_dir))
        self.train_set, self.val_set = random_split(
            dataset,
            [0.9, 0.1],
            # generator=torch.Generator().manual_seed(42),  # Should be deterministic when using seed_everything
        )

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
        return val_dataloader
