import json
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        # TODO: Add normalization to img_tfm
        img_tfm: transforms.Compose = transforms.Compose([transforms.ToTensor()]),
        mask_tfm: transforms.Compose = transforms.Compose([transforms.ToTensor()]),
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
            mask = self.mask_tfm(mask)
        return img, mask


def training_function(cfg):
    dataset = SegmentationDataset(
        root_dir=Path(cfg["data_dir"]),
        # img_tfm=None,
        # mask_tfm=None,
    )
    print(len(dataset))
    # print(dataset.df.head(20))
    img, mask = dataset[0]
    print("img.shape:", img.shape)
    print("mask.shape:", mask.shape)


def main(
    data_dir: str = os.environ["HOME"] + "/Data/FacialAttributes/celebamask_hq",
):
    cfg = locals()
    training_function(cfg)


if __name__ == "__main__":
    main()
