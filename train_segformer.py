import json
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from accelerate import Accelerator
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    Trainer,
    TrainingArguments,
)


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

    def __getitem__(self, idx) -> dict:
        entry = self.df.loc[idx]
        img = Image.open(self.img_dir / entry["img_fp"])
        if self.img_tfm is not None:
            img = self.img_tfm(img)
        mask = Image.open(self.mask_dir / entry["mask_fp"])
        if self.mask_tfm is not None:
            mask = self.mask_tfm(mask)
        return {"pixel_values": img, "label": mask}


def training_function(cfg, training_args):
    torch_dataset = SegmentationDataset(
        root_dir=Path(cfg["data_dir"]),
        img_tfm=None,
        mask_tfm=None,
    )
    # print(len(dataset))
    # print(dataset.df.head(20))
    # img, mask = dataset[0]
    # print("img.shape:", img.shape)
    # print("mask.shape:", mask.shape)
    id2label = torch_dataset.id2label
    label2id = {v: k for k, v in id2label.items()}

    # Convert torch_dataset into hf_dataset
    def gen():
        for idx in range(len(torch_dataset)):
            yield torch_dataset[idx]  # this has to be a dictionary

    ds = HFDataset.from_generator(gen)

    # Display the Hugging Face Dataset
    print(ds)

    # Shuffle the dataset and split in train and test set
    ds = ds.shuffle(seed=cfg["seed"])
    ds = ds.train_test_split(test_size=0.1)
    train_ds = ds["train"]
    test_ds = ds["test"]

    # We use the SegformerImageProcessor for the datasets after converting the dataset to hf datasets
    processor = SegformerImageProcessor()
    jitter = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

    def train_transforms(example_batch):
        images = [jitter(x) for x in example_batch["pixel_values"]]
        labels = [x for x in example_batch["label"]]
        inputs = processor(images, labels)
        return inputs

    def val_transforms(example_batch):
        images = [x for x in example_batch["pixel_values"]]
        labels = [x for x in example_batch["label"]]
        inputs = processor(images, labels)
        return inputs

    # Set transforms
    train_ds.set_transform(train_transforms)
    test_ds.set_transform(val_transforms)

    # train_set, val_set = random_split(
    #     torch_dataset,
    #     [0.9, 0.1],
    #     generator=torch.Generator().manual_seed(cfg["seed"]),
    # )
    # train_dataloader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=8)
    # val_dataloader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=8)
    # batch = next(iter(train_dataloader))
    # imgs, masks = batch
    # print(imgs.shape)

    # Load model from hf hub and store it locally
    # model = SegformerForSemanticSegmentation.from_pretrained(
    #     "nvidia/mit-b0",
    #     id2label=id2label,
    #     label2id=label2id,
    # )
    # model.save_pretrained("./checkpoints/mit-b0")

    # Load local model
    model = SegformerForSemanticSegmentation.from_pretrained("./checkpoints/mit-b0/")
    # print(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        # compute_metrics=compute_metrics,
    )

    trainer.train()


def main(
    data_dir: str = os.environ["HOME"] + "/Data/FacialAttributes/celebamask_hq",
    seed: int = 42,
):
    cfg = locals()
    training_args = TrainingArguments(
        output_dir="logs/",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
    )
    training_function(cfg, training_args)


if __name__ == "__main__":
    main()
