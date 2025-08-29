import io
import json
import os
import shutil
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm


def download_dataset(
    parquet_dir: str = os.environ["HOME"] + "/Data/FacialAttributes/celebamask_hq_hf",
):
    # Load dataset from huggingface
    # Link: https://huggingface.co/datasets/v-xchen-v/celebamask_hq
    hf_dataset_identifier = "v-xchen-v/celebamask_hq"
    ds = datasets.load_dataset(hf_dataset_identifier)
    print(ds)

    # Load and store id2label
    id2label = json.load(
        open(
            hf_hub_download(
                repo_id=hf_dataset_identifier,
                filename="id2label.json",
                repo_type="dataset",
            ),
        )
    )
    with open(parquet_dir / "id2label.json", "w") as f:
        json.dump(id2label, f, indent=4)

    # Store dataset in custom folder
    parquet_dir = Path(parquet_dir)  # type: Path
    parquet_dir.mkdir(exist_ok=True)
    ds["train"].to_parquet(parquet_dir / "celebamaskhq.parquet")


def convert_parquet(
    root_dir: str = os.environ["HOME"] + "/Data/FacialAttributes",
    parquet_fp: str = "celebamask_hq_hf/celebamaskhq.parquet",
    out_folder: str = "celebamask_hq",
):
    # Copy json file
    root_dir = Path(root_dir)  # type: Path
    parquet_fp = root_dir / parquet_fp  # type: Path
    out_dir = root_dir / out_folder
    out_dir.mkdir(exist_ok=True)
    shutil.copy(
        parquet_fp.parent / "id2label.json",
        out_dir / "id2label.json",
    )

    df = pd.read_parquet(parquet_fp)
    # print(df)
    image_dir = out_dir / "images"
    image_dir.mkdir()
    mask_dir = out_dir / "masks"
    mask_dir.mkdir()
    for idx in tqdm(range(len(df))):
        image = Image.open(io.BytesIO(df.loc[idx]["image"]["bytes"]))
        label = Image.open(io.BytesIO(df.loc[idx]["label"]["bytes"]))

        image.save(image_dir / f"{idx:05}.jpg")
        label.save(mask_dir / f"{idx:05}.png")


if __name__ == "__main__":
    download_dataset()
    convert_parquet()
