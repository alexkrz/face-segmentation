import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from safetensors.torch import save_file

from src.datamodule import SegmentationDatamodule
from src.pl_module import SegformerModule


def main(
    data_dir: str = os.environ["HOME"] + "/Data/FacialAttributes/celebamask_hq",
    ckpt_dir: str = "./checkpoints/mit-b0/",
    seed: int = 42,
):
    pl.seed_everything(seed)

    datamodule = SegmentationDatamodule(data_dir)

    pl_module = SegformerModule.from_pretrained(ckpt_dir)
    model_config = pl_module.config
    # print(model_config)

    tb_logger = TensorBoardLogger("./logs_pl", name="mit-b0")

    cp_callback = ModelCheckpoint(
        monitor="val_mean_iou",
        filename="{epoch:d}-{val_mean_iou:.4f}",
        mode="max",
        save_weights_only=True,
    )

    trainer = Trainer(
        logger=[tb_logger],
        callbacks=[cp_callback],
        max_epochs=20,
        num_sanity_val_steps=0,
        limit_train_batches=0.1,
    )

    log_dir = Path(trainer.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Writing logs to:", str(log_dir))
    # Write config of pretrained model to .json file
    model_config.to_json_file(log_dir / "model_config.json")

    trainer.fit(
        model=pl_module,
        datamodule=datamodule,
    )

    # Convert last checkpoint to safetensors after training
    print("Best model:", cp_callback.best_model_path)
    ckpt = torch.load(cp_callback.best_model_path, weights_only=True)
    # print([key for key in ckpt["state_dict"].keys() if "decode_head" in key])

    # Remove "model." prefix from checkpoint keys
    state_dict = {key.replace("model.", ""): value for key, value in ckpt["state_dict"].items()}
    save_file(state_dict, log_dir / "best_model.safetensors")


if __name__ == "__main__":
    main()
