import os
from pathlib import Path

import jsonargparse
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from safetensors.torch import save_file

from src.datamodule import SegmentationDatamodule
from src.pl_module import SegformerModule


def main(parser: jsonargparse.ArgumentParser):
    cfg = parser.parse_args()
    pl.seed_everything(cfg.seed)

    datamodule = SegmentationDatamodule(**cfg.datamodule)

    pl_module = SegformerModule.from_pretrained(cfg.ckpt_dir)
    model_config = pl_module.config
    # print(model_config)

    # Modify logger
    cfg_logger = cfg.trainer.pop("logger")
    tb_logger = TensorBoardLogger("./logs_pl", name="mit-b0")

    # Modify checkpoint callback
    cfg_callbacks = cfg.trainer.pop("callbacks")
    cp_callback = ModelCheckpoint(
        monitor="val_mean_iou",
        filename="{epoch:d}-{val_mean_iou:.4f}",
        mode="max",
        save_weights_only=True,
    )

    trainer = Trainer(**cfg.trainer, logger=[tb_logger], callbacks=[cp_callback])

    log_dir = Path(trainer.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Writing logs to:", str(log_dir))
    # Write config of pretrained model to .json file
    model_config.to_json_file(log_dir / "model_config.json")
    # Also write experiment config
    parser.save(cfg, log_dir / "config.yaml")

    trainer.fit(model=pl_module, datamodule=datamodule)

    # Convert last checkpoint to safetensors after training
    print("Best model:", cp_callback.best_model_path)
    ckpt = torch.load(cp_callback.best_model_path, weights_only=True)
    # print([key for key in ckpt["state_dict"].keys() if "decode_head" in key])

    # Remove "model." prefix from checkpoint keys
    state_dict = {key.replace("model.", ""): value for key, value in ckpt["state_dict"].items()}
    save_file(state_dict, log_dir / "best_model.safetensors")


if __name__ == "__main__":
    from jsonargparse import ActionConfigFile, ArgumentParser

    parser = ArgumentParser(parser_mode="omegaconf")
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/mit-b0/")
    parser.add_class_arguments(
        SegmentationDatamodule,
        "datamodule",
        default={
            "data_dir": f"{os.environ['HOME']}/Data/FacialAttributes/celebamask_hq",
            "batch_size": 8,
        },
    )
    parser.add_class_arguments(
        Trainer,
        "trainer",
        default={
            "max_epochs": 2,
            "limit_train_batches": 0.1,
        },
    )

    main(parser)
