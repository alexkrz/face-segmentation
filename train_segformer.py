import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

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

    logger = TensorBoardLogger("./logs_pl", name="segformer")

    trainer = Trainer(
        logger=logger,
        max_epochs=2,
        num_sanity_val_steps=0,
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


if __name__ == "__main__":
    main()
