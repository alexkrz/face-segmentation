import os

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

    pl_module = SegformerModule(ckpt_dir)

    logger = TensorBoardLogger("./logs_pl", name="segformer")

    trainer = Trainer(
        logger=logger,
        max_epochs=2,
        num_sanity_val_steps=0,
    )

    trainer.fit(
        model=pl_module,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    main()
