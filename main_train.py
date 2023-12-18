import pytorch_lightning as pl
from attr import evolve
from pytorch_lightning.loggers import TensorBoardLogger

from moco import SelfSupervisedMethod
from model_params import EigRegParams
from model_params import VICRegParams


def main():
    configs = {
        # "vicreg_320": VICRegParams(),
        "eigen_320_subset_0.7": evolve(EigRegParams(), eigen_subset=0.7),
        "eigen_320_subset_0.8": evolve(EigRegParams(), eigen_subset=0.8),
        "eigen_320_subset_0.9": evolve(EigRegParams(), eigen_subset=0.9)
    }
    for seed in range(1):
        for name, config in configs.items():
            method = SelfSupervisedMethod(config)
            logger = TensorBoardLogger("tb_logs", name=f"{name}_{seed}")

            trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=320, logger=logger)

            trainer.fit(method)


if __name__ == "__main__":
    main()
