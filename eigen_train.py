import pytorch_lightning as pl
from attr import evolve
from pytorch_lightning.loggers import TensorBoardLogger

from moco import SelfSupervisedMethod
from model_params import EigRegParams


def main():
    base_config = EigRegParams()
    configs = {
        # "eigval_logger" : evolve(base_config, eigen_subset=70)
        # "10": evolve(base_config, eigen_loss_weight=10),
        # "1": evolve(base_config, eigen_loss_weight=1),
        # "0.1": evolve(base_config, eigen_loss_weight=0.1),
        # "0.01": evolve(base_config, eigen_loss_weight=0.01),
        # "0.001": base_config,
        # "0.0001": evolve(base_config, eigen_loss_weight=0.0001),
        # "0.00001": evolve(base_config, eigen_loss_weight=0.00001),
        "subset_100": base_config,
        "subset_90": evolve(base_config, eigen_subset=0.9),
        "subset_80": evolve(base_config, eigen_subset=0.8),
        "subset_70": evolve(base_config, eigen_subset=0.7),
        "subset_60": evolve(base_config, eigen_subset=0.6),
        "subset_50": evolve(base_config, eigen_subset=0.5),
        # "pred_only": evolve(base_config, mlp_normalization=None, prediction_mlp_normalization="bn"),
        # "proj_only": evolve(base_config, mlp_normalization="bn", prediction_mlp_normalization=None),
        # "no_norm": evolve(base_config, mlp_normalization=None),
        # "layer_norm": evolve(base_config, mlp_normalization="ln"),
        # "xent": evolve(
        #     base_config, use_negative_examples_from_queue=True, loss_type="ce", mlp_normalization=None, lr=0.02
        # ),
    }
    for seed in range(1):
        for name, config in configs.items():
            method = SelfSupervisedMethod(config)
            logger = TensorBoardLogger("tb_logs", name=f"{name}_{seed}")

            trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=100, logger=logger)

            trainer.fit(method)


if __name__ == "__main__":
    main()
