import pytorch_lightning as pl
from attr import evolve
from pytorch_lightning.loggers import TensorBoardLogger

from moco import SelfSupervisedMethod
from model_params import EigRegParams
from model_params import VICRegParams


def main():
    configs = {
        "eigen": evolve(EigRegParams(), eigen_subset=70)
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

            trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=320, logger=logger)

            trainer.fit(method)


if __name__ == "__main__":
    main()
