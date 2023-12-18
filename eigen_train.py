import pytorch_lightning as pl
from attr import evolve
from pytorch_lightning.loggers import TensorBoardLogger

from moco import SelfSupervisedMethod
from model_params import EigRegParams


def main():
    base_config = EigRegParams()
    configs = {
        # "eigen_tester": evolve(EigRegParams(), eigen_subset=1.0, eigen_loss_weight=0.001)
        "eigen_320": evolve(EigRegParams(), eigen_subset=0.7)
        # "beta_1.1": evolve(base_config, beta=1.1),
        # "beta_5": evolve(base_config, beta=5),
        # "beta_2": evolve(base_config, beta=2)
        # "unif_penalty_0.1" : base_config,
        # "unif_penalty_0.01" : evolve(base_config, eigen_uniform=0.01),
        # "unif_penalty_0.001" : evolve(base_config, eigen_uniform=0.001),
        # "no_unif_penalty" : evolve(base_config, eigen_uniform=0)
        # "eigval_logger" : evolve(base_config, eigen_subset=70)
        # "1": evolve(base_config, eigen_loss_weight=1),
        # "eigen_sum_weight_0.001": base_config,
        # "eigen_sum_weight_0.01": evolve(base_config, eigen_loss_weight=0.01),
        # "0.1": evolve(base_config, eigen_loss_weight=0.1),
        # "0.01": evolve(base_config, eigen_loss_weight=0.01),
        # "0.001": base_config,
        # "0.0001": evolve(base_config, eigen_loss_weight=0.0001),
        # "0.00001": evolve(base_config, eigen_loss_weight=0.00001),
        # "subset_100": base_config,
        # "subset_sum_90_drop_most": evolve(base_config, eigen_subset=0.9),
        # "subset_sum_80_drop_most": evolve(base_config, eigen_subset=0.8),
        # "subset_sum_70_drop_most": evolve(base_config, eigen_subset=0.7),
        # "subset_sum_60_drop_most": evolve(base_config, eigen_subset=0.6),
        # "subset_50_drop_most": evolve(base_config, eigen_subset=0.5),
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

            trainer.fit(method, ckpt_path="tb_logs/eigen_320_0/version_0/checkpoints/epoch=232-step=95530.ckpt")


if __name__ == "__main__":
    main()
