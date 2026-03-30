import logging
import sys
import warnings

import hydra
import torch
import torch.utils
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler, Subset
from pathlib import Path

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from vesselfm.seg.dataset import UnionDataset
from vesselfm.seg.utils.evaluation import Evaluator

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def load_pretrained_weights(model, ckpt_path, device):
    """
    Supports both:
    1) Lightning checkpoints containing "state_dict" with keys prefixed by "model."
    2) Raw model state_dict checkpoints (e.g. vesselFM_base.pt from Hugging Face)
    """
    ckpt_path = Path(ckpt_path)
    logger.info(f"Loading pretrained checkpoint from {ckpt_path}")
    # Always load checkpoint on CPU first to avoid invalid CUDA index issues
    # when users pass physical GPU ids instead of visible-device-relative ids.
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    model.load_state_dict(state_dict, strict=True)
    logger.info(f"Loaded {len(state_dict)} model tensors.")


@hydra.main(config_path="configs", config_name="finetune", version_base="1.3.2")
def main(cfg):
    seed_everything(cfg.seed, True)
    torch.set_float32_matmul_precision("medium")
    dataset_name = list(cfg.data.keys())[0]
    run_name = f'finetune_{cfg.num_shots}shot_{dataset_name}_' + cfg.run_name

    # init logger
    wnb_logger = WandbLogger(
        project=cfg.wandb_project,
        name=run_name,
        config=OmegaConf.to_container(cfg),
        offline=cfg.offline,
    )

    # callbacks
    lr_monitor = LearningRateMonitor()
    monitor_metric = "val_DiceMetric"
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.chkpt_folder + "/" + cfg.wandb_project + "/" + run_name,
        monitor=monitor_metric,
        save_top_k=1,
        mode="max",
        filename=f"{run_name}_" + "{step}_{" + monitor_metric + ":.2f}",
        auto_insert_metric_name=True,
        save_last=True
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = ":"
    checkpoint_callback.CHECKPOINT_NAME_LAST = run_name + "_last"

    # init trainer
    trainer = hydra.utils.instantiate(cfg.trainer.lightning_trainer)
    trainer_additional_kwargs = {
        "logger": wnb_logger,
        "callbacks": [lr_monitor, checkpoint_callback],
        "devices": cfg.devices
    }
    trainer = trainer(**trainer_additional_kwargs)

    # init dataloader
    train_dataset = UnionDataset(cfg.data, "train", finetune=True)
    train_dataset = Subset(train_dataset, range(cfg.num_shots))
    random_sampler = RandomSampler(train_dataset, replacement=True, num_samples=int(1e6))
    train_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=train_dataset, sampler=random_sampler)
    logger.info(f"Train dataset size mapped to {len(train_dataset)} samples")

    val_dataset = UnionDataset(cfg.data, "val", finetune=True)
    val_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=val_dataset, batch_size=1)
    logger.info(f"Val dataset size: {len(val_dataset)}")

    test_dataset = UnionDataset(cfg.data, "test", finetune=True)
    test_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=test_dataset, batch_size=1)
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # init model
    model = hydra.utils.instantiate(cfg.model)
    if cfg.path_to_chkpt is not None:
        load_pretrained_weights(model, cfg.path_to_chkpt, device=f"cuda:{cfg.devices[0]}")

    # init lightning module
    evaluator = Evaluator()
    lightning_module = hydra.utils.instantiate(cfg.trainer.lightning_module)(
        model=model, evaluator=evaluator, dataset_name=dataset_name
    )

    # train loop and eval
    wnb_logger.watch(model, log="all", log_freq=20)
    if cfg.num_shots == 0:
        trainer.test(lightning_module, test_loader) # eval on test set
    else:
        logger.info("Starting training")
        trainer.validate(lightning_module, val_loader)
        trainer.fit(lightning_module, train_loader, val_loader)
        logger.info("Finished training")
        trainer.test(lightning_module, test_loader, ckpt_path="best")


if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
    main()
