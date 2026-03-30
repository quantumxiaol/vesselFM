import logging
import sys
import warnings

import hydra
import torch
import torch.utils
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from pathlib import Path

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from vesselfm.seg.dataset import UnionDataset
from vesselfm.seg.utils.evaluation import PretrainEvaluationDataset, Evaluator

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def load_pretrained_weights(model, ckpt_path, device):
    """
    Supports both:
    1) Lightning checkpoints containing "state_dict" with keys prefixed by "model."
    2) Raw model state_dict checkpoints
    """
    ckpt_path = Path(ckpt_path)
    logger.info(f"Loading pretrained checkpoint from {ckpt_path}")
    # Always load checkpoint on CPU first to avoid invalid CUDA index issues.
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    model.load_state_dict(state_dict, strict=True)
    logger.info(f"Loaded {len(state_dict)} model tensors.")


@hydra.main(config_path="configs", config_name="train", version_base="1.3.2")
def main(cfg):
    seed_everything(cfg.seed, True)
    torch.set_float32_matmul_precision("medium")
    run_name = 'pretrain_' + cfg.run_name

    # init logger
    wnb_logger = WandbLogger(
        project=cfg.wandb_project,
        name=run_name,
        config=OmegaConf.to_container(cfg),
        offline=cfg.offline,
    )

    # callbacks
    lr_monitor = LearningRateMonitor()
    monitor_metric = "val_loss"
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.chkpt_folder + "/" + cfg.wandb_project + "/" + cfg.run_name,
        monitor=monitor_metric,
        save_top_k=3,
        mode="min",
        filename=(f"{run_name}_" + "{step}_{" + monitor_metric + ":.2f}"),
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
    train_dataset = UnionDataset(cfg.data, "train")
    train_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=train_dataset)
    logger.info(f"Train dataset size mapped to {len(train_dataset)} samples")
    
    val_dataset = PretrainEvaluationDataset(cfg.eval_path)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, pin_memory=True,
        persistent_workers=True, num_workers=1
    )
    logger.info(f"Val dataset size mapped to {len(val_dataset)} samples")

    # init model
    model = hydra.utils.instantiate(cfg.model)
    if cfg.path_to_chkpt is not None:
        load_pretrained_weights(model, cfg.path_to_chkpt, device=f"cuda:{cfg.devices[0]}")

    # init lightning module
    evaluator = Evaluator()
    lightning_module = hydra.utils.instantiate(cfg.trainer.lightning_module)(
        model=model, evaluator=evaluator
    )

    # train loop
    wnb_logger.watch(model, log="all", log_freq=200)
    logger.info("Starting training")
    trainer.validate(lightning_module, val_loader)
    trainer.fit(lightning_module, train_loader, val_loader)
    logger.info("Finished training")


if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
    main()
