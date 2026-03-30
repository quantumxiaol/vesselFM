import logging
import os

import torch
import lightning
import numpy as np
from monai.inferers.inferer import SlidingWindowInfererAdapt


logger = logging.getLogger(__name__)


class PLModule(lightning.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss,
        optimizer_factory,
        prediction_threshold: float,
        scheduler_configs=None,
        evaluator=None
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer_factory = optimizer_factory
        self.scheduler_configs = scheduler_configs
        self.prediction_threshold = prediction_threshold
        self.rank = int(os.environ.get("LOCAL_RANK", 0))
        self.evaluator = evaluator

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(params=self.parameters())

        if self.scheduler_configs is not None:
            schedulers = []
            logger.info(f"Initializing schedulers: {self.scheduler_configs}")
            for scheduler_name, scheduler_config in self.scheduler_configs.items():
                if scheduler_config is None:
                    continue    # skip empty configs during finetuning

                logger.info(f"Initializing scheduler: {scheduler_name}")
                scheduler_config["scheduler"] = scheduler_config["scheduler"](optimizer=optimizer)
                scheduler_config = dict(scheduler_config)
                schedulers.append(scheduler_config)
            return [optimizer], schedulers
        return optimizer

    def training_step(self, batch, batch_idx):
        image, mask = batch
        pred_mask = self.model(image)
        loss = self.loss(pred_mask, mask)
        self.log("train_loss", loss, logger=(self.rank == 0), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, mask, name = batch
        pred_mask = self.model(image)
        loss = self.loss(pred_mask, mask)
        self.log("val_loss", loss, logger=(self.rank == 0), sync_dist=True)

        metrics = self.evaluator.estimate_metrics(
            pred_mask.sigmoid().squeeze(), mask.squeeze(), threshold=self.prediction_threshold
        )
        for metric, value in metrics.items():
            if isinstance(value, np.ndarray):
                value = float(value)
            self.log(f"val_{name[0]}_{metric}", value, logger=(self.rank == 0), sync_dist=True)


class PLModuleFinetune(PLModule):
    def __init__(
        self,
        dataset_name: str = None,
        input_size: tuple = None,
        batch_size: int = None,
        num_shots: int = None,
        *args,
        **kwargs
    ):
        # Remove dataset_name from kwargs
        self.dataset_name = dataset_name
        logger.info(f"Dataset name: {self.dataset_name}")
        super().__init__(*args, **kwargs)
        self.sliding_window_inferer = SlidingWindowInfererAdapt(
            roi_size=input_size, sw_batch_size=batch_size, overlap=0.5,
        )
        self.num_shots = num_shots

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        with torch.no_grad():
            pred_mask = self.sliding_window_inferer(image, self.model)
            loss = self.loss(pred_mask, mask)
            self.log(
                f"{self.dataset_name}_val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

            metrics = self.evaluator.estimate_metrics(
                pred_mask.sigmoid().squeeze(), mask.squeeze(), threshold=self.prediction_threshold, fast=True
            )
            for name, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.detach().float()
                elif isinstance(value, np.ndarray):
                    value = float(value)
                self.log(
                    f"{self.dataset_name}_val_{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                
                if name == "dice":
                    self.log("val_DiceMetric", value, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        image, mask = batch
        with torch.no_grad():
            pred_mask = self.sliding_window_inferer(image, self.model)
            loss = self.loss(pred_mask, mask)
            self.log(
                f"{self.dataset_name}_test_loss",
                loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

            metrics = self.evaluator.estimate_metrics(
                pred_mask.sigmoid().squeeze(), mask.squeeze(), threshold=self.prediction_threshold
            )
            for name, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.detach().float()
                elif isinstance(value, np.ndarray):
                    value = float(value)
                self.log(
                    f"{self.dataset_name}_test_{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        return loss
