import torch
from torch import nn

from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import instantiate_class
from torchmetrics import MetricCollection, Accuracy, Recall, F1Score
from torchmetrics.classification import MultilabelROC, MultilabelAUROC

from torchvision.models import resnet50, ResNet50_Weights

import matplotlib.pyplot as plt
# from palettable.tableau import TableauMedium_10
import mpltex
from pathlib import Path
import os
import pandas as pd

from libhaa.classification.config import CLASS_CONFIG
from libhaa.classification.dataloader import HAA_DataModule


def inference_folder(
    inference_folder: Path, save_path: Path, load_path: Path, device: str
):
    data_csv = list(inference_folder.glob("*_file_list.csv"))[0]
    config = CLASS_CONFIG
    config["data"]["test_data_csv"] = data_csv
    config["trainer"]["default_root_dir"] = save_path
    config["trainer"]["devices"] = device
    config["trainer"]["logger"] = False
    model = BaseVisionSystem.load_from_checkpoint(load_path)
    (save_path / inference_folder.name).mkdir(parents=True, exist_ok=False)
    model.set_save_path(save_path / inference_folder.name)
    data = HAA_DataModule(**config["data"])
    trainer = Trainer(**config["trainer"])
    trainer.test(model=model, datamodule=data)


class BaseVisionSystem(LightningModule):
    def __init__(
        self,
        num_labels: int,
        num_step: int,
        class_weights: list,
        max_epochs: int,
        optimizer_init: dict,
        lr_scheduler_init: dict,
    ):
        """Define base vision classification system
        backbone_init: dict,
        :arg
            backbone_init: feature extractor
            num_labels: number of class of dataset
            num_step: number of step
            gpus: gpus id
            max_epoch: max number of epoch
            optimizer_init: optimizer class path and init args
            lr_scheduler_init: learning rate scheduler class path and init args
        """
        super().__init__()
        self.save_path = None
        # step 2. define model
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.fc = nn.Linear(self.backbone.fc.in_features, num_labels)
        self.backbone.fc = (
            nn.Sequential()
        )  # quick and dirty hack to keep compatible with other models
        self.backbone.requires_grad_(False)

        self.backbone.fc.requires_grad_(True)
        self.fc.requires_grad_(True)

        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.criterion = nn.BCEWithLogitsLoss(weight=class_weights)

        self.preds = []

    def set_save_path(self, save_path):
        self.save_path = save_path

    def forward(self, x):
        return self.fc(self.backbone(x))

    def on_test_end(self):
        preds = pd.DataFrame(self.preds, columns=["name", "y", "y_hat"])
        preds.to_csv(os.path.join(self.save_path, "preds.csv"))

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        x, y, name = batch
        y_hat = self.forward(x)

        self.preds += [
            (name, y_item, y_hat_item)
            for name, y_item, y_hat_item in zip(name, y.tolist(), y_hat.tolist())
        ]

