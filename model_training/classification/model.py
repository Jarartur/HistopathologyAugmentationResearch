import torch 
from torch import nn

from pytorch_lightning import LightningModule
from pytorch_lightning.cli import instantiate_class
from torchmetrics import MetricCollection, Accuracy, Recall, F1Score
from torchmetrics.classification import MultilabelROC, MultilabelAUROC

from torchvision.models import resnet50, ResNet50_Weights

import matplotlib.pyplot as plt
from palettable.tableau import TableauMedium_10
import mpltex


import os

class BaseVisionSystem(LightningModule):
    def __init__(self, num_labels: int, num_step: int, class_weights: list, max_epochs: int,
                 optimizer_init: dict, lr_scheduler_init: dict):
        """ Define base vision classification system
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
        self.save_hyperparameters()

        # step 2. define model
        # self.backbone = torch.hub.load('hankyul2/EfficientNetV2-pytorch', **backbone_init)
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.fc = nn.Linear(self.backbone.fc.in_features, num_labels)
        self.backbone.fc = nn.Sequential() # quick and dirty hack to keep compatible with other models 
                                           # and have separate fc layer
                                           # and have lower lr for the rest of the model
        # self.fc = nn.Linear(self.backbone.out_channels, num_labels)

        # step 2.5 freeze the model
        self.backbone.requires_grad_(False)
        # self.backbone.layer4[-1].conv1.requires_grad_(True)
        # self.backbone.layer4[-1].bn1.requires_grad_(True)
        # self.backbone.layer4[-1].conv2.requires_grad_(True)
        # self.backbone.layer4[-1].bn2.requires_grad_(True)
        # self.backbone.layer4[-1].conv3.requires_grad_(True)
        # self.backbone.layer4[-1].bn3.requires_grad_(True)
        # self.backbone.layer4[-1].relu.requires_grad_(True)
        # Im not sure if needed but just in case not to break the grad flow
        self.backbone.fc.requires_grad_(True)
        self.fc.requires_grad_(True)
        # self.backbone.blocks[-1].requires_grad_(False)
        # self.fc.requires_grad_(True)

        print('-'*50)
        for name, param in self.backbone.named_parameters():
            print(name,param.requires_grad)
        print('-'*50)

        # step 3. define lr tools (optimizer, lr scheduler)
        self.optimizer_init_config = optimizer_init
        self.lr_scheduler_init_config = lr_scheduler_init

        class_weights = torch.tensor(self.hparams.class_weights, dtype=torch.float32)
        self.criterion = nn.BCEWithLogitsLoss(weight=class_weights)

        # step 4. define metric
        metrics = MetricCollection({
            'accuracy': Accuracy(task='multilabel', num_labels=num_labels), 
            'recall': Recall(task='multilabel', num_labels=num_labels),
            'f1': F1Score(task='multilabel', num_labels=num_labels),
            })
        self.train_metric = metrics.clone(prefix='train/')
        self.valid_metric = metrics.clone(prefix='valid/')
        self.test_metric = metrics.clone(prefix='test/')

    def forward(self, x):
        return self.fc(self.backbone(x))

    def on_test_start(self):
        self.roc = MultilabelROC(num_labels=self.hparams.num_labels, thresholds=None)
        self.auc = MultilabelAUROC(num_labels=self.hparams.num_labels, average='none')
        self.preds_accum = []
        self.targets_accum = []

    def on_test_end(self):
        preds = torch.cat(self.preds_accum)
        targets = torch.cat(self.targets_accum)
        fpr, tpr, thresholds = self.roc(preds, targets.int())
        auc = self.auc(preds, targets.int())
        save_path = self.logger.log_dir
        num_labels = self.hparams.num_labels

        print('-'*50)
        print(f'fpr: {fpr[0].shape}')
        print(f'tpr: {tpr[0].shape}')
        print('-'*50)

        labels = ['air', 'dust', 'tissue', 'ink', 'marker', 'focus']

        lowest_shape = 10_000_000
        for f in fpr:
            print(f.shape[0])
            if lowest_shape > f.shape[0]:
                lowest_shape = f.shape[0]
        
        fpr = [f[:lowest_shape] for f in fpr]
        tpr = [t[:lowest_shape] for t in tpr]

        fpr_mean = torch.stack(fpr).mean(dim=0)
        tpr_mean = torch.stack(tpr).mean(dim=0)
        auc_mean = torch.mean(auc)

        plot(fpr, tpr, save_path, num_labels, labels, auc, auc_mean, fpr_mean, tpr_mean)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        return self.shared_step(batch, self.train_metric, 'train', add_dataloader_idx=False)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self.shared_step(batch, self.valid_metric, 'valid', add_dataloader_idx=True)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self.shared_step(batch, self.test_metric, 'test', add_dataloader_idx=True)

    def shared_step(self, batch, metric, mode, add_dataloader_idx):
        x, y = batch
        loss, y_hat = self.compute_loss(x, y) if mode == 'train' else self.compute_loss_eval(x, y)
        self.log_dict({f'{mode}/loss': loss}, add_dataloader_idx=add_dataloader_idx)

        metrics = metric(y_hat, y)
        self.log_dict(metrics, add_dataloader_idx=add_dataloader_idx, prog_bar=True)

        if mode == 'valid':
            self.log("hp_metric", metrics['valid/f1'])

        if mode == 'test':
            self.preds_accum += [y_hat.detach().cpu()]
            self.targets_accum += [y.detach().cpu()]

        return loss

    def compute_loss(self, x, y):
        return self.compute_loss_eval(x, y)

    def compute_loss_eval(self, x, y):
        y_hat = self.fc(self.backbone(x))
        loss = self.criterion(y_hat, y)
        return loss, y_hat

    def configure_optimizers(self):
        optimizer = instantiate_class([
            {'params': self.backbone.parameters(), 'lr': self.optimizer_init_config['init_args']['lr'] * 0.1},
            {'params': self.fc.parameters()},
        ], self.optimizer_init_config)

        lr_scheduler = {
            'scheduler': instantiate_class(optimizer, self.update_and_get_lr_scheduler_config()),
            'interval': self.lr_scheduler_init_config['step']
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def update_and_get_lr_scheduler_config(self):
        if 'T_max' in self.lr_scheduler_init_config['init_args']:
            self.lr_scheduler_init_config['init_args']['T_max'] = self.hparams.num_step * self.hparams.max_epochs
        return self.lr_scheduler_init_config

@mpltex.acs_decorator
def plot(fpr, tpr, save_path, num_labels, labels, auc, auc_mean, fpr_mean, tpr_mean):

    fig, ax = plt.subplots(1, 1, figsize=(5.3, 3))
    linestyles = mpltex.linestyle_generator(colors = TableauMedium_10.mpl_colors)

    for i in range(num_labels):
            ax.plot(fpr[i], tpr[i], linewidth=1, label=f'$ {labels[i]}-{auc[i]:.3f} $', **next(linestyles), markevery=1500)

    ax.plot(fpr_mean, tpr_mean, linewidth=3, label=f'$ all\ classes-{auc_mean:.3f} $', markevery=1500)
    
    ax.set_xlabel("$ False\ Positive\ Rate $")
    ax.set_ylabel("$ True\ Positive\ Rate $")
    # plt.title("ROC")
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    # ax.legend()
    fig.tight_layout(pad=0.1)
    fig.savefig(os.path.join(save_path, "ROC.png"))