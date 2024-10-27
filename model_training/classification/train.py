from pytorch_lightning import Trainer
from pytorch_lightning.tuner.tuning import Tuner

from config import config

from dataloader import HAA_DataModule
from model import BaseVisionSystem

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import os

def update_config(config, data):
    config['model']['num_labels'] = data.num_classes
    config['model']['num_step'] = data.num_step
    config['model']['class_weights'] = data.class_weights
    config['model']['max_epochs'] = config['trainer']['max_epochs']


if __name__ == '__main__':
    import sys
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    save_path = sys.argv[2]
    resume = sys.argv[3]

    config['data']['train_data_csv'] = sys.argv[1]
    config['trainer']['default_root_dir'] = save_path
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="hp_metric", save_last=True)
    
    config['trainer']['callbacks'] = [lr_monitor, checkpoint_callback]
    
    data = HAA_DataModule(**config['data'])
    data.prepare_data()
    update_config(config, data)
    model = BaseVisionSystem(**config['model'])
    trainer = Trainer(**config['trainer'])
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(model, datamodule=data)
    trainer.fit(model, datamodule=data, ckpt_path=resume)
    # trainer.test(datamodule=data)
