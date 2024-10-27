from typing import Tuple, Type, Any

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from torchvision.models import ResNet50_Weights

import pandas as pd
from PIL import Image

class HAA_Dataset(Dataset):
    def __init__(self, root: str, transform: Type[Any]):
        super().__init__()
        self.root = root
        self.transform = transform

        self.data = pd.read_csv(root)
        self.targets = []

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        file_path, *labels = self.data.iloc[idx, :].to_list()
        img = Image.open(file_path)

        img = self.transform(img)
        labels = torch.tensor(labels, dtype=torch.float32)

        return img, labels

class HAA_DataModule(LightningDataModule):
    def __init__(self,
                 dataset_name: str,
                 img_size: Tuple[int, int] = (224, 224),
                 batch_size: int = 2,
                 num_workers: int = 4,
                 train_data_csv: str = 'data.csv',
                 val_data_csv: str = 'data.csv',
                 test_data_csv: str = 'data.csv',):
        """
        Base Data Module
        :arg
            Dataset: Enter Dataset
            batch_size: Enter batch size
            num_workers: Enter number of workers
            size: Enter resized image
            data_root: Enter root data folder name
            valid_ratio: Enter valid dataset ratio
        """
        super().__init__()
        self.save_hyperparameters()
        self.dataset = HAA_Dataset
        self.train_transform, self.test_transform = self.get_transforms()

    def prepare_data(self) -> None:
        data = pd.read_csv(self.hparams.train_data_csv)
        file_path, *labels = data.iloc[0, :].to_list()
        num_samples = data.shape[0]
        self.num_classes = len(labels)
        self.num_step = num_samples // self.hparams.batch_size

        self.class_quant = []
        num_samples_nonempty = 0
        for i in range(self.num_classes):
            num = data.iloc[:, i+1].sum()
            self.class_quant += [num]
            num_samples_nonempty += num
        self.class_weights = [1-(quant/num_samples_nonempty) for quant in self.class_quant]

        print('-' * 50)
        print('* {} dataset class num: {}'.format(self.hparams.dataset_name, self.num_classes))
        print('* {} dataset class quantity: {}'.format(self.hparams.dataset_name, self.class_quant))
        print('* {} dataset class weights: {}'.format(self.hparams.dataset_name, self.class_weights))
        print('* {} dataset len: {}'.format(self.hparams.dataset_name, num_samples))
        print('-' * 50)

    def setup(self, stage: str = None):
        if stage in (None, 'fit'):
            self.train_ds = self.dataset(root=self.hparams.train_data_csv, transform=self.train_transform)
            self.valid_ds = self.dataset(root=self.hparams.val_data_csv, transform=self.test_transform)

        elif stage in (None, 'test', 'predict'):
            self.test_ds = self.dataset(root=self.hparams.test_data_csv, transform=self.test_transform)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_ds, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    def get_transforms(self):
        
        # # as we will use pretrained net on imagenet:
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]

        size = self.hparams.img_size

        train = ResNet50_Weights.DEFAULT.transforms()
        test = ResNet50_Weights.DEFAULT.transforms()

        # train = transforms.Compose([
        #     transforms.Resize(size),
        #     transforms.Pad(4, padding_mode='reflect'),
        #     transforms.RandomCrop(size),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean, std=std)
        # ])
        # test = transforms.Compose([
        #     transforms.Resize(size),
        #     transforms.CenterCrop(size),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean, std=std)
        # ])
        return train, test