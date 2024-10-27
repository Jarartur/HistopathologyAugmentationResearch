from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights

import pandas as pd
from PIL import Image

from typing import Tuple, Type, Any
from pathlib import Path
from tqdm import trange
from sys import platform

if platform == "win32":
    import os

    vipsbin = r"c:\ProgramData\libvips\bin"
    os.environ["PATH"] = os.pathsep.join((vipsbin, os.environ["PATH"]))
import pyvips


def cut_patches_for_inference(
    input_image: Path,
    save_dir: Path,
    patch_size: Tuple[int, int],
    openslide_level: int = 2,
):
    # img: pyvips.Image = pyvips.Image.new_from_file(str(input_image), access='sequential')
    img: pyvips.Image = pyvips.Image.openslideload(
        str(input_image), level=openslide_level
    )
    if img.bands > 3:
        img = img.flatten(background=255)
    # get metadata from img
    width = img.width
    height = img.height
    # get number of patches based on patch size and image size

    # get padding size
    pad_x = patch_size[0] - (width % patch_size[0])
    pad_y = patch_size[1] - (height % patch_size[1])
    # pad image to fit patches
    img2 = img.embed(0, 0, width + pad_x, height + pad_y, extend="white")

    num_patches_x = img2.width // patch_size[0]
    num_patches_y = img2.height // patch_size[1]

    file_list = []

    for i in trange(num_patches_x):
        for j in range(num_patches_y):
            patch = img2.crop(
                i * patch_size[0], j * patch_size[1], patch_size[0], patch_size[1]
            )
            file = str(save_dir / f"{input_image.stem}_{i}_{j}.png")
            patch.write_to_file(file)
            file_list.append(file)

    with open(save_dir / f"{input_image.stem}_info.txt", "w") as f:
        f.write(f"width: {width}\n")
        f.write(f"height: {height}\n")
        f.write(f"num_patches_x: {num_patches_x}\n")
        f.write(f"num_patches_y: {num_patches_y}\n")
        f.write(f"pad_x: {pad_x}\n")
        f.write(f"pad_y: {pad_y}\n")

    with open(save_dir / ".." / f"{input_image.stem}_file_list.csv", "w") as f:
        f.write("File,air,dust,tissue,ink,marker,focus\n")
        for file in file_list:
            f.write(f"{file},0,0,0,0,0,0\n")

    print(f"Done! {num_patches_x*num_patches_y} patches saved to {save_dir}.")


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
    def __init__(
        self,
        dataset_name: str,
        img_size: Tuple[int, int] = (224, 224),
        batch_size: int = 2,
        num_workers: int = 4,
        train_data_csv: str = "data.csv",
        val_data_csv: str = "data.csv",
        test_data_csv: str = "data.csv",
    ):
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
        num_samples = data.shape[0]
        self.num_classes = len(labels)
        self.num_step = num_samples // self.hparams.batch_size

        self.class_quant = []
        num_samples_nonempty = 0
        for i in range(self.num_classes):
            num = data.iloc[:, i + 1].sum()
            self.class_quant += [num]
            num_samples_nonempty += num
        self.class_weights = [
            1 - (quant / num_samples_nonempty) for quant in self.class_quant
        ]

        print("-" * 50)
        print(
            "* {} dataset class num: {}".format(
                self.hparams.dataset_name, self.num_classes
            )
        )
        print(
            "* {} dataset class quantity: {}".format(
                self.hparams.dataset_name, self.class_quant
            )
        )
        print(
            "* {} dataset class weights: {}".format(
                self.hparams.dataset_name, self.class_weights
            )
        )
        print("* {} dataset len: {}".format(self.hparams.dataset_name, num_samples))
        print("-" * 50)

    def setup(self, stage: str = None):
        if stage in (None, "fit"):
            self.train_ds = self.dataset(
                root=self.hparams.train_data_csv, transform=self.train_transform
            )
            self.valid_ds = self.dataset(
                root=self.hparams.val_data_csv, transform=self.test_transform
            )

        elif stage in (None, "test", "predict"):
            self.test_ds = self.dataset(
                root=self.hparams.test_data_csv, transform=self.test_transform
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.valid_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def get_transforms(self):
        train = ResNet50_Weights.DEFAULT.transforms()
        test = ResNet50_Weights.DEFAULT.transforms()

        return train, test