# Imports
from dataloader import PatchDataset
from model import UNet

import numpy as np
from ruamel.yaml import YAML
from pathlib import Path
from dvclive import Live
import torch as tc
import torchio as tio
import torch.nn as nn
import torch.optim as optim
import torchmetrics as metrics
from torchvision.utils import make_grid
import torchvision.transforms.functional as F

from datetime import datetime
from tqdm import trange, tqdm

def load_checkpoint(path, model, optimizer, scheduler, load_scheduler:bool = True):
    checkpoint = tc.load(path)
    epoch = checkpoint['epoch']

    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded model")

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Loaded optimizer")

    if (scheduler is not None) and (load_scheduler):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Loaded scheduler")
    else:
        print("Scheduler omitted")

    model.train()

    return model, optimizer, scheduler, epoch

@tc.inference_mode()
def dice_loss(prediction, target):
    """
    Dice as PyTorch cost function. TODO: move
    """
    smooth = 1
    prediction = prediction.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = tc.sum(prediction * target)
    return ((2 * intersection + smooth) / (prediction.sum() + target.sum() + smooth))

def train(config_yaml: Path):
    # Parameters initialization
    params = YAML().load(config_yaml)
    optim_params = params["optimizer"]
    data_params = params["data"]
    model_params = params["model"]
    logging_params = params["logging"]

    data = Path(data_params["train_path"])
    data_val = Path(data_params["val_path"])
    print(f"{data=}")

    # src_list = list(data.glob("*[!mask].tiff"))
    mask_list = list(data.glob("*mask.tiff"))
    src_list = [(path.parent / path.name.replace('_mask', '')) for path in mask_list]

    # src_val_list = list(data_val.glob("*[!mask].tiff"))
    mask_val_list = list(data_val.glob("*mask.tiff"))
    src_val_list = [(path.parent / path.name.replace('_mask', '')) for path in mask_val_list]


    # Data initialization
    preproces = [tio.RescaleIntensity(out_min_max=(0, 1))]
    # augment = [tio.RescaleIntensity(out_min_max=(0, 1))]

    Dataset = PatchDataset(
        src_list=src_list,
        mask_list=mask_list,
        patch_size=data_params["patch_size"],
        max_samples_per_volume=data_params["max_samples_per_volume"],
        num_samples_downsampling_mode=data_params["samples_downsampling_mode"],
        levels_to_skip=data_params["levels_to_skip"],
        preproces=preproces,
    )
    queue = Dataset.get_training_queue(
        max_length=data_params["max_length"],
        batch_size=data_params["batch_size"],
        num_workers=data_params["num_workers"],
    )

    Dataset_val = PatchDataset(
        src_list=src_val_list,
        mask_list=mask_val_list,
        patch_size=data_params["patch_size"],
        max_samples_per_volume=data_params["max_samples_per_volume"],
        num_samples_downsampling_mode=data_params["samples_downsampling_mode"],
        levels_to_skip=data_params["levels_to_skip"],
        preproces=preproces,
    )
    queue_val = Dataset_val.get_training_queue(
        max_length=None,
        batch_size=data_params["batch_size"],
        num_workers=2,
    )

    # Model initialization
    DEVICE = model_params["device"]
    model = UNet(1, 1).to(DEVICE)

    # Optimizer initialization
    objective = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=model.parameters(), **optim_params)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.993, verbose=True)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 50, 0.1, verbose=True)
    scheduler = None


    # Metric and logging initialization
    now = str(datetime.now()).replace(':', ";")
    dvc_path = Path("data/logging") / now
    checkpoint_path = Path(data_params["checkpoint_path"]) / now
    checkpoint_path.mkdir()

    if data_params['resume_path'] is not None:
        model, optimizer, scheduler, epoch = load_checkpoint(data_params['resume_path'], model ,optimizer, scheduler, data_params["load_scheduler"])
        print("Loaded checkpiont...✔️")
    else:
        epoch = 0

    with Live(dir=str(dvc_path), save_dvc_exp=True) as live:
        live.log_params(params)
        best_dice = -1e5

        # Trainloop
        print(f"Starting with {epoch=}")
        for epoch_index in trange(epoch, model_params["epochs"], desc="Epochs"):
            live.log_param("current_epoch", epoch_index)
            batch_loss = []
            batch_dice = []
            for patches_batch in tqdm(queue, desc="Batch"):
                optimizer.zero_grad()

                # Batch data setup
                src = patches_batch["src"][tio.DATA].squeeze(-1).to(DEVICE)
                trg = patches_batch["mask"][tio.DATA].squeeze(-1).to(DEVICE)

                # Inference and loss calculations
                result = model(src)

                loss = objective(result, trg)
                batch_loss += [loss.item()]

                # Backward pass
                loss.backward()
                optimizer.step()

                # dice = 0.1
                dice = dice_loss(tc.sigmoid(result), trg).item()
                # print(f"{dice=}")
                batch_dice += [dice]

            if scheduler is not None:
                scheduler.step()

            # Metric logging
            print(f"Last dice: {dice}")
            batch_dice = np.mean(batch_dice)
            batch_loss = np.mean(batch_loss)
            live.log_metric("minibatch_dice", batch_dice)
            live.log_metric("batch_loss", batch_loss)

            # Plotting every #n epochs
            if not epoch_index % logging_params["plot_every"]:
                print("Ploting...", end="")

                res = tc.sigmoid(result)
                res = tc.where(res > 0.5, 1.0, 0.0)

                max_imgs = 30
                sub_grids = [
                    make_grid(img, nrow=img.shape[0])
                    for img in [src[:max_imgs, ...], res[:max_imgs, ...], trg[:max_imgs, ...]]
                ]

                print(f"{res[:max_imgs, ...].max()=}")

                grid = make_grid(sub_grids, nrow=1)
                grid_img = F.to_pil_image(grid)
                # grid_img.save(f"temp/imgs/PIL_src_vs_result_vs_trg_test#.png")
                live.log_image(f"src_vs_result_vs_trg_#{live.step}.png", grid_img)
                print("✔️")

            if not epoch_index % logging_params["val_every"]:
                print("Validating...")
                val_dice = []
                model.eval()
                with tc.inference_mode():
                    for patches_batch_val in tqdm(queue_val, desc="Validation"):
                        src_val = patches_batch_val["src"][tio.DATA].squeeze(-1).to(DEVICE)
                        trg_val = patches_batch_val["mask"][tio.DATA].squeeze(-1).to(DEVICE)
                        result_val = model(src_val)
                        res_val = tc.sigmoid(result_val) #NOTE: to remove after testing
                        val_dice += [dice_loss(res_val, trg_val).item()]
                model.train()
                val_dice = np.mean(val_dice)
                live.log_metric("validation_dice", val_dice)
                print("✔️")

            # Checkpointing every #n epochs
            if not epoch_index % model_params["checkpoint_every"]:
                print("Ckeckpointing...", end="")

                save_dict = {
                    "epoch": epoch_index,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None
                }
                path = checkpoint_path / "checkpoint_last.tar"
                tc.save(save_dict, path)
                print("✔️")

                if batch_dice > best_dice:
                    print("Ckeckpointing best...", end="")

                    save_dict = {
                        "epoch": epoch_index,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None
                    }
                    path = checkpoint_path / "checkpoint_best.tar"
                    tc.save(save_dict, path)

                    best_dice = batch_dice
                    print("✔️")

            # Finishing batch
            live.next_step()


if __name__ == "__main__":
    config = Path("segmentation/config.yaml")
    train(config)
