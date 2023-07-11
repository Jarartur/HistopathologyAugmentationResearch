# %% Imports
from types import NoneType
from typing import Iterable, List, Tuple
import warnings
import torch as tc
import torchio as tio
from pathlib import Path
import numpy as np
import cv2
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm


from sys import platform
if platform == "linux" or platform == "linux2":
    from openslide import open_slide
elif platform == "win32":
    import os
    with os.add_dll_directory(r"C:\ProgramData\openslide\bin"):
        from openslide import open_slide


# %%
class PatchDataset:
    def __init__(
        self,
        src_list: Iterable,
        mask_list: Iterable,
        patch_size: Tuple[int, int, int],
        max_samples_per_volume: int,
        num_samples_downsampling_mode: str = 'quadratic',
        levels_to_skip: List[int] = [],
        preproces: List | NoneType = None,
        augments: List | NoneType = None,
    ) -> None:
        self.patch_size = patch_size
        self.max_samples_per_volume = max_samples_per_volume
        preproces = preproces if preproces else []
        augments = augments if augments else []
        subject_list = []

        for src, mask in zip(src_list, mask_list):
            slide = open_slide(src)
            levels = slide.level_count
            shapes = slide.level_dimensions
            downsamples = slide.level_downsamples
            print(f"{slide.level_dimensions=}")
            for level, shape, downsample in zip(range(levels), shapes, downsamples):

                if level in levels_to_skip: continue
                
                if num_samples_downsampling_mode=='linear' or num_samples_downsampling_mode==1:
                    down = int(downsample)
                elif num_samples_downsampling_mode=='quadratic' or num_samples_downsampling_mode==2:
                    down = int(downsample**2)
                elif num_samples_downsampling_mode=='none' or num_samples_downsampling_mode==0:
                    down = 1
                else:
                    raise Exception(f"num_samples_downsampling_mode: {num_samples_downsampling_mode} not supported.")

                num_samples = self.max_samples_per_volume // down

                if down == 0:
                    warnings.warn(
                        f'''Chosen `num_samples_downsampling_mode`={num_samples_downsampling_mode} and `max_samples_per_volume={max_samples_per_volume}` made `num_samples` for subject of size={shape} equal 0.
                        Changing to at least 1.
                        If you have lower `max_samples_per_volume` it is recommended to use linear mode.
                        '''
                        )
                    down = 1

                subject_list += [
                    tio.Subject(
                        src=tio.ScalarImage(
                            src, reader=partial(reader, level=level, mask_dims=None)
                        ),
                        mask=tio.LabelMap(
                            mask, reader=partial(reader, level=0, mask_dims=shape)
                        ),
                        sample_probability=tio.Image(
                            mask,
                            reader=partial(probability_reader, mask_dims=shape),
                            type=tio.SAMPLING_MAP,
                        ),
                        name=src.stem,
                        level=level,
                        num_samples=num_samples
                    )
                ]

        print(f"Found: {len(subject_list)} subjects.")
        self.len_of_subjects = len(subject_list)

        transforms_train = tio.Compose(preproces + augments)
        transforms_infer = tio.Compose(preproces)

        self.subjects_dataset_train = tio.SubjectsDataset(
            subject_list, transform=transforms_train
        )
        self.subjects_dataset_infer = tio.SubjectsDataset(
            subject_list, transform=transforms_infer
        )

    def get_training_queue(
        self,
        batch_size: int,
        num_workers: int = 0,
        max_length: int | NoneType = None,
    ):
        max_length = (
            max_length
            if max_length is not None
            else self.max_samples_per_volume * self.len_of_subjects
        )

        sampler = tio.data.WeightedSampler(
            self.patch_size, "sample_probability"
        )  # random uniform sampling

        patches_training_set = tio.Queue(
            subjects_dataset=self.subjects_dataset_train,
            sampler=sampler,
            max_length=max_length,
            samples_per_volume=self.max_samples_per_volume,
            shuffle_subjects=True,
            shuffle_patches=True,
            verbose=False,
            num_workers=num_workers,
        )

        patches_loader = DataLoader(
            patches_training_set,
            # pin_memory=True,
            batch_size=batch_size,
            num_workers=0,  # this must be 0
        )

        return patches_loader

    @tc.inference_mode()
    def get_inference_aggregator(
        self,
        patch_overlap: Tuple[int, int, int],
        model: tc.nn.Module,
        device: str = "cpu",
        indices: List[int] | NoneType = None,
        squeeze_last_dim: bool = True,
        batch_size: int = 16
    ):
        dataset = (
            self.subjects_dataset_infer
            if indices is None
            else [self.subjects_dataset_infer[i] for i in indices]
        )
        for subject in tqdm(dataset, desc="Subjects"):
            grid_sampler = tio.inference.GridSampler(
                subject,
                self.patch_size,
                patch_overlap,
            )

            patch_loader = tc.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
            aggregator = tio.inference.GridAggregator(grid_sampler)

            for patches_batch in tqdm(patch_loader, desc="Batch"):
                input_tensor = patches_batch["src"][tio.DATA].to(device)

                if squeeze_last_dim:
                    output = model(input_tensor.squeeze(-1)).unsqueeze(-1)
                    output = tc.sigmoid(output)
                    output = tc.where(output > 0.5, 1.0, 0.0)
                else:
                    output = model(input_tensor)
                    output = tc.sigmoid(output)
                    output = tc.where(output > 0.5, 1.0, 0.0)

                locations = patches_batch[tio.LOCATION]
                aggregator.add_batch(output, locations)

            output_tensor = aggregator.get_output_tensor()
            yield output_tensor

def reader(
    path: Path | str,
    level: int = 0,
    mask_dims: Tuple[int, int] | NoneType = None,
):
    slide = open_slide(str(path))
    # # HACK  --> #
    # # NOTE: change openslide to sth else for masks
    # try:
    #     image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    # except IndexError as e:
    #     image = slide.read_region((0, 0), 0, slide.level_dimensions[0])
    # # HACK  <-- #

    image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    image = np.asarray(image)  # .copy()

    if image.shape[2] == 4:
        image = image[:, :, :-1]

    if mask_dims:
        image = image[:, :, 0] / 255
        image = (
            cv2.resize(image, mask_dims, interpolation=cv2.INTER_NEAREST)
            if image.shape[::-1] != mask_dims
            else image
        )
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    affine = np.eye(4)
    return image.astype(np.float32), affine

def probability_reader(path: Path | str, mask_dims: Tuple[int, int]):
    image, affine = reader(path, 0, mask_dims)
    diff = inv_gauss(image)
    image = np.full_like(image, 0.25)
    image += diff
    return image, affine

def inv_gauss(mask: cv2.Mat):
    out = cv2.GaussianBlur(mask, (3, 3), 0)
    diff = np.abs(mask - out)
    return diff


# %% Check
if __name__ == "__main__":
    from matplotlib import pyplot as plt

    output = None

    LEVEL = 1
    data = Path(r"C:\Users\Artur Jurgas\Documents\Workshop\HAA\data\segmentation\input\train")

    src_list = list(data.glob("*[!mask].tiff"))
    mask_list = list(data.glob("*mask.tiff"))

    preproces = [tio.RescaleIntensity(out_min_max=(0, 1))]

    dataset = PatchDataset(src_list, mask_list, (32, 32, 1), preproces, None)

    queue = dataset.get_training_queue(100, 120, 2)
    for epoch_index in range(2):
        for i, patches_batch in enumerate(queue):
            output_src = patches_batch["src"][tio.DATA]
            output_mask = patches_batch["mask"][tio.DATA]
            output_sampling = patches_batch["sample_probability"][tio.DATA]

    # Should yield single grayscale patch
    plt.subplot(1, 3, 1)
    plt.imshow(np.array(output_src[0]).squeeze(), cmap="gray")
    plt.subplot(1, 3, 2)
    plt.imshow(np.array(output_mask[0]).squeeze(), cmap="gray")
    plt.subplot(1, 3, 3)
    plt.imshow(np.array(output_sampling[0]).squeeze(), cmap="gray")
    plt.show()

    def model(input):
        return input

    inference = dataset.get_inference_aggregator((4, 4, 0), model, None)
    for image in inference:
        output = image
    # Should yield whole grayscale image
    plt.imshow(np.array(output).squeeze(), cmap="gray")
    plt.show()

# %%
