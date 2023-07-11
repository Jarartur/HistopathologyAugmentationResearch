# %% Imports
import torch as tc
import torchio as tio
from pathlib import Path
import numpy as np
import cv2
from torch.utils.data import DataLoader

import os
with os.add_dll_directory(r"C:\ProgramData\openslide\bin"):
    from openslide import open_slide

# %% Setup
LEVEL = 1
data = Path(r"C:\Users\Artur Jurgas\Documents\Workshop\HAA\data\segmentation\input")

src_list = data.glob("*[!mask].tiff")
mask_list = data.glob("*mask.tiff")

def inv_gauss(mask):
    out = cv2.GaussianBlur(mask, (3,3), 0)
    diff = np.abs(mask-out)
    return diff

def reader(path, level=0, mask=True):
    slide = open_slide(str(path))
    image = slide.read_region((0,0), level, slide.level_dimensions[level])
    image = np.asarray(image).copy()

    if image.shape[2] == 4:
        image = image[:,:,:-1]

    if mask:
        image = image[:,:,0]/254
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    affine = np.eye(4)
    return image, affine

def probability_reader(path):
    image, affine = reader(path)
    diff = inv_gauss(image)
    image = np.full_like(image, 0.25)
    image += diff
    return image, affine

# %%
subject_list = []
for src, mask in zip(src_list, mask_list):
    subject_list += [tio.Subject(
        src=tio.ScalarImage(src, reader=lambda p: reader(p, LEVEL, False)),
        mask=tio.ScalarImage(mask, reader=reader),
        sample_probability=tio.Image(mask, reader=probability_reader, type=tio.SAMPLING_MAP),
        name=src.stem,
    )]

# %%
transforms = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1)),
])

# %%
subjects_dataset = tio.SubjectsDataset(subject_list, transform=transforms)

# %%
patch_size = (32, 32, 1)
samples_per_volume = 100
sampler = tio.data.WeightedSampler(patch_size, "sample_probability") # random uniform sampling

#%%
patches_training_set = tio.Queue(
        subjects_dataset=subjects_dataset,
        sampler=sampler,
        max_length=120,
        samples_per_volume=samples_per_volume,
        shuffle_subjects=True,
        shuffle_patches=True,
        verbose=False
        )

#%%
patches_loader = DataLoader(
    patches_training_set,
    batch_size=2,
    num_workers=0,  # this must be 0
)

#%%
for epoch_index in range(2):
    for i, patches_batch in enumerate(patches_loader):
        print(f"=== {i} ===")
        # print(patches_batch)
        print(f"Finished")

        if i > 4: break

# %%
subject = subjects_dataset[0]

patch_overlap = (4, 4, 0)
grid_sampler = tio.inference.GridSampler(
    subject,
    patch_size,
    patch_overlap,
)

patch_loader = tc.utils.data.DataLoader(grid_sampler, batch_size=4)
aggregator = tio.inference.GridAggregator(grid_sampler)

for patches_batch in patch_loader:
    input_tensor = patches_batch['src'][tio.DATA]
    locations = patches_batch[tio.LOCATION]

    aggregator.add_batch(input_tensor, locations)

output_tensor = aggregator.get_output_tensor()
# %%
