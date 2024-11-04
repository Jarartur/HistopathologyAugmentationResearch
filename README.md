This repository contains the code used in the studies listed in the [Citing](#citing) section.

# Instalation
It is recommended to install this repository inside a conda virtual environment. To do so, run the following commands:

```bash
conda create -n haa python=3.11
```

For easier installation of [openslide-python](https://openslide.org/api/python/) and [pyvips](https://libvips.github.io/pyvips/intro.html) libraries, it is recommended to install the following packages before installing the project:

> [!IMPORTANT]  
> There is no `openslide-python` and `pyvips` package for windows on conda-forge. To install it on windows you need to manually supply the [openslide binary](https://github.com/openslide/openslide-bin/releases/tag/v4.0.0.6) to `C:/ProgramData/openslide` and [pyvips binary](https://www.libvips.org/install.html) to `C:/ProgramData/libvips`.

```bash
conda install openslide-python pyvips -c conda-forge
```


Finally, the `libhaa` library is installable by running the following command:

```bash
pip install -e .
```

Additional options are available after installation, such as installing the library with the `classification` or `segmentation` modules. To do so, first make sure you have installed the base package, then install [PyTorch](https://pytorch.org/get-started/locally/) and then run the following command. You can choose to instal one of the following options:

- both modules by running the command with the `all` options.
- only one module with either `histo-seg` or `histo-class` option.

```bash
# (e.g.) conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
# After this you can choose to install optional dependencies:
pip install -e .[all] # or [histo-seg] or [histo-class]
```

We provide the segmentation and classification model weights in the [releases page](https://github.com/Jarartur/HistopathologyAugmentationResearch/releases).

# Repository information

The `libhaa` folder contains code from this library, including segmentation and classification (wip) inference scripts.

The `model_training` folder contains the original code used during the segmentation and classification study.

# Usage guide

For now a few scripts are available as cli commands after instalation:

- `build-collection`:
allows for building a custom artifact library from a set of WSIs with their annotations done in [ASAP](https://computationalpathologygroup.github.io/ASAP/).

- `segment`:
allows for segmenting histopathology images with the segmentation model available in the [releases page](https://github.com/Jarartur/HistopathologyAugmentationResearch/releases/tag/segmentation).

- `generate-dataset`:
allows for augmenting a chosen dataset with the previously generated artifact library and segmentations.

- `cut-patches-training`:
generates patches of training data from the chosen pyramid level of images from the augmented (or not) dataset.

- `cut-patches-inference`:
generates patches from a single WSI for inference of the classification model available in the [releases page](https://github.com/Jarartur/HistopathologyAugmentationResearch/releases/tag/classification).

## Typical workflow

To follow along download `test_image.tiff` from the releases page and replace the file `data/test_data/images/dataset_name/replace_with_test_image`.

### Building a collection

First to build our artifact collection we run
```bash
build-collection --wsi-path data/test_data/images --annotations-path data/test_data/annotations --save-path data/test_collection --relative-paths
```

- `--wsi-path` specifies our image directory.
- `--annotations-path` specifies our annotation directories. Keep in mind that if we were to keep our annotations inside the image folder we can just supply the same path as to `--wsi-path`.
- `--save-path` specifies where to save the collection.
- `--relative-paths` tells our program to include the `dataset_name` folder when searching for annotations. If you had a folder structure where each of the image is inside a subfolder (like here) but the annotations are aggregated in a single folder then do not use this option.

This will create an artifact library in `data/test_collection`. It will consist of folders of artifact types and in each folder there will be artifact images and their annotation.

### Segmenting images

Now before augmenting our images we also need to generate segmentations for our images, so the program will know where to place each artifact. To do so, first download the segmentation model weights and place them, e.g., in the `data/models` directory. *Do not extract the downloaded file.* Now we run:
```bash
segment --wsi-path data/test_data/images --model-weights data/models/weights_v11.07.2023.tar --save-path data/test_data/segmentations --openslide-level 4 --device 'cuda'
```

- `--wsi-path` specifies our image directory.
- `--model-weights` specifies our weights path.
- `--save-path` specifies the save folder. Any subfolders in `--wsi-path` will be propagated here.
- `--openslide-level` specifies the pyramid level to load. Running `segment -h` will give you recommended values for datasets used in the study.
- `--device` can be a cpu or cuda. You can also specify which GPU to use by, e.g., `cuda:2`.

In our case this will create a folder `data/test_data/segmentations/dataset_name` with a `test_image.xml` file in it contianing the segmentation in ASAP format.

### Augmenting a dataset

Now we will augment our image with new artifacts.

```bash
generate-dataset --wsi-path data/test_data/images --segmentations-path data/test_data/segmentations --artifact-collection-path data/test_collection --save-path data/test_augmented --root-annotation data/test_data/annotations
```

- `--wsi-path` specifies our image directory.
- `--segmentations-path` specifies our previously generated segmentations.
- `--artifact-collection-path` specifies our previously generated artifact collection.
- `--save-path` specifies the save folder.
- `--root-annotation` (optional). With this argument you can supply already present annotations. The new, augmented annotations will be merged with the existing ones.

This will generate a `data/test_augmented/dataset_name` folder containing our augmented image. You can inspect the before and after with [ASAP](https://computationalpathologygroup.github.io/ASAP/) software. Keep in mind not all already present artifacts in this particular image are annotated from the beginning as it is a simple example.

### Classification

The classification model works a bit differently from the segmentation model. First we need to manually cut patches from the augmented image. To do so, we run:

```bash
cut-patches-inference --wsi-path data/test_augmented/dataset_name/test_image.tiff --save-path data/test_inference --patch-size 224 --openslide-level 1
```

Finally, we can classify our image. To do so, first download the classification model weights and unzip them. Place the chosen model ending in `.ckpt` in the `data/models` directory. Now we run:

```bash
classify --wsi-folder data/test_inference/test_image --save-path data/test_inference/infra --weights-path "data/models/ACR'.ckpt"
```

This will produce a `data/test_inference/infra/test_image` folder with the classification results in the form of a `preds.csv` file containing the class probabilities for each patch under the `y_hat` column. 

In the future we plan to add a script that will merge the patch predictions into a single image.

# Citing

If you find our work usefull, please cite:

```
@article{jurgasImprovingQualityControl2024,
  title = {Improving Quality Control of Whole Slide Images by Explicit Artifact Augmentation},
  author = {Jurgas, Artur and Wodzinski, Marek and D'Amato, Marina and {van der Laak}, Jeroen and Atzori, Manfredo and M{/"u}ller, Henning},
  year = {2024},
  month = aug,
  journal = {Scientific Reports},
  volume = {14},
  number = {1},
  pages = {17847},
  publisher = {Nature Publishing Group},
  issn = {2045-2322},
  doi = {10.1038/s41598-024-68667-2},
  urldate = {2024-08-27},
  copyright = {2024 The Author(s)},
  langid = {english},
  keywords = {Data processing,Machine learning,Quality control},
}

@article{jurgasArtifactAugmentationLearningbased2023,
  title = {Artifact {{Augmentation}} for {{Learning-based Quality Control}} of {{Whole Slide Images}}},
  author = {Jurgas, Artur and Wodzinski, Marek and Celniak, Weronika and Atzori, Manfredo and Muller, Henning},
  year = {2023},
  month = jul,
  journal = {Annual International Conference of the IEEE Engineering in Medicine and Biology Society. IEEE Engineering in Medicine and Biology Society. Annual International Conference},
  volume = {2023},
  pages = {1--4},
  issn = {2694-0604},
  doi = {10.1109/EMBC40787.2023.10340997},
  langid = {english},
  pmid = {38082977},
  keywords = {Algorithms,Artifacts,Humans,Image Processing Computer-Assisted,Neoplasms},
}
```

For segmentation:
```
@inproceedings{jurgasRobustMultiresolutionMultistain2024,
  title = {Robust {{Multiresolution}} and {{Multistain Background Segmentation}} in {{Whole Slide Images}}},
  booktitle = {The {{Latest Developments}} and {{Challenges}} in {{Biomedical Engineering}}},
  author = {Jurgas, Artur and Wodzinski, Marek and Atzori, Manfredo and M{/"u}ller, Henning},
  editor = {Strumi{/l}{/l}o, Pawe{/l} and Klepaczko, Artur and Strzelecki, Micha{/l} and Boci{/k a}ga, Dorota},
  year = {2024},
  series = {Lecture {{Notes}} in {{Networks}} and {{Systems}}},
  pages = {29--40},
  publisher = {Springer Nature Switzerland},
  address = {Cham},
  doi = {10.1007/978-3-031-38430-1_3},
  isbn = {978-3-031-38430-1},
  langid = {english},
  keywords = {Computational pathology,Deep learning,Digital pathology Segmentation,Whole-slide images,WSI},
}
```
