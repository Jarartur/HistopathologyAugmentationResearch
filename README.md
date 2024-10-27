This repository contains the code used in the studies listed in the [Citing](#citing) section.

# Instalation
It is recommended to install this repository inside a conda virtual environment. To do so, run the following commands:

```bash
conda create -n haa python=3.11
```

For easier installation of [openslide-python](https://openslide.org/api/python/) and [pyvips](https://libvips.github.io/pyvips/intro.html) libraries, it is recommended to install the following packages before installing the project:

> [!IMPORTANT]  
> There is no `openslide-python` and `pyvips` package for windows on conda-forge. To install it on windows you need to manually supply the [openslide binary](https://github.com/openslide/openslide-bin/releases/tag/v4.0.0.6) to `C:\ProgramData\openslide` and [pyvips binary](https://www.libvips.org/install.html) to `C:\ProgramData\libvips`.

```bash
conda install openslide-python pyvips -c conda-forge
```


Finally, the `libhaa` library is installable by running the following command:

```bash
pip install -e .
```

Additional options are available for installation, such as installing the library with the `classification` or `segmentation` modules. To do so, first install [PyTorch](https://pytorch.org/get-started/locally/) and then run the following command. You can choose to instal one of the following options:

- both modules by running the command with the `all` options.
- only one module with either `histo-seg` or `histo-class` option.

```bash
# (e.g.) conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
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

Run `command -h` to learn about their available arguments. More detailed step-by-step guide will come in the near future. A simplified classification inference script is currently being worked on.

# Citing

If you find our work usefull, please cite:

```
@article{jurgasImprovingQualityControl2024,
  title = {Improving Quality Control of Whole Slide Images by Explicit Artifact Augmentation},
  author = {Jurgas, Artur and Wodzinski, Marek and D'Amato, Marina and {van der Laak}, Jeroen and Atzori, Manfredo and M{\"u}ller, Henning},
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
  author = {Jurgas, Artur and Wodzinski, Marek and Atzori, Manfredo and M{\"u}ller, Henning},
  editor = {Strumi{\l}{\l}o, Pawe{\l} and Klepaczko, Artur and Strzelecki, Micha{\l} and Boci{\k a}ga, Dorota},
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
