Contained is the base library for augmentation as of publication.

Additionally code for the classification module and segmentation is included.

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


Finally, libhaa library is installable by running the following command:

```bash
pip install -e .
```

Additional options are available for installation, such as installing the library with the `classification` or `segmentation` modules. To do so, first install [PyTorch](https://pytorch.org/get-started/locally/) and then run the following command. You can choose to install both modules by running the command with the `all` options, or only one of them with `histo-seg` or `histo-class`.

```bash
# (e.g.) conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -e .[all] # [histo-seg] or [histo-class]
```


For segmentation inference we provide the weights in the `Releases` github panel.
Classification weights will be made available soon.

# Repository information

`libhaa` folder contains this library's code, including segmentation and classification (wip) inference scripts.

`model_training` folder contains the original code used during the segmentation and classification study.

# Usage guide

For now a few scripts are available as cli commands after instalation:

- `build-collection`
- `generate-dataset`
- `segment`
- `cut-patches-training `
- `cut-patches-inference`

Run `command -h` to lear about their available arguments.

More detailed step-by-step guide will come in the near future.

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
  abstract = {The problem of artifacts in whole slide image acquisition, prevalent in both clinical workflows and research-oriented settings, necessitates human intervention and re-scanning. Overcoming this challenge requires developing quality control algorithms, that are hindered by the limited availability of relevant annotated data in histopathology. The manual annotation of ground-truth for artifact detection methods is expensive and time-consuming. This work addresses the issue by proposing a method dedicated to augmenting whole slide images with artifacts. The tool seamlessly generates and blends artifacts from an external library to a given histopathology dataset. The augmented datasets are then utilized to train artifact classification methods. The evaluation shows their usefulness in classification of the artifacts, where they show an improvement from 0.10 to 0.01 AUROC depending on the artifact type. The framework, model, weights, and ground-truth annotations are freely released to facilitate open science and reproducible research.},
  copyright = {2024 The Author(s)},
  langid = {english},
  keywords = {Data processing,Machine learning,Quality control},
}
```