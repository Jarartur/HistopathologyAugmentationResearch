Contained is the base library for augmentation as of publication.

Additionally code for the classification module and segmentation is included.

# Instalation
It is recommended to install this repository inside a conda virtual environment. To do so, run the following commands:

```bash
conda create -n haa python=3.11
```

For easier installation of [openslide-python]() library, it is recommended to install the following packages before installing the project:

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
