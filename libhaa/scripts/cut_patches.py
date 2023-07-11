from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

from numpy import save

from libhaa.base.config import (
    PATCHES_SIZE_DETECTION,
    WSI_EXTs,
    ARTIFACT_LIBRARY_ANNOTATION_EXT,
)
from libhaa.base.io import AnnotatedWSI
from pathlib import Path
import itertools
from tqdm import tqdm

def cut_training(
    root_wsi: Path,
    root_segmentation: Path,
    root_annotation: Path,
    save_root: Path,
    num_background: int,
    num_tissue: int,
    num_artifact: int,
    openslide_levels: Optional[List] = None,
    yolo: bool = False,
    patch_size: Optional[Tuple[int, int]] = None
):
    
    patch_size = PATCHES_SIZE_DETECTION if patch_size is None else patch_size

    pbar = tqdm(
        wsi_iterator(
        root_wsi=root_wsi,
        root_segmentation=root_segmentation,
        root_annotation=root_annotation,
        save_root=save_root,
        )
    )
    for case in pbar:
        pbar.set_description(f"Processing {case.wsi_path.name}...")

        if openslide_levels is not None:
            for openslide_level in openslide_levels:
                downsample = openslide_level - openslide_levels[0]
                downsample = 2**downsample

                num_background_level = num_background // downsample
                num_tissue_level = num_tissue // downsample
                num_artifact_level = num_artifact // downsample

                extract_patches(case, patch_size, num_background_level, num_tissue_level, num_artifact_level, openslide_level, yolo)
        else:
            extract_patches(case, patch_size, num_background, num_tissue, num_artifact, None, yolo)


def cut_inference(wsi_path):
    ...


@dataclass
class Case:
    wsi_path: Path
    segmentation_path: Path
    annotation_path: Path
    save_path: Path


def extract_patches(case: Case, patch_size:[Tuple[int, int]], num_background: int, num_tissue: int, num_artifact: int, openslide_level = None, yolo = False):
    wsi = AnnotatedWSI.from_file(
                    case.wsi_path, case.annotation_path, case.segmentation_path, openslide_level
    )
    try:
        wsi.cutout_patches(
            patch_size,
            case.save_path,
            num_background,
            num_tissue,
            num_artifact,
            'both'
        )
    except Exception as e:
        print(e)
        print(f"Skipping {case.wsi_path}...")
        pass

def wsi_iterator(
    root_wsi: Path, root_segmentation: Path, root_annotation: Path, save_root: Path
) -> Generator[Case, None, None]:
    """
    Iterate over a directory of WSIs and match it's annotation
    """
    # chain all wsi possible extensions together into an iterator
    for wsi_path in itertools.chain(*[root_wsi.rglob(f"*{ext}") for ext in WSI_EXTs]):
        relative_path = wsi_path.relative_to(root_wsi)
        segmentation_path = (root_segmentation / relative_path).with_suffix(
            ARTIFACT_LIBRARY_ANNOTATION_EXT
        )
        annotation_path = (root_annotation / relative_path).with_suffix(
            ARTIFACT_LIBRARY_ANNOTATION_EXT
        )
        save_path = save_root / relative_path.parent / relative_path.stem
        save_path.mkdir(parents=True, exist_ok=True)

        yield Case(wsi_path, segmentation_path, annotation_path, save_path)
