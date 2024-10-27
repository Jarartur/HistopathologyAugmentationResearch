from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

from libhaa.base.config import (
    PATCHES_SIZE_DETECTION,
    WSI_EXTs,
    ARTIFACT_LIBRARY_ANNOTATION_EXT,
)
from libhaa.base.io import AnnotatedWSI
from pathlib import Path
import itertools
from tqdm import tqdm
import argparse

from libhaa.classification.dataloader import cut_patches_for_inference
from libhaa.base.config import LABELS_SAVING_FORMATS


def cut_training(
    root_wsi: Path,
    root_segmentation: Path,
    root_annotation: Path,
    save_root: Path,
    num_background: int,
    num_tissue: int,
    num_artifact: int,
    openslide_levels: Optional[List] = None,
    format: LABELS_SAVING_FORMATS = "both",
    patch_size: Optional[Tuple[int, int]] = None,
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

                extract_patches(
                    case,
                    patch_size,
                    num_background_level,
                    num_tissue_level,
                    num_artifact_level,
                    openslide_level,
                    format,
                )
        else:
            extract_patches(
                case, patch_size, num_background, num_tissue, num_artifact, None, format
            )


def cut_inference(
    input_image: Path,
    save_dir: Path,
    patch_size: Tuple[int, int],
    openslide_level: int = 2,
):
    cut_patches_for_inference(input_image, save_dir, patch_size, openslide_level)


@dataclass
class Case:
    wsi_path: Path
    segmentation_path: Path
    annotation_path: Path
    save_path: Path


def extract_patches(
    case: Case,
    patch_size: Tuple[int, int],
    num_background: int,
    num_tissue: int,
    num_artifact: int,
    openslide_level=None,
    format: LABELS_SAVING_FORMATS = "both",
):
    wsi = AnnotatedWSI.from_file(
        case.wsi_path, case.annotation_path, case.segmentation_path, openslide_level
    )
    try:
        wsi.cutout_patches(
            patch_size, case.save_path, num_background, num_tissue, num_artifact, format
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


def cli_cut_for_training():
    parser = argparse.ArgumentParser(
        description="Cut patches from the WSI for training.",
        epilog="Artur Jurgas.",
    )
    parser.add_argument("--wsi-path", type=Path, required=True, help="Path to WSIs.")
    parser.add_argument(
        "--segmentations-path",
        type=Path,
        required=True,
        help="Path to segmentations of WSIs.",
    )
    parser.add_argument(
        "--annotations-path",
        type=Path,
        required=True,
        help="Path to annotations of WSIs.",
    )
    parser.add_argument(
        "--save-path", type=Path, required=True, help="Path to save to."
    )

    parser.add_argument(
        "--num-background",
        type=int,
        required=True,
        help="Number of background patches to extract.",
    )
    parser.add_argument(
        "--num-tissue",
        type=int,
        required=True,
        help="Number of tissue (without artifacts) patches to extract.",
    )
    parser.add_argument(
        "--num-artifact",
        type=int,
        required=True,
        help="Number of artifact patches to extract.",
    )

    parser.add_argument(
        "--openslide-levels",
        type=List,
        required=True,
        help="List of openslide levels to extract from. Usefull for multiresolution models.",
    )

    parser.add_argument(
        "--format",
        type=str,
        required=True,
        help="Format of the resulting annotations. Can be `yolo`, `asap`, or `both`. `yolo` is compatile with the YOLOv5 model. `asap` is comaptible with the ASAP histopathology annotation program.",
    )

    parser.add_argument(
        "--patch-size",
        type=List,
        required=True,
        help="List of ints. Defines the size of patches that will be extracted.",
    )

    args = parser.parse_args()

    cut_training(
        root_wsi=args.wsi_path,
        root_segmentation=args.segmentation_path,
        root_annotation=args.annotation_path,
        save_root=args.save_path,
        num_background=args.num_background,
        num_tissue=args.num_tissue,
        num_artifact=args.num_artifact,
        openslide_levels=args.openslide_levels,
        format=args.format,
        patch_size=args.patch_size,
    )


def cli_cut_for_inference():
    parser = argparse.ArgumentParser(
        description="Cut patches from the WSI for inference.",
        epilog="Artur Jurgas.",
    )
    parser.add_argument(
        "--wsi-path", type=Path, required=True, help="Path to WSI image (one)."
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        required=True,
        help="Path where to save the processed patches from the WSI.",
    )
    parser.add_argument(
        "--patch-size",
        type=List,
        required=True,
        help="List of ints. Size of the extracted patches.",
    )
    parser.add_argument(
        "--openslide-level",
        type=Path,
        required=True,
        help="Pyramid level from which to extract the patches.",
    )

    args = parser.parse_args()

    cut_inference(
        input_image=args.wsi_path,
        save_dir=args.save_path,
        patch_size=args.path_size,
        openslide_level=args.openslide_level,
    )