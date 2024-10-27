import itertools
from pathlib import Path
from typing import Generator
from dataclasses import dataclass
from tqdm import tqdm
import warnings

import multiprocessing as mp

from libhaa.base.io import (
    AnnotatedWSI,
    ImageWSI,
    AnnotationCollection,
    ArtifactsCollection,
)
from libhaa.base.config import ARTIFACT_LIBRARY_ANNOTATION_EXT, WSI_EXTs
from libhaa.base.parsing import match_ext2class


def main(
    root_wsi: Path, root_segmentation: Path, artifact_collection: Path, save_root: Path, skip_existing: bool = True, root_annotation = None
) -> None:
    """
    Build a collection of artifact images from a directory of WSIs images with annotations.
    """
    # iterate over root directory
    Artifacts = ArtifactsCollection.from_folder(artifact_collection)
    pbar = tqdm(wsi_iterator(
        root_wsi=root_wsi, root_segmentation=root_segmentation, save_root=save_root, root_annotation=root_annotation
    ))
    for case in pbar:
        pbar.set_description(f"Processing {case.wsi_path.name}...")

        if case.save_path.is_file() and skip_existing:
            pbar.set_description(f"{case.wsi_path.name} already exists, skipping...")
            continue

        proc = mp.Process(target=execute_then_release, args=(case, Artifacts))
        proc.start()
        proc.join()


def execute_then_release(case, Artifacts):
    #this trick seems to have worked at least in part
    wsi = AnnotatedWSI.from_file(case.wsi_path, case.annotation_path, case.segmentation_path)

    if (not len(wsi.segmentation.annotations) >= 1) or (not len(wsi.segmentation.annotations[0].coordinates) >= 1): #HACK: delete later as we allow augmentation with no segmentations
        warnings.warn(f"No segmentation provided or the segmentation is empty for file {case.segmentation_path}")
        return

    save = True
    for artifact in Artifacts.iterate():
        try: #HACK: delete later as .augment_with modifies in place and potentially can lead to corruption
            wsi = wsi.augment_with(artifact)
        except Exception as e:
            print(e)
            print(f"Failed for: {case}")
            save = False
            break
    
    if save:
        wsi.to_file(case.save_path)

@dataclass
class Case:
    wsi_path: Path
    segmentation_path: Path
    save_path: Path
    annotation_path: Path|None


def wsi_iterator(
    root_wsi: Path,
    root_segmentation: Path,
    save_root: Path,
    root_annotation: Path|None,
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
        save_path = (save_root / relative_path).with_suffix(".tiff") #HACK: delete later as currently we save exclusevily in tiffs
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if root_annotation is not None:
            annotation_path = (root_annotation / relative_path).with_suffix(
            ARTIFACT_LIBRARY_ANNOTATION_EXT
        )
            if not annotation_path.is_file():
                annotation_path = None
        else:
            annotation_path = None

        yield Case(wsi_path, segmentation_path, save_path, annotation_path)

def cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a dataset.", epilog="Artur Jurgas."
    )
    parser.add_argument("--wsi-path", type=Path, required=True, help="Path to WSIs.")
    parser.add_argument(
        "--segmentations-path", type=Path, required=True, help="Path to annotations."
    )
    parser.add_argument(
        "--artifact-collection-path",
        type=Path,
        required=True,
        help="Path to the previously extracted artifact collection.",
    )
    parser.add_argument(
        "--save-path", type=Path, required=True, help="Path to save to."
    )
    parser.add_argument(
        "--root-annotation",
        type=Path,
        required=False,
        default=None,
        help="""
If specified, the annotation path will be `root-annotation/wsi-folder/wsi-name`.
If not specified the dataset will not include annotations.""",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip already existing files. Usefull for re-runs or appending new cases.",
    )
    args = parser.parse_args()

    main(
        root_wsi=args.wsi_path,
        root_segmentation=args.segmentations_path,
        artifact_collection=args.artifact_collection_path,
        save_root=args.save_path,
        skip_existing=args.skip_existing,
        root_annotation=args.root_annotation,
    )