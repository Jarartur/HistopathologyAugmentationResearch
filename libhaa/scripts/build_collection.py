import itertools
from pathlib import Path
from typing import Generator, Tuple, List

from libhaa.base.io import AnnotationCollection, Artifact, ImageWSI, AnnotatedWSI
from libhaa.base.config import WSI_EXTs, ANNOTATION_EXTs
import uuid

from tqdm import tqdm

def main(
    root_wsi: Path, root_annotations: Path, root_save: Path, relative_paths: bool = False
) -> None:
    """
    Build a collection of artifact images from a directory of WSIs images with annotations.
    """
    # iterate over root directory
    pbar = tqdm(case_iterator(
        root_wsi=root_wsi,
        root_annotations=root_annotations,
        relative_paths=relative_paths,
    ))
    for (wsi_path, annotations_path) in pbar:
        pbar.set_description(f"Processing {wsi_path.name}")
        # print(wsi_path)#HACK: delete later

        image = ImageWSI.from_file(path=wsi_path)
        annotations = AnnotationCollection.from_file(path=annotations_path)
        case = AnnotatedWSI(wsi=image, annotations=annotations)

        artifacts = extract_artifacts(case)
        save_artifacts(artifacts, root_save)


def save_artifacts(artifacts: List[Artifact], root_save: Path) -> None:
    for artifact in artifacts:
        unique_filename = str(uuid.uuid4().hex)
        (root_save/artifact.group).mkdir(parents=True, exist_ok=True)
        artifact.to_file((root_save/artifact.group/unique_filename).with_suffix(artifact.ext))

def extract_artifacts(case: AnnotatedWSI) -> List[Artifact]:

    artifacts = []
    for artifact in case.iterate_artifacts():
        artifacts += [artifact]

    return artifacts


def case_iterator(
    root_wsi: Path, root_annotations: Path, relative_paths: bool
) -> Generator[Tuple[Path, Path], None, None]:
    """
    Iterate over a directory of WSIs and match it's annotation
    """
    # chain all wsi possible extensions together into an iterator
    for wsi_image in itertools.chain(*[root_wsi.rglob(f"*{ext}") for ext in WSI_EXTs]):

        annotations = get_annotation_path(wsi_image, root_wsi, root_annotations, relative_paths)
        
        yield (wsi_image, annotations)


def get_annotation_path(wsi: Path, wsi_root: Path, root_annotations: Path, relative_paths: bool) -> Path:
    """
    Get the path to the annotation file for a given wsi.
    """
    for annotation_ext in ANNOTATION_EXTs:
        if not relative_paths:
            annotation = (root_annotations / wsi.stem).with_suffix(annotation_ext)
        else:
            # get annotation path relative to root_annotations (the same folder structure as wsi)
            annotation = root_annotations / wsi.relative_to(
                wsi_root
            ).with_suffix(annotation_ext)
        
        if annotation.is_file():
            return annotation
    else:
        raise ValueError(f"Annotation not found for {wsi} with extensions {ANNOTATION_EXTs}")


def cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build artifact collection", epilog="Artur Jurgas."
    )
    parser.add_argument("--wsi-path", type=Path, required=True, help="Path to WSIs.")
    parser.add_argument(
        "--annotations-path", type=Path, required=True, help="Path to annotations."
    )
    parser.add_argument(
        "--save-path", type=Path, required=True, help="Path to save to."
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        default=False,
        help="""Get annotation path relative to the WSI being processed. 
If false the annotation path will be `annotations-path/wsi-name`. 
If true it will be `annotations-path/wsi-folder/wsi-name`""",
    )
    args = parser.parse_args()

    main(
        root_wsi=args.wsi_path,
        root_annotations=args.annotations_path,
        root_save=args.save_path,
        relative_paths=args.relative_paths,
    )