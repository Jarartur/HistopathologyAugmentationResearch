from pathlib import Path

from libhaa.base.config import ANNOTATION_EXTs
from libhaa.base.config import WSI_EXTs
from libhaa.base.io import Annotation, ImageWSI, Artifact, AnnotationCollection, AnnotatedWSI


def match_ext2class(path: Path) -> ImageWSI | AnnotationCollection:
    """
    Match a file extension to the corresponding class.
    """
    if path.suffix in WSI_EXTs:
        return ImageWSI.from_file(path=path)
    elif path.suffix in ANNOTATION_EXTs:
        return AnnotationCollection.from_file(path=path)
    else:
        raise ValueError(f"Unknown file extension: {path.suffix}")
    
def create_artifact(path: Path) -> Artifact:
    return Artifact.from_file(path, access='sequential')