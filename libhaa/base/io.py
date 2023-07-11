from __future__ import annotations
from dataclasses import dataclass
import itertools
import random
import uuid
import warnings
from tqdm import tqdm

import numpy as np
from libhaa.base.config import (
    ANNOTATION_SEGMENTATION_GROUP,
    ARTIFACT_LIBRARY_ANNOTATION_EXT,
    ARTIFACT_TYPES_MAX_NUMBERS,
    IMAGE_DATA_TYPE,
    LABELS_SAVING_FORMATS,
    LEGAL_ARTIFACT_TYPES,
    LEGAL_BLENDING_PLACES,
    YOLO_DATASET_ANNOTATION_EXT,
    WSI_EXTs,
)
from libhaa.base.loaders import (
    AbsoluteCoordinate,
    CornerCoordinate,
    NormalizedCoordinate,
    Scaling,
    augment_wsi,
    create_segmentation_mask,
    cut_artifact,
    cut_patch,
    dispatch_blending_method,
    dispatch_blending_place,
    get_affine_transform,
    get_annotations_in_patch,
    get_padding_size,
    get_sampled_corner,
    get_wsi_dims,
    load_annotations,
    load_image,
    pad_wsi,
    point_in_tissue,
    sample_coord,
    sample_point,
    save_annotations_xml,
    save_annotations_yolo,
    save_image,
    Annotation,
    scale_artifact_image,
    transform_artifact,
    # scale_artifact,
    point_in_artifact,
    point_in_background,
)

from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    TypeVar,
    get_args,
)

# TODO: replace @properties with class decorator or __getattr__(self, name)
# current @property-based implementation gives us nice type hints, but produces a lot of boilerplate
# __getattribute__ works but is not type-hintable very well
# - maybe just disregard it and require it at __init__ time?


class ImageWSI:
    """
    A wrapper class for backend Image object and its metadata.
    Used for WSIs and as a base class for Artifacts.

    `.data` should contain all metadata needed to properly save.
    """

    Self = TypeVar(
        "Self", bound="ImageWSI"
    )  # HACK: change when going to python 3.11 to `from typing import Self`

    def __init__(self) -> None:
        self._data: Optional[IMAGE_DATA_TYPE] = None
        self._ext: Optional[str] = None
        # self._segmentation: Optional[List[Annotation]] = None # TODO: make this a lazy evaluation from annotaiton to segmentation mask

    @property
    def data(self) -> IMAGE_DATA_TYPE:
        if not self._data:
            raise ValueError("Image data not loaded.")
        return self._data

    # @property
    # def segmentation(self) -> Optional[IMAGE_DATA_TYPE]:
    #     return self._segmentation

    @property
    def ext(self) -> str:
        if not self._ext:
            raise ValueError("Image ext not loaded.")
        return self._ext

    @classmethod
    def from_obj(cls, obj: ImageWSI) -> ImageWSI:
        """
        Create new Image object from another Image object.
        """
        new = cls()
        new._data = obj._data
        new._ext = obj._ext
        return new

    @classmethod
    def from_file(
        cls: type[Self], path: Path, openslide_level: Optional[int] = None
    ) -> Self:
        new = cls()
        new._data = load_image(path, openslide_level)
        new._ext = path.suffix
        return new

    def to_file(self, path: Path) -> None:
        save_image(self.data, path)

    def augment_with(
        self: Self,
        artifact: Artifact,
        insertion_point: CornerCoordinate | Tuple[int, int],
    ) -> Self:
        """
        Augment the image with an artifact.
        """
        insertion_point = (
            insertion_point
            if isinstance(insertion_point, CornerCoordinate)
            else CornerCoordinate(*insertion_point)
        )
        self._data = augment_wsi(
            self.data,
            artifact.data,
            artifact.annotation,
            insertion_point,
            artifact.blending_method,
        )

        return self

    def get_patch(
        self: Self, point: CornerCoordinate, patch_size: CornerCoordinate
    ) -> Self:
        """
        Get a patch from the image.
        """
        new = type(self)()
        new._data = cut_patch(self.data, point, patch_size)
        new._ext = self.ext
        return new

    def pad(self, pad_size: Tuple[int, int, int, int]) -> ImageWSI:
        """
        Pad the image data with zeros.
        pad_size: (left, top, right, bottom)
        """
        new = type(self)()
        new._data = pad_wsi(self.data, pad_size)
        new._ext = self.ext
        return new

class AnnotationCollection:
    """
    A collection of annotations for a single WSI.
    """

    Self = TypeVar(
        "Self", bound="AnnotationCollection"
    )  # HACK: change when going to python 3.11 to `from typing import Self`

    def __init__(self) -> None:
        self._annotations: Optional[List[Annotation]] = None

    @property
    def annotations(self) -> List[Annotation]:
        if self._annotations is None:
            raise ValueError("Image annotations not loaded.")
        return self._annotations

    @annotations.setter
    def annotations(self, annotations: List[Annotation]) -> None:
        self._annotations = annotations

    @classmethod
    def from_file(cls, path: Path) -> AnnotationCollection:
        new = cls()
        new._annotations = load_annotations(path)
        return new

    @classmethod
    def new_empty(cls: type[Self]) -> Self:
        new = cls()
        new._annotations = []
        return new

    def iterate_annotations(self) -> Generator[Annotation, None, None]:
        for annotation in self.annotations:
            yield annotation

    def to_file(self, path: Path) -> None:
        save_annotations_xml(self.annotations, path)

    def to_file_yolo(self, path: Path, patch_size: Tuple[int, int]) -> None:
        annotations = [annotation.get_yolo_label(patch_size) for annotation in self.annotations]
        save_annotations_yolo(annotations, path)

    def get_patch(self, point: CornerCoordinate, patch: ImageWSI, patch_shape: Tuple[int, int]) -> AnnotationCollection:
        annotations = get_annotations_in_patch(self.annotations, point, patch.data)
        annotations = [
            annotation.get_cropped_annotation(
                point, (patch.data.width, patch.data.height)
            )
            for annotation in annotations
        ]
        annotations = [
            annotation.recalculate_coordinates_after_crop(patch.data, point, patch_shape)
            for annotation in annotations
        ]
        new = AnnotationCollection()
        new._annotations = annotations
        return new
    
    def pad(self, pad_size: Tuple[int, int]) -> AnnotationCollection:
        annotations = [
            annotation.get_moved_annotation(CornerCoordinate(*pad_size))
            for annotation in self.annotations
        ]
        new = AnnotationCollection()
        new._annotations = annotations
        return new
    
    def filter_to_known_types(self) -> AnnotationCollection:
        annotations = [annotation for annotation in self.annotations if annotation.group in get_args(LEGAL_ARTIFACT_TYPES)]
        new = AnnotationCollection()
        new._annotations = annotations
        return new

    def scale(self: type[Self], scaling: Scaling) -> Self:
        annotations = [
            annotation.get_scaled_by_factor_annotation(scaling)
            for annotation in self.annotations
        ]
        new = AnnotationCollection()
        new._annotations = annotations
        return new


class SegmentationAnnotation(AnnotationCollection):
    @classmethod
    def from_file(cls, path: Path) -> SegmentationAnnotation:
        new = cls()
        new._annotations = [
            annotation
            for annotation in load_annotations(path)
            if annotation.group == ANNOTATION_SEGMENTATION_GROUP
        ]
        return new
    

@dataclass
class AnnotatedWSI:
    wsi: ImageWSI
    annotations: AnnotationCollection = AnnotationCollection.new_empty()
    segmentation: SegmentationAnnotation = SegmentationAnnotation.new_empty()

    @classmethod
    def from_file(
        cls,
        path_wsi: Path,
        path_annotation: Optional[Path],
        path_segmentation: Optional[Path],
        openslide_level: Optional[int] = None,
    ) -> AnnotatedWSI:
        
        wsi = ImageWSI.from_file(path_wsi, openslide_level)


        annotations = (
            AnnotationCollection.new_empty()
            if path_annotation is None
            else AnnotationCollection.from_file(path_annotation)
        )

        segmentation = (
            SegmentationAnnotation.new_empty()
            if path_segmentation is None
            else SegmentationAnnotation.from_file(path_segmentation)
        )

        if openslide_level is not None:
            width, height = get_wsi_dims(path_wsi)
            scaling = Scaling(
                height = wsi.data.height / height,
                width = wsi.data.width / width,
            )
            annotations = annotations.scale(scaling)
            segmentation = segmentation.scale(scaling)

        return cls(wsi, annotations, segmentation)

    def iterate_artifacts(self) -> Generator[Artifact, None, None]:
        for annotation in self.annotations.iterate_annotations():
            if annotation.group in get_args(LEGAL_ARTIFACT_TYPES):
                try:
                    yield Artifact.from_wsi_annotation(wsi=self.wsi, annotation=annotation)
                except Exception as e:
                    print(f"Error while loading artifact: {annotation.group}, {annotation.name}, skipping...")
                    continue

    def translate_insertion_point(
        self, patch_shape: Tuple[int, int], insertion_point: NormalizedCoordinate
    ) -> CornerCoordinate:
        corner = get_sampled_corner(
            (self.wsi.data.width, self.wsi.data.height),
            (patch_shape[0], patch_shape[1]),
            insertion_point,
        )

        return corner

    def to_file(self, path: Path) -> None:
        self.wsi.to_file(path)
        self.annotations.to_file(path.with_suffix(ARTIFACT_LIBRARY_ANNOTATION_EXT))

    def augment_with(
        self, artifact: Artifact, coordinate: Optional[Tuple[float, float]] = None
    ) -> AnnotatedWSI:
        if coordinate is None:
            insertion_point = sample_coord(
                wsi_image=self.wsi.data,
                segmentation_annotation=self.segmentation.annotations,
                blending_place=artifact.blending_place,
            )
        else:
            insertion_point = NormalizedCoordinate(*coordinate)

        artifact = artifact.scale_artifact(self.wsi)

        affine_matrix = get_affine_transform(artifact.data, 10, 1.0)
        artifact = artifact.transform_artifact(affine_matrix)

        insertion_point = self.translate_insertion_point(
            (artifact.data.width, artifact.data.height), insertion_point
        )

        self.wsi = self.wsi.augment_with(artifact, insertion_point)
        annotation = artifact.annotation
        annotation = annotation.get_moved_annotation(insertion_point)
        annotation = annotation.get_cropped_annotation(
            CornerCoordinate(left=0, top=0), (self.wsi.data.width, self.wsi.data.height)
        )

        self.annotations.annotations += [annotation]
        return self

    def cutout_patches(
        self,
        patch_size: Tuple[int, int],
        root_save_path: Path,
        num_background: int,
        num_tissue: int,
        num_artifact: int,
        format: LABELS_SAVING_FORMATS = 'asap',
    ):
        # root_save_path.mkdir(parents=True, exist_ok=False) #HACK: add later
        # (root_save_path/"debug").mkdir(parents=True, exist_ok=True) #HACK: add later
        pbar = tqdm(total=num_background+num_tissue+num_artifact)

        pbar.set_description("Limiting artifacts to known types...")
        known_artifact_annotations = self.annotations.filter_to_known_types()

        self._pad_size = get_padding_size(patch_size)
        self._padded_wsi = self.wsi.pad(self._pad_size)
        self._padded_annotations = known_artifact_annotations.pad(self._pad_size[:2])

        pbar.set_description("Sampling background...")
        for point in sample_point(
            point_in_background,
            num_background,
            self.wsi.data,
            annotations=self.segmentation.annotations,
            exclusions=known_artifact_annotations.annotations,
            limit_sampling_range=False,
        ):
            self.save_cut(point, patch_size, root_save_path/"background", format)
            pbar.update(1)

        pbar.set_description("Sampling tissue...")
        for point in sample_point(
            point_in_tissue,
            num_tissue,
            self.wsi.data,
            annotations=self.segmentation.annotations,
            exclusions=known_artifact_annotations.annotations,
            limit_sampling_range=True,
        ):
            self.save_cut(point, patch_size, root_save_path/'tissue', format)
            pbar.update(1)


        pbar.set_description("Sampling known artifacts...")
        for point in sample_point(
            point_in_artifact,
            num_artifact,
            self.wsi.data,
            annotations=known_artifact_annotations.annotations,
            limit_sampling_range=True,
        ):
            self.save_cut(point, patch_size, root_save_path/'artifact', format)
            pbar.update(1)

        pbar.close()

    def save_cut(
        self, point: NormalizedCoordinate, patch_size: Tuple[int, int], root_path: Path, format: LABELS_SAVING_FORMATS = 'asap'
    ):
        root_path.mkdir(parents=True, exist_ok=True)

        unique_filename = str(uuid.uuid4().hex)
        insertion_point = self.translate_insertion_point(patch_size, point)
        insertion_point.left += self._pad_size[0]
        insertion_point.top += self._pad_size[1]

        patch = self._padded_wsi.get_patch(insertion_point, patch_size)
        annotations = self._padded_annotations.get_patch(insertion_point, patch, patch_size)

        patch.to_file((root_path / unique_filename).with_suffix(self.wsi.ext))

        if format == 'yolo' or format == 'both':
            annotations.to_file_yolo(
            (root_path / unique_filename).with_suffix(YOLO_DATASET_ANNOTATION_EXT),
            patch_size
            )
        if format == 'asap' or format == 'both':
            annotations.to_file(
                (root_path / unique_filename).with_suffix(ARTIFACT_LIBRARY_ANNOTATION_EXT)
            )


class Artifact(ImageWSI):
    """
    Class for artifacts, which are images stored in an artifact library or extracted from WSIs.
    """

    def __init__(self) -> None:
        super().__init__()
        self._annotation: Optional[Annotation] = None
        self._blending_method: Optional[Callable] = None
        self._blending_place: Optional[LEGAL_BLENDING_PLACES] = None

    @property
    def annotation(self) -> Annotation:
        if not self._annotation:
            raise ValueError("Artifact annotation not loaded.")
        return self._annotation

    @property
    def blending_method(self) -> Callable:
        if not self._blending_method:
            raise ValueError("Artifact blending method not specified.")
        return self._blending_method

    @property
    def blending_place(self) -> LEGAL_BLENDING_PLACES:
        if not self._blending_place:
            raise ValueError("Artifact blending place not specified.")
        return self._blending_place

    @property
    def group(self) -> str:
        if not self._annotation:
            raise ValueError("Artifact group not loaded.")
        return self._annotation.group

    def to_file(self, path: Path) -> None:
        save_image(self.data, path.with_suffix(self.ext))
        save_annotations_xml(
            [self.annotation], path.with_suffix(ARTIFACT_LIBRARY_ANNOTATION_EXT)
        )

    def transform_artifact(self, affine_matrix: np.ndarray) -> Artifact:
        new = Artifact()

        image_augmented, annotation_augmented = transform_artifact(
            self.data, self.annotation, affine_matrix
        )

        new._data = image_augmented
        new._annotation = annotation_augmented
        new._ext = self.ext
        new._blending_method = self.blending_method
        new._blending_place = self.blending_place
        return new

    def scale_artifact(self, wsi: ImageWSI) -> Artifact:
        """
        Scale the artifact to the size of the WSI.
        """
        new = Artifact()

        image_scaled = scale_artifact_image(self.data, wsi.data)
        annotation_scaled = self.annotation.get_scaled_annotation(wsi.data, self.data)
        new._data = image_scaled
        new._annotation = annotation_scaled
        new._ext = self.ext
        new._blending_method = self.blending_method
        new._blending_place = self.blending_place
        return new

    @classmethod
    def from_wsi_annotation(cls, wsi: ImageWSI, annotation: Annotation) -> Artifact:
        new = cls()
        new_data, corner, patch_size = cut_artifact(wsi.data, annotation.coordinates)
        new_annotation = annotation.recalculate_coordinates_after_crop(new_data, corner, patch_size)

        new._data = new_data
        new._annotation = new_annotation
        new._ext = wsi.ext

        # new._segmentation = create_segmentation_mask(
        #     new_data, new_annotation.coordinates
        # ) #NOTE: deprecated

        new._blending_method = dispatch_blending_method(new_annotation.group)
        new._blending_place = dispatch_blending_place(new_annotation.group)

        return new

    @classmethod
    def from_file(cls, path: Path) -> Artifact:
        new = cls()
        new._data = load_image(path)
        annotation = load_annotations(path.with_suffix(ARTIFACT_LIBRARY_ANNOTATION_EXT))
        assert len(annotation) == 1, "Artifact should have exactly one annotation."
        new._annotation = annotation[0]
        new._ext = path.suffix
        #TODO: replace upper section with super().from_file(path)
        new._blending_method = dispatch_blending_method(new._annotation.group)
        new._blending_place = dispatch_blending_place(new._annotation.group)
        return new


class ArtifactsCollection:
    def __init__(self) -> None:
        super().__init__()
        self._all_artifacts: List[Artifact] = []
        self._artifacts_dict: Dict[LEGAL_ARTIFACT_TYPES, List[Artifact]] = {
            key: [] for key in get_args(LEGAL_ARTIFACT_TYPES)
        }

    @property
    def all_artifacts(self) -> List[Artifact]:
        if not self._all_artifacts:
            raise ValueError("Artifacts not loaded.")
        return self._all_artifacts

    @property
    def artifacts_dict(self) -> Dict[LEGAL_ARTIFACT_TYPES, List[Artifact]]:
        if not self._artifacts_dict:
            raise ValueError("Artifacts not loaded.")
        return self._artifacts_dict

    @classmethod
    def from_folder(cls, path: Path) -> ArtifactsCollection:
        new = cls()
        new._all_artifacts = [
            Artifact.from_file(file)
            for file in itertools.chain(*[path.rglob(f"*{ext}") for ext in WSI_EXTs])
        ]
        new._artifacts_dict = {
            key: [artifact for artifact in new._all_artifacts if artifact.group == key]
            for key in get_args(LEGAL_ARTIFACT_TYPES)
        }
        return new

    def iterate(self) -> Generator[Artifact, None, None]:
        for key in self.artifacts_dict:
            random.shuffle(self.artifacts_dict[key])
            num_max_artifact = random.randint(0, ARTIFACT_TYPES_MAX_NUMBERS[key])
            for i, artifact in enumerate(self.artifacts_dict[key]):
                if i >= num_max_artifact:
                    break
                yield artifact
