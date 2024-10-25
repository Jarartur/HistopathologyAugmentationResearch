# TODO:
# - refactor into more files and smaller categorized functions

from __future__ import annotations
from collections import namedtuple
import colorsys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Callable, Generator, List, Optional, Tuple
import warnings
from PIL import Image as PILimage
from PIL import ImageDraw
import numpy as np
import xml.etree.ElementTree as ET
import math
import cv2

from sys import platform
if platform == "win32":
    import os
    vipsbin = r'c:\ProgramData\libvips\bin'
    os.environ['PATH'] = os.pathsep.join((vipsbin, os.environ['PATH']))
import pyvips

from libhaa.base.config import (
    ANNOTATION_GROUP_XML_NAME,
    ANNOTATION_XML_NAME,
    ARTIFACT_TYPES_ENCODING,
    LEGAL_ARTIFACT_TYPES,
    LEGAL_BLENDING_PLACES,
    BLENDING_METHODS_DISPATCH,
    BLENDING_PLACES_DISPATCH,
    AUGMENTATION_PARAMS,
    SAMPLING_MAX_ITER
)
import matplotlib.path as mpltPath


@dataclass(slots=True)
class ValueRange:
    min: float | int
    max: float | int

Scaling = namedtuple("Scaling", ["height", "width"])


@dataclass(order=True, slots=True)
class AbsoluteCoordinate:
    """
    x,y should be in absolute [0..inf] values
    x: horizontal coodinate
    y: vertical coordinate
    """

    x: Annotated[float, ValueRange(0, math.inf)] = field(compare=False)
    y: Annotated[float, ValueRange(0, math.inf)] = field(compare=False)
    order: int = field(compare=True, default=0)

    @classmethod
    def parse_coordinates(cls, coordinates: ET.Element) -> List[AbsoluteCoordinate]:
        coordinates_list = []
        for coordinate in coordinates:
            coordinates_list += [
                cls(
                    x=float(coordinate.attrib["X"]),
                    y=float(coordinate.attrib["Y"]),
                    order=int(coordinate.attrib["Order"]),
                )
            ]

        return sorted(coordinates_list)


@dataclass(order=True, slots=True)
class NormalizedCoordinate:
    """
    x,y should be in normalized [0..1] values
    """

    x: Annotated[float, ValueRange(0, 1)] = field(compare=False)
    y: Annotated[float, ValueRange(0, 1)] = field(compare=False)
    order: int = field(compare=True, default=0)


@dataclass(order=True, slots=True)
class CornerCoordinate:
    left: Annotated[int, ValueRange(-math.inf, math.inf)]
    top: Annotated[int, ValueRange(-math.inf, math.inf)]

@dataclass
class YoloLabel:
    class_id: int
    x: float
    y: float
    width: float
    height: float

@dataclass(slots=True)
class Annotation:
    name: str
    group: str
    shape: str
    coordinates: List[AbsoluteCoordinate]

    def recalculate_coordinates_after_crop(
        self, cropped_artifact: pyvips.Image, corner: CornerCoordinate, wsi_shape: Tuple[int, int]
    ) -> Annotation:
        """
        Assumes that coordinates are in the whole patch range.
        """
        self.coordinates = denormalize_coordinates_to_image(
            cropped_artifact,
            normalize_coordinates_to_patch(self.coordinates, corner, wsi_shape),
        )

        return self
    
    def get_yolo_label(self, patch_size: Tuple[int, int]) -> YoloLabel:

        left, top, right, bottom = get_minmax_coords(self.coordinates)
        width, height = patch_size[0], patch_size[1]

        return YoloLabel(
            class_id = ARTIFACT_TYPES_ENCODING[self.group],
            x = ((left + right) / 2) / width,
            y = ((top + bottom) / 2) / height,
            width = (right - left) / width,
            height = (bottom - top) / height,
        )

    def get_moved_annotation(self, corner: CornerCoordinate) -> Annotation:
        coordinates = [
            AbsoluteCoordinate(
                x=coordinate.x + corner.left,
                y=coordinate.y + corner.top,
                order=coordinate.order,
            )
            for coordinate in self.coordinates
        ]

        return type(self)(
            name=self.name,
            group=self.group,
            shape=self.shape,
            coordinates=coordinates,
        )

    def get_cropped_annotation(self, corner: CornerCoordinate, wsi_shape: Tuple[int, int]) -> Annotation:
        coordinates = crop_annotation(self.coordinates, corner, wsi_shape)

        return type(self)(
            name=self.name,
            group=self.group,
            shape=self.shape,
            coordinates=coordinates,
        )
    
    def get_scaled_by_factor_annotation(self, factor: Scaling) -> Annotation:
        coordinates = [
            AbsoluteCoordinate(
                x=coordinate.x * factor.width,
                y=coordinate.y * factor.height,
                order=coordinate.order,
            )
            for coordinate in self.coordinates
        ]

        return type(self)(
            name=self.name,
            group=self.group,
            shape=self.shape,
            coordinates=coordinates,
        )

    def get_scaled_annotation(
        self, wsi_image: pyvips.Image, artifact_image: pyvips.Image
    ) -> Annotation:
        x_scale, y_scale = get_artifact_scaling(artifact_image, wsi_image)

        coordinates = [
            AbsoluteCoordinate(
                x=coordinate.x * x_scale,
                y=coordinate.y * y_scale,
                order=coordinate.order,
            )
            for coordinate in self.coordinates
        ]

        return type(self)(
            name=self.name,
            group=self.group,
            shape=self.shape,
            coordinates=coordinates,
        )

    def get_transformed_annotation(
        self,
        affine_matrix: np.ndarray,
    ) -> Annotation:
        points = [transform_coord(coord, affine_matrix) for coord in self.coordinates]

        transformed_coordinates = [
            AbsoluteCoordinate(
                x=point[0],
                y=point[1],
                order=coordinate.order,
            )
            for point, coordinate in zip(points, self.coordinates)
        ]

        new_annotation = type(self)(
            name=self.name,
            group=self.group,
            shape=self.shape,
            coordinates=transformed_coordinates,
        )
        return new_annotation


def transform_coord(
    coord: AbsoluteCoordinate, matrix: np.ndarray
) -> Tuple[float, float]:
    out = [[coord.x, coord.y, 1]] @ matrix.T
    out = out[0, :-1].tolist()
    return out


def get_annotations_in_patch(
    annotations: List[Annotation], point: CornerCoordinate, patch: pyvips.Image
) -> List[Annotation]:
    
    annotations_in_patch = []
    patch_size = (patch.width, patch.height)

    for annotation in annotations:
        if annotation_intersects_patch(annotation, point, patch_size):
            annotations_in_patch += [annotation]

    return annotations_in_patch


# def get_annotation_patch_coordinates(annotation: Annotation, point: CornerCoordinate, patch_size: Tuple[int, int]) -> List[AbsoluteCoordinate]:
#     coordinates = []
#     for coordinate in annotation.coordinates:

#     return coordinates


def annotation_intersects_patch(
    annotation: Annotation, point: CornerCoordinate, patch_size: Tuple[int, int]
) -> bool:
    annotation_path = mpltPath.Path(
        [(coord.x, coord.y) for coord in annotation.coordinates]
    )
    patch_path = mpltPath.Path(
        [
            (point.left, point.top),
            (point.left + patch_size[0], point.top),
            (point.left + patch_size[0], point.top + patch_size[1]),
            (point.left, point.top + patch_size[1]),
        ]
    )
    return annotation_path.intersects_path(patch_path, filled=True)


def load_image(path: Path, openslide_level: Optional[int] = None) -> pyvips.Image:
    if openslide_level is None:
        img = pyvips.Image.new_from_file(str(path), access="sequential")
        if img.bands > 3:
            img = img.flatten(background=255)
        return img
    else:
        img = pyvips.Image.openslideload(str(path), level=openslide_level)
        if img.bands > 3:
            img = img.flatten(background=255)
        return img

def get_wsi_dims(wsi_path: Path) -> tuple[int, int]:
    pyvips_img = pyvips.Image.new_from_file(wsi_path)
    return pyvips_img.width, pyvips_img.height


# def load_metadata(path: Path) -> Dict:
#     image = pyvips.Image.new_from_file(str(path))
#     metadata = {}
#     for key in image.get_fields():
#         metadata[key] = image.get(key)
#     metadata["name"] = path.name
#     return metadata


def save_image(image: pyvips.Image, path: Path, **kwargs) -> None:
    # Hardcoded for now, TODO: dispatch saving method
    image.tiffsave(
        path.with_suffix(".tiff"),
        tile=True,
        pyramid=True,
        compression="jpeg",
        Q=75,
        **kwargs,
    )


# def save_annotation_xml(annotation: Annotation, path: Path) -> None:
#     root = ET.Element("Annotations")
#     annotation_xml = ET.SubElement(root, ANNOTATION_XML_NAME)
#     annotation_xml.attrib["Name"] = annotation.name
#     annotation_xml.attrib["PartOfGroup"] = annotation.group
#     annotation_xml.attrib["Type"] = annotation.shape
#     annotation_xml.attrib["Color"] = "#F4FA58"

#     coordinates_xml = ET.SubElement(annotation_xml, "Coordinates")
#     for coordinate in annotation.coordinates:
#         coordinate_xml = ET.SubElement(coordinates_xml, "Coordinate")
#         coordinate_xml.attrib["Order"] = str(coordinate.order)
#         coordinate_xml.attrib["X"] = str(coordinate.x)
#         coordinate_xml.attrib["Y"] = str(coordinate.y)

#     tree = ET.ElementTree(root)
#     ET.indent(tree, '\t')
#     tree.write(path, encoding="UTF-8", xml_declaration=True)

def save_annotations_yolo(labels: List[YoloLabel], path: Path) -> None:
    with open(path, "w") as f:
        for label in labels:
            f.write(
                f"{label.class_id} {label.x} {label.y} {label.width} {label.height}\n"
            )

def save_annotations_xml(annotations: List[Annotation], path: Path) -> None:
    # TODO: test

    groups = [annotation.group for annotation in annotations]
    groups_unique = list(set(groups))
    colors = iter(get_N_HexCol(len(groups_unique)))

    root = ET.Element("ASAP_Annotations")

    root_annotations = ET.SubElement(root, "Annotations")
    for annotation in annotations:
        annotation_xml = ET.SubElement(root_annotations, ANNOTATION_XML_NAME)
        annotation_xml.attrib["Name"] = annotation.name
        annotation_xml.attrib["PartOfGroup"] = annotation.group
        annotation_xml.attrib["Type"] = annotation.shape
        annotation_xml.attrib["Color"] = "#F4FA58"

        coordinates_xml = ET.SubElement(annotation_xml, "Coordinates")
        for coordinate in annotation.coordinates:
            coordinate_xml = ET.SubElement(coordinates_xml, "Coordinate")
            coordinate_xml.attrib["Order"] = str(coordinate.order)
            coordinate_xml.attrib["X"] = str(coordinate.x)
            coordinate_xml.attrib["Y"] = str(coordinate.y)

    root_groups = ET.SubElement(root, "AnnotationGroups")
    for group in groups_unique:
        group_xml = ET.SubElement(root_groups, ANNOTATION_GROUP_XML_NAME)
        group_xml.attrib["Name"] = group
        group_xml.attrib["PartOfGroup"] = "None"
        group_xml.attrib["Color"] = next(colors)
        attr = ET.SubElement(group_xml, "Attributes")

    tree = ET.ElementTree(root)
    ET.indent(tree, "\t")
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def get_N_HexCol(N=5):
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append("#%02x%02x%02x" % tuple(rgb))
    return hex_out


def load_annotations(path: Path) -> List[Annotation]:
    root = get_xml_tree(path)
    annotation_list = []

    for annotation in root.iter(ANNOTATION_XML_NAME):
        annotation = Annotation(
            name=annotation.attrib["Name"],
            group=annotation.attrib["PartOfGroup"].lower(),
            shape=annotation.attrib["Type"],
            coordinates=AbsoluteCoordinate.parse_coordinates(annotation[0]),
        )

        annotation_list += [annotation]

    return annotation_list


# def load_annotation(path: Path) -> Annotation:
#     root = get_xml_tree(path)

#     annotation = Annotation(
#         name=root[0].attrib["Name"],
#         group=root[0].attrib["PartOfGroup"],
#         shape=root[0].attrib["Type"],
#         coordinates=AbsoluteCoordinate.parse_coordinates(root[0][0]),
#     )

#     return annotation


def get_xml_tree(path: Path) -> ET.Element:
    tree = ET.parse(path)
    root = tree.getroot()
    return root


def cut_artifact(
    image: pyvips.Image,
    coordinates: List[AbsoluteCoordinate],
) -> Tuple[pyvips.Image, CornerCoordinate, Tuple[int, int]]:
    left, top, right, bottom = get_minmax_coords(coordinates)

    width = right - left
    height = bottom - top

    args = [math.floor(coord) for coord in [left, top, width, height]]

    if left+width > image.width or top+height > image.height:
        warnings.warn(f"""Artifact annotation out of bounds, for image {image.width}x{image.height}
and artifact {left}x{top} x {left+width}x{top+height}
and minmax {left}x{top} x {right}x{bottom}""")

    return image.crop(*args), CornerCoordinate(*args[:2]), (args[2:])


def get_affine_transform(
    artifact_img: pyvips.Image, rotation: float, scale: float
) -> np.ndarray:
    
    center = (artifact_img.width / 2, artifact_img.height / 2)

    scale_params = AUGMENTATION_PARAMS['scale']
    rotation_params = AUGMENTATION_PARAMS['rotation']

    scale = np.random.uniform(scale_params[0], scale_params[1])
    rotation = np.random.uniform(rotation_params[0], rotation_params[1])

    mat = np.eye(3)
    rot = cv2.getRotationMatrix2D(center, rotation, scale)
    mat[0:2, :] = rot
    return mat


def get_minmax_coords(
    coordinates: List[AbsoluteCoordinate],
) -> Tuple[float, float, float, float]:
    sorted_x = sorted(coordinates, key=lambda coord: coord.x)
    sorted_y = sorted(coordinates, key=lambda coord: coord.y)

    left = sorted_x[0].x
    top = sorted_y[0].y

    right = sorted_x[-1].x
    bottom = sorted_y[-1].y

    return left, top, right, bottom


def normalize_coordinates_to_image(
    image: pyvips.Image, coordinates: List[AbsoluteCoordinate]
) -> List[NormalizedCoordinate]:
    return [
        NormalizedCoordinate(
            x=coordinate.x / image.width,
            y=coordinate.y / image.height,
            order=coordinate.order,
        )
        for coordinate in coordinates
    ]


def normalize_coordinates_to_patch(coordinates: List[AbsoluteCoordinate], corner: CornerCoordinate, patch_size: Tuple[int, int]) -> List[NormalizedCoordinate]:
    
    min_x = corner.left
    max_x = corner.left + patch_size[0]
    min_y = corner.top
    max_y = corner.top + patch_size[1]
    
    return [
        NormalizedCoordinate(
            x=(coordinate.x - min_x) / (max_x - min_x),
            y=(coordinate.y - min_y) / (max_y - min_y),
            order=coordinate.order,
        )
        for coordinate in coordinates
    ]

def normalize_coordinates_minmax(
    coordinates: List[AbsoluteCoordinate],
) -> List[NormalizedCoordinate]:
    """
    Normalize list of coordinates to 0..1 range using min,max from coordinate list itself.
    """
    sorted_x = sorted(coordinates, key=lambda coord: coord.x)
    sorted_y = sorted(coordinates, key=lambda coord: coord.y)

    min_x = sorted_x[0].x  # min(coordinate, key=lambda coord: coord.x).x
    max_x = sorted_x[-1].x  # max(coordinate, key=lambda coord: coord.x).x
    min_y = sorted_y[0].y  # min(coordinate, key=lambda coord: coord.y).y
    max_y = sorted_y[-1].y  # max(coordinate, key=lambda coord: coord.y).y

    return [
        NormalizedCoordinate(
            x=(coordinate.x - min_x) / (max_x - min_x),
            y=(coordinate.y - min_y) / (max_y - min_y),
            order=coordinate.order,
        )
        for coordinate in coordinates
    ]


def denormalize_coordinates_to_image(
    image: pyvips.Image, coordinates: List[NormalizedCoordinate]
) -> List[AbsoluteCoordinate]:
    return [
        AbsoluteCoordinate(
            x=coordinate.x * image.width,
            y=coordinate.y * image.height,
            order=coordinate.order,
        )
        for coordinate in coordinates
    ]


def augment_wsi(
    wsi_image: pyvips.Image,
    artifact_image: pyvips.Image,
    artifact_annotation: Annotation,
    insertion_point: CornerCoordinate,
    blending_method: Callable,
) -> pyvips.Image:
    artifact_mask = create_segmentation_mask(
        artifact_image, artifact_annotation.coordinates
    )

    left, top = insertion_point.left, insertion_point.top

    try:
        wsi_image_cropped, artifact_image_cropped, artifact_mask_cropped = crop_images(
            wsi_image, artifact_image, artifact_mask, (left, top)
        )
    except Exception as e:
        print(f"{artifact_annotation.group=}")
        raise e

    insert = blend_artifact(
        artifact_image_cropped,
        wsi_image_cropped,
        artifact_mask_cropped,
        blending_method,
    )

    wsi_image = wsi_image.insert(insert, max(0, left), max(0, top))

    return wsi_image


def transform_artifact(
    artifact_image: pyvips.Image,
    artifact_annotation: Annotation,
    affine_matrix: np.ndarray,
) -> tuple[pyvips.Image, Annotation]:
    artifact_image_augmented = augment_image(artifact_image, affine_matrix)

    center_scaled = (
        artifact_image.width / 2,
        artifact_image.height / 2,
    )
    center_augmented = (
        artifact_image_augmented.width / 2,
        artifact_image_augmented.height / 2,
    )

    affine_matrix[0, -1] += center_augmented[0] - center_scaled[0]
    affine_matrix[1, -1] += center_augmented[1] - center_scaled[1]

    artifact_annotation_augmented = artifact_annotation.get_transformed_annotation(
        affine_matrix
    )

    return artifact_image_augmented, artifact_annotation_augmented


def crop_annotation(
    coordinates: List[AbsoluteCoordinate], corner: CornerCoordinate, patch_shape: Tuple[int, int]
) -> List[AbsoluteCoordinate]:
    
    coords = []
    for coordinate in coordinates:

        if coordinate.x < corner.left or coordinate.x > corner.left + patch_shape[0]:
            continue
        if coordinate.y < corner.top or coordinate.y > corner.top + patch_shape[1]:
            continue

        coords += [AbsoluteCoordinate(coordinate.x, coordinate.y, coordinate.order)]

        # coords += [
        #     AbsoluteCoordinate(
        #         x=(
        #             corner.left
        #             if coordinate.x < corner.left
        #             else corner.left + patch_shape[0]
        #             if coordinate.x > corner.left + patch_shape[0]
        #             else coordinate.x
        #         ),
        #         y=(
        #             corner.top
        #             if coordinate.y < corner.top
        #             else corner.top + patch_shape[1]
        #             if coordinate.y > corner.top + patch_shape[1]
        #             else coordinate.y
        #         ),
        #         order=coordinate.order,
        #     )
        # ]

    for i, coord in enumerate(coords):
        coord.order = i

    if len(coords) == 0:
        # if all coordinates are outside of patch, add corners
        #WARNING: this assumes that the annotations are filtered and only those with overlap are passed into this function

        corners = [[0, 0], [0, patch_shape[1]], [patch_shape[0], 0], [patch_shape[0], patch_shape[1]]]
        for i, corner in enumerate(corners):
            coords += [AbsoluteCoordinate(corner[0], corner[1], i)]

    return coords


def crop_images(
    wsi_image: pyvips.Image,
    artifact_image: pyvips.Image,
    mask_image: pyvips.Image,
    corner: Tuple[int, int],
) -> Tuple[pyvips.Image, pyvips.Image, pyvips.Image]:
    left, top = corner

    artifact_coords = (
        art_left := 0 - min(0, left),
        art_top := 0 - min(0, top),
        art_width := min(artifact_image.width - art_left, wsi_image.width - max(0, left)),
        art_height := min(artifact_image.height - art_top, wsi_image.height - max(0, top)),
    )

    wsi_coords = (
        max(0, left),
        max(0, top),
        # min(wsi_image.width - left, artifact_image.width - art_left),
        # min(wsi_image.height - top, artifact_image.height - art_top),
        art_width,
        art_height
    )

    artifact_image_cropped = artifact_image.crop(*artifact_coords)
    mask_image_cropped = mask_image.crop(*artifact_coords)

    try:
        wsi_image_cropped = wsi_image.crop(*wsi_coords)
    except Exception as e:
        print(f"{corner=}")
        print(f"{wsi_coords=}")
        print(f"{artifact_coords=}")
        print(f"{wsi_image.width=}, {wsi_image.height}")
        print(f"{artifact_image.width=}, {artifact_image.height}")
        print(f"{artifact_image_cropped.width=}, {artifact_image_cropped.height}")
        print(e)
        raise e

    return wsi_image_cropped, artifact_image_cropped, mask_image_cropped


# def scale_artifact(
#     wsi_image: pyvips.Image,
#     artifact_image: pyvips.Image,
#     artifact_annotation: Annotation,
# ) -> Tuple[pyvips.Image, Annotation]:
#     artifact_image_scaled = scale_artifact_image(artifact_image, wsi_image)
#     artifact_annotation_scaled = artifact_annotation.get_augmented_annotation(
#         artifact_image_scaled, artifact_image, None
#     )

#     return artifact_image_scaled, artifact_annotation_scaled


def get_sampled_corner(
    wsi_shape: Tuple[float | int, float | int],
    artifact_shape: Tuple[float | int, float | int],
    insertion_point: NormalizedCoordinate,
) -> CornerCoordinate:
    """
    Get the top left corner of the sampled region.
    Input shapes from pyvips in the form of (width, height).

    Can return values outside of the wsi range.
    """

    center_x, center_y = (
        insertion_point.x * wsi_shape[0],
        insertion_point.y * wsi_shape[1],
    )

    left = center_x - artifact_shape[0] / 2
    top = center_y - artifact_shape[1] / 2

    corner = CornerCoordinate(round(left), round(top))

    return corner


def blend_artifact(
    artifact_image: pyvips.Image,
    wsi_image: pyvips.Image,
    artifact_mask: pyvips.Image,
    blending_method: Callable,
) -> pyvips.Image:
    artifact = artifact_image.numpy()
    artifact = artifact[..., :3] if artifact.shape[-1] == 4 else artifact
    wsi = wsi_image.numpy()
    wsi = wsi[..., :3] if wsi.shape[-1] == 4 else wsi

    if wsi.dtype == np.float64:
        wsi = wsi.astype(np.uint8)
        warnings.warn("WSI image is of type float64, converting to uint8.")

    blended_insert = blending_method(artifact, wsi, artifact_mask.numpy())
    insert = pyvips.Image.new_from_array(blended_insert)

    return insert


def augment_image(image: pyvips.Image, affine_matrix: np.ndarray) -> pyvips.Image:
    """
    Augment images with affine transformation.
    """
    # TODO
    augmented_image = image.affine(
        [
            affine_matrix[0, 0],
            affine_matrix[0, 1],
            affine_matrix[1, 0],
            affine_matrix[1, 1],
        ]
    )
    return augmented_image


def scale_artifact_image(
    artifact_image: pyvips.Image, wsi_image: pyvips.Image
) -> pyvips.Image:
    """
    Scaling of the artifact is done based on xres/yres attributes of the images.
    That means the input to the framework needs to have those parameters.
    Keep in mind that loading jpeg with pyvips will also produce those parameters but they will be bogus in this context.
    So every WSI_EXTs listed in config needs to properly return xres/yres in pixels/mm (according to pyvips.Image).
    """
    x_scale, y_scale = get_artifact_scaling(artifact_image, wsi_image)

    artifact_image = artifact_image.resize(x_scale, vscale=y_scale)
    return artifact_image


def get_artifact_scaling(
    artifact_image: pyvips.Image, wsi_image: pyvips.Image
) -> Tuple[float, float]:
    x_scale = wsi_image.xres / artifact_image.xres
    y_scale = wsi_image.yres / artifact_image.yres

    return x_scale, y_scale


def sample_coord(
    wsi_image: pyvips.Image,
    segmentation_annotation: Optional[List[Annotation]],
    blending_place: LEGAL_BLENDING_PLACES,
) -> NormalizedCoordinate:
    # TODO: change to sample from an xml

    wsi_shape = (wsi_image.width, wsi_image.height)

    if blending_place == "all":
        idx_x, idx_y = (np.random.rand(), np.random.rand())

    elif segmentation_annotation is not None:
        # annotation = np.random.choice(segmentation_annotation) # will produce uneven sampling (e.g. there is one big and multiple smaller annotations)
        polygons = get_polygons(segmentation_annotation)
        # segmentation_image_np: np.ndarray = segmentation_image.numpy()

        if blending_place == "inside":
            # idx_x, idx_y = np.where(segmentation_image_np > 0)
            idx_x, idx_y = sample_random_coord(
                polygons=polygons, inside=True, wsi_shape=wsi_shape
            )

        elif blending_place == "outside":
            # idx_x, idx_y = np.where(segmentation_image_np == 0)
            idx_x, idx_y = sample_random_coord(
                polygons=polygons, inside=False, wsi_shape=wsi_shape
            )

        elif blending_place == "edge":
            # edges = feature.canny(segmentation_image_np)  # replace with opencv
            # idx_x, idx_y = np.where(edges > 0)
            # https://stackoverflow.com/questions/42023522/random-sampling-of-points-along-a-polygon-boundary
            # Just sample a random coordinate from the polygon
            polygon = np.random.choice(polygons)
            high = polygon.vertices.shape[0]
            indx = np.random.randint(0, high)
            idx_x, idx_y = polygon.vertices[indx, ...]

        idx_x, idx_y = idx_x / wsi_shape[0], idx_y / wsi_shape[1]
    else:
        raise ValueError(
            f"blending_place {blending_place} not supported or segmentation not provided for place other than all"
        )

    return NormalizedCoordinate(
        x=idx_x,  # WARNING: numpy's x is pyvips' y
        y=idx_y,  # Now change as we use xml
    )


def get_polygons(annotations: List[Annotation]) -> List[mpltPath.Path]:
    polygons = []
    for annotation in annotations:
        polygons += [mpltPath.Path([(c.x, c.y) for c in annotation.coordinates])]
    return polygons


def sample_random_coord(
    polygons: List[mpltPath.Path], inside: bool, wsi_shape: Tuple[float, float]
) -> Tuple[float, float]:
    """
    wsi_shape: (width, height)
    """
    left, top, right, bottom = 0, 0, wsi_shape[0], wsi_shape[1]

    while True:
        x = np.random.uniform(left, right)
        y = np.random.uniform(top, bottom)
        if check_if_inside_polygons(polygons, (x, y)) == inside:
            return (x, y)


def cut_patch(
    image: pyvips.Image, coord: CornerCoordinate, patch_size: Tuple[int, int]
) -> pyvips.Image:
    try:
        return image.crop(coord.left, coord.top, patch_size[0], patch_size[1])
    except Exception as e:
        #TODO: remove later
        print(f"{coord.left=}, {coord.top=}, {patch_size[0]=}, {patch_size[1]=}")
        print(f"{image.width=}, {image.height=}")
        raise e

def pad_wsi(image: pyvips.Image, pad_size: Tuple[int, int, int, int]) -> pyvips.Image:
    '''
    pad_size: (left, top, right, bottom)
    '''
    img = pyvips.Image.black(image.width+pad_size[0]+pad_size[2], image.height+pad_size[1]+pad_size[3])
    img = img.insert(image, pad_size[0], pad_size[1])
    return img

def get_padding_size(patch_size):
    return (patch_size[0]//2, patch_size[1]//2, patch_size[0]//2, patch_size[1]//2)

def check_if_inside_polygons(
    polygons: List[mpltPath.Path], point: Tuple[float, float], pop_on_match: bool = False
) -> bool:
    """
    point: (x, y)
    """
    for i, polygon in enumerate(polygons):
        if polygon.contains_point(point):
            if pop_on_match:
                _ = polygons.pop(i)
            return True
    return False


def point_in_background(
    point: Tuple[float, float], 
    annotations: List[mpltPath.Path], 
    exclusions: List[mpltPath.Path],
):
    return (not check_if_inside_polygons(annotations, point)) and (
        not check_if_inside_polygons(exclusions, point)
    )


def point_in_tissue(
    point: Tuple[float, float],
    annotations: List[mpltPath.Path],
    exclusions: List[mpltPath.Path],
):
    return check_if_inside_polygons(annotations, point) and (
        not check_if_inside_polygons(exclusions, point)
    )


def point_in_artifact(
    point: Tuple[float, float], annotations: List[mpltPath.Path], **kwargs
):
    """
    To prevent skewing towards bigger artifacts we pop the artifact from the list after match.
    In sample_point we check if the list is empty and if so we reinitialize it.
    Ok that will actually skew the sampling towards smaller artifacts, so setting pop_on_match to False.
    But if we set it to True it will result in an even sampling of the types of artifacts more.
    #NOTE: decide later what is better.
    """
    return check_if_inside_polygons(annotations, point, True)


def get_bbox_of_annotation(polygons: List[mpltPath.Path]) -> Tuple[float, float, float, float]:
    """
    Returns the bounding box of the annotation
    """
    left, top, right, bottom = 0, 0, 0, 0
    for polygon in polygons:
        left = min(left, np.min(polygon.vertices[:, 0]))
        top = min(top, np.min(polygon.vertices[:, 1]))
        right = max(right, np.max(polygon.vertices[:, 0]))
        bottom = max(bottom, np.max(polygon.vertices[:, 1]))
    return (left, top, right, bottom)

def sample_point(
    func,
    num,
    image: pyvips.Image,
    annotations: List[Annotation],
    exclusions: List[Annotation] = None,
    limit_sampling_range: bool = False
) -> Generator[NormalizedCoordinate, None, None]:
    polygons = {
        "annotations": get_polygons(annotations),
        "exclusions": get_polygons(exclusions) if exclusions else None,
    }

    left, top, right, bottom = 0, 0, image.width, image.height

    for i in range(num):
        point = None

        if len(polygons['annotations']) == 0:
                polygons['annotations'] = get_polygons(annotations)

        if limit_sampling_range:
            left, top, right, bottom = get_bbox_of_annotation(polygons['annotations'])

        i = 0
        while not point:
            i+=1
            point_candidate = (
                np.random.uniform(left, right),
                np.random.uniform(top, bottom),
            )

            if func(point_candidate, **polygons):
                point = point_candidate

            if i > SAMPLING_MAX_ITER:
                raise ValueError(f"Maximum sampling time reached - could not sample point given func: {func.__name__}.")

        yield NormalizedCoordinate(x=point[0] / image.width, y=point[1] / image.height)


def create_segmentation_mask(
    image: pyvips.Image, annotation_coordinates: List[AbsoluteCoordinate]
) -> pyvips.Image:
    # NOTE: it should draw the polygon inside the points no matter the order
    #      we then lose the information about the order of the points
    #      we can fix that later if needed

    polygon = [(coord.x, coord.y) for coord in annotation_coordinates]

    mask_img = PILimage.new("L", (image.width, image.height), 0)
    ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
    # mask_np = np.array(mask_img)

    mask_image = pyvips.Image.new_from_array(mask_img)

    return mask_image


def dispatch_blending_method(artifact_group: LEGAL_ARTIFACT_TYPES) -> Callable:
    return BLENDING_METHODS_DISPATCH[artifact_group]


def dispatch_blending_place(
    artifact_group: LEGAL_ARTIFACT_TYPES,
) -> LEGAL_BLENDING_PLACES:
    return BLENDING_PLACES_DISPATCH[artifact_group]
