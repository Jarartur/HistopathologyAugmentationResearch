from typing import Callable, Dict, Literal, TypeAlias
from libhaa.base.blending import cvClone, do_nothing, blend_output_dst, focus_distortion, hard_clone, reinhard_staining

from sys import platform
if platform == "win32":
    import os
    vipsbin = r'c:\ProgramData\libvips\bin'
    os.environ['PATH'] = os.pathsep.join((vipsbin, os.environ['PATH']))
import pyvips

# -- Base -- #
WSI_EXTs = [".tiff", ".tif", ".mrxs"]
ANNOTATION_EXTs = [".xml"]
ARTIFACT_LIBRARY_ANNOTATION_EXT = ".xml"

IMAGE_DATA_TYPE: TypeAlias = pyvips.Image

ANNOTATION_XML_NAME = "Annotation"
ANNOTATION_GROUP_XML_NAME = "Group"
ANNOTATION_SEGMENTATION_GROUP = 'foreground'

LABELS_SAVING_FORMATS = Literal["asap", "yolo", 'both']

#NOTE: this probably could go away as we do random cropping mainly
PYVIPS_ACCESS_TYPES_DICT = {
    "sequential": pyvips.Access.SEQUENTIAL,
}

LEGAL_ARTIFACT_TYPES = Literal["air", "dust", "tissue", "ink", "marker", "glass", "focus"] #, "focus", "background"
LEGAL_BLENDING_PLACES = Literal["inside", "outside", "edge", "all"] #TODO: think about adding "border" location

BLENDING_METHODS_DISPATCH: Dict[LEGAL_ARTIFACT_TYPES, Callable] = {
    "air" : hard_clone,
    "dust" : hard_clone, #cvClone
    "tissue" : hard_clone,
    "ink" : reinhard_staining,
    "marker" : hard_clone,
    "focus" : focus_distortion,
    # "glass" : do_nothing,
}

BLENDING_PLACES_DISPATCH: Dict[LEGAL_ARTIFACT_TYPES, LEGAL_BLENDING_PLACES] = {
    "air" : "all",
    "dust" : "all",
    "tissue" : "inside",
    "ink" : "edge",
    "marker" : "outside",
    "focus" : "inside",
    # "glass" : "all",
}

SAMPLING_MAX_ITER = 32_000_000

ARTIFACT_TYPES_MAX_NUMBERS = {
    "air": 4,
    "dust": 7,
    "tissue": 4,
    "ink": 4,
    "marker": 4,
    "glass": 1,
    "focus": 2,
}

AUGMENTATION_PARAMS = {
    'scale': [0.4, 1.4],
    'rotation': [-90, 90]
}

# -- Detection -- #
PATCHES_SIZE_DETECTION = (640, 640)

ARTIFACT_TYPES_ENCODING = {
    "air": 0,
    "dust": 1,
    "tissue": 2,
    "ink": 3,
    "marker": 4,
    "glass": 5,
    "focus": 6,
}

YOLO_DATASET_ANNOTATION_EXT = '.txt'

# -- Segmentation -- #
NUM_UNSQUEEZED_DIMS = 2

WSI_WHOLE_INFRA_PARAMS = {
    "patch_size": (256, 256, 1),
    "patch_overlap": (50, 50, 0),
    "batch_size": 16,
}

XML_SEG_NAME = "Annotation "
XML_SEG_TYPE = "foreground"
XML_SEG_SHAPE = "Polygon"