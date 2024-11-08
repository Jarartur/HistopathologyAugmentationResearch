import cv2
import pyvips
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Tuple, List, Dict

def parse_annotation(xml_filepath : Path,
                     unit_scaling_factor = Tuple[float, float]
                      ) -> Dict[str, List[List[Tuple[float, float]]]]:
    """
    Parse annotations coordinates from xaml file. 

    Parameters
    ----------
    xml_filepath : Path
        Path to the specific xml file.
    
    unit_scaling_factor = Tuple[float, float]
        Scalling factors, depending on unit in xml eg. real-world pixel size in um.

    Returns
    -------
    Dict[str, List[List[Tuple[float, float]]]]
    A dictionary where keys are the values of 'PartOfGroup' and values are lists of coordinates.
    """
    tree = ET.parse(xml_filepath)
    root = tree.getroot()
    
    categorized_coordinates = {}
    for annotation in root.findall('.//Annotation'):
        part_of_group = annotation.get('PartOfGroup')
        coords_temp = []
        coords = annotation.find('Coordinates')
        if coords is None:
            continue
        for coord in coords.findall('Coordinate'):
            x = float(coord.get('X')) / unit_scaling_factor[0]
            y = float(coord.get('Y')) / unit_scaling_factor[1]
            coords_temp.append((x,y))

        if part_of_group not in categorized_coordinates:
            categorized_coordinates[part_of_group] = []

        categorized_coordinates[part_of_group].append(coords_temp)
    return categorized_coordinates

def fill_array_with_poly(mask: np.ndarray,
                         polys: Dict[str, List[List[Tuple[float, float]]]],
                         group_to_value: Dict[str, int]
                         ) -> np.ndarray:
    """
    Fill mask according to poly coordinates using values assigned by the group_to_value dictionary.

    Parameters
    ----------
    mask : np.ndarray
        Input mask array.

    polys : Dict[str, List[List[Tuple[float, float]]]]
        Dictionary with group types and lists of all found polygons and coordinates of their contours.

    group_to_value : Dict[str, int]
        Dictionary mapping group names to unique integer identifiers.

    Returns
    -------
    np.ndarray
        Updated mask array with polygons filled according to the group_to_value mapping.
    """
    for group, polygons in polys.items():
        value = group_to_value.get(group, 0) 
        for polygon in polygons:
            pts = np.array(polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], value)

    return mask 

def find_unique_part_of_groups(xml_folder_path: Path) -> Dict[str, int]:
    """
    Find all unique 'PartOfGroup' values from XML files in a specified folder and 
    map them to unique integer values starting from 1, with 0 reserved for background.

    Parameters
    ----------
    xml_folder_path : Path
        Path to the folder containing XML files.

    Returns
    -------
    Dict[str, int]
        A dictionary mapping unique 'PartOfGroup' values to unique integer identifiers.
    """
    unique_groups = set()
    
    for xml_file in xml_folder_path.glob('*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for annotation in root.findall('.//Annotation'):
            part_of_group = annotation.get('PartOfGroup')
            if part_of_group:
                unique_groups.add(part_of_group)
    
    unique_groups_dict = {group: idx + 1 for idx, group in enumerate(sorted(unique_groups))}
    unique_groups_dict['background'] = 0
    
    return unique_groups_dict

def create_array_from_coordinates(image_dim : Tuple[float, float],
                                  all_polys_coordinates : List[Tuple[float, float]],
                                  group_to_value: Dict[str, int]
                                  ) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Create mask with given dimension, based on coordinates of the found polys contours. 

    Parameters
    ----------
    image_dim : Tuple[float, float]
        Dimension of the input image (width, height). 

    polys : List[List[Tuple[float, float]]]
        List of all found polys and coordinates of their contours.
    
    group_to_value : Dict[str, int]
        Dictionary mapping group names to unique integer identifiers.

    Returns
    -------
    Tuple[np.ndarray, List[int], List[int]]
    """
    width, height = image_dim
    dimension_converted_from_pyvips_to_numpy = (height, width)
    mask = np.zeros(dimension_converted_from_pyvips_to_numpy, 
                    dtype=np.uint8)
    filled_mask = fill_array_with_poly(mask, all_polys_coordinates, group_to_value)
    mask_rgba = convert_binary_array_to_rgb(filled_mask)

    return mask_rgba

def prepare_metadata(image : pyvips.Image) -> Dict[any, any]:
    """
    Copy metadata from loaded image, change format from list to dictionary. 

    Parameters
    ----------
    image : pyvips.Image
    Input image

    Returns
    -------
    Dict[any, any]
    
    """
    image_copy = image.copy()
    properties = image_copy.get_fields()
    metadata = {}
    for field in properties:
        metadata[field] = image_copy.get(field)
    
    return metadata

def save_mask_as_tiff(metadata : Dict,
                      mask : np.ndarray,
                      output_path : Path):
    """
    Save created mask in .tiff format using pyvips library.  

    Parameters
    ----------
    metadata
        Metadata of input image. 

    mask : np.ndarray
        Created mask to save.

    output_path : Path
        Path for file saving. 

    Returns
    -------
    
    """
    binary_mask = pyvips.Image.new_from_memory(mask.tobytes(), mask.shape[1], mask.shape[0], 3, 'uchar')
    binary_mask.tiffsave(output_path.with_suffix(".tiff"), 
                         tile=True,
                         pyramid=True,
                         compression="jpeg",
                         Q=75)
   
    binary_mask.set_type(pyvips.GValue.gstr_type, "image-description", metadata.get("openslide.comment", ""))

def convert_binary_array_to_rgb(img : np.ndarray) -> np.ndarray:
    """
    Convert binary array to rgb.

    Parameters
    ----------
    img : np.ndarray
        Input numpy array.

    Returns
    -------
    np.ndarray
    """
    height, width = img.shape
    rgba_image = np.zeros((height, width, 3), dtype=np.uint8)
    rgba_image[img == 1, :] = [255, 255, 255]  
    return rgba_image

    