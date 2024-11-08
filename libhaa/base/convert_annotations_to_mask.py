import tqdm
from typing import Dict, List, Tuple
from pathlib import Path

from config import (
    WSI_EXTs,
    ANNOTATION_EXTs,
    AnnotationDataType
)

from loading_utils import (
    load_image,
    find_files_with_extension,
    map_images_to_annotations,
    get_pixel_size_scalling_factor_in_um
)

from converting_utils import (
    prepare_metadata,
    parse_annotation,
    create_array_from_coordinates,
    save_mask_as_tiff,
    find_unique_part_of_groups,
)

def generate_and_save_mask_as_tiff(image_path : Path,
                                   annotation_path : Path,
                                   output_path : Path,
                                   group_to_value: Dict[str, int],
                                   xml_data_type : AnnotationDataType) -> Tuple[List[int], List[int]]:
    """
    Generate mask from xml annotation, and then save it as pyramidal tiff file. 
    It produces levels with downsampling factor only equal to 2. 

    Parameters
    ----------
    image_path : Path
        Image path - to copy metadata.

    annotation_path : Path
        Path to xml with annotations

    output_path : Path
        Path to output file, with filename ending with extension ".tiff"

    group_to_value : Dict[str, int]
        Dictionary mapping group names to unique integer identifiers.

    xml_data_type : AnnotationDataType
        Information about data saved in xml, wheather coordinates are saved in pixels at level 0 resolution, or in um.

    Returns
    -------
    Tuple[list[int], list[int]]

    """
    image = load_image(image_path)
    if image == None :
        print(f"An unexpected error occurred while loading image.")
        return
    
    metadata = prepare_metadata(image)

    factor = (1.0, 1.0) if xml_data_type == AnnotationDataType.PIXELS_LVL_0 else get_pixel_size_scalling_factor_in_um(image)

    coords_polys_lvl0 = parse_annotation(annotation_path,
                                         factor)
    mask = create_array_from_coordinates((image.width, image.height), 
                                                         coords_polys_lvl0,
                                                         group_to_value)
    save_mask_as_tiff(metadata,
                      mask, 
                      output_path)

def generate_masks_from_annotations(annotations_dir : Path, 
                                    images_dir : Path,
                                    output_mask_dir : Path,
                                    xml_data_type : AnnotationDataType):
    """
    Generate mask from xml annotations, and then save it as pyramidal tiff file. 
    It produces levels with downsampling factor only equal to 2. 

    Parameters
    ----------
    annotations_dir : Path
        Dir with annotations.

    images_dir : Path
        Dir with images.

    output_mask_dir : Path
        Output dir for masks.

    xml_type : AnnotationDataType
        Information about data saved in xml, wheather coordinates are saved in pixels at level 0 resolution, or in um.

    Returns
    -------

    """

    if not images_dir.exists() or not annotations_dir.exists():
        return

    if not output_mask_dir.exists():
        output_mask_dir.mkdir(parents=True)

    classes = find_unique_part_of_groups(annotations_dir)

    img_paths = find_files_with_extension(images_dir, WSI_EXTs)
    annotations_paths = find_files_with_extension(annotations_dir, ANNOTATION_EXTs)
    input_data = map_images_to_annotations(img_paths, annotations_paths)

    if not input_data:
        raise FileNotFoundError(f"Input list of paths is empty. Cannot find corresponding pairs: image - annotation.")
    
    for record in tqdm.tqdm(input_data, desc="Generating masks"):
        try:
            annotation_path = record['annotation']
            image_path = record['image']
            filename = image_path.stem + "_mask"
            output_mask_path = output_mask_dir / filename
            generate_and_save_mask_as_tiff(image_path,
                                           annotation_path,
                                           output_mask_path,
                                           classes,
                                           xml_data_type)
        except Exception as e:
            print(f"An unexpected error occurred: {e} while proccesing {record['image']} file")
            continue
