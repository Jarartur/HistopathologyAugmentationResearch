import os
import pyvips
from pathlib import Path
from typing import Dict, Tuple

def load_image(path: Path ) -> pyvips.Image:
    """
    Load WSI image using PyVips library.  

    Parameters
    ----------
    path : Path
        Path to the specific file.

    Returns
    -------
    pyvips.Image
    """
    img = pyvips.Image.new_from_file(str(path), access="sequential")
    if img.bands > 3:
        img = img.flatten(background=255)
    return img

def get_pixel_size_scalling_factor_in_um(image : pyvips.Image) -> Tuple[float, float]:
    """
    Get pixel size at level 0. 

    Parameters
    ----------
    image : pyvips.Image
        Input pyvips image.

    Returns
    -------
    Tuple[float, float]
        Tuple of pixel size.

    """

    x_res = get_property_with_default(image, "xres")
    y_res = get_property_with_default(image, "yres")
    resolution_unit = get_property_with_default(image, "resolution-unit", "inch")

    if x_res and y_res:
        if resolution_unit == "inch":
            UNIT_IN_DPI = 25400
            x_size_um = UNIT_IN_DPI / x_res
            y_size_um = UNIT_IN_DPI / y_res
            return (x_size_um, y_size_um)
        elif resolution_unit == "cm":
            UNIT_IN_CM = 10000
            x_size_um = UNIT_IN_CM / x_res
            y_size_um = UNIT_IN_CM / y_res
            return (x_size_um, y_size_um)
        else:
            print(f"Unknown resolution unit: {resolution_unit}. The scalling factor is set to (1.0, 1.0)")
            return (1.0, 1.0)
    else:
        print("Resolution information not available in metadata. The scalling factor is set to (1.0, 1.0)")
        return (1.0, 1.0)

 
def get_property_with_default(image : pyvips.Image, 
                              property_name : str, 
                              default_value = None):
    """
    Retrieve a property from pyvips image with a fallback to a default value.

    Parameters
    ----------
    image : pyvips.Image
        Input image
    property_name : str
        Property name as string
    default_value = None
        Default value if property is not defined.

    Returns
    -------
    value or None
    """
    
    try:
        return image.get(property_name)
    except Exception as e:
        print(f"Error retrieving property {property_name}. Assigning default value: {default_value}")
        return default_value   

def find_files_with_extension(root_dir : Path, 
                              extensions : list[str]) -> list[Path]:
    """
    Search root dir to find files with declared extensions.

    Parameters
    ----------
    root_dir : Path
        Path to the main dir you want to be searched.
    extensions : list of string
        List of desirable files' extensions.

    Returns
    -------
    list[Path]
        List of all found files with corresponding extension.

    Raises
    ------
    ExceptionType
        FileNotFoundError: If rootDir was not found.
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Dir '{root_dir}' does not exist.")

    file_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            for extension in extensions:
                if filename.endswith(extension):
                    file_paths.append(Path(os.path.join(dirpath, filename)))
    
    return file_paths

def map_images_to_annotations(img_paths : list[Path], 
                              annotation_paths : list[Path]) -> list[Dict[str, Path]]:
    """
    Map images and their corresponding annotations.

    Parameters
    ----------
    img_paths : list[Path]
        List of all found images' paths.
    annotation_paths : list[Path]
        List of all found annotations' paths.

    Returns
    -------
    list[dict]
        List of dictionaries containing all found pairs: paths to corresponding images and annotations.

    Raises
    ------
    ExceptionType
        ValueError: If at least one of the lists is empty.
    """
    if not img_paths or not annotation_paths:
        raise FileNotFoundError(f"Input list is empty")
    
    mappedData = []
    for img_path in img_paths:
        for annotation_path in annotation_paths:
            if str(img_path.stem) in str(annotation_path):
                annotation_paths.remove(annotation_path)
                mappedData.append({'image' : img_path, 
                                   'annotation' : annotation_path })
                break
    return mappedData
