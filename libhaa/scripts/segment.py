from pathlib import Path
from typing import Generator, Tuple
from libhaa.base.config import WSI_EXTs, ARTIFACT_LIBRARY_ANNOTATION_EXT
from libhaa.segmentation.dataloader import SegmentationModule
import itertools

def main(root_wsi: Path, model_weitghts_path: Path, save_root: Path, openslide_level: int = 3):
    '''
    openslide_level: int, default 3
        for ANHIR: 2-3
        for ACROBAT: 3-4
        for BigPicture: 6-7
    '''
    segmod = SegmentationModule(model_weitghts_path, False, device="cuda")

    for wsi_path, save_path in wsi_iterator(root_wsi=root_wsi, save_root=save_root):
        try:
            segmod.inference_wsi(wsi_path, openslide_level, save_path)
        except Exception as e:
            print(f"\n\nSkipping processing for file '{wsi_path}' due to an error: {e}\n\n")
            continue 

def wsi_iterator(
    root_wsi: Path,
    save_root: Path,
) -> Generator[Tuple[Path, Path], None, None]:
    """
    Iterate over a directory of WSIs and match it's annotation
    """
    # chain all wsi possible extensions together into an iterator
    for wsi_path in itertools.chain(*[root_wsi.rglob(f"*{ext}") for ext in WSI_EXTs]):
        relative_path = wsi_path.relative_to(root_wsi)
        save_path = (save_root / relative_path).with_suffix(
            ARTIFACT_LIBRARY_ANNOTATION_EXT
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)

        yield wsi_path, save_path