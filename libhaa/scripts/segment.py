from pathlib import Path
from typing import Generator, Tuple

from libhaa.base.config import WSI_EXTs, ARTIFACT_LIBRARY_ANNOTATION_EXT
from libhaa.segmentation.dataloader import SegmentationModule
import itertools

def main(
    root_wsi: Path,
    model_weights_path: Path,
    save_root: Path,
    openslide_level: int = 3,
    device: str = "cpu",
):
    '''
    openslide_level: int, default 3
        for ANHIR: 2-3
        for ACROBAT: 3-4
        for BigPicture: 6-7
    '''
    segmod = SegmentationModule(model_weights_path, False, device=device)

    for wsi_path, save_path in wsi_iterator(root_wsi=root_wsi, save_root=save_root):
        segmod.inference_wsi(wsi_path, openslide_level, save_path)

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


def cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build artifact collection", epilog="Artur Jurgas."
    )

    parser.add_argument("--wsi-path", type=Path, required=True, help="Path to WSIs.")
    parser.add_argument(
        "--model-weights", type=Path, required=True, help="Path to model weights."
    )
    parser.add_argument(
        "--save-path", type=Path, required=True, help="Path to save to."
    )
    parser.add_argument(
        "--openslide-level",
        type=int,
        required=False,
        default=3,
        help="""
Which level of the WSI to load with openslide.
Recommended values for tested datasets:
    for ANHIR: 2-3
    for ACROBAT: 3-4
    for BigPicture: 6-7
""",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",  # Default value if none is provided
        help="Specify the device to use (cpu or cuda). Default is cpu.",
    )

    args = parser.parse_args()
    main(
        root_wsi=args.wsi_path,
        model_weights_path=args.model_weights,
        save_root=args.save_path,
        openslide_level=args.openslide_level,
    )