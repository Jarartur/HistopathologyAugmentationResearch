from collections import namedtuple
from typing import List, Optional
import numpy as np
import torch as tc
import torchio as tio
from pathlib import Path
from libhaa.segmentation.model import load_network
from libhaa.base.loaders import AbsoluteCoordinate, Annotation, save_annotations_xml, get_wsi_dims, Scaling
import cv2
from tqdm import tqdm
import warnings

from sys import platform
if platform == "win32":
    import os
    vipsbin = r'c:\ProgramData\libvips\bin'
    os.environ['PATH'] = os.pathsep.join((vipsbin, os.environ['PATH']))
import pyvips

from libhaa.base.config import (
    NUM_UNSQUEEZED_DIMS,
    WSI_WHOLE_INFRA_PARAMS,
    XML_SEG_NAME,
    XML_SEG_TYPE,
    XML_SEG_SHAPE,
)



class PatchSampler:
    ...


class PatchAggregator:
    ...


class SegmentationModule:
    def __init__(
        self, model_path: Path, streaming_mode: bool, device: tc.device | str = "cpu"
    ) -> None:
        self.model = load_network(model_path, device)
        self.device = device
        self.stream = streaming_mode

    def inference_wsi(
        self, wsi_path: Path, level: int, xml_save_path: Optional[Path] = None
    ) -> None:
        # TODO: if streaming_mode then use custom PatchSampler and Patch Aggregator.
        if self.stream:
            raise NotImplementedError(
                "Streaming mode not implemented yet."
            )  # TODO: implement streaming mode.
        else:
            print(f"Starting inference on {wsi_path.name}...")

            wsi = wsi_level_reader(wsi_path, level)
            width, height = get_wsi_dims(wsi_path)
            scaling = Scaling(
                height = height / wsi.shape[0],
                width = width / wsi.shape[1],
            )

            output_tensor = self.aggregate_whole_wsi(
                wsi, **WSI_WHOLE_INFRA_PARAMS
            )
            output_array = tensor_to_numpy(output_tensor)
            output_contours = self.mask_to_contour(output_array)
            output_contours = self.filter_coordinates(output_contours)
            save_path = (
                wsi_path.with_suffix(".xml") if xml_save_path is None else xml_save_path
            )
            self.contour_to_xml(output_contours, save_path, scaling)

    @tc.inference_mode()
    def inference_patch(self, patch: np.ndarray | tc.Tensor) -> np.ndarray | tc.Tensor:
        return singular_inference(patch, self.model, device=self.device)
    

    def filter_coordinates(self, contours: List[np.ndarray]) -> List[np.ndarray]:

        contour_lenghts = [contour.shape[0] for contour in contours]
        threshold = np.quantile(contour_lenghts, 0.99)
        return [contour for contour in contours if contour.shape[0] > threshold]

    def contour_to_xml(
        self, contours: List[np.ndarray], xml_save_path: Path, scaling: Scaling
    ) -> None:
        annotations = []
        for i, contour in enumerate(contours):
            coordinates = []
            for j, coord in enumerate(contour):
                coordinates += [
                    AbsoluteCoordinate(
                        x=coord[0, 0] * scaling.width, y=coord[0, 1] * scaling.height, order=j
                    )
                ]  # TODO: check x,y order.

            annotations += [Annotation(
                name=XML_SEG_NAME + str(i),
                group=XML_SEG_TYPE,
                shape=XML_SEG_SHAPE,
                coordinates=coordinates,
            )]
        save_annotations_xml(annotations, xml_save_path)

    def mask_to_contour(self, mask: np.ndarray) -> List[np.ndarray]:
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    @tc.inference_mode()
    def aggregate_whole_wsi(
        self,
        wsi: tc.Tensor,
        patch_size: int,
        patch_overlap: int,
        batch_size: int = 1,
    ) -> tc.Tensor:
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=wsi.unsqueeze(0).unsqueeze(-1)),
        )

        grid_sampler = tio.inference.GridSampler(
            subject,
            patch_size,
            patch_overlap,
        )

        patch_loader = tc.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
        aggregator = tio.inference.GridAggregator(grid_sampler)

        for patches_batch in tqdm(patch_loader, desc="WSI processing"):
            input_tensor: tc.Tensor = patches_batch["image"][tio.DATA].to(self.device).squeeze(-1)
            output = singular_inference(input_tensor, self.model, device=self.device)

            locations = patches_batch[tio.LOCATION]
            aggregator.add_batch(output.unsqueeze(-1), locations)

        return aggregator.get_output_tensor().squeeze(-1)


def wsi_level_reader(
    path: Path | str,
    level: int = 0,
):
    try:
        img = pyvips.Image.openslideload(path, level=level)
    except Exception as e:
        print(e)
        warnings.warn(f"Probably the level {level} does not exist. Using level 0.")
        img = pyvips.Image.openslideload(path, level=0)

    image = img.numpy()

    if image.shape[2] == 4:
        image = image[:, :, :-1]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

    return tc.from_numpy(image)


def singular_inference(
    patch: np.ndarray | tc.Tensor, model: tc.nn.Module, device: tc.device | str = "cpu"
) -> np.ndarray | tc.Tensor:
    """Perform inference on a single patch.

    Args:
        patch (np.ndarray | tc.Tensor): The patch to perform inference on. Should be of shape (256, 256) for numpy and (B, 1, 256, 256) for Tensor.
        model (tc.nn.Module): The model to use for inference.
        device (tc.device): The device to use for inference.

    Returns:
        np.ndarray: The prediction of the model.
    """

    numpy = isinstance(patch, np.ndarray)
    if numpy:
        patch = numpy_to_tensor(patch, device)

    prediction = tc.sigmoid(model(patch))
    prediction = tc.where(prediction > 0.5, 1.0, 0.0)

    if numpy:
        prediction = tensor_to_numpy(prediction)

    return prediction


def numpy_to_tensor(patch: np.ndarray, device: tc.device | str) -> tc.Tensor:
    """Convert a numpy array to a tensor.

    Args:
        patch (np.ndarray): The numpy array to convert.
        device (tc.device): The device to use.

    Returns:
        tc.Tensor: The converted tensor.
    """
    # convert to tensor
    patch = tc.from_numpy(patch).float().to(device)

    # add batch dimension
    for _ in range(NUM_UNSQUEEZED_DIMS):
        patch = patch.unsqueeze(0)

    return patch


def tensor_to_numpy(patch: tc.Tensor) -> np.ndarray:
    for _ in range(NUM_UNSQUEEZED_DIMS):
        patch = patch.squeeze(0)
    patch = patch.detach().cpu().numpy().astype(np.uint8)
    return patch
