from libhaa.classification.dataloader import (
    cut_patches_for_inference,
    reconstruct_prediction_from_preds,
)
from libhaa.classification.model import inference_folder

import argparse
from pathlib import Path


def run_inference():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument(
        "--wsi-folder", type=Path, required=True, help="Path to data csv."
    )
    parser.add_argument(
        "--save-path", type=Path, required=True, help="Path to save to."
    )
    parser.add_argument(
        "--weights-path", type=Path, required=True, help="Path to load weights from."
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="auto",
        help="Device. Can be cpu, cuda, or auto.",
    )
    args = parser.parse_args()

    inference_folder(args.wsi_folder, args.save_path, args.weights_path, args.device)


def reconstruct_prediction():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument(
        "--prediction-folder",
        type=Path,
        required=True,
        help="Path to predicted WSI's csv.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        default=0.5,
        help="Threshold for prediction.",
    )
    args = parser.parse_args()

    reconstruct_prediction_from_preds(args.prediction_folder, args.threshold)
