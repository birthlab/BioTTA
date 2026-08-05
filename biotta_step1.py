"""Step 1: predict 11 fetal brain biometric measurements."""

import os
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from lib.step1_length_prediction import (
    EnhancedLengthPredictor,
    LocalAppearance,
    block_ind,
    extract_brain,
)


DEFAULT_BIOMETRY_LIST = [
    "RLV_A",
    "LLV_A",
    "RLV_C",
    "LLV_C",
    "BBD",
    "CBD",
    "TCD",
    "FOD",
    "AVD",
    "VH",
    "CCL",
]


def predict_lengths(
    image_paths: List[str],
    model_path: str,
    device: torch.device,
    batch_size: int = 4,
    biometry_list: Optional[List[str]] = None,
    length_weights: Optional[List[float]] = None,
    ventricle_indices: Optional[List[int]] = None,
    label_csv_path: Optional[str] = None,
    num_workers: int = 2,
) -> pd.DataFrame:
    """Predict biometric measurements for one or more NIfTI images."""
    if biometry_list is None:
        biometry_list = DEFAULT_BIOMETRY_LIST
    if length_weights is None:
        length_weights = [10.0, 10.0, 10.0, 10.0, 1.5, 1.5, 0.6, 1.0, 3.0, 3.0, 1.0]
    if ventricle_indices is None:
        ventricle_indices = [0, 1, 2, 3]

    class SimpleImageDataset(Dataset):
        def __init__(self, paths):
            self.image_paths = paths
            self.image_names = []
            for image_path in paths:
                image_name = Path(image_path).name
                if image_name.lower().endswith(".nii.gz"):
                    image_name = image_name[:-7]
                elif image_name.lower().endswith(".nii"):
                    image_name = image_name[:-4]
                self.image_names.append(image_name)

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, index):
            image_path = self.image_paths[index]
            image_name = self.image_names[index]

            image = nb.load(image_path).get_fdata()
            mask = image > 0

            foreground = image[mask]
            mean = np.mean(foreground) if len(foreground) > 0 else 0
            std = np.std(foreground) if len(foreground) > 0 else 1
            normalized = (foreground - mean) / std if len(foreground) > 0 else foreground
            image[mask] = normalized
            image[~mask] = 0

            image = extract_brain(image, block_ind(mask), [128, 160, 128])
            data = image.reshape((1,) + image.shape)
            data = torch.tensor(data, dtype=torch.float32)

            label = torch.full((len(biometry_list),), float("nan"), dtype=torch.float32)
            return data, label, image_name

    dataset = SimpleImageDataset(image_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    base_model = LocalAppearance(in_channels=1, num_classes=22)
    model = EnhancedLengthPredictor(base_model, num_classes=11).to(device)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"[OK] Step 1 model loaded: {model_path}")
    else:
        raise FileNotFoundError(f"Step 1 model was not found: {model_path}")

    model.eval()

    results = []
    with torch.no_grad():
        for batch_index, (images, labels, image_names) in enumerate(dataloader):
            images = images.to(device)
            outputs, _ = model(images)

            for index in range(len(image_names)):
                sample_result = {
                    "label_name": image_names[index],
                    **{
                        biometry_list[offset]: outputs[index][offset].item()
                        for offset in range(11)
                    },
                }
                results.append(sample_result)

            print(f"Processed Step 1 batch {batch_index + 1}/{len(dataloader)}")

    result_frame = pd.DataFrame(results)
    columns = ["label_name"] + [biometry_list[index] for index in range(11)]
    return result_frame[columns]


def predict_single_image(
    image_path: str,
    model_path: str,
    device: torch.device,
    biometry_list: Optional[List[str]] = None,
    length_weights: Optional[List[float]] = None,
    ventricle_indices: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Predict biometric measurements for one NIfTI image."""
    results = predict_lengths(
        [image_path],
        model_path,
        device,
        batch_size=1,
        biometry_list=biometry_list,
        length_weights=length_weights,
        ventricle_indices=ventricle_indices,
    )
    return results.iloc[0].to_dict()
