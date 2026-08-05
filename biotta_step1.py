"""Step 1: predict 11 fetal brain biometric measurements."""

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


class _ImageDataset(Dataset):
    """Load and normalize the NIfTI volumes used by the Step 1 model."""

    def __init__(self, image_paths: List[str], measurement_count: int):
        self.image_paths = image_paths
        self.measurement_count = measurement_count

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image_name = Path(image_path).name
        if image_name.lower().endswith(".nii.gz"):
            image_name = image_name[:-7]
        elif image_name.lower().endswith(".nii"):
            image_name = image_name[:-4]

        image = nb.load(image_path).get_fdata()
        mask = image > 0
        if not np.any(mask):
            raise ValueError(f"The input image has no positive-valued foreground: {image_path}")

        foreground = image[mask]
        mean = float(np.mean(foreground))
        std = float(np.std(foreground))
        if std == 0:
            std = 1.0
        image[mask] = (foreground - mean) / std
        image[~mask] = 0

        image = extract_brain(image, block_ind(mask), [128, 160, 128])
        data = torch.as_tensor(image.reshape((1,) + image.shape), dtype=torch.float32)
        placeholder = torch.full((self.measurement_count,), float("nan"), dtype=torch.float32)
        return data, placeholder, image_name


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
    """Predict all configured measurements for one or more NIfTI images."""
    del length_weights, ventricle_indices, label_csv_path
    measurements = biometry_list or DEFAULT_BIOMETRY_LIST
    if len(measurements) != 11:
        raise ValueError(f"Step 1 expects 11 measurements, received {len(measurements)}.")

    dataset = _ImageDataset(image_paths, len(measurements))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Step 1 model was not found: {model_file}")

    base_model = LocalAppearance(in_channels=1, num_classes=22)
    model = EnhancedLengthPredictor(base_model, num_classes=11).to(device)
    checkpoint = torch.load(model_file, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"[OK] Step 1 model loaded: {model_file}")

    results = []
    with torch.no_grad():
        for batch_index, (images, _, image_names) in enumerate(dataloader, start=1):
            outputs, _ = model(images.to(device))
            for index, image_name in enumerate(image_names):
                result = {"label_name": image_name}
                result.update(
                    {name: outputs[index, offset].item() for offset, name in enumerate(measurements)}
                )
                results.append(result)
            print(f"Processed Step 1 batch {batch_index}/{len(dataloader)}")

    return pd.DataFrame(results, columns=["label_name", *measurements])


def predict_single_image(
    image_path: str,
    model_path: str,
    device: torch.device,
    biometry_list: Optional[List[str]] = None,
    length_weights: Optional[List[float]] = None,
    ventricle_indices: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Convenience wrapper for predicting one NIfTI image."""
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
