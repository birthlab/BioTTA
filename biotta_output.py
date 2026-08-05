"""Save BioTTA measurements and 22 landmark coordinates as JSON."""

import json
import os
from typing import Dict, List, Optional

import numpy as np


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


def save_results(
    results: List[Dict],
    output_dir: str,
    output_format: str = "json",
    biometry_list: Optional[List[str]] = None,
):
    """Save inference results to a JSON file."""
    if biometry_list is None:
        biometry_list = DEFAULT_BIOMETRY_LIST

    os.makedirs(output_dir, exist_ok=True)
    json_data = []

    for result in results:
        image_name = result["image_name"]
        age = result.get("age", None)

        lengths = result.get("lengths", {})
        if isinstance(lengths, dict):
            length_dict = lengths
        elif isinstance(lengths, (list, np.ndarray)):
            length_dict = {
                biometry_list[index]: float(lengths[index])
                for index in range(len(lengths))
            }
        else:
            length_dict = {}

        landmarks = result.get("landmarks", None)
        if landmarks is not None:
            if isinstance(landmarks, np.ndarray):
                landmarks = landmarks.tolist()
        else:
            landmarks = [[0.0, 0.0, 0.0]] * 22

        json_entry = {
            "image_name": image_name,
            "lengths": length_dict,
            "landmarks": landmarks,
        }

        if age is not None:
            json_entry["age"] = float(age)

        json_data.append(json_entry)

    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(json_data, file, indent=2, ensure_ascii=False)
    print(f"[OK] Results saved to: {json_path}")

    return json_data


def format_landmarks_for_display(landmarks: np.ndarray) -> str:
    """Format up to 22 landmark coordinates for terminal display."""
    if landmarks is None:
        return "No landmarks"

    if isinstance(landmarks, list):
        landmarks = np.array(landmarks)

    lines = ["Landmark coordinates (22 points):", "-" * 50]
    point_names = [
        "RLV_A_start", "RLV_A_end",
        "LLV_A_start", "LLV_A_end",
        "RLV_C_start", "RLV_C_end",
        "LLV_C_start", "LLV_C_end",
        "BBD_start", "BBD_end",
        "CBD_start", "CBD_end",
        "TCD_start", "TCD_end",
        "FOD_start", "FOD_end",
        "AVD_start", "AVD_end",
        "VH_start", "VH_end",
        "CCL_start", "CCL_end",
    ]

    for index in range(min(22, len(landmarks))):
        point = landmarks[index]
        point_name = point_names[index] if index < len(point_names) else f"Point_{index + 1}"
        lines.append(
            f"{point_name:20s}: ({point[0]:8.2f}, {point[1]:8.2f}, {point[2]:8.2f})"
        )

    return "\n".join(lines)
