"""Serialize BioTTA measurements and landmark coordinates."""

import json
from pathlib import Path
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
) -> List[Dict]:
    """Normalize inference results and save them as UTF-8 JSON."""
    if output_format.lower() != "json":
        raise ValueError(f"Unsupported output format: {output_format}")

    measurements = biometry_list or DEFAULT_BIOMETRY_LIST
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_data = []

    for result in results:
        lengths = result.get("lengths", {})
        if isinstance(lengths, dict):
            length_dict = {name: float(value) for name, value in lengths.items()}
        elif isinstance(lengths, (list, tuple, np.ndarray)):
            length_dict = {
                measurements[index]: float(value)
                for index, value in enumerate(lengths)
                if index < len(measurements)
            }
        else:
            length_dict = {}

        landmarks = result.get("landmarks")
        if landmarks is None:
            landmark_list = [[0.0, 0.0, 0.0] for _ in range(22)]
        else:
            landmark_list = np.asarray(landmarks, dtype=float).tolist()

        entry = {
            "image_name": str(result["image_name"]),
            "lengths": length_dict,
            "landmarks": landmark_list,
        }
        if result.get("age") is not None:
            entry["age"] = float(result["age"])
        json_data.append(entry)

    json_path = destination / "results.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(json_data, file, indent=2, ensure_ascii=False)
    print(f"[OK] Results saved to: {json_path}")
    return json_data


def format_landmarks_for_display(landmarks: np.ndarray) -> str:
    """Format up to 22 landmark coordinates for terminal display."""
    if landmarks is None:
        return "No landmarks"

    points = np.asarray(landmarks)
    point_names = [
        "RLV_A_start",
        "RLV_A_end",
        "LLV_A_start",
        "LLV_A_end",
        "RLV_C_start",
        "RLV_C_end",
        "LLV_C_start",
        "LLV_C_end",
        "BBD_start",
        "BBD_end",
        "CBD_start",
        "CBD_end",
        "TCD_start",
        "TCD_end",
        "FOD_start",
        "FOD_end",
        "AVD_start",
        "AVD_end",
        "VH_start",
        "VH_end",
        "CCL_start",
        "CCL_end",
    ]

    lines = ["Landmark coordinates (up to 22 points):", "-" * 50]
    for index, point in enumerate(points[:22]):
        name = point_names[index] if index < len(point_names) else f"Point_{index + 1}"
        lines.append(f"{name:20s}: ({point[0]:8.2f}, {point[1]:8.2f}, {point[2]:8.2f})")
    return "\n".join(lines)
