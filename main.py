"""Command-line entry point for the complete BioTTA inference pipeline."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from natsort import natsorted

from biotta_output import save_results
from biotta_step1 import predict_lengths
from biotta_step2 import run_tta_and_predict


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def _resolve_config_path(value: str, base_dir: Path) -> str:
    """Resolve a non-empty configuration path relative to the config file."""
    if not value:
        return value
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def load_config(config_path: str) -> Dict:
    """Load YAML configuration and normalize every configured filesystem path."""
    resolved_config = Path(config_path).expanduser().resolve()
    with resolved_config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    base_dir = resolved_config.parent
    paths = config.setdefault("paths", {})
    for key in ("input_data", "output_dir", "step1_model", "step2_model", "development_curves"):
        if key in paths:
            paths[key] = _resolve_config_path(paths[key], base_dir)

    template = paths.get("template", {})
    for key in ("image_folder", "label_folder", "registered_folder"):
        if key in template:
            template[key] = _resolve_config_path(template[key], base_dir)

    data = config.setdefault("data", {})
    if "age_file" in data:
        data["age_file"] = _resolve_config_path(data["age_file"], base_dir)

    return config


def image_identifier(image_path: str) -> str:
    """Return a stable image name with the NIfTI suffix removed."""
    name = Path(image_path).name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    return Path(image_path).stem


def get_image_paths(input_path: str) -> List[str]:
    """Return naturally sorted NIfTI paths from one file or one directory."""
    path = Path(input_path).expanduser()
    if path.is_file():
        if path.name.lower().endswith((".nii", ".nii.gz")):
            return [str(path.resolve())]
        raise ValueError(f"Unsupported input format: {path.name}")

    if path.is_dir():
        images = list(path.glob("*.nii.gz")) + list(path.glob("*.nii"))
        return natsorted(str(image.resolve()) for image in images)

    raise FileNotFoundError(f"Input path does not exist: {path}")


def _read_age_table(age_file: str, age_format: Dict) -> pd.DataFrame:
    """Read a CSV or TSV gestational-age table with stable string identifiers."""
    name_column = age_format.get("name_column", "name")
    suffix = Path(age_file).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(age_file, dtype={name_column: str})
    if suffix == ".tsv":
        return pd.read_csv(age_file, sep="\t", dtype={name_column: str})
    raise ValueError("The gestational-age table must be a CSV or TSV file.")


def get_age_for_image(
    image_path: str,
    age_file: Optional[str] = None,
    age_format: Optional[Dict] = None,
) -> Optional[float]:
    """Look up an image's gestational age in an optional CSV or TSV table."""
    if not age_file or not Path(age_file).is_file():
        return None

    age_format = age_format or {"name_column": "name", "age_column": "age"}
    try:
        age_df = _read_age_table(age_file, age_format)
        name_column = age_format.get("name_column", "name")
        age_column = age_format.get("age_column", "age")
        if name_column not in age_df.columns or age_column not in age_df.columns:
            raise KeyError(f"Required columns are missing: {name_column}, {age_column}")

        matches = age_df[age_df[name_column] == image_identifier(image_path)]
        if not matches.empty:
            return float(matches.iloc[0][age_column])
    except Exception as error:
        print(f"[WARNING] Could not read gestational age for {image_path}: {error}")

    return None


def process_single_image(
    image_path: str,
    config: Dict,
    device: torch.device,
    age: Optional[float] = None,
) -> Dict:
    """Run length prediction and landmark localization for one NIfTI image."""
    print("\n" + "=" * 60)
    print(f"Processing image: {Path(image_path).name}")
    print("=" * 60)

    step1_config = config["step1"]
    biometry_list = config["data"]["biometry_list"]

    print("\n[Step 1] Predicting biometric measurements...")
    try:
        prediction_df = predict_lengths(
            [image_path],
            config["paths"]["step1_model"],
            device,
            batch_size=step1_config["batch_size"],
            biometry_list=biometry_list,
            length_weights=config["data"]["length_weights"],
            ventricle_indices=config["data"].get("ventricle_indices", [0, 1, 2, 3]),
            num_workers=config["system"].get("num_workers", 2),
        )
        predicted_lengths = prediction_df.iloc[0]
        print("[OK] Step 1 completed")
        for biometry in biometry_list:
            print(f"  {biometry}: {predicted_lengths[biometry]:.2f}")

        save_intermediate = config["system"].get("save_intermediate", {})
        if save_intermediate.get("step1_results", False):
            output_dir = Path(config["paths"]["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / "step1_lengths.csv"
            if csv_path.exists():
                existing_df = pd.read_csv(csv_path)
                prediction_df = pd.concat([existing_df, prediction_df], ignore_index=True)
            prediction_df.to_csv(csv_path, index=False)
            print(f"[OK] Step 1 results saved to: {csv_path}")
    except Exception as error:
        print(f"[ERROR] Step 1 failed: {error}")
        raise

    if age is None:
        data_config = config["data"]
        age = get_age_for_image(
            image_path,
            data_config.get("age_file") or None,
            data_config.get("age_file_format", {}),
        )

    template_paths = None
    template_config = config["paths"].get("template", {})
    image_folder = template_config.get("image_folder", "")
    label_folder = template_config.get("label_folder", "")
    if Path(image_folder).is_dir() and Path(label_folder).is_dir():
        template_paths = {
            "image_folder": image_folder,
            "label_folder": label_folder,
            "registered_folder": template_config.get("registered_folder", ""),
        }

    image_name = image_identifier(image_path)
    output_dir = Path(config["paths"]["output_dir"])
    image_output_dir = output_dir / image_name
    image_output_dir.mkdir(parents=True, exist_ok=True)

    step2_config = config["step2"]
    use_existing_tta = step2_config.get("use_existing_tta", False)
    tta_model_path = image_output_dir / "tent_TTA.pth"
    if not use_existing_tta or not tta_model_path.exists():
        tta_model_path = None

    print("\n[Step 2] Running test-time adaptation and landmark localization...")
    try:
        landmarks, _ = run_tta_and_predict(
            image_path,
            predicted_lengths[biometry_list],
            config["paths"]["step2_model"],
            device,
            step2_config,
            template_paths=template_paths,
            age=age,
            skip_tta=step2_config.get("skip_tta", False),
            tta_model_path=str(tta_model_path) if tta_model_path else None,
            output_dir=str(image_output_dir),
            curve_dir=config["paths"]["development_curves"],
            save_intermediate=config["system"].get("save_intermediate", {}),
        )
        print(f"[OK] Step 2 completed with {len(landmarks)} landmarks")
        for index, point in enumerate(landmarks[:3], start=1):
            print(f"  Point {index}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")
    except Exception as error:
        print(f"[ERROR] Step 2 failed: {error}")
        raise

    return {
        "image_name": image_name,
        "age": age,
        "lengths": predicted_lengths[biometry_list].to_dict(),
        "landmarks": landmarks,
    }


def process_batch(image_paths: List[str], config: Dict, device: torch.device) -> List[Dict]:
    """Process multiple images while allowing an individual sample to fail."""
    print(f"\nStarting batch processing for {len(image_paths)} images...")
    results = []
    data_config = config["data"]
    age_file = data_config.get("age_file") or None
    age_format = data_config.get("age_file_format", {})

    for index, image_path in enumerate(image_paths, start=1):
        print(f"\n[{index}/{len(image_paths)}]")
        age = get_age_for_image(image_path, age_file, age_format)
        try:
            results.append(process_single_image(image_path, config, device, age))
        except Exception as error:
            print(f"[ERROR] Processing failed for {image_path}: {error}")

    return results


def build_parser() -> argparse.ArgumentParser:
    """Create the BioTTA command-line parser."""
    parser = argparse.ArgumentParser(
        description="BioTTA fetal brain biometry and landmark localization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python main.py --input path/to/image.nii.gz --age 31 --output ./results
  python main.py --input path/to/folder --age_csv path/to/ages.csv --output ./results
  python main.py --config path/to/config.yaml
""",
    )
    parser.add_argument("--input", "-i", help="NIfTI file or directory")
    parser.add_argument(
        "--config",
        "-c",
        default=str(DEFAULT_CONFIG),
        help=f"Configuration file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--gpu", help="Visible GPU index")
    parser.add_argument("--age", type=float, help="Gestational age for one input file")
    parser.add_argument("--age_csv", help="CSV or TSV gestational-age table for batch input")
    parser.add_argument(
        "--age_file_format",
        help='JSON column mapping, for example: {"name_column":"name","age_column":"age"}',
    )
    parser.add_argument(
        "--registered_folder",
        help="Directory used to cache registered atlas labels",
    )
    parser.add_argument("--output", "-o", help="Output directory")
    return parser


def main() -> None:
    """Parse arguments, run BioTTA, and save structured results."""
    args = build_parser().parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_file():
        print(f"[ERROR] Configuration file does not exist: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))

    if args.age_csv:
        age_csv = Path(args.age_csv).expanduser().resolve()
        if not age_csv.is_file():
            print(f"[ERROR] Gestational-age file does not exist: {age_csv}")
            sys.exit(1)
        config["data"]["age_file"] = str(age_csv)

    if args.age_file_format:
        try:
            config["data"]["age_file_format"].update(json.loads(args.age_file_format))
        except (json.JSONDecodeError, TypeError) as error:
            print(f"[ERROR] Invalid --age_file_format JSON: {error}")
            sys.exit(1)

    if args.registered_folder:
        config["paths"]["template"]["registered_folder"] = str(
            Path(args.registered_folder).expanduser().resolve()
        )

    if args.output:
        config["paths"]["output_dir"] = str(Path(args.output).expanduser().resolve())

    gpu_id = args.gpu if args.gpu is not None else config["system"].get("gpu_id")
    if gpu_id is not None and str(gpu_id) != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    seed = int(config["system"].get("seed", 1))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_path = args.input or config["paths"].get("input_data", "")
    if not input_path:
        print("[ERROR] No input path was provided by --input or paths.input_data.")
        sys.exit(1)

    try:
        image_paths = get_image_paths(input_path)
    except Exception as error:
        print(f"[ERROR] Could not collect input images: {error}")
        sys.exit(1)

    if not image_paths:
        print("[ERROR] No NIfTI images were found.")
        sys.exit(1)
    print(f"Found {len(image_paths)} image file(s)")

    if len(image_paths) == 1:
        results = [process_single_image(image_paths[0], config, device, age=args.age)]
    else:
        if args.age is not None:
            print("[WARNING] --age is ignored for batch input; use --age_csv instead.")
        results = process_batch(image_paths, config, device)

    if not results:
        print("[ERROR] No images were processed successfully.")
        sys.exit(1)

    save_results(
        results,
        config["paths"]["output_dir"],
        config["output"].get("format", "json"),
        config["data"]["biometry_list"],
    )
    print(f"[OK] Processed {len(results)} image(s)")
    print(f"Results saved to: {config['paths']['output_dir']}")


if __name__ == "__main__":
    main()
