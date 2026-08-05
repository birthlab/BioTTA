# BioTTA: Fetal Biometry and Landmark Localization

BioTTA is a source-free, unsupervised test-time adaptation framework for robust automatic fetal brain biometry. It adapts pretrained models to out-of-distribution fetal MRI volumes without source data or target annotations.

The pipeline has two stages:

1. A shared encoder with regression and heatmap decoders predicts 11 biometric measurements and 22 landmarks.
2. Test-time adaptation minimizes entropy, length-consistency, and boundary-gradient losses. When gestational age and atlas templates are available, registration supplies an additional anatomical prior.

![BioTTA architecture and workflow](src/github_fig1_press.jpg)

Normative growth trajectories map the 11 predicted measurements to gestational-age-specific centiles. The included curves cover the reference range used by this release, while bundled atlas image and label pairs are available for 23-38 weeks.

![Fetal brain biometry growth trajectories](src/github_fig2.jpg)

The accompanying reporting workflow can display the predicted measurements, landmarks, centiles, and slice-level visualizations.

[Watch the BioTTA reporting demo](src/demo_video.mp4)

## Features

- Predicts 11 standard fetal brain biometric measurements and 22 landmark coordinates.
- Adapts at inference time without source data or target labels.
- Supports single-file and batch NIfTI input.
- Uses optional gestational-age atlas registration.
- Exports JSON results, landmark coordinates, visualizations, and centile plots.
- Provides a GPU-enabled, non-root Docker image for reproducible execution.

## Installation

### Existing A40 environment

The dependency set is based on the validated `lyj_clone` environment on the A40 server. The code is validated with PyTorch 2.5.1 in `lyj_clone` and PyTorch 2.6.0 in Docker. Clean installations use NumPy 1.26.4 to satisfy NiBabel's declared dependency while remaining below NumPy 2:

```bash
conda activate lyj_clone
python main.py --help
```

### Python environment

Python 3.10 is recommended.

```bash
python -m pip install -r requirements.txt
python main.py --help
```

### Docker

Build the image with the host user's UID and GID so files written to bind mounts are not owned by root:

```bash
docker build \
  --build-arg APP_UID=$(id -u) \
  --build-arg APP_GID=$(id -g) \
  -t biotta:latest .
```

See [DOCKER.md](DOCKER.md) for GPU and volume examples.

## Directory Structure

```text
BioTTA/
|-- main.py
|-- config.yaml
|-- requirements.txt
|-- Dockerfile
|-- DOCKER.md
|-- biotta_step1.py
|-- biotta_step2.py
|-- biotta_output.py
|-- lib/
|   |-- step1_length_prediction.py
|   |-- step2_source_network.py
|   `-- step2_supp.py
|-- models/
|   |-- step1_length_MinValLoss.pth
|   `-- step2_biometry_MinValLoss.pth
|-- templates/
|   |-- template_image/
|   `-- template_label/
|-- development_curves/
`-- src/
    |-- github_fig1_press.jpg
    |-- github_fig2.jpg
    `-- demo_video.mp4
```

## Configuration

Paths in `config.yaml` are resolved relative to the configuration file, not the current working directory. Command-line paths are resolved normally by the shell.

```yaml
paths:
  input_data: ""
  output_dir: "./results"
  step1_model: "./models/step1_length_MinValLoss.pth"
  step2_model: "./models/step2_biometry_MinValLoss.pth"
  template:
    image_folder: "./templates/template_image"
    label_folder: "./templates/template_label"
    registered_folder: "./templates_registered"
```

Set `system.gpu_id` to `null` to use the CUDA devices exposed by the host or container. Use `--gpu` to select a specific visible device.

## Usage

Process one NIfTI file:

```bash
python main.py \
  --input path/to/sample.nii.gz \
  --age 31 \
  --registered_folder ./templates_registered \
  --output ./results
```

Process a folder using a gestational-age CSV file:

```bash
python main.py \
  --input path/to/nifti_folder \
  --age_csv path/to/ages.csv \
  --age_file_format '{"name_column": "name", "age_column": "age"}' \
  --registered_folder ./templates_registered \
  --output ./results
```

Use a custom configuration file:

```bash
python main.py --config path/to/config.yaml --input path/to/sample.nii.gz
```

The input may also be defined by `paths.input_data` in the configuration file:

```bash
python main.py --config path/to/config.yaml
```

### Gestational-age CSV

```csv
name,age
sample1,23.5
sample2,24.0
```

Age selection follows this priority:

1. `--age` for a single input file.
2. The CSV or TSV file supplied with `--age_csv` or `data.age_file`.
3. No atlas prior when age is unavailable.

## Output

The main output is `results.json`:

```json
[
  {
    "image_name": "sample1",
    "age": 24.0,
    "lengths": {
      "RLV_A": 12.5,
      "LLV_A": 12.3,
      "RLV_C": 8.2
    },
    "landmarks": [
      [12.0, 34.0, 56.0],
      [18.0, 35.0, 57.0]
    ]
  }
]
```

Depending on `system.save_intermediate`, each image output directory can also contain:

- `tent_TTA.pth`
- `landmarks.txt`
- `landmarks_original.txt`
- `landmarks.png`
- `length_centile.csv`
- `length_centile.png`
- `length_centile_web.png`

## Model Safety

Only load model checkpoints from trusted sources. PyTorch checkpoints can contain serialized data and should be treated as executable artifacts when loaded with compatibility fallbacks.
