# BioTTA Docker Guide

The Docker image uses PyTorch 2.6.0 with CUDA 12.4 and runs as a non-root user.

## Prerequisites

- Docker 20.10 or later
- NVIDIA Container Toolkit for GPU access
- A compatible NVIDIA driver
- Sufficient GPU memory for 3D inference and test-time adaptation

## Build

Pass the current user's UID and GID at build time. This is required when the output directory is on NFS or another bind-mounted filesystem.

```bash
docker build \
  --build-arg APP_UID=$(id -u) \
  --build-arg APP_GID=$(id -g) \
  -t biotta:latest .
```

Verify the image and its non-root identity:

```bash
docker run --rm biotta:latest python main.py --help
docker run --rm biotta:latest id
```

## Run One File

```bash
mkdir -p ./results ./templates_registered

docker run --rm --gpus '"device=0"' \
  -v "$(pwd)/path/to/input:/data/input:ro" \
  -v "$(pwd)/results:/data/output" \
  -v "$(pwd)/templates_registered:/data/registered" \
  biotta:latest \
  python main.py \
    --input /data/input/sample.nii.gz \
    --age 31 \
    --registered_folder /data/registered \
    --output /data/output \
    --gpu 0
```

The image already contains the bundled models, atlas templates, and development curves. To substitute any of them, mount a read-only directory over the corresponding path:

```bash
-v /path/to/models:/app/models:ro
-v /path/to/templates:/app/templates:ro
-v /path/to/development_curves:/app/development_curves:ro
```

## Batch Processing

```bash
mkdir -p ./results ./templates_registered

docker run --rm --gpus '"device=0"' \
  -v "$(pwd)/path/to/input:/data/input:ro" \
  -v "$(pwd)/path/to/ages.csv:/data/ages.csv:ro" \
  -v "$(pwd)/results:/data/output" \
  -v "$(pwd)/templates_registered:/data/registered" \
  biotta:latest \
  python main.py \
    --input /data/input \
    --age_csv /data/ages.csv \
    --registered_folder /data/registered \
    --output /data/output \
    --gpu 0
```

## Permission Checks

The files below should have the same numeric owner as the host user:

```bash
id -u
id -g
ls -ln ./results
```

If the image was built for a different host user, rebuild it with the correct `APP_UID` and `APP_GID`. Do not work around NFS permission errors by running the container as root.

## Troubleshooting

Check GPU access:

```bash
docker run --rm --gpus all biotta:latest \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Check ANTsPyX:

```bash
docker run --rm biotta:latest \
  python -c "import ants; print(ants.__version__)"
```

Check the installed package versions:

```bash
docker run --rm biotta:latest python -m pip list
```
