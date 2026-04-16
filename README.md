# MURMUR — Backend Server

MURMUR is a breath-based Augmentative and Alternative Communication (AAC) system for individuals with severe motor impairments. This repository contains the ML training pipeline and inference backend that classifies short and long breath gestures using a Multi-Scale Temporal Convolutional Network (MS-TCN) with Prototypical Network personalisation. The backend exposes the model as a REST API served via FastAPI.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn backend_server.main:app --reload
```

## Model Checkpoint

The trained model checkpoint (`ms_tcn_no_cmvn.pt`) is not included in this repository due to file size. Download it from Google Drive and place it in the `models/` directory before running the server.

> **Download:** [Google Drive link]

## Folder Structure

```
breath_aac/
├── backend_server/           # FastAPI application
│   ├── main.py               # API routes and server entry point
│   ├── model.py              # Model loading and inference wrapper
│   ├── audio_processor.py    # Audio preprocessing for incoming requests
│   └── config.py             # Runtime configuration
├── src/
│   ├── audio/                # Audio preprocessing pipeline
│   │   ├── segment_gestures.py        # Segments Coswara recordings into short/long clips
│   │   ├── standardize.py             # Resamples and pads clips to fixed length
│   │   └── make_subjectwise_split.py  # Builds train/val/test split by subject
│   ├── features/             # Feature extraction
│   │   ├── mel_delta.py      # Log-Mel + delta + delta-delta features
│   │   ├── batch_mel.py      # Batch extraction over a manifest CSV
│   │   ├── protonet.py       # Prototype computation and cosine prediction
│   │   └── viz_mel.py        # Saves three-panel feature plots
│   ├── train/                # Model training
│   │   ├── train_ms_tcn_2c.py         # MS-TCN architecture and training loop
│   │   ├── train_ablation.py          # Ablation variants (no delta, no augment, etc.)
│   │   └── protonet_cal_2c.py         # Few-shot personalisation evaluation
│   ├── infer/                # Inference utilities
│   │   ├── infer_ms_tcn_2c.py         # Single-file inference script
│   │   ├── generate_diagrams.py       # Augmentation visualisation diagrams
│   │   └── generate_verification.py   # Saves augmented WAVs for manual checking
│   ├── explain/              # Explainability
│   │   ├── gradcam_ms_tcn_2c.py       # Grad-CAM heatmaps for MS-TCN
│   │   └── gradcam_protonet_2c.py     # Grad-CAM heatmaps for ProtoNet path
│   └── utils/
│       ├── device.py         # Picks best available device (MPS / CUDA / CPU)
│       └── build_manifest.py # Scans standardised WAVs and writes a manifest CSV
├── manifests/                # Data split CSVs (tracked)
├── models/                   # Checkpoint directory (files not tracked — see above)
├── notebooks/                # Exploratory Jupyter notebooks
├── viz/                      # Saved figures and Grad-CAM outputs
└── requirements.txt
```
