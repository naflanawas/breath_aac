# MURMUR: Breath-Based AAC System

A breath-based **Augmentative and Alternative Communication (AAC)** system for individuals with severe motor impairments. This repository contains the ML training pipeline for classifying breath patterns using a Multi-Scale Temporal Convolutional Network (MS-TCN) with Prototypical Network personalization.

## Overview

MURMUR enables communication through breath control by:
1. Classifying breath patterns into **SHORT** (puff) or **LONG** (sustained) gestures
2. Personalizing to individual users with just **5 calibration samples** using few-shot learning
3. Providing real-time inference with **< 100ms latency**

## Project Structure

```
breath_aac/
├── src/
│   ├── audio/                    # Audio preprocessing
│   │   ├── segment_gestures.py   # Segment Coswara data into short/long
│   │   ├── standardize.py        # Normalize audio to fixed lengths
│   │   └── make_subjectwise_split.py
│   ├── features/                 # Feature extraction
│   │   ├── mel_delta.py          # Mel-spectrogram + delta features
│   │   ├── batch_mel.py          # Batch feature extraction
│   │   ├── protonet.py           # Prototypical network utilities
│   │   └── viz_mel.py            # Visualization
│   ├── train/                    # Model training
│   │   ├── train_ms_tcn_2c.py    # MS-TCN training script
│   │   └── protonet_cal_2c.py    # ProtoNet calibration evaluation
│   ├── infer/                    # Inference
│   │   └── infer_ms_tcn_2c.py    # Audio file inference
│   └── explain/                  # Explainability
│       └── gradcam_ms_tcn_2c.py  # Grad-CAM visualization
├── models/                       # Trained model checkpoints (.pt)
├── manifests/                    # Data split CSVs
├── features/                     # Extracted mel-spectrogram features (.npy)
├── data_segments/                # Segmented audio clips
├── data_std/                     # Standardized audio clips
├── notebooks/                    # Jupyter notebooks
│   └── protonet_eval.ipynb       # ProtoNet evaluation notebook
└── Coswara-Data-master/          # Raw Coswara dataset
```

## Model Architecture

### MS-TCN (Multi-Scale Temporal Convolutional Network)

```
Input: [B, 3, 64, 256]  (3 channels × 64 mel bins × 256 time frames)
    ↓
┌─────────────────────────────────────┐
│  STEM: Conv2d(3→64) + Conv2d(64→64) │
└─────────────────────────────────────┘
    ↓
┌───────────┬───────────┬───────────┬───────────┐
│ TCN d=1   │ TCN d=2   │ TCN d=4   │ TCN d=8   │
│ (local)   │ (short)   │ (medium)  │ (long)    │
└─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────┘
      └───────────┴───────────┴───────────┘
                     ↓
        ┌─────────────────────────────┐
        │ Fusion + Pool + Classifier  │
        └─────────────────────────────┘
                     ↓
          Output: [B, 2] (short/long)
```

**Key Features:**
- **Multi-scale temporal processing** with dilation rates 1, 2, 4, 8
- **SpecAugment** data augmentation for robustness
- **64-dimensional embeddings** for ProtoNet personalization

## Quick Start

### Prerequisites

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 1. Data Preprocessing

```bash
# Segment Coswara breathing recordings
python -m src.audio.segment_gestures \
    --in_root Coswara-Data-master \
    --out_root data_segments \
    --manifest manifests/segments_2c.csv

# Standardize audio to fixed lengths
python -m src.audio.standardize \
    --in_root data_segments \
    --out_root data_std
```

### 2. Feature Extraction

```bash
# Extract mel-spectrogram + delta features
python -m src.features.batch_mel \
    --manifest manifests/segments_2c.csv \
    --out_root features/mel_dd_subjectwise
```

### 3. Training

```bash
# Train MS-TCN model
python -m src.train.train_ms_tcn_2c \
    --split_csv manifests/split_2c_subjectwise.csv \
    --epochs 40 \
    --bs 8 \
    --max_len 1024 \
    --ckpt models/ms_tcn_2c_coswara.pt
```

### 4. Inference

```bash
# Classify a single audio file
python -m src.infer.infer_ms_tcn_2c \
    --wav path/to/breath.wav \
    --ckpt models/ms_tcn_2c_coswara.pt
```

### 5. Visualization (Grad-CAM)

```bash
# Generate Grad-CAM explanation
python -m src.explain.gradcam_ms_tcn_2c \
    --wav path/to/breath.wav \
    --ckpt models/ms_tcn_2c_coswara.pt \
    --out viz/gradcam/output.png
```

## Features

| Feature | Description |
|---------|-------------|
| **Mel Spectrogram** | 64 mel bins, 50-8000 Hz |
| **Delta** | First derivative (velocity) |
| **Delta-Delta** | Second derivative (acceleration) |
| **Sample Rate** | 16 kHz |
| **FFT Window** | 1024 samples (~64ms) |
| **Hop Length** | 256 samples (~16ms) |

## Personalization (ProtoNet)

The system uses Prototypical Networks for few-shot personalization:

1. **Calibration**: User provides 5 samples per breath type
2. **Embedding**: MS-TCN extracts 64-dim embeddings
3. **Prototype**: Mean embedding per class
4. **Prediction**: Nearest-prototype classification (cosine similarity)

```python
# Evaluate ProtoNet personalization
python -m src.train.protonet_cal_2c \
    --split_csv manifests/split_2c_subjectwise.csv \
    --ckpt models/ms_tcn_2c_coswara.pt \
    --shots 5
```

## Results

| Metric | Global Model | Personalized (5-shot) |
|--------|--------------|----------------------|
| Accuracy | ~85% | ~95%+ |
| F1 Score | ~0.85 | ~0.95+ |

## Dataset

This project uses the [Coswara Dataset](https://github.com/iiscleap/Coswara-Data):
- **breathing-shallow.wav** → SHORT breath class
- **breathing-deep.wav** → LONG breath class

## Key Dependencies

- `torch` >= 2.0 - Deep learning framework
- `librosa` - Audio processing & feature extraction
- `scikit-learn` - Evaluation metrics
- `pandas` - Data manipulation
- `matplotlib` - Visualization

## License

MIT License

## Acknowledgments

- [Coswara Project](https://coswara.iisc.ac.in/) for the breathing dataset
- [IISc Bangalore](https://www.iisc.ac.in/) for data collection