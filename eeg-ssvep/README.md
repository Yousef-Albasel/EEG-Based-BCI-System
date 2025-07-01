# SSVEP EEG Classification Pipeline

A comprehensive Python pipeline for **Steady-State Visual Evoked Potential (SSVEP)** EEG classification using **Filter Bank Canonical Correlation Analysis (FBCCA)** and **Support Vector Machines (SVM)**.

## Overview

This project implements a state-of-the-art SSVEP classification system that can distinguish between different visual stimulation frequencies corresponding to movement directions:

- **Forward**: 7 Hz
- **Backward**: 8 Hz  
- **Left**: 10 Hz
- **Right**: 13 Hz

## Project Structure

```
eeg-ssvep/
├── config.py                    # Configuration parameters
├── extract_features.py          # Feature extraction pipeline
├── main.py                     # Main entry point
├── models.py                   # Classification models
├── requirements.txt            # Dependencies
├── README.md                  # This file
├── data/
│   ├── __init__.py
│   ├── EEGDataset.py          # Dataset handling
│   └── features/              # Extracted features
├── models/                    # Trained models
├── preprocessing/
│   ├── __init__.py
│   └── EEGPreprocessor.py     # EEG preprocessing
├── scripts/
│   ├── __init__.py
│   ├── train.py              # Training script
│   ├── validation.py         # Validation script
│   └── inference.py          # Inference script
└── utils/
    ├── __init__.py
    ├── visualization_utils.py  # Plotting utilities
    └── train_utils.py         # Training utilities
```

## Installation

1. **Clone or download the project:**
   ```bash
   cd "d:\python projects\eeg-ssvep"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python main.py
   ```

### Available Modes

1. **extract** - Extract FBCCA features only
2. **train** - Train SVM classifier (auto-extracts if needed)
3. **full_train** - Complete pipeline (extract + train)
4. **validate** - Validate model (auto-extracts if needed)
5. **loso** - Leave-One-Subject-Out cross-validation
6. **infer** - Make predictions on new data

### 1. Feature Extraction

Extract FBCCA features from raw EEG data:

```bash
python main.py --mode extract --raw_data_path /path/to/dataset
```

This will:
- Load CSV files (train.csv, validation.csv, test.csv)
- Filter for SSVEP tasks
- Extract enhanced FBCCA features
- Save features to `data/features/ssvep_features.pkl`

### 2. Model Training

Train the SVM classifier with enhanced options:

```bash
# Train with existing features (uses defaults)
python main.py --mode train

# Train with auto-extraction
python main.py --mode train --raw_data_path /path/to/dataset

# Train with custom paths
python main.py --mode train --data_path custom_features.pkl --save_dir custom_models

# Full training pipeline (recommended)
python main.py --mode full_train --raw_data_path /path/to/dataset
```

### 3. Model Validation

Validate the trained model:

```bash
# Validate with defaults
python main.py --mode validate

# Validate with auto-extraction
python main.py --mode validate --raw_data_path /path/to/dataset

# Validate custom model
python main.py --mode validate --model_path custom_model.joblib --raw_data_path /path/to/dataset
```

### 4. Leave-One-Subject-Out Cross-Validation

```bash
# LOSO-CV with defaults
python main.py --mode loso --raw_data_path /path/to/dataset

# LOSO-CV with custom model
python main.py --mode loso --model_path custom_model.joblib --raw_data_path /path/to/dataset
```

### 5. Inference

Make predictions on new data:

```bash
# Inference with defaults
python main.py --mode infer --raw_data_path /path/to/dataset

# Inference with custom output
python main.py --mode infer --raw_data_path /path/to/dataset --output_path my_predictions.csv

# Inference with custom model
python main.py --mode infer --model_path custom_model.joblib --raw_data_path /path/to/dataset
```

### 🔧 Default Paths

The pipeline uses smart defaults to reduce command complexity:

- **Features**: `data/features/ssvep_features.pkl`
- **Model**: `models/ssvep_model.joblib`
- **Scaler**: `models/scaler.joblib`
- **Predictions**: `predictions.csv`