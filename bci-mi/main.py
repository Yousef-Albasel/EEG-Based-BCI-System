import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from config import *
from preprocessing import preprocess_and_save_all, EEGPreprocessor
from dataset.SlidingWindowEEGDataset import SlidingWindowEEGDataset
from feature_selection import (
    collect_subject_spectra,
    select_common_features,
    compute_relation_spectrum,
)
from visualizations import (
    plot_mean_coefficients,
    plot_full_heatmap,
    plot_common_features_heatmap,
    plot_feature_distribution,
    plot_std_coefficients,
    save_tsne_plot,
)
from models.UFE import UFE
from models.CSC import CrossSubjectClassifier
from feature_selection import FeatureTransformLayer
from train import (
    get_subject_loader,
    train_model,
    train_cross_subject_classifier,
    finetune_ddfilter_per_subject,
)
from inference import run_inference

# 1. RANDOM SEEDS FOR REPRODUCIBILITY

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
print("Torch version:", torch.__version__)

# 2. LOAD DATAFRAMES AND FILTER ONLY MI TASKS

train_df = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
validation_df = pd.read_csv(os.path.join(BASE_PATH, "validation.csv"))
test_df = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))

train_df = train_df[train_df["task"] == "MI"]
validation_df = validation_df[validation_df["task"] == "MI"]
test_df = test_df[test_df["task"] == "MI"]

# 3. PREPROCESS & CACHE DATA

train_cache = r".\data\preprocessed_trials_v1"
val_cache = r".\data\preprocessed_trials_val_v1"
test_cache = r".\data\preprocessed_trials_test_v1"

preprocess_and_save_all(train_df, BASE_PATH, EEGPreprocessor(), cache_dir=train_cache)
preprocess_and_save_all(validation_df, BASE_PATH, EEGPreprocessor(), cache_dir=val_cache)

# 4. CREATE DATASETS & LOADERS

train_dataset = SlidingWindowEEGDataset(
    cache_dir=Path(train_cache),
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    augment=True,
)
val_dataset = SlidingWindowEEGDataset(
    cache_dir=Path(val_cache),
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    augment=True,
)

print(f"Training samples: {len(train_dataset)} | Shape: {train_dataset[0][0].shape}")
print(f"Validation samples: {len(val_dataset)} | Shape: {val_dataset[0][0].shape}")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64)

# 5. CREATE AND LOAD BASE FEATURE EXTRACTOR (UFE)

model = UFE(in_chans=3, n_classes=2).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters in UFE model: {total_params}")

optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
criterion = nn.CrossEntropyLoss()

# Load pretrained UFE weights
model.load_state_dict(torch.load("best_model.pth", map_location=device))

# -- If you want to fine-tune or re-train UFE --
# train_model(
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     criterion=criterion,
#     num_epochs=250,
#     device=device,
#     lr=0.0005,
#     save_path="best_model_v2.pth"
# )

model.eval()

# -- If you want to fine-tune per subject --
# for param in model.parameters():
#     param.requires_grad = False

# finetune_ddfilter_per_subject(
#     subjects=range(1, 30),
#     get_subject_loader=lambda idx, batch_size=64: get_subject_loader(
#         subject_idx=idx,
#         cache_dir=train_cache,
#         window_size=WINDOW_SIZE,
#         stride=STRIDE,
#         channel_names=['C3', 'CZ', 'C4'],
#         batch_size=batch_size,
#         augment=True
#     ),
#     ufe=model,
#     device=device,
#     num_epochs=100,
#     lr=0.0002
# )

# 6. FEATURE SELECTION (BASED ON TRAINED DDFilter OUTPUTS)

subjects_indices = list(range(1, 30))
all_spectra = collect_subject_spectra(
    subjects_indices,
    ddfilter_dir="outputs/ddfilter",
    compute_relation_spectrum_fn=compute_relation_spectrum,
)

selected_indices_path = "outputs/selected_indices_v2.npy"
if os.path.exists(selected_indices_path):
    selected_indices = np.load(selected_indices_path)
    print(f"Loaded {len(selected_indices)} selected indices from {selected_indices_path}")
else:
    selected_indices, _ = select_common_features(
        all_spectra, alpha_threshold=0.1, save_path=selected_indices_path
    )

# Visualization of feature selection
# plot_mean_coefficients(all_spectra, selected_indices)
# plot_full_heatmap(all_spectra)
# plot_common_features_heatmap(all_spectra, selected_indices)
# plot_feature_distribution(selected_indices)
# plot_std_coefficients(all_spectra, selected_indices)

# 7. CROSS-SUBJECT CLASSIFIER (CSC)

transformer = FeatureTransformLayer(selected_indices)
clf = CrossSubjectClassifier(m=len(selected_indices), num_classes=2).to(device)
clf.load_state_dict(torch.load("outputs/clf_last_v2.pth", map_location=device))


# -- If you want to train cross-subject classifier --
# train_cross_subject_classifier(
#     ufe=model,
#     transformer=transformer,
#     clf=clf,
#     train_loader=train_loader,
#     device=device,
#     num_epochs=400,
#     lr=0.001
# )

# 8. INFERENCE ON TEST SET

preprocess_and_save_all(
    test_df, BASE_PATH, EEGPreprocessor(), cache_dir=test_cache, is_test=True
)

run_inference(
    cache_dir=test_cache,
    model_path="best_model.pth",
    clf_path="outputs/clf_last_v2.pth",
    selected_indices=selected_indices,
    device=device,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    channel_names=["C3", "CZ", "C4"],
    batch_size=100,
    output_csv="outputs/test_predictions.csv",
)
