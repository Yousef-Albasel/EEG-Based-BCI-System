import os
import numpy as np
import torch
import pandas as pd
from dataset.SlidingWindowEEGDataset import SlidingWindowEEGDataset
from models.UFE import UFE
from models.CSC import CrossSubjectClassifier
from feature_selection import FeatureTransformLayer
import torch.nn.functional as F

def get_window_to_trial_mapping(dataset):
    mapping = []
    for trial_idx, trial_path in enumerate(dataset.trial_paths):
        data_dict = np.load(trial_path, allow_pickle=True).item()
        data = data_dict['data']
        T, C = data.shape
        if T >= dataset.window_size:
            n_windows = (T - dataset.window_size) // dataset.stride + 1
        else:
            n_windows = 1
        mapping.extend([trial_idx] * n_windows)
    return mapping

def aggregate_predictions(window_logits, window_to_trial, num_trials):
    trial_probs = [[] for _ in range(num_trials)]
    for logit, trial_idx in zip(window_logits, window_to_trial):
        prob = F.softmax(torch.tensor(logit), dim=0).numpy()
        trial_probs[trial_idx].append(prob)
    trial_preds = []
    for probs in trial_probs:
        if probs:
            mean_prob = np.mean(probs, axis=0)
            trial_pred = np.argmax(mean_prob)
        else:
            trial_pred = -1
        trial_preds.append(trial_pred)
    return trial_preds

def run_inference(
    cache_dir,
    model_path,
    clf_path,
    selected_indices,
    device,
    window_size=750,
    stride=250,
    channel_names=['C3', 'CZ', 'C4'],
    batch_size=64,
    output_csv='test_predictions.csv'
):
    # Load dataset
    dataset = SlidingWindowEEGDataset(
        cache_dir=cache_dir,
        window_size=window_size,
        stride=stride,
        channel_names=channel_names,
        augment=True,
        transforms=None
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Track window-to-trial mapping
    window_to_trial = get_window_to_trial_mapping(dataset)
    num_trials = len(dataset.trial_paths)

    # Load models
    model = UFE(in_chans=len(channel_names), n_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transformer = FeatureTransformLayer(selected_indices)
    clf = CrossSubjectClassifier(m=len(selected_indices), num_classes=2).to(device)
    clf.load_state_dict(torch.load(clf_path, map_location=device))
    clf.eval()

    # Inference
    window_logits = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            _, feat32 = model(data)
            Z = transformer.transform(feat32)
            logits = clf(Z)
            window_logits.extend(logits.cpu().numpy())

    # Aggregate window predictions to trial predictions
    trial_preds = aggregate_predictions(window_logits, window_to_trial, num_trials)
    trial_labels = ['Left' if p == 0 else 'Right' if p == 1 else 'unknown' for p in trial_preds]
    trial_ids = [os.path.splitext(os.path.basename(p))[0].replace('trial_', '') for p in dataset.trial_paths]

    df = pd.DataFrame({'id': trial_ids, 'label': trial_labels})
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

