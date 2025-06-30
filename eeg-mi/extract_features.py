import pickle
from data.EEGDataset import *
from preprocessing.EEGPreprocessor import EEGPreprocessor
import pandas as pd
from tqdm import tqdm
import numpy as np
from utils.motion_artificat_utils import *
import os

def collect_features_labels(dataset, has_labels=True):
    features = []
    labels = []

    for i in tqdm(range(len(dataset)), desc="Collecting features"):
        x = dataset[i][0]
        features.append(x)

        if has_labels:
            y = dataset[i][1]
            labels.append(y)

    if has_labels:
        return np.array(features), np.array(labels)
    else:
        return np.array(features)

def main(base_path):
    # base_path = "F:\MTC-AIC3\mtcaic3"

    train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
    validation_df = pd.read_csv(os.path.join(base_path, 'validation.csv'))
    test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))

    # Filtering the dataframe for only MI task

    train_df = train_df[train_df['task'] == 'MI']
    validation_df = validation_df[validation_df['task'] == 'MI']
    test_df = test_df[test_df['task'] == 'MI']
    
    # Filtering Trainning data from motion artificats 

    train_df, _, _ = filter_defective_trials(train_df, base_path)

    m_preprocessor = EEGPreprocessor()

    train_psd = EEGDataset(train_df, base_path, preprocessor=m_preprocessor)
    val_psd = EEGDataset(validation_df, base_path, preprocessor=m_preprocessor)
    subject_ids = train_df['subject_id'].values
    
    X_train, y_train = collect_features_labels(train_psd)
    X_val, y_val = collect_features_labels(val_psd)
    os.makedirs("data/features", exist_ok=True)
    with open("data/features/mi_data.pkl", "wb") as f:
        pickle.dump((X_train, y_train, X_val, y_val,subject_ids), f)
