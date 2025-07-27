import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from extract_features import extract_features
from preprocessing.EEGPreprocessor import load_trial
from tqdm import tqdm
from config import *

class EEGDataLoader:
    def __init__(self, base_path=''):
        self.base_path = base_path
        self.train_df = None
        self.validation_df = None
        self.test_df = None
        self.sample_submission_df = None
        
    def load_csv_files(self):
        """Load main CSV files"""
        print("📥 Loading dataset CSV files...")
        
        self.train_df = pd.read_csv(os.path.join(self.base_path, 'train.csv'))
        self.validation_df = pd.read_csv(os.path.join(self.base_path, 'validation.csv'))
        self.test_df = pd.read_csv(os.path.join(self.base_path, 'test.csv'))
        self.sample_submission_df = pd.read_csv(os.path.join(self.base_path, 'sample_submission.csv'))
        
        
        if self.train_df is None:
            raise ValueError("Dataset not loaded")
            
        self.ssvep_train_df = self.train_df[self.train_df['task'] == 'SSVEP'].copy()
        self.ssvep_validation_df = self.validation_df[self.validation_df['task'] == 'SSVEP'].copy()
        self.ssvep_test_df = self.test_df[self.test_df['task'] == 'SSVEP'].copy()
        
        print(f"📊 Training examples: {len(self.ssvep_train_df)}")
        print(f"📊 Validation examples: {len(self.ssvep_validation_df)}")
        print(f"📊 Test examples: {len(self.ssvep_test_df)}")
            
        return self.ssvep_train_df, self.ssvep_validation_df, self.ssvep_test_df
        
class EEGFeatureDataset(Dataset):
    def __init__(self, df, encoder, data_path):
        self.encoder = encoder
        self.labels = encoder.transform(df['label'])
        if os.path.exists(FEATURES_PATH):
            print("📥 Loading cached features...")
            cache = np.load(FEATURES_PATH)
            self.features = cache['X']
        else:
            print("⚙️ Extracting features...")
            self.features = np.array([extract_features(load_trial(data_path, r, "train")) for _, r in tqdm(df.iterrows(), total=len(df))])
            np.savez(FEATURES_PATH, X=self.features)

    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx])

