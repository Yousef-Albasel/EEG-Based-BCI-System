"""
EEG Dataset handling for SSVEP classification
"""

import pandas as pd
import numpy as np
import os
from config import SAMPLES_PER_TRIAL, EEG_CHANNELS


class SSVEPDataset:
    """
    Dataset class for handling SSVEP EEG data
    """
    
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
        
        print("✅ Dataset loaded successfully!")
        
    def filter_ssvep_data(self):
        """Filter for SSVEP task only"""
        if self.train_df is None:
            raise ValueError("Dataset not loaded. Call load_csv_files() first.")
            
        self.ssvep_train_df = self.train_df[self.train_df['task'] == 'SSVEP'].copy()
        self.ssvep_validation_df = self.validation_df[self.validation_df['task'] == 'SSVEP'].copy()
        self.ssvep_test_df = self.test_df[self.test_df['task'] == 'SSVEP'].copy()
        
        print(f"📊 Training examples: {len(self.ssvep_train_df)}")
        print(f"📊 Validation examples: {len(self.ssvep_validation_df)}")
        print(f"📊 Test examples: {len(self.ssvep_test_df)}")
        
        # Display label distribution
        print("\n📈 Label Distribution in Training Set:")
        label_counts = self.ssvep_train_df['label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(self.ssvep_train_df)) * 100
            print(f"  {label}: {count} samples ({percentage:.1f}%)")
            
        return self.ssvep_train_df, self.ssvep_validation_df, self.ssvep_test_df
    
    def load_trial_data(self, row, base_path=None):
        """
        Load individual trial data
        
        Parameters:
        -----------
        row : pandas.Series
            Row from the dataset CSV
        base_path : str
            Base path to the dataset
            
        Returns:
        --------
        pandas.DataFrame or None
            Trial data or None if file not found
        """
        if base_path is None:
            base_path = self.base_path
            
        # Determine dataset folder based on ID
        id_num = row['id']
        if id_num <= 4800:
            dataset_folder = 'train'
        elif id_num <= 4900:
            dataset_folder = 'validation'
        else:
            dataset_folder = 'test'

        # Construct file path
        eeg_path = os.path.join(
            base_path, row['task'], dataset_folder,
            row['subject_id'], str(row['trial_session']), 'EEGdata.csv'
        )

        # Check if file exists
        if not os.path.exists(eeg_path):
            print(f"❌ Error: File not found at {eeg_path}")
            return None

        # Load EEG data
        eeg_data = pd.read_csv(eeg_path)

        # Extract specific trial
        trial_num = int(row['trial'])
        start_idx = (trial_num - 1) * SAMPLES_PER_TRIAL
        end_idx = start_idx + SAMPLES_PER_TRIAL

        trial_data = eeg_data.iloc[start_idx:end_idx].copy()
        return trial_data
    
    def get_dataset_info(self):
        """Get information about the loaded dataset"""
        if self.train_df is None:
            return "Dataset not loaded"
            
        info = {
            'total_train': len(self.train_df),
            'total_validation': len(self.validation_df),
            'total_test': len(self.test_df),
            'ssvep_train': len(self.ssvep_train_df) if hasattr(self, 'ssvep_train_df') else 0,
            'ssvep_validation': len(self.ssvep_validation_df) if hasattr(self, 'ssvep_validation_df') else 0,
            'ssvep_test': len(self.ssvep_test_df) if hasattr(self, 'ssvep_test_df') else 0,
        }
        return info
