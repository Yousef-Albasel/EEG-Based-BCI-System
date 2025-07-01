"""
Enhanced FBCCA feature extraction for SSVEP classification
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from preprocessing.EEGPreprocessor import SSVEPPreprocessor
from config import SSVEP_FREQS, EXTENDED_FILTER_BANK, SAMPLING_RATE


def generate_reference_signals(freq, n_samples, fs, n_harmonics=4):
    """
    Generate reference sine and cosine signals for CCA.

    Parameters:
    -----------
    freq : float
        Fundamental frequency
    n_samples : int
        Number of samples
    fs : int
        Sampling frequency
    n_harmonics : int
        Number of harmonics to include

    Returns:
    --------
    numpy.ndarray
        Reference signals matrix
    """
    t = np.arange(n_samples) / fs
    ref_signals = []

    for harmonic in range(1, n_harmonics + 1):
        ref_signals.append(np.sin(2 * np.pi * freq * harmonic * t))
        ref_signals.append(np.cos(2 * np.pi * freq * harmonic * t))

    return np.array(ref_signals).T


def extract_enhanced_fbcca_features(eeg_data, fs=250, preprocessor=None):
    """
    Extract comprehensive FBCCA features using multiple filter banks.

    Parameters:
    -----------
    eeg_data : pandas.DataFrame
        Raw EEG data
    fs : int
        Sampling frequency
    preprocessor : SSVEPPreprocessor
        Preprocessor instance

    Returns:
    --------
    numpy.ndarray
        Enhanced FBCCA feature vector
    """
    if preprocessor is None:
        preprocessor = SSVEPPreprocessor(fs=fs)
    
    # Apply advanced preprocessing first
    preprocessed_data = preprocessor.advanced_preprocess_eeg(eeg_data)

    # Create MNE raw object
    raw = preprocessor.create_mne_raw_object(preprocessed_data, fs)
    n_samples = preprocessed_data.shape[0]
    feature_vector = []

    # Process each filter bank
    for l_freq, h_freq in EXTENDED_FILTER_BANK:
        try:
            # Apply bandpass filter
            raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, method='iir', verbose=False)
            filtered_signals = raw_filtered.get_data().T

            correlations_for_bank = []

            # Calculate correlations for each target frequency
            for freq in SSVEP_FREQS.values():
                ref_signals = generate_reference_signals(freq, n_samples, fs, n_harmonics=4)

                try:
                    cca = CCA(n_components=1)
                    cca.fit(filtered_signals, ref_signals)
                    eeg_c, ref_c = cca.transform(filtered_signals, ref_signals)
                    correlation = np.corrcoef(eeg_c.T, ref_c.T)[0, 1]
                    correlations_for_bank.append(correlation if not np.isnan(correlation) else 0)
                except:
                    correlations_for_bank.append(0)

            # Add both linear and squared correlations as features
            feature_vector.extend(correlations_for_bank)
            squared_correlations = [corr**2 for corr in correlations_for_bank]
            feature_vector.extend(squared_correlations)

        except Exception as e:
            # If filtering fails, add zeros
            feature_vector.extend([0] * (len(SSVEP_FREQS) * 2))

    return np.array(feature_vector)


def create_fbcca_dataset(df, dataset, base_path):
    """
    Create enhanced FBCCA feature dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset DataFrame
    dataset : SSVEPDataset
        Dataset instance for loading trial data
    base_path : str
        Base path to data
        
    Returns:
    --------
    tuple
        Features array and labels array (if available)
    """
    features_list = []
    labels_list = []
    has_label = 'label' in df.columns
    
    preprocessor = SSVEPPreprocessor()

    print(f"🔄 Processing {len(df)} samples for Enhanced FBCCA features...")

    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 20 == 0:
            print(f"   Progress: {idx}/{len(df)} ({100*idx/len(df):.1f}%)")

        trial_data = dataset.load_trial_data(row, base_path)
        if trial_data is not None:
            try:
                features = extract_enhanced_fbcca_features(trial_data, SAMPLING_RATE, preprocessor)
                features_list.append(features)

                if has_label:
                    labels_list.append(row['label'])
            except Exception as e:
                print(f"   Warning: Error processing sample {idx}: {e}")
                continue

    if has_label:
        return np.array(features_list), np.array(labels_list)
    else:
        return np.array(features_list), None


def main(base_path):
    """
    Main function for feature extraction
    
    Parameters:
    -----------
    base_path : str
        Base path to the dataset
    """
    from data.EEGDataset import SSVEPDataset
    import pickle
    
    # Initialize dataset
    dataset = SSVEPDataset(base_path)
    dataset.load_csv_files()
    ssvep_train_df, ssvep_validation_df, ssvep_test_df = dataset.filter_ssvep_data()
    
    print("🚀 Starting Enhanced FBCCA Feature Extraction...")
    print("="*60)

    # Training set
    print("📚 Processing training set...")
    X_train, y_train = create_fbcca_dataset(ssvep_train_df, dataset, base_path)

    # Validation set
    print("📝 Processing validation set...")
    X_val, y_val = create_fbcca_dataset(ssvep_validation_df, dataset, base_path)

    print(f"\n✅ Feature extraction completed!")
    print(f"📊 Training features shape: {X_train.shape}")
    print(f"📊 Validation features shape: {X_val.shape}")
    print(f"📊 Number of features per sample: {X_train.shape[1]}")
    
    # Save features
    features_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'train_df': ssvep_train_df,
        'val_df': ssvep_validation_df
    }
    
    with open('data/features/ssvep_features.pkl', 'wb') as f:
        pickle.dump(features_data, f)
    
    print("💾 Features saved to data/features/ssvep_features.pkl")
    
    return features_data


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = ''
    
    main(base_path)
