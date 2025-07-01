"""
Inference script for SSVEP classification
"""

import pickle
import pandas as pd
import numpy as np
import os
from models import SSVEPClassifier
from data.EEGDataset import SSVEPDataset
from extract_features import extract_enhanced_fbcca_features
from preprocessing.EEGPreprocessor import SSVEPPreprocessor


def infer(model_path, data_path, base_path, output_path="predictions.csv", scaler_type="standard"):
    """
    Perform inference on test data
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model
    data_path : str
        Path to the features data (for test DataFrame)
    base_path : str
        Base path to raw EEG data
    output_path : str
        Path to save predictions
    scaler_type : str
        Type of scaler used during training
    """
    print("🔮 Starting SSVEP Inference...")
    print("="*40)
    
    # Load test data info
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Initialize dataset and load test DataFrame
    dataset = SSVEPDataset(base_path)
    dataset.load_csv_files()
    _, _, ssvep_test_df = dataset.filter_ssvep_data()
    
    print(f"📊 Test samples: {len(ssvep_test_df)}")
    
    # Initialize classifier and load model
    classifier = SSVEPClassifier()
    
    # Extract model and scaler paths
    model_dir = os.path.dirname(model_path)
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    
    # Check if scaler exists, if not use default location
    if not os.path.exists(scaler_path):
        scaler_path = "models/scaler.joblib"
    
    if os.path.exists(scaler_path):
        classifier.load_model(model_path, scaler_path)
    else:
        print(f"⚠️  Warning: Scaler not found at {scaler_path}. Loading model only.")
        classifier.load_model(model_path)
    
    # Extract features for test data
    print("\n🔄 Extracting features for test data...")
    preprocessor = SSVEPPreprocessor()
    
    test_features = []
    test_ids = []
    
    for idx, (_, row) in enumerate(ssvep_test_df.iterrows()):
        if idx % 20 == 0:
            print(f"   Progress: {idx}/{len(ssvep_test_df)} ({100*idx/len(ssvep_test_df):.1f}%)")
        
        trial_data = dataset.load_trial_data(row, base_path)
        if trial_data is not None:
            try:
                features = extract_enhanced_fbcca_features(trial_data, 250, preprocessor)
                test_features.append(features)
                test_ids.append(row['id'])
            except Exception as e:
                print(f"   Warning: Error processing sample {idx}: {e}")
                continue
    
    X_test = np.array(test_features)
    print(f"\n📊 Test features shape: {X_test.shape}")
    
    # Make predictions
    print("\n🎯 Making predictions...")
    predictions = classifier.predict(X_test)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': test_ids,
        'label': predictions
    })
    
    # Save predictions
    submission_df.to_csv(output_path, index=False)
    print(f"💾 Predictions saved to {output_path}")
    
    # Display prediction distribution
    print("\n📈 Prediction Distribution:")
    pred_counts = pd.Series(predictions).value_counts()
    for label, count in pred_counts.items():
        percentage = (count / len(predictions)) * 100
        print(f"  {label}: {count} samples ({percentage:.1f}%)")
    
    return submission_df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python inference.py <model_path> <data_path> <base_path> [output_path] [scaler_type]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    base_path = sys.argv[3]
    output_path = sys.argv[4] if len(sys.argv) > 4 else "predictions.csv"
    scaler_type = sys.argv[5] if len(sys.argv) > 5 else "standard"
    
    infer(model_path, data_path, base_path, output_path, scaler_type)
