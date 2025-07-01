"""
Validation script for SSVEP classification
"""

import pickle
import numpy as np
from models import SSVEPClassifier


def loso_eval(model_path, data_path):
    """
    Perform Leave-One-Subject-Out evaluation
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model
    data_path : str
        Path to the features data
    """
    print("🧪 Starting Leave-One-Subject-Out Evaluation...")
    print("="*50)
    
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    y_train = data['y_train']
    train_df = data['train_df']
    
    # Initialize classifier and load model
    classifier = SSVEPClassifier()
    
    # Extract model and scaler paths
    import os
    model_dir = os.path.dirname(model_path)
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    
    classifier.load_model(model_path, scaler_path)
    
    # Perform LOSO-CV
    if 'subject_id' in train_df.columns:
        subject_ids = train_df['subject_id'].values
        loso_results = classifier.leave_one_subject_out_cv(X_train, y_train, subject_ids)
        
        # Save results
        results_path = os.path.join(model_dir, 'loso_evaluation_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(loso_results, f)
        
        print(f"💾 LOSO evaluation results saved to {results_path}")
        return loso_results
    else:
        print("❌ Subject ID information not available for LOSO evaluation")
        return None


def validate_model(model_path, data_path):
    """
    Validate trained model on validation set
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model
    data_path : str
        Path to the features data
    """
    print("📈 Starting Model Validation...")
    print("="*40)
    
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X_val = data['X_val']
    y_val = data['y_val']
    
    # Initialize classifier and load model
    classifier = SSVEPClassifier()
    
    # Extract model and scaler paths
    import os
    model_dir = os.path.dirname(model_path)
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    
    classifier.load_model(model_path, scaler_path)
    
    # Evaluate
    results = classifier.evaluate(X_val, y_val)
    
    # Save results
    results_path = os.path.join(model_dir, 'validation_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"💾 Validation results saved to {results_path}")
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python validation.py <mode> <model_path> <data_path>")
        print("Modes: loso, validate")
        sys.exit(1)
    
    mode = sys.argv[1]
    model_path = sys.argv[2]
    data_path = sys.argv[3]
    
    if mode == "loso":
        loso_eval(model_path, data_path)
    elif mode == "validate":
        validate_model(model_path, data_path)
    else:
        print("Invalid mode. Use 'loso' or 'validate'")
        sys.exit(1)
