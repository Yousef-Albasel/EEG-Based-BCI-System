"""
Training script for SSVEP classification
"""

import pickle
import numpy as np
import os
from models import SSVEPClassifier
from sklearn.preprocessing import StandardScaler


def main(data_path, save_dir="models"):
    """
    Main training function
    
    Parameters:
    -----------
    data_path : str
        Path to the pickled features file
    save_dir : str
        Directory to save trained models
    """
    print("🚀 Starting SSVEP Model Training...")
    print("="*50)
    
    # Load features
    print("📂 Loading features...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    train_df = data['train_df']
    
    print(f"📊 Training features shape: {X_train.shape}")
    print(f"📊 Validation features shape: {X_val.shape}")
    
    # Initialize classifier
    classifier = SSVEPClassifier()
    
    # Train the model
    print("\n🎯 Training SVM classifier...")
    classifier.fit(X_train, y_train)
    
    # Evaluate on validation set
    print("\n📈 Evaluating on validation set...")
    val_results = classifier.evaluate(X_val, y_val)
    
    # Cross-validation
    print("\n🔄 Performing cross-validation...")
    cv_results = classifier.cross_validate(X_train, y_train, cv=5)
    
    # Leave-One-Subject-Out Cross-Validation
    if 'subject_id' in train_df.columns:
        print("\n🧪 Performing Leave-One-Subject-Out CV...")
        subject_ids = train_df['subject_id'].values
        loso_results = classifier.leave_one_subject_out_cv(X_train, y_train, subject_ids)
    
    # Save the model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'ssvep_model.joblib')
    scaler_path = os.path.join(save_dir, 'scaler.joblib')
    
    classifier.save_model(model_path, scaler_path)
    
    # Save training results
    results = {
        'validation_results': val_results,
        'cv_results': cv_results,
        'loso_results': loso_results if 'subject_id' in train_df.columns else None,
        'model_path': model_path,
        'scaler_path': scaler_path
    }
    
    results_path = os.path.join(save_dir, 'training_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Training results saved to {results_path}")
    print("✅ Training completed successfully!")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python train.py <data_path> [save_dir]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    save_dir = sys.argv[2] if len(sys.argv) > 2 else "models"
    
    main(data_path, save_dir)
