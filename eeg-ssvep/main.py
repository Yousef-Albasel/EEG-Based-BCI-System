"""
Main entry point for SSVEP EEG Classification Pipeline
"""

import argparse
import os
import sys
from extract_features import main as extract_main
from scripts.train import main as train_main
from scripts.validation import loso_eval, validate_model
from scripts.inference import infer


def main():
    parser = argparse.ArgumentParser(description="SSVEP EEG Classification Pipeline")
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['extract', 'train', 'full_train', 'validate', 'loso', 'infer'],
                       help="Pipeline mode to run")
    
    # Data paths
    parser.add_argument('--raw_data_path', type=str, 
                       help="Path to raw EEG dataset (for extraction/inference/auto-extraction)")
    parser.add_argument('--data_path', type=str, 
                       help="Path to pickled features file (for training/validation/inference). "
                            "If not provided, will use default: data/features/ssvep_features.pkl")
    
    # Model paths
    parser.add_argument('--model_path', type=str, 
                       help="Path to trained model (for validation/inference). "
                            "If not provided, will use default: models/ssvep_model.joblib")
    parser.add_argument('--scaler_path', type=str,
                       help="Path to trained scaler (for validation/inference). "
                            "If not provided, will use default: models/scaler.joblib")
    parser.add_argument('--save_dir', type=str, default="models", 
                       help="Directory to save trained models")
    
    # Output paths
    parser.add_argument('--output_path', type=str, default="predictions.csv", 
                       help="Path to save predictions (inference)")
    
    # Other parameters
    parser.add_argument('--scaler', type=str, default="standard", 
                       choices=["standard", "minmax"], 
                       help="Scaler type used during training")
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    if not args.data_path:
        args.data_path = "data/features/ssvep_features.pkl"
    
    if not args.model_path:
        args.model_path = "models/ssvep_model.joblib"
    
    if not args.scaler_path:
        args.scaler_path = "models/scaler.joblib"
    
    print("🧠 SSVEP EEG Classification Pipeline")
    print("="*50)
    print(f"Mode: {args.mode}")
    print(f"Data path: {args.data_path}")
    if args.mode in ['validate', 'loso', 'infer']:
        print(f"Model path: {args.model_path}")
        print(f"Scaler path: {args.scaler_path}")
    print("="*50)
    
    try:
        if args.mode == 'extract':
            if not args.raw_data_path:
                raise ValueError("--raw_data_path is required for feature extraction")
            
            print("🔄 Starting feature extraction...")
            extract_main(args.raw_data_path)
            
        elif args.mode == 'train':
            # Check if features exist, if not require raw_data_path for extraction
            if not os.path.exists(args.data_path):
                if not args.raw_data_path:
                    raise ValueError(f"Features file not found at {args.data_path}. "
                                   "Please provide --raw_data_path for automatic feature extraction, "
                                   "or run feature extraction first with --mode extract")
                
                print("📁 Features not found. Extracting features first...")
                extract_main(args.raw_data_path)
            
            print("🎯 Starting model training...")
            train_main(args.data_path, args.save_dir)
            
        elif args.mode == 'full_train':
            if not args.raw_data_path:
                raise ValueError("--raw_data_path is required for full training pipeline")
            
            print("🚀 Starting full training pipeline (extract + train)...")
            print("\n🔄 Step 1: Feature extraction...")
            extract_main(args.raw_data_path)
            
            print("\n🎯 Step 2: Model training...")
            train_main(args.data_path, args.save_dir)
            
        elif args.mode == 'validate':
            # Auto-extract features if needed
            if not os.path.exists(args.data_path):
                if not args.raw_data_path:
                    raise ValueError(f"Features file not found at {args.data_path}. "
                                   "Please provide --raw_data_path for automatic feature extraction")
                
                print("📁 Features not found. Extracting features first...")
                extract_main(args.raw_data_path)
            
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f"Model not found at {args.model_path}")
            
            print("📈 Starting model validation...")
            validate_model(args.model_path, args.data_path)
            
        elif args.mode == 'loso':
            # Auto-extract features if needed
            if not os.path.exists(args.data_path):
                if not args.raw_data_path:
                    raise ValueError(f"Features file not found at {args.data_path}. "
                                   "Please provide --raw_data_path for automatic feature extraction")
                
                print("📁 Features not found. Extracting features first...")
                extract_main(args.raw_data_path)
            
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f"Model not found at {args.model_path}")
            
            print("🧪 Starting Leave-One-Subject-Out evaluation...")
            loso_eval(args.model_path, args.data_path)
            
        elif args.mode == 'infer':
            # Auto-extract features if needed
            if not os.path.exists(args.data_path):
                if not args.raw_data_path:
                    raise ValueError(f"Features file not found at {args.data_path}. "
                                   "Please provide --raw_data_path for automatic feature extraction")
                
                print("📁 Features not found. Extracting features first...")
                extract_main(args.raw_data_path)
            
            if not args.raw_data_path:
                raise ValueError("--raw_data_path is required for inference")
            
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f"Model not found at {args.model_path}")
            
            print("🔮 Starting inference...")
            infer(args.model_path, args.data_path, args.raw_data_path, 
                  args.output_path, args.scaler)
        
        print("\n✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


def quick_demo():
    """
    Quick demo function for testing the pipeline
    """
    print("🚀 SSVEP EEG Classification Demo")
    print("="*40)
    
    print("📋 Available modes:")
    print("  - extract: Extract features only")
    print("  - train: Train model (auto-extracts features if needed)")
    print("  - full_train: Full pipeline (extract + train)")
    print("  - validate: Validate model (auto-extracts features if needed)")
    print("  - loso: Leave-One-Subject-Out evaluation")
    print("  - infer: Make predictions on new data")
    
    print("\n💡 Example usage:")
    
    print("\n1. Quick start - Full training pipeline:")
    print("   python main.py --mode full_train --raw_data_path ../eeg-data")
    
    print("\n2. Train with existing features:")
    print("   python main.py --mode train")
    print("   # Uses default: data/features/ssvep_features.pkl")
    
    print("\n3. Train with custom data path:")
    print("   python main.py --mode train --data_path custom_features.pkl")
    
    print("\n4. Extract features only:")
    print("   python main.py --mode extract --raw_data_path ../eeg-data")
    
    print("\n5. Validate with defaults:")
    print("   python main.py --mode validate")
    print("   # Uses: models/ssvep_model.joblib, data/features/ssvep_features.pkl")
    
    print("\n6. Validate with auto feature extraction:")
    print("   python main.py --mode validate --raw_data_path ../eeg-data")
    
    print("\n7. LOSO evaluation:")
    print("   python main.py --mode loso --raw_data_path ../eeg-data")
    
    print("\n8. Inference with custom model:")
    print("   python main.py --mode infer --raw_data_path ../eeg-data --model_path custom_model.joblib")
    
    print("\n🔧 Default paths:")
    print("  - Features: data/features/ssvep_features.pkl")
    print("  - Model: models/ssvep_model.joblib")
    print("  - Scaler: models/scaler.joblib")
    print("  - Output: predictions.csv")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        quick_demo()
    else:
        main()
