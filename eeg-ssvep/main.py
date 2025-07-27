import argparse
import os
import sys
from extract_features import extract_features
from scripts.train import train_main
from scripts.inference import infer
from models import load_model


def main():
    parser = argparse.ArgumentParser(description="SSVEP EEG Classification Pipeline")
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['extract', 'train', 'infer'],
                       help="Pipeline mode to run")
    
    # Data paths
    parser.add_argument('--data_path', type=str, 
                       help="Path to raw EEG dataset (for extraction/inference/auto-extraction)")
    
    # Model paths
    parser.add_argument('--model_path', type=str, 
                       help="Path to trained model (for validation/inference). "
                            "If not provided, will use default: models/ssvep_model.pt")
    parser.add_argument('--encoder_path', type=str,
                       help="Path to trained encoder (for validation/inference). "
                            "If not provided, will use default: models/encoder.joblib")
    
    args = parser.parse_args()
    
    if not args.data_path:
        raise ValueError("--data_path data set path is required")


    # Set default paths if not provided
    if not args.model_path:
        args.model_path = "models/ssvep_model.pt"
    
    if not args.encoder_path:
        args.encoder_path = "models/encoder.joblib"
    
    print("🧠 SSVEP EEG Classification Pipeline")
    print("="*50)
    print(f"Mode: {args.mode}")
    print(f"Data path: {args.data_path}")
    if args.mode == 'infer':
        print(f"Model path: {args.model_path}")
        print(f"Encoder path: {args.encoder_path}")
    print("="*50)
    
    try:
        if args.mode == 'extract':

            print("🔄 Starting feature extraction...")
            extract_features(args.data_path)

        elif args.mode == 'train':
            print("🎯 Starting model training...")
            train_main(args.data_path)
            

        elif args.mode == 'infer':
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f"Model not found at {args.model_path}")
            
            print("🔮 Starting inference...")
            model = load_model(args.model_path)
            infer(model, args.data_path , args.encoder_path)
        
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
    print("  - train: Train model")
    print("  - infer: Make predictions on the test data")
    


if __name__ == "__main__":
    if len(sys.argv) == 1:
        quick_demo()
    else:
        main()
