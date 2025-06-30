import argparse
from extract_features import main as extract_main
from scripts.train import main as train_main
from scripts.validation import loso_eval
from scripts.inference import infer

import torch
import numpy as np
import os
torch.manual_seed(42)
np.random.seed(42)

def main():
    parser = argparse.ArgumentParser(description="EEG MI Classification Pipeline Entry Point")
    parser.add_argument('--mode', type=str, required=True, choices=['extract', 'train', 'loso', 'infer'])

    parser.add_argument('--data_path', type=str, help="Path to pickled features file (for training/LOSO/inference)")
    parser.add_argument('--save_dir', type=str, default="models", help="Where to save trained models")
    parser.add_argument('--raw_data_path', type=str, help="Path to raw EEG dataset (for extraction/inference)")
    parser.add_argument('--model_path', type=str, help="Path to the trained model for LOSO evaluation/inference")
    parser.add_argument('--output_path', type=str, default="predictions.csv", help="Path to save predictions (inference)")
    parser.add_argument('--scaler', type=str, default="standard", choices=["standard", "minmax"], help="Scaler type used during training")

    args = parser.parse_args()

    if args.mode == 'extract':
        if not args.raw_data_path:
            raise ValueError("You must provide --raw_data_path for extraction")
        extract_main(base_path=args.raw_data_path)

    elif args.mode == 'train':
        if not args.data_path:
            raise ValueError("You must provide --data_path for training")
        train_main(data_path=args.data_path, save_dir=args.save_dir)

    elif args.mode == 'loso':
        if not args.data_path or not args.model_path:
            raise ValueError("You must provide both --data_path and --model_path for LOSO evaluation")
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model not found at {args.model_path}")
        loso_eval(model_path=args.model_path, data_path=args.data_path)
        
    elif args.mode == 'infer':
        if not args.model_path or not args.data_path or not args.raw_data_path:
            raise ValueError("For inference, provide --model_path, --data_path, and --raw_data_path")
        infer(
            model_path=args.model_path,
            data_path=args.data_path,
            base_path=args.raw_data_path,
            output_path=args.output_path,
            scaler_type=args.scaler
        )
if __name__ == "__main__":
    main()
