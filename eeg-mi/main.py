import argparse
from extract_features import main as extract_main
from scripts.train import main as train_main
from scripts.validation import loso_eval
from scripts.inference import infer

import random
import torch
import numpy as np
import os
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['extract', 'train', 'loso', 'infer'])
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--raw_data_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--output_path', type=str, default="predictions.csv")
    parser.add_argument('--scaler', type=str, default="standard")

    args = parser.parse_args()

    if args.mode == 'extract':
        extract_main(base_path=args.raw_data_path)

    elif args.mode == 'train':
        train_main(data_path=args.data_path, save_dir=args.save_dir)

    elif args.mode == 'loso':
        loso_eval(model_path=args.model_path, data_path=args.data_path)
        
    elif args.mode == 'infer':
        infer(
            model_path=args.model_path,
            data_path=args.data_path,
            base_path=args.raw_data_path,
            output_path=args.output_path,
            scaler_type=args.scaler
        )
if __name__ == "__main__":
    main()
