import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import load as joblib_load
from data.EEGDataset import EEGDataset
from preprocessing.EEGPreprocessor import EEGPreprocessor

def infer(model_path, data_path, base_path, output_path, scaler_type="standard"):
    print(f"Loading model from: {model_path}")
    model = joblib_load(model_path)

    with open(data_path, "rb") as f:
        X_train, _, _, _ = pickle.load(f)

    test_df = pd.read_csv(f"{base_path}/test.csv")
    test_df = test_df[test_df['task'] == 'MI']

    test_dataset = EEGDataset(df=test_df, base_path=base_path, preprocessor=EEGPreprocessor(), has_labels=False)

    # Extract features
    test_features = []
    print("Extracting test features...")
    for i in tqdm(range(len(test_dataset)), desc="Extracting Test PSD"):
        x = test_dataset[i]
        test_features.append(x)

    X_test = np.array(test_features)

    # Scale
    if scaler_type.lower() == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Inference
    print("Running inference...")
    preds = model.predict(X_test_scaled)
    label_map = {0: "Left", 1: "Right"}
    label_preds = [label_map[p] for p in preds]

    submission_df = pd.DataFrame({
        "id": test_df["id"].values,
        "prediction": label_preds
    })

    submission_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")
