import joblib
import pandas as pd
import torch
from tqdm import tqdm
from preprocessing.EEGPreprocessor import load_trial
from extract_features import extract_features
from data.EEGDataset import EEGDataLoader


def infer(model , data_path , encoder_path):
    dataset = EEGDataLoader(data_path)
    _,_ , ssvep_test_df = dataset.load_csv_files()

    
    encoder = joblib.load(encoder_path)  

    submission_ids = []
    submission_preds = []

    print(f"📤 Generating predictions for {len(ssvep_test_df)} SSVEP test samples...")

    for i, row in tqdm(ssvep_test_df.iterrows(), total=len(ssvep_test_df)):
        trial = load_trial(data_path, row, "test")
        if trial is None:
            print(f"⚠️ Missing EEG data for ID {row['id']}, skipping...")
            continue
        features = extract_features(trial)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  
        with torch.no_grad():
            y_pred = model(x).argmax(dim=1).item()
            pred_label = encoder.inverse_transform([y_pred])[0]
            submission_ids.append(row['id'])
            submission_preds.append(pred_label)

    submission_df = pd.DataFrame({
        'id': submission_ids,
        'label': submission_preds
    })
    submission_df.to_csv("submission.csv", index=False)
    print("✅ submission.csv saved!")
