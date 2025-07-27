import pandas as pd
import numpy as np 
from tqdm import tqdm
def motion_artifact_score(trial_df):
    acc_mean = trial_df[['AccX', 'AccY', 'AccZ']].mean().values
    gyro_mean = trial_df[['Gyro1', 'Gyro2', 'Gyro3']].mean().values

    acc_deviation = np.linalg.norm(acc_mean - np.array([0, 0, 1]))
    gyro_deviation = np.linalg.norm(gyro_mean - np.array([0, 0, 0]))

    return acc_deviation + 0.5 * gyro_deviation

def filter_defective_trials(df, base_path, score_threshold_z=2.5):
    scores = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring trials"):
        id_num = row['id']
        if id_num <= 4800:
            dataset = 'train'
        elif id_num <= 4900:
            dataset = 'validation'
        else:
            dataset = 'test'

        eeg_path = f"{base_path}/{row['task']}/{dataset}/{row['subject_id']}/{row['trial_session']}/EEGdata.csv"
        eeg_data = pd.read_csv(eeg_path)

        trial_num = int(row['trial'])
        samples_per_trial = 2250 if row['task'] == 'MI' else 1750
        start_idx = (trial_num - 1) * samples_per_trial
        end_idx = start_idx + samples_per_trial

        trial_df = eeg_data.iloc[start_idx:end_idx]
        score = motion_artifact_score(trial_df)
        scores.append(score)

    # Convert to NumPy and Z-score
    scores = np.array(scores)
    z_scores = (scores - scores.mean()) / scores.std()

    # Filter rows where |z| < threshold
    clean_mask = np.abs(z_scores) < score_threshold_z
    return df[clean_mask].reset_index(drop=True), scores, z_scores


