import os
import pickle
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import load

def loso_eval(model_path, data_path):
    print(f"Loading model from: {model_path}")
    model = load(model_path)

    print(f"Loading LOSO data from: {data_path}")
    with open(data_path, "rb") as f:
        X_train, y_train, _, _, subject_ids = pickle.load(f)

    X, y = X_train, y_train
    logo = LeaveOneGroupOut()
    scaler = StandardScaler()

    all_accuracies = []
    print("Starting LOSO Evaluation...\n")

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, subject_ids)):
        X_train_fold = scaler.fit_transform(X[train_idx])
        X_test_fold = scaler.transform(X[test_idx])
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]

        model.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_test_fold)
        acc = accuracy_score(y_test_fold, preds)

        print(f"Fold {fold_idx + 1} | Subject {np.unique(subject_ids[test_idx])[0]} | Accuracy: {acc:.4f}")
        print("Classification Report:\n", classification_report(y_test_fold, preds))
        all_accuracies.append(acc)

    avg_acc = np.mean(all_accuracies)
    print(f"\nLOSO Evaluation Complete — Avg Accuracy: {avg_acc:.4f}")
    return all_accuracies
