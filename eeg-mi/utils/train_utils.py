from models import get_model_configs
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler
)
import os

def train_with_scalers(X_train, y_train, X_val, y_val, save_dir=None):
    scalers = {
        "MinMax": MinMaxScaler(),
        "Standard": StandardScaler()
    }

    models = get_model_configs()
    all_results = {}

    for scaler_name, scaler in scalers.items():
        print(f"\n🔧 Testing with Scaler: {scaler_name}")
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        best_model = None
        best_model_name = None
        best_score = -1

        for name, (model, params) in models.items():
            print(f"Training: {name} with {scaler_name}")
            grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train_scaled, y_train)

            best = grid.best_estimator_
            preds = best.predict(X_val_scaled)
            acc = accuracy_score(y_val, preds)

            print("Classification Report:\n", classification_report(y_val, preds))
            print(f"[{scaler_name}] {name} Best Params: {grid.best_params_}")
            print(f"[{scaler_name}] {name} Val Accuracy: {acc:.4f}")

            if acc > best_score:
                best_score = acc
                best_model = best
                best_model_name = name

        print(f"\Best Model for {scaler_name}: {best_model_name} ({best_score:.4f})")
        print("-" * 60)

        all_results[scaler_name] = {
            "best_model_name": best_model_name,
            "best_score": best_score,
            "model": best_model
        }

        if save_dir:
            import joblib
            os.makedirs(save_dir, exist_ok=True)
            model_path = f"{save_dir}/model_{scaler_name}_{best_model_name}.pkl"
            joblib.dump(best_model, model_path)
            print(f"Saved model to {model_path}")

    return all_results
