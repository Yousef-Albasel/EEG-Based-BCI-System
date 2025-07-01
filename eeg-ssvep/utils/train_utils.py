"""
Training utilities for SSVEP classification
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


def stratified_split(X, y, test_size=0.2, random_state=42):
    """
    Create stratified train-test split
    
    Parameters:
    -----------
    X : numpy.ndarray
        Features
    y : numpy.ndarray
        Labels
    test_size : float
        Test set size
    random_state : int
        Random state
        
    Returns:
    --------
    tuple
        Train and test indices
    """
    from sklearn.model_selection import train_test_split
    
    train_idx, test_idx = train_test_split(
        range(len(X)), test_size=test_size, 
        stratify=y, random_state=random_state
    )
    
    return train_idx, test_idx


def calculate_metrics(y_true, y_pred, average='weighted'):
    """
    Calculate comprehensive classification metrics
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    average : str
        Averaging method for multi-class metrics
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    return metrics


def save_experiment_results(results, experiment_name, save_dir="experiments"):
    """
    Save experiment results
    
    Parameters:
    -----------
    results : dict
        Results dictionary
    experiment_name : str
        Name of the experiment
    save_dir : str
        Directory to save results
    """
    import os
    
    os.makedirs(save_dir, exist_ok=True)
    
    results_path = os.path.join(save_dir, f"{experiment_name}_results.pkl")
    
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"💾 Experiment results saved to {results_path}")


def load_experiment_results(experiment_name, save_dir="experiments"):
    """
    Load experiment results
    
    Parameters:
    -----------
    experiment_name : str
        Name of the experiment
    save_dir : str
        Directory containing results
        
    Returns:
    --------
    dict
        Results dictionary
    """
    import os
    
    results_path = os.path.join(save_dir, f"{experiment_name}_results.pkl")
    
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    return results


def grid_search_params():
    """
    Get parameter grid for hyperparameter tuning
    
    Returns:
    --------
    dict
        Parameter grid for SVM
    """
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'class_weight': ['balanced', None]
    }
    
    return param_grid


def feature_importance_analysis(X, y, feature_names=None):
    """
    Analyze feature importance using various methods
    
    Parameters:
    -----------
    X : numpy.ndarray
        Features
    y : numpy.ndarray
        Labels
    feature_names : list
        Feature names
        
    Returns:
    --------
    dict
        Feature importance results
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.ensemble import RandomForestClassifier
    
    results = {}
    
    # Univariate feature selection
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    univariate_scores = selector.scores_
    
    # Random forest feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    results['univariate_scores'] = dict(zip(feature_names, univariate_scores))
    results['random_forest_importance'] = dict(zip(feature_names, rf_importance))
    
    return results


def cross_subject_validation(X, y, subject_ids, n_folds=5):
    """
    Perform cross-subject validation
    
    Parameters:
    -----------
    X : numpy.ndarray
        Features
    y : numpy.ndarray
        Labels
    subject_ids : numpy.ndarray
        Subject IDs
    n_folds : int
        Number of folds
        
    Returns:
    --------
    dict
        Cross-validation results
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    
    gkf = GroupKFold(n_splits=n_folds)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=subject_ids)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        # Train model
        model = SVC(kernel='linear', class_weight='balanced', random_state=42)
        model.fit(X_train_scaled, y_train_fold)
        
        # Predict
        y_pred = model.predict(X_val_scaled)
        
        # Calculate metrics
        metrics = calculate_metrics(y_val_fold, y_pred)
        metrics['fold'] = fold
        metrics['val_subjects'] = np.unique(subject_ids[val_idx])
        
        fold_results.append(metrics)
        
        print(f"Fold {fold+1}: Accuracy = {metrics['accuracy']:.4f}")
    
    # Aggregate results
    mean_metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        values = [result[metric] for result in fold_results]
        mean_metrics[f'mean_{metric}'] = np.mean(values)
        mean_metrics[f'std_{metric}'] = np.std(values)
    
    return {
        'fold_results': fold_results,
        'mean_metrics': mean_metrics
    }
