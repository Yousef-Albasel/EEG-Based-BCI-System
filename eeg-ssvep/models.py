"""
Machine learning models for SSVEP classification
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
import joblib
from config import MODEL_CONFIG


class SSVEPClassifier:
    """
    SVM-based classifier for SSVEP EEG data
    """
    
    def __init__(self, model_config=None):
        if model_config is None:
            model_config = MODEL_CONFIG
            
        self.model = SVC(**model_config)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X_train, y_train):
        """
        Train the classifier
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training labels
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        print("✅ Model training completed!")
        
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : numpy.ndarray
            Features to predict
            
        Returns:
        --------
        numpy.ndarray
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Parameters:
        -----------
        X : numpy.ndarray
            Features to predict
            
        Returns:
        --------
        numpy.ndarray
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X_val, y_val):
        """
        Evaluate the model
        
        Parameters:
        -----------
        X_val : numpy.ndarray
            Validation features
        y_val : numpy.ndarray
            Validation labels
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        y_pred = self.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"✅ Validation Accuracy: {accuracy:.4f}")
        print("\n📊 Classification Report:")
        print(classification_report(y_val, y_pred))
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'classification_report': classification_report(y_val, y_pred, output_dict=True)
        }
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        
        Parameters:
        -----------
        X : numpy.ndarray
            Features
        y : numpy.ndarray
            Labels
        cv : int
            Number of folds
            
        Returns:
        --------
        dict
            Cross-validation results
        """
        X_scaled = self.scaler.transform(X) if self.is_fitted else self.scaler.fit_transform(X)
        
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
        
        print(f"📊 CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
    
    def leave_one_subject_out_cv(self, X, y, subject_ids):
        """
        Perform Leave-One-Subject-Out Cross-Validation
        
        Parameters:
        -----------
        X : numpy.ndarray
            Features
        y : numpy.ndarray
            Labels
        subject_ids : numpy.ndarray
            Subject IDs for each sample
            
        Returns:
        --------
        dict
            LOSO-CV results
        """
        logo = LeaveOneGroupOut()
        accuracies = []
        
        print("\n🧪 Starting Leave-One-Subject-Out Cross-Validation (LOSO-CV)")
        print("="*60)
        
        for i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=subject_ids)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            
            # Create temporary model for this fold
            temp_model = SVC(**MODEL_CONFIG)
            temp_scaler = StandardScaler()
            
            X_tr_scaled = temp_scaler.fit_transform(X_tr)
            X_te_scaled = temp_scaler.transform(X_te)
            
            temp_model.fit(X_tr_scaled, y_tr)
            y_pred = temp_model.predict(X_te_scaled)
            acc = accuracy_score(y_te, y_pred)
            accuracies.append(acc)
            
            print(f"👤 Subject {subject_ids[test_idx[0]]} - Accuracy: {acc:.4f}")
        
        mean_acc = np.mean(accuracies)
        print("\n📊 LOSO-CV Summary:")
        print(f"✅ Mean Accuracy Across Subjects: {mean_acc:.4f}")
        
        return {
            'mean_accuracy': mean_acc,
            'std_accuracy': np.std(accuracies),
            'subject_accuracies': accuracies
        }
    
    def save_model(self, model_path, scaler_path=None):
        """
        Save the trained model
        
        Parameters:
        -----------
        model_path : str
            Path to save the model
        scaler_path : str
            Path to save the scaler (optional)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        joblib.dump(self.model, model_path)
        print(f"✅ Trained model saved to {model_path}")
        
        if scaler_path:
            joblib.dump(self.scaler, scaler_path)
            print(f"✅ Scaler saved to {scaler_path}")
    
    def load_model(self, model_path, scaler_path=None):
        """
        Load a trained model
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model
        scaler_path : str
            Path to the saved scaler (optional)
        """
        self.model = joblib.load(model_path)
        print(f"✅ Model loaded from {model_path}")
        
        if scaler_path:
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Scaler loaded from {scaler_path}")
        
        self.is_fitted = True
