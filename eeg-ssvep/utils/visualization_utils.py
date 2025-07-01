"""
Visualization utilities for SSVEP EEG analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from config import SSVEP_FREQS

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_confusion_matrix(y_true, y_pred, classes=None, title="Confusion Matrix", figsize=(8, 6)):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    classes : list
        Class names
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if classes is None:
        classes = np.unique(y_true)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


def plot_feature_distribution(features, labels, feature_names=None, title="Feature Distribution"):
    """
    Plot feature distribution by class
    
    Parameters:
    -----------
    features : numpy.ndarray
        Feature matrix
    labels : numpy.ndarray
        Labels
    feature_names : list
        Feature names
    title : str
        Plot title
    """
    n_features = min(features.shape[1], 10)  # Show max 10 features
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    unique_labels = np.unique(labels)
    
    for i in range(n_features):
        for label in unique_labels:
            mask = labels == label
            axes[i].hist(features[mask, i], alpha=0.7, label=f'Class {label}', bins=20)
        
        feature_name = feature_names[i] if feature_names else f'Feature {i}'
        axes[i].set_title(f'{feature_name}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_accuracy_by_subject(subject_accuracies, subject_ids=None, title="Accuracy by Subject"):
    """
    Plot accuracy by subject
    
    Parameters:
    -----------
    subject_accuracies : list
        Accuracy for each subject
    subject_ids : list
        Subject IDs
    title : str
        Plot title
    """
    if subject_ids is None:
        subject_ids = range(len(subject_accuracies))
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(subject_accuracies)), subject_accuracies, 
                   color=sns.color_palette("husl", len(subject_accuracies)))
    
    plt.xlabel('Subject')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.xticks(range(len(subject_accuracies)), subject_ids, rotation=45)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, subject_accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.axhline(y=np.mean(subject_accuracies), color='red', linestyle='--', 
                label=f'Mean: {np.mean(subject_accuracies):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_ssvep_frequencies():
    """
    Plot SSVEP stimulus frequencies
    """
    frequencies = list(SSVEP_FREQS.values())
    directions = list(SSVEP_FREQS.keys())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(directions, frequencies, color=sns.color_palette("husl", len(frequencies)))
    
    plt.xlabel('Direction')
    plt.ylabel('Frequency (Hz)')
    plt.title('SSVEP Stimulus Frequencies')
    
    # Add value labels on bars
    for bar, freq in zip(bars, frequencies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{freq} Hz', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_training_history(results, title="Training Results"):
    """
    Plot training results
    
    Parameters:
    -----------
    results : dict
        Training results dictionary
    title : str
        Plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Validation accuracy
    if 'validation_results' in results:
        val_acc = results['validation_results']['accuracy']
        axes[0].bar(['Validation'], [val_acc], color='skyblue')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Validation Accuracy')
        axes[0].set_ylim([0, 1])
        axes[0].text(0, val_acc + 0.02, f'{val_acc:.3f}', ha='center', va='bottom')
    
    # Cross-validation scores
    if 'cv_results' in results:
        cv_scores = results['cv_results']['cv_scores']
        axes[1].bar(range(len(cv_scores)), cv_scores, color='lightgreen')
        axes[1].axhline(y=np.mean(cv_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(cv_scores):.3f}')
        axes[1].set_xlabel('Fold')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Cross-Validation Scores')
        axes[1].legend()
    
    # LOSO results
    if 'loso_results' in results and results['loso_results'] is not None:
        loso_acc = results['loso_results']['subject_accuracies']
        axes[2].bar(range(len(loso_acc)), loso_acc, color='salmon')
        axes[2].axhline(y=np.mean(loso_acc), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(loso_acc):.3f}')
        axes[2].set_xlabel('Subject')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_title('LOSO-CV Accuracy by Subject')
        axes[2].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
