import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from sklearn.manifold import TSNE
import os
import seaborn as sns

def visualize_trial(trial_tensor, label, sample_rate=250, channel_names=None):
    if hasattr(trial_tensor, 'numpy'):
        data = trial_tensor.numpy()
    else:
        data = trial_tensor

    n_samples, n_channels = data.shape
    time = np.arange(n_samples) / sample_rate
    label_str = 'Right Hand' if label == 1 else 'Left Hand'

    if channel_names is None or len(channel_names) != n_channels:
        channel_names = [f'Ch-{i}' for i in range(n_channels)]

    fig, axs = plt.subplots(n_channels + 1, 1, figsize=(12, 2.5 * (n_channels + 1)), sharex=False)
    fig.suptitle(f"EEG Trial Visualization - {label_str}", fontsize=16)

    for i in range(n_channels):
        axs[i].plot(time, data[:, i], label=f'EEG-{channel_names[i]}')
        axs[i].set_ylabel('Amplitude (µV)')
        axs[i].legend(loc='upper right')
        axs[i].grid(True)

    axs[n_channels - 1].set_xlabel('Time (s)')

    # PSD plot (last one)
    axs[-1].set_title("Power Spectral Density (Welch)")
    for i in range(n_channels):
        f, Pxx = welch(data[:, i], fs=sample_rate, nperseg=500)
        axs[-1].semilogy(f, Pxx, label=f'EEG-{channel_names[i]}')

    axs[-1].set_xlabel('Frequency (Hz)')
    axs[-1].set_ylabel('Power')
    axs[-1].legend()
    axs[-1].grid(True)

    plt.tight_layout()
    plt.show()


def save_tsne_plot(features, labels, save_path='tsne_plots/tsne_features.png', title='t-SNE of extracted features'):
    tsne = TSNE(n_components=2, random_state=42)
    feat_2d = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    for cls in np.unique(labels):
        idx = labels == cls
        plt.scatter(feat_2d[idx, 0], feat_2d[idx, 1], label=f'Class {cls}', alpha=0.6)

    plt.legend()
    plt.title(title)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_mean_coefficients(all_spectra, selected_indices, save_path="outputs/mean_coefficients.png"):
    means = np.mean(all_spectra, axis=0)
    x = np.arange(len(means))
    colors = ['red' if i in selected_indices else 'gray' for i in x]

    plt.figure(figsize=(18,6))
    plt.bar(x, means, color=colors)
    plt.xlabel("Feature index (0–527)")
    plt.ylabel("Mean coefficient across subjects")
    plt.title("Common features (red) vs subject-specific (gray)")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_full_heatmap(all_spectra, save_path="outputs/full_heatmap.png"):
    plt.figure(figsize=(18,6))
    sns.heatmap(all_spectra, cmap='coolwarm', center=0)
    plt.xlabel("Feature index")
    plt.ylabel("Subject index")
    plt.title("Relation spectrum coefficients per subject")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_common_features_heatmap(all_spectra, selected_indices, save_path="outputs/common_features_heatmap.png"):
    common_matrix = all_spectra[:, selected_indices]
    plt.figure(figsize=(10,6))
    sns.heatmap(common_matrix, cmap='coolwarm', center=0)
    plt.xlabel("Common feature index")
    plt.ylabel("Subject index")
    plt.title("Common features coefficients per subject")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_feature_distribution(selected_indices, total_features=528, save_path="outputs/feature_distribution_pie.png"):
    sizes = [len(selected_indices), total_features - len(selected_indices)]
    labels = ['Common', 'Subject-specific']
    colors = ['red','gray']

    plt.figure(figsize=(4,4))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title("Distribution of features")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_std_coefficients(all_spectra, selected_indices, save_path="outputs/std_coefficients.png"):
    stds = np.std(all_spectra, axis=0)
    x = np.arange(len(stds))
    colors = ['red' if i in selected_indices else 'gray' for i in x]

    plt.figure(figsize=(18,6))
    plt.bar(x, stds, color=colors)
    plt.xlabel("Feature index")
    plt.ylabel("Std across subjects")
    plt.title("Variability of coefficients (common in red)")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
