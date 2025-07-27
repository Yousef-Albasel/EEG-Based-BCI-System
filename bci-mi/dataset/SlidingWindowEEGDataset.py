import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from tqdm import tqdm

class SlidingWindowEEGDataset(Dataset):
    def __init__(self, cache_dir, window_size=500, stride=50, channel_names=None, augment=True,transforms=None):
        
        self.cache_dir = cache_dir
        self.window_size = window_size
        self.stride = stride
        self.channel_names = channel_names
        self.augment = augment
        self.transforms = transforms if transforms is not None else []

        self.trial_paths = [
            os.path.join(cache_dir, fname)
            for fname in sorted(os.listdir(cache_dir))
            if fname.endswith('.npy')
        ]

        if not self.trial_paths:
            raise RuntimeError(f"No .npy files found in {cache_dir}")

        # Pre-compute sliding windows
        self.windows = []
        self.labels = []
        self._create_sliding_windows()

    def _create_sliding_windows(self):
        print(f"Creating sliding windows (window_size={self.window_size}, stride={self.stride})...")

        for trial_path in tqdm(self.trial_paths, desc="Processing trials"):
            data_dict = np.load(trial_path, allow_pickle=True).item()
            data = data_dict['data']  # Shape: (time, channels)
            label = data_dict.get('label',-1)  # Use -1 for test
            T, C = data.shape

            if self.augment:
                if T < self.window_size:
                    padded = np.zeros((self.window_size, C))
                    padded[:T] = data
                    self.windows.append(padded)
                    self.labels.append(label)
                else:
                    for start in range(0, T - self.window_size + 1, self.stride):
                        self.windows.append(data[start:start + self.window_size])
                        self.labels.append(label)
            else:
                if T >= self.window_size:
                    start = (T - self.window_size) // 2
                    window = data[start:start + self.window_size]
                else:
                    window = np.zeros((self.window_size, C))
                    window[:T] = data
                self.windows.append(window)
                self.labels.append(label)

        print(f"Created {len(self.windows)} windows from {len(self.trial_paths)} trials")
        print(f"Augmentation ratio: {len(self.windows) / len(self.trial_paths):.1f}x")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        data = torch.tensor(self.windows[idx].T, dtype=torch.float32)  # Shape: (channels, time)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.augment and self.transforms:
            for t in self.transforms:
                data = t(data)
        return data, label
