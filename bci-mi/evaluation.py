import torch
from torch.utils.data import DataLoader
import os
from dataset import SlidingWindowEEGDataset


def get_subject_val_loader(subject_idx, cache_dir, window_size=750, stride=250, channel_names=None, batch_size=64):
    trials_per_subject = 10
    start_idx = subject_idx * trials_per_subject
    end_idx = start_idx + trials_per_subject

    all_trial_paths = [
        os.path.join(cache_dir, fname)
        for fname in sorted(os.listdir(cache_dir))
        if fname.endswith('.npy')
    ]
    subject_trial_paths = all_trial_paths[start_idx:end_idx]

    # Just pass the filtered trial paths to the dataset
    dataset = SlidingWindowEEGDataset(
        cache_dir=None,  # Not used since we pass trial_paths directly
        window_size=window_size,
        stride=stride,
        channel_names=channel_names,
        augment=False,
        transforms=None
    )
    dataset.trial_paths = subject_trial_paths
    dataset.windows = []
    dataset.labels = []
    dataset._create_sliding_windows()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def evaluate_all_subjects(ufe, clf, transformer, device, cache_dir, num_subjects=30, window_size=750, stride=250, channel_names=None, batch_size=64):
    subject_accuracies = []
    subject_losses = []
    for subject_idx in range(num_subjects):
        loader = get_subject_val_loader(
            subject_idx, cache_dir, window_size, stride, channel_names, batch_size, augment=False
        )
        clf.eval()
        ufe.eval()
        criterion = torch.nn.CrossEntropyLoss()

        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for data, label in loader:
                data, label = data.to(device), label.to(device)
                _, feat32 = ufe(data)
                Z = transformer.transform(feat32)
                out = clf(Z)
                loss = criterion(out, label)
                val_loss += loss.item() * data.size(0)
                _, pred = out.max(1)
                val_correct += pred.eq(label).sum().item()
                val_total += label.size(0)

        acc = 100.0 * val_correct / val_total if val_total > 0 else 0
        subject_accuracies.append(acc)
        subject_losses.append(val_loss / val_total if val_total > 0 else 0)
        print(f"Subject {subject_idx}: Accuracy={acc:.2f}%, Loss={val_loss / val_total:.4f}")

    mean_acc = sum(subject_accuracies) / len(subject_accuracies)
    print(f"\nMean accuracy across {num_subjects} subjects: {mean_acc:.2f}%")
