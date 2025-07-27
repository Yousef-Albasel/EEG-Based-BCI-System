from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
import os

from dataset.SlidingWindowEEGDataset import SlidingWindowEEGDataset
from models.DDFilter import DDFilter

def train_model(model, train_loader, val_loader, criterion, device, num_epochs=400, lr=0.0001, save_path="best_model.pth"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0

    for epoch in range(1, num_epochs+1):
        # --------------------
        # Training phase
        # --------------------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        for data, label in train_bar:
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()
            out, feat = model(data)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * data.size(0)
            _, predicted = out.max(1)
            correct += predicted.eq(label).sum().item()
            total += label.size(0)

            train_bar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "acc": f"{100. * correct / total:.2f}%"
            })

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        # --------------------
        # Validation phase
        # --------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                out, feat = model(data)
                loss = criterion(out, label)

                val_loss += loss.item() * data.size(0)
                _, predicted = out.max(1)
                val_correct += predicted.eq(label).sum().item()
                val_total += label.size(0)

        val_loss /= val_total
        val_acc = 100. * val_correct / val_total

        # --------------------
        # Save best model
        # --------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch} with Val Acc: {val_acc:.2f}%")

        # --------------------
        # Print results
        # --------------------
        print(f"[Epoch {epoch}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% || "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")


def get_subject_loader(subject_idx, cache_dir, window_size=750, stride=250, channel_names=None, batch_size=64, augment=True, transforms=None):
    trials_per_subject = 80
    start_idx = subject_idx * trials_per_subject
    end_idx = start_idx + trials_per_subject

    all_trial_paths = [
        os.path.join(cache_dir, fname)
        for fname in sorted(os.listdir(cache_dir))
        if fname.endswith('.npy')
    ]
    subject_trial_paths = all_trial_paths[start_idx:end_idx]

    class SubjectSlidingWindowEEGDataset(SlidingWindowEEGDataset):
        def __init__(self, trial_paths, window_size, stride, channel_names, augment, transforms):
            self.trial_paths = trial_paths
            self.window_size = window_size
            self.stride = stride
            self.channel_names = channel_names
            self.augment = augment
            self.transforms = transforms if transforms is not None else []
            self.windows = []
            self.labels = []
            self._create_sliding_windows()

    dataset = SubjectSlidingWindowEEGDataset(
        subject_trial_paths, window_size, stride, channel_names, augment, transforms
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def finetune_ddfilter_per_subject(subjects, get_subject_loader, ufe, device, num_epochs=100, lr=0.0005):
    for subj in subjects:  # subj is an integer index
        loader = get_subject_loader(subj, batch_size=64)
        dd_filter = DDFilter(input_dim=32, hidden_dim=16, num_classes=2).to(device)
        optimizer = torch.optim.Adam(dd_filter.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        print(f"\n=== Fine-tuning DD Filter for subject {subj} ===")
        for epoch in range(num_epochs):
            dd_filter.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for data, label in loader:
                data, label = data.to(device), label.to(device)
                with torch.no_grad():
                    _, feat32 = ufe(data)  # extract 32-D features
                out = dd_filter(feat32)
                loss = criterion(out, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * data.size(0)
                _, pred = out.max(1)
                correct += pred.eq(label).sum().item()
                total += label.size(0)

            epoch_loss = running_loss / total
            epoch_acc = 100.0 * correct / total
            print(f"[{subj}] Epoch {epoch+1:02d}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")

        # Save subject-specific fine-tuned model
        torch.save(dd_filter.state_dict(), f"outputs/ddfilter/ddfilter_{subj}_v2.pth")
        print(f"Saved DD Filter for {subj} as ddfilter_{subj}_v2.pth")


# Trainning the classifier 

def train_cross_subject_classifier(
    ufe, transformer, clf, train_loader, device, num_epochs=50, lr=0.0001
):
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    ufe.eval()
    for epoch in range(num_epochs):
        clf.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            with torch.no_grad():
                _, feat32 = ufe(data)  # (batch, 32)
            Z = transformer.transform(feat32)  # (batch, m)
            out = clf(Z)  # (batch, num_classes)

            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            _, pred = out.max(1)
            correct += pred.eq(label).sum().item()
            total += label.size(0)

        print(f"Epoch {epoch+1}: Loss={running_loss/total:.4f}, Acc={100*correct/total:.2f}%")
    torch.save(clf.state_dict(), f"outputs/clf_last_v2.pth")
