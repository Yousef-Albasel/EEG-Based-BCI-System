import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from data.EEGDataset import EEGFeatureDataset , EEGDataLoader
from torch.utils.data import DataLoader, Subset
from models import MLP

def train(model, train_loader, val_loader, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(10):
        model.train()
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            pred = model(xb).argmax(1).cpu().numpy()
            all_preds.extend(pred)
            all_true.extend(yb.numpy())
    print(classification_report(all_true, all_preds))
    acc = accuracy_score(all_true, all_preds)
    return model, acc

def train_main(data_path):
    raw_dataset = EEGDataLoader(data_path)
    ssvep_train_df,_,_ = raw_dataset.load_csv_files()
    label_encoder = LabelEncoder().fit(ssvep_train_df['label'])
    dataset = EEGFeatureDataset(ssvep_train_df, label_encoder , data_path)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    trained_model = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n📁 Fold {fold+1}/5")
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32)
        model = MLP(dataset[0][0].shape[0], len(label_encoder.classes_))
        model, acc = train(model, train_loader, val_loader, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        trained_model = model
        accs.append(acc)

    print(f"\n📊 Mean Accuracy: {np.mean(accs):.2%} ± {np.std(accs):.2%}")
    torch.save(trained_model.state_dict(), "deep_ssvep_model.pt")
    print("✅ Saved model as deep_ssvep_model.pt")