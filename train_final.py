# train_final.py
"""
Trains the final deployment model on the complete dataset.
Generates tcn_gesture.pth for inference.
"""
from email import generator
import os
import glob
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tcn_model import TCN

WINDOW = 30
FEATURES = 63

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class GestureDataset(Dataset):
    def __init__(self, data_dir, classes, transform=None):
        self.files = []
        self.labels = []
        self.classes = classes
        for i, cls in enumerate(classes):
            for f in glob.glob(os.path.join(data_dir, f'{cls}_*.npz')):
                self.files.append(f)
                self.labels.append(i)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])['X']  # shape (WINDOW, FEATURES)
        if arr.shape[0] != WINDOW:
            # pad or trim
            if arr.shape[0] < WINDOW:
                pad = np.zeros((WINDOW - arr.shape[0], arr.shape[1]), dtype=np.float32)
                arr = np.vstack([arr, pad])
            else:
                arr = arr[:WINDOW]
        if self.transform:
            arr = self.transform(arr)
        return torch.tensor(arr, dtype=torch.float32), self.labels[idx]

def normalize(arr):
    # arr: (T,63)
    seq = arr.reshape(WINDOW, 21, 3)
    wrist = seq[:, 0:1, :].copy()
    seq = seq - wrist
    scale = np.linalg.norm(seq, axis=2).max() + 1e-6
    seq = seq / scale
    return seq.reshape(WINDOW, -1)

def collate_fn(batch):
    Xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return Xs, ys

def train_loop(data_dir, classes, epochs=7, batch_size=32, lr=1e-3, device='cpu'):
    ds = GestureDataset(data_dir, classes, transform=normalize)
    train_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    model = TCN(
        num_inputs=FEATURES,
        num_channels=[128,128,128],
        kernel_size=3,
        dropout=0.2,
        num_classes=len(classes)
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for ep in range(1, epochs+1):

        model.train()

        running_loss = 0

        train_preds = []
        train_labels = []

        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device)

            opt.zero_grad()

            logits = model(X)

            loss = criterion(logits, y)

            loss.backward()

            opt.step()

            running_loss += loss.item() * X.size(0)

            train_preds.extend(
                logits.argmax(1).cpu().numpy()
            )

            train_labels.extend(
                y.cpu().numpy()
            )

        train_loss = running_loss / len(ds)

        train_acc = accuracy_score(
            train_labels,
            train_preds
        )

        print(
            f"Epoch {ep:02d} | "
            f"Loss {train_loss:.4f} | "
            f"Accuracy {train_acc:.4f}"
        )
    
    torch.save(model.state_dict(), 'tcn_gesture.pth')
    print("Saved model to tcn_gesture.pth")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data')
    p.add_argument('--classes', nargs='+', required=True, help='gesture class names e.g. swipe_left swipe_right')
    p.add_argument('--epochs', type=int, default=7)
    args = p.parse_args()
    train_loop(args.data, args.classes, epochs=args.epochs, device=('cuda' if torch.cuda.is_available() else 'cpu'))
