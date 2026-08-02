# train.py
"""
Evaluates the model using 5-fold Stratified Cross Validation.
Produces evaluation_report.txt.
Not intended to generate the deployment model.
"""
from email import generator
import os
import glob
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
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
    labels = ds.labels

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    
    all_acc = []
    all_precision = []
    all_recall = []
    all_f1 = []
    all_preds = []
    all_labels = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):

        print(f"\n========== Fold {fold}/5 ==========\n")

        train_ds = Subset(ds, train_idx)
        val_ds = Subset(ds, val_idx)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator= torch.Generator().manual_seed(SEED), collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, generator= torch.Generator().manual_seed(SEED), collate_fn=collate_fn)

        model = TCN(num_inputs=FEATURES, num_channels=[128,128,128], kernel_size=3, dropout=0.2, num_classes=len(classes)).to(device)
        opt = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for ep in range(1, epochs+1):
            model.train()
            train_loss = 0.0
            all_preds, all_labels = [], []
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                opt.zero_grad()
                logits = model(X)
                loss = criterion(logits, y)
                loss.backward()
                opt.step()
                train_loss += loss.item() * X.size(0)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(y.cpu().numpy().tolist())
            train_loss /= train_idx.size
            train_acc = accuracy_score(all_labels, all_preds)

            # Validation
            model.eval()
            val_loss = 0.0
            vpreds, vlabels = [], []
            with torch.no_grad():
                for Xv, yv in val_loader:
                    Xv, yv = Xv.to(device), yv.to(device)
                    logits = model(Xv)
                    loss = criterion(logits, yv)
                    val_loss += loss.item() * Xv.size(0)
                    vpreds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                    vlabels.extend(yv.cpu().numpy().tolist())
            val_loss /= val_idx.size
            val_acc = accuracy_score(vlabels, vpreds)
            precision = precision_score(vlabels, vpreds, average='weighted', zero_division=0)
            recall = recall_score(vlabels, vpreds, average='weighted', zero_division=0)
            f1 = f1_score(vlabels, vpreds, average='weighted', zero_division=0)

            print(f"Epoch {ep} TrainLoss {train_loss:.4f} TrainAcc {train_acc:.4f} ValLoss {val_loss:.4f} ValAcc {val_acc:.4f} ValF1 {f1:.4f}")
            
        all_acc.append(val_acc)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        all_preds.extend(vpreds)
        all_labels.extend(vlabels)
        
    avg_acc = np.mean(all_acc)
    std_acc = np.std(all_acc)

    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)
    avg_f1 = np.mean(all_f1)
    
    best_fold = np.argmax(all_acc) + 1
    best_acc = max(all_acc)
    
    #torch.save(model.state_dict(), 'tcn_gesture.pth')
    
    # Save evaluation report after final epoch
    if ep == epochs:

        precision = precision_score(vlabels, vpreds, average='weighted')
        recall = recall_score(vlabels, vpreds, average='weighted')
        f1 = f1_score(vlabels, vpreds, average='weighted')

        report_dict = classification_report(
            all_labels,
            all_preds,
            labels=list(range(len(classes))),
            target_names=classes,
            zero_division=0,
            output_dict=True
        )
        
        report_text = classification_report(
            all_labels,
            all_preds,
            labels=list(range(len(classes))),
            target_names=classes,
            digits=4,
            zero_division=0,
        )
        
        accuracy = report_dict['accuracy']
        precision = report_dict['weighted avg']['precision']
        recall = report_dict['weighted avg']['recall']
        f1 = report_dict['weighted avg']['f1-score']
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))
        '''import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=classes
        )

        fig, ax = plt.subplots(figsize=(7, 6))

        disp.plot(
            cmap="Blues",
            ax=ax,
            colorbar=True,
            values_format="d"
        )

        plt.title("Confusion Matrix")
        plt.tight_layout()

        # Save the image
        plt.savefig("assets/confusion_matrix.png", dpi=300)
        plt.close() '''

        # Per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)

        # Model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters()
            if p.requires_grad
        )

        model_size = os.path.getsize("tcn_gesture.pth") / (1024 * 1024) \
            if os.path.exists("tcn_gesture.pth") else 0

        with open("evaluation_report.txt", "w") as f:

            f.write("="*60 + "\n")
            f.write("5-Fold Stratified Cross Validation Results\n")
            f.write("="*60 + "\n\n")

            for i, acc in enumerate(all_acc, 1):
                f.write(f"Fold {i} Accuracy : {acc*100:.2f}%\n")
                
            f.write(f"Best Fold : {best_fold} with Accuracy : {best_acc*100:.2f}%\n\n")
            
            f.write(f"Average Accuracy : {avg_acc*100:.2f}%\n")
            f.write(f"Precision        : {avg_precision*100:.2f}%\n")
            f.write(f"Recall           : {avg_recall*100:.2f}%\n")
            f.write(f"Weighted F1      : {avg_f1*100:.2f}%\n\n")

            f.write("\n")
            f.write("=" * 60 + "\n")
            f.write("Gesture Recognition Model Evaluation\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Per-Class Accuracy\n")
            f.write("-" * 60 + "\n")

            for cls, acc in zip(classes, per_class_acc):
                f.write(f"{cls:15s}: {acc*100:.2f}%\n")

            f.write("\n")

            f.write(f"Total Parameters    : {total_params:,}\n")
            f.write(f"Trainable Params    : {trainable_params:,}\n")
            f.write(f"Model Size (MB)     : {model_size:.2f}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("Classification Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(report_text)

            f.write("\n")

            f.write(f"Accuracy : {accuracy*100:.2f}%\n")
            f.write(f"Precision: {precision*100:.2f}%\n")
            f.write(f"Recall   : {recall*100:.2f}%\n")
            f.write(f"F1 Score : {f1*100:.2f}%\n")

            f.write("\n")

            f.write("=" * 60 + "\n")
            f.write("Confusion Matrix\n")
            f.write("=" * 60 + "\n")
            
            f.write(str(cm))

    #torch.save(model.state_dict(), 'tcn_gesture.pth')
    #print("Saved model to tcn_gesture.pth")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data')
    p.add_argument('--classes', nargs='+', required=True, help='gesture class names e.g. swipe_left swipe_right')
    p.add_argument('--epochs', type=int, default=7)
    args = p.parse_args()
    train_loop(args.data, args.classes, epochs=args.epochs, device=('cuda' if torch.cuda.is_available() else 'cpu'))
