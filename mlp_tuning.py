import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset, Subset
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# Load latent vectors, labels, subject IDs
X = np.load("vae_latents.npy")
y = np.load("vae_labels.npy")
subject_ids = np.load("vae_subject_ids.npy")

print("📊 Label distribution:", np.bincount(y))
print("🧠 Total subjects:", len(set(subject_ids)))


def normalize_latents(X):
    global_min = np.min(X)
    global_max = np.max(X)
    print(f"🔍 Latents Global Min: {global_min:.4f}, Max: {global_max:.4f}")

    X_norm = (X - global_min) / (global_max - global_min + 1e-8)
    return X_norm


X = normalize_latents(X)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()

# MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=800, num_classes=4):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(


            nn.Linear(128, 64),
            nn.Sigmoid(),


            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Dropout(0.3),

            nn.Linear(32, 16),
            nn.Sigmoid(),


            nn.Linear(64, num_classes)
           # nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

def check_subject_split(subject_ids, labels, k=5, seed=42):
    folds = subject_stratified_kfold(subject_ids, labels, k, seed)

    print("\n🔎 Subject Splits Across Folds:")
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_subjects = set(subject_ids[i] for i in train_idx)
        test_subjects = set(subject_ids[i] for i in test_idx)

        print(f"\n📂 Fold {fold_idx + 1}:")
        print(f"  🏋️‍♂️ Train Subjects ({len(train_subjects)}): {sorted(train_subjects)}")
        print(f"  🧪 Test Subjects ({len(test_subjects)}): {sorted(test_subjects)}")

        # 🛡️ Safety check: No subject should be in both train and test!
        common = train_subjects.intersection(test_subjects)
        if common:
            print(f"⚠️ WARNING: {len(common)} common subjects found between train and test!")
            print(common)
        else:
            print("✅ No common subjects between train and test.")


# Custom subject-wise stratified split
def subject_stratified_kfold(subject_ids, labels, k=5, seed=42):
    random.seed(seed)
    subject_to_indices = defaultdict(list)
    subject_to_label = {}

    for idx, sid in enumerate(subject_ids):
        subject_to_indices[sid].append(idx)
        if sid not in subject_to_label:
            subject_to_label[sid] = labels[idx]

    subjects = list(subject_to_label.keys())
    subject_labels = [subject_to_label[sid] for sid in subjects]

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    folds = []
    for train_sub_idx, test_sub_idx in skf.split(subjects, subject_labels):
        train_ids = [subjects[i] for i in train_sub_idx]
        test_ids = [subjects[i] for i in test_sub_idx]

        train_idx = [i for sid in train_ids for i in subject_to_indices[sid]]
        test_idx = [i for sid in test_ids for i in subject_to_indices[sid]]

        folds.append((train_idx, test_idx))
    return folds

# Train and evaluate with subject-based K-Fold

def train_kfold(X, y, subject_ids, k=5, epochs=100, batch_size=32, lr=1e-3, weight_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = subject_stratified_kfold(subject_ids, y.numpy(), k)

    all_fold_f1 = []

    for fold, (train_idx, test_idx) in enumerate(folds):
        print(f"\n🔁 Fold {fold+1}/{k}")


        model = MLPClassifier(input_dim=X.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr )
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(Subset(TensorDataset(X, y), train_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(TensorDataset(X, y), test_idx), batch_size=batch_size, shuffle=False)

        train_accs, test_accs = [], []
        train_losses, test_losses = [], []

        for epoch in range(epochs):
            model.train()
            total_train_loss, correct_train, total_train = 0, 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                correct_train += (preds.argmax(dim=1) == yb).sum().item()
                total_train += yb.size(0)
            train_accs.append(correct_train / total_train)
            train_losses.append(total_train_loss / len(train_loader))

            model.eval()
            total_test_loss, correct_test, total_test = 0, 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    total_test_loss += loss.item()
                    correct_test += (preds.argmax(dim=1) == yb).sum().item()
                    total_test += yb.size(0)
            test_accs.append(correct_test / total_test)
            test_losses.append(total_test_loss / len(test_loader))

        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                preds = model(xb)
                all_preds.extend(preds.argmax(dim=1).cpu().numpy())
                all_labels.extend(yb.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        cm = confusion_matrix(all_labels, all_preds)

        print(f"🎯 Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
        print("🧩 Confusion Matrix:\n", cm)

        all_fold_f1.append(f1)

        # Plot for current fold
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(test_accs, label='Test Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Fold {fold+1} Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Fold {fold+1} Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    print(f"\n📊 Average Macro F1 across {k} folds: {np.mean(all_fold_f1):.4f} ± {np.std(all_fold_f1):.4f}")


if __name__ == "__main__":
    train_kfold(X, y, subject_ids, k=4, epochs=100, batch_size=32, lr=1e-4, weight_decay=1e-4)
