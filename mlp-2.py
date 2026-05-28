import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load saved latents
X = np.load("vae_latents.npy")  # shape: [samples, latent_dim]
y = np.load("vae_labels.npy")   # shape: [samples]
subject_ids = np.load("vae_subject_ids.npy")  # shape: [samples]

print(f"✅ Loaded AE latent vectors: {X.shape}")

manual_subject_split = {
    "CN": {"Train": ["002_S_0413", "002_S_4264", "002_S_4262", "010_S_4442", "013_S_4580", "006_S_0731"],
           "Test":  ["002_S_1261", "010_S_4345"]},
    "EMCI": {"Train": ["002_S_4237", "002_S_2073", "002_S_4473", "006_S_4679", "002_S_2010", "002_S_2043"],
             "Test":  ["012_S_4012"]},
    "LMCI": {"Train": ["002_S_4229", "002_S_4521", "002_S_4746", "012_S_4094", "006_S_4713", "006_S_4960"],
             "Test":  ["002_S_4251", "010_S_4135"]},
    "AD": {"Train": ["002_S_5018", "006_S_4153", "013_S_5071", "018_S_4696", "010_S_5163", "006_S_4867"],
           "Test":  ["006_S_4192", "018_S_4733"]}
}

# Build subject-to-split map
train_subjects = set()
test_subjects = set()
for cls in manual_subject_split:
    train_subjects.update(manual_subject_split[cls]['Train'])
    test_subjects.update(manual_subject_split[cls]['Test'])

# Split based on subject_ids
X_train, y_train = [], []
X_test, y_test = [], []

for xi, yi, sid in zip(X, y, subject_ids):
    if sid in train_subjects:
        X_train.append(xi)
        y_train.append(yi)
    elif sid in test_subjects:
        X_test.append(xi)
        y_test.append(yi)

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)






print(f"📊 Train samples: {X_train.shape}, Test samples: {X_test.shape}")



# ==== Create DataLoaders ====
batch_size = 32
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()), batch_size=batch_size, shuffle=False)

# ==== Define MLP ====
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=180, num_classes=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(0.3),




            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# ==== Train ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPClassifier(input_dim=180).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

epochs = 80
train_losses, test_losses = [], []
train_accs, test_accs = [], []

for epoch in range(epochs):
    # Train
    model.train()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (preds.argmax(dim=1) == yb).sum().item()
        total += yb.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Test
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item()
            correct += (preds.argmax(dim=1) == yb).sum().item()
            total += yb.size(0)

    test_loss = total_loss / len(test_loader)
    test_acc = 100 * correct / total
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    # ✅ Epoch output
    print(f"[Epoch {epoch+1}/{epochs}] "
          f"Train Acc: {train_acc:.2f}%, Train Loss: {train_loss:.4f} | "
          f"Test Acc: {test_acc:.2f}%, Test Loss: {test_loss:.4f}")


# ==== Plot results ====
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Test Acc')

plt.legend()
plt.grid(True)
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.grid(True)
plt.title('Loss')

plt.tight_layout()
plt.show()
