import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
import os
import scipy.io as sio


# ===============================
# Utilities
# ===============================
def extract_subject_id(filename):
    parts = filename.split("_")
    return parts[1] + "_" + parts[2] + "_" + parts[3]  # Example: "018_S_4733"


def compute_global_min_max(data_folder):
    classes = ["CN", "EMCI", "LMCI", "AD"]
    global_min = float('inf')
    global_max = float('-inf')

    for class_name in classes:
        class_path = os.path.join(data_folder, class_name)
        for file in os.listdir(class_path):
            if file.endswith(".mat"):
                file_path = os.path.join(class_path, file)
                mat_data = sio.loadmat(file_path)
                kl_matrix = mat_data[list(mat_data.keys())[-1]]
                if kl_matrix.shape == (90, 90):
                    global_min = min(global_min, kl_matrix.min())
                    global_max = max(global_max, kl_matrix.max())

    print(f"🌐 Global min: {global_min:.4f}, Global max: {global_max:.4f}")
    return global_min, global_max


# ===============================
# Dataset
# ===============================
class FullKLDataset(Dataset):
    def __init__(self, data_folder, global_min=None, global_max=None):
        self.data = []
        self.labels = []
        self.subject_ids = []
        self.classes = ["CN", "EMCI", "LMCI", "AD"]
        self.global_min = global_min
        self.global_max = global_max
        self.load_all_data(data_folder)

    def load_all_data(self, data_folder):
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(data_folder, class_name)
            for file in os.listdir(class_path):
                if file.endswith(".mat"):
                    file_path = os.path.join(class_path, file)
                    mat_data = sio.loadmat(file_path)
                    kl_matrix = mat_data[list(mat_data.keys())[-1]]

                    if kl_matrix.shape == (90, 90):
                        flat_kl = kl_matrix.flatten().astype(np.float32)
                        if self.global_min is not None and self.global_max is not None:
                            flat_kl = (flat_kl - self.global_min) / (self.global_max - self.global_min + 1e-8)
                        subject_id = extract_subject_id(file)
                        self.data.append(flat_kl)
                        self.labels.append(class_idx)
                        self.subject_ids.append(subject_id)

        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx]), self.subject_ids[idx]


# ===============================
# Joint VAE + MLP Model
# ===============================
class JointVAEMLP(nn.Module):
    def __init__(self, input_dim=8100, latent_dim=180, num_classes=4):
        super(JointVAEMLP, self).__init__()

        # ===== VAE Encoder =====
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Sigmoid(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # ===== VAE Decoder =====
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

        # ===== MLP Classifier Head =====
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # VAE reconstruction
        recon = self.decoder(z)

        # MLP classification
        class_logits = self.classifier(z)

        return recon, mu, logvar, class_logits, z


# ===============================
# Joint Loss Function
# ===============================
def joint_loss(recon_x, x, mu, logvar, class_logits, y, beta=1.0, alpha=1.0):
    """
    Combined loss for VAE reconstruction + KL divergence + classification
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Classification loss
    class_loss = F.cross_entropy(class_logits, y)

    # Combined loss
    total_loss = recon_loss + beta * kl_div + alpha * class_loss

    return total_loss, recon_loss, kl_div, class_loss


# ===============================
# Data Splitting
# ===============================
def create_train_test_split(dataset):
    """Create train/test split based on your manual subject split"""
    manual_subject_split = {
        "CN": {"Train": ["002_S_0413", "002_S_4264", "002_S_4262", "010_S_4442", "013_S_4580", "006_S_0731"],
               "Test": ["002_S_1261", "010_S_4345"]},
        "EMCI": {"Train": ["002_S_4237", "002_S_2073", "002_S_4473", "006_S_4679", "002_S_2010", "002_S_2043"],
                 "Test": ["012_S_4012"]},
        "LMCI": {"Train": ["002_S_4229", "002_S_4521", "002_S_4746", "012_S_4094", "006_S_4713", "006_S_4960"],
                 "Test": ["002_S_4251", "010_S_4135"]},
        "AD": {"Train": ["002_S_5018", "006_S_4153", "013_S_5071", "018_S_4696", "010_S_5163", "006_S_4867"],
               "Test": ["006_S_4192", "018_S_4733"]}
    }

    # Build subject-to-split map
    train_subjects = set()
    test_subjects = set()
    for cls in manual_subject_split:
        train_subjects.update(manual_subject_split[cls]['Train'])
        test_subjects.update(manual_subject_split[cls]['Test'])

    # Split data
    train_indices, test_indices = [], []

    for idx in range(len(dataset)):
        _, _, subject_id = dataset[idx]
        if subject_id in train_subjects:
            train_indices.append(idx)
        elif subject_id in test_subjects:
            test_indices.append(idx)

    return train_indices, test_indices


def create_dataloaders(dataset, train_indices, test_indices, batch_size=32):
    """Create train and test dataloaders"""
    train_data = [dataset[i] for i in train_indices]
    test_data = [dataset[i] for i in test_indices]

    # Extract tensors
    X_train = torch.stack([x for x, _, _ in train_data])
    y_train = torch.stack([y for _, y, _ in train_data])

    X_test = torch.stack([x for x, _, _ in test_data])
    y_test = torch.stack([y for _, y, _ in test_data])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ===============================
# Training Function
# ===============================
def train_joint_model(model, train_loader, test_loader, epochs=100, lr=1e-4, beta=18.0, alpha=8.0):
    """Joint training of VAE + MLP"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # Tracking metrics
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    recon_losses, kl_losses, class_losses = [], [], []

    for epoch in range(epochs):
        # ===== Training Phase =====
        model.train()
        total_train_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_class_loss = 0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            recon, mu, logvar, class_logits, _ = model(xb)

            loss, recon_loss, kl_div, class_loss = joint_loss(
                recon, xb, mu, logvar, class_logits, yb, beta, alpha
            )

            loss.backward()
            optimizer.step()

            # Track losses
            total_train_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_div.item()
            total_class_loss += class_loss.item()

            # Track accuracy
            train_correct += (class_logits.argmax(dim=1) == yb).sum().item()
            train_total += yb.size(0)

        # Average losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
        avg_class_loss = total_class_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        train_losses.append(avg_train_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)
        class_losses.append(avg_class_loss)
        train_accs.append(train_acc)

        # ===== Testing Phase =====
        model.eval()
        total_test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                recon, mu, logvar, class_logits, _ = model(xb)

                loss, _, _, _ = joint_loss(recon, xb, mu, logvar, class_logits, yb, beta, alpha)

                total_test_loss += loss.item()
                test_correct += (class_logits.argmax(dim=1) == yb).sum().item()
                test_total += yb.size(0)

        avg_test_loss = total_test_loss / len(test_loader)
        test_acc = 100 * test_correct / test_total

        test_losses.append(avg_test_loss)
        test_accs.append(test_acc)

        # Print progress
        print(f"[Epoch {epoch + 1}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} (Recon: {avg_recon_loss:.4f}, "
              f"KL: {avg_kl_loss:.4f}, Class: {avg_class_loss:.4f}) | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'recon_losses': recon_losses,
        'kl_losses': kl_losses,
        'class_losses': class_losses
    }


# ===============================
# Plotting Function
# ===============================
def plot_training_results(history):
    """Plot training results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Accuracy plot
    axes[0, 0].plot(history['train_accs'], label='Train Accuracy', color='blue')
    axes[0, 0].plot(history['test_accs'], label='Test Accuracy', color='red')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Total loss plot
    axes[0, 1].plot(history['train_losses'], label='Train Loss', color='blue')
    axes[0, 1].plot(history['test_losses'], label='Test Loss', color='red')
    axes[0, 1].set_title('Total Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Component losses plot
    axes[1, 0].plot(history['recon_losses'], label='Reconstruction Loss', color='green')
    axes[1, 0].plot(history['kl_losses'], label='KL Divergence', color='orange')
    axes[1, 0].plot(history['class_losses'], label='Classification Loss', color='purple')
    axes[1, 0].set_title('Component Losses')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Loss ratios
    axes[1, 1].plot(np.array(history['recon_losses']) / np.array(history['train_losses']),
                    label='Recon/Total', color='green')
    axes[1, 1].plot(np.array(history['kl_losses']) / np.array(history['train_losses']),
                    label='KL/Total', color='orange')
    axes[1, 1].plot(np.array(history['class_losses']) / np.array(history['train_losses']),
                    label='Class/Total', color='purple')
    axes[1, 1].set_title('Loss Component Ratios')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


# ===============================
# Main Function
# ===============================
def main():
    # Data path
    data_path = "/kaggle/input/adnida/"  # Update this path

    # Load dataset
    print("📊 Loading dataset...")
    dataset = FullKLDataset(data_path, global_min=0.0, global_max=26.8768)
    print(f"📊 Loaded {len(dataset)} samples.")

    # Create train/test split
    print("🔄 Creating train/test split...")
    train_indices, test_indices = create_train_test_split(dataset)
    train_loader, test_loader = create_dataloaders(dataset, train_indices, test_indices, batch_size=32)

    print(f"📊 Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")

    # Model parameters
    input_dim = 90 * 90  # 8100
    latent_dim = 180
    num_classes = 4
    lr = 1e-4
    epochs = 300
    beta = 18.0  # Weight for KL divergence
    alpha = 8.0  # Weight for classification loss

    # Initialize model
    print("🤖 Initializing joint VAE+MLP model...")
    model = JointVAEMLP(input_dim=input_dim, latent_dim=latent_dim, num_classes=num_classes)

    # Train model
    print("🚀 Starting joint training...")
    history = train_joint_model(model, train_loader, test_loader, epochs=epochs,
                                lr=lr, beta=beta, alpha=alpha)

    # Plot results
    plot_training_results(history)

    print("✅ Training completed!")

    # Optionally save the model
    torch.save(model.state_dict(), "joint_vae_mlp_model.pth")
    print("💾 Model saved as 'joint_vae_mlp_model.pth'")


if __name__ == "__main__":
    main()