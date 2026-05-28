import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import scipy.io as sio
from torch.utils.data import Dataset

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

# Modified VAE with classifier head
class SemiSupervisedVAE(nn.Module):
    def __init__(self, input_dim=8100, latent_dim=50, num_classes=4):
        super(SemiSupervisedVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Sigmoid(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

        # Classifier head
        self.classifier = nn.Linear(latent_dim, num_classes)

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
        recon = self.decoder(z)
        class_logits = self.classifier(z)
        return recon, mu, logvar, class_logits


# Modified loss function with classification
def semi_supervised_vae_loss(recon_x, x, mu, logvar, class_logits, y, beta=1.0, alpha=0.5):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    class_loss = F.cross_entropy(class_logits, y)
    total_loss = recon_loss + beta * kl_div + alpha * class_loss
    return total_loss, recon_loss, kl_div, class_loss


# Training function
def train_semi_supervised_vae(model, dataloader, epochs, lr, beta=1.0, alpha=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_losses, recon_losses, kl_losses, class_losses = [], [], [], []

    for epoch in range(epochs):
        model.train()
        total_epoch_loss, recon_epoch_loss, kl_epoch_loss, class_epoch_loss = 0, 0, 0, 0

        for x, y, _ in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            recon, mu, logvar, class_logits = model(x)
            loss, recon_loss, kl_div, class_loss = semi_supervised_vae_loss(recon, x, mu, logvar, class_logits, y, beta, alpha)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            recon_epoch_loss += recon_loss.item()
            kl_epoch_loss += kl_div.item()
            class_epoch_loss += class_loss.item()

        total_losses.append(total_epoch_loss / len(dataloader))
        recon_losses.append(recon_epoch_loss / len(dataloader))
        kl_losses.append(kl_epoch_loss / len(dataloader))
        class_losses.append(class_epoch_loss / len(dataloader))

        print(f"Epoch {epoch+1}/{epochs} | Total: {total_losses[-1]:.4f} | Recon: {recon_losses[-1]:.4f} | KL: {kl_losses[-1]:.4f} | Cls: {class_losses[-1]:.4f}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(total_losses, label="Total Loss")
    plt.plot(recon_losses, label="Reconstruction Loss")
    plt.plot(kl_losses, label="KL Divergence")
    plt.plot(class_losses, label="Classification Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Semi-Supervised VAE Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model
def extract_latents_vae(model, dataloader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    z_list, labels, subject_ids = [], [], []

    with torch.no_grad():
        for x, y, s_id in dataloader:
            x = x.to(device)
            mu, logvar = model.encode(x)
            z = model.reparameterize(mu, logvar)
            z_list.append(z.cpu().numpy())
            labels.append(y.numpy())
            subject_ids.extend(s_id)

    return np.concatenate(z_list), np.concatenate(labels), subject_ids


def main():
    data_path = "/kaggle/input/adnida/"
    dataset = FullKLDataset(data_path, global_min=0.0, global_max=26.8768)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = 90 * 90
    latent_dim = 180
    lr = 1e-4
    epochs = 300
    beta = 18.0  # Weight for KL divergence
    alpha = 8  # Weight for classification loss

    print(f"📊 Loaded {len(dataset)} samples.")
    model = SemiSupervisedVAE(input_dim=input_dim, latent_dim=latent_dim, num_classes=4)
    model = train_semi_supervised_vae(model, dataloader, epochs, lr, beta, alpha)

    z_latents, labels, subject_ids = extract_latents_vae(model, dataloader)
    np.save("vae_latents.npy", z_latents)
    np.save("vae_labels.npy", labels)
    np.save("vae_subject_ids.npy", np.array(subject_ids))
    print("✅ VAE latent vectors saved.")


main()