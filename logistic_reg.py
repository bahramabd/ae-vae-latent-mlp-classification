import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# === Load Data ===
X = np.load("vae_latents.npy")     # shape: (11200, latent_dim)
y = np.load("vae_labels.npy")      # shape: (11200,)
sids = np.load("vae_subject_ids.npy")  # shape: (11200,), dtype=str

# === Subject-based Manual Split ===
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

train_subjects = set()
test_subjects = set()
for cls in manual_subject_split:
    train_subjects.update(manual_subject_split[cls]["Train"])
    test_subjects.update(manual_subject_split[cls]["Test"])

# === Split ===
X_train, y_train, X_test, y_test = [], [], [], []
for xi, yi, sid in zip(X, y, sids):
    if sid in train_subjects:
        X_train.append(xi)
        y_train.append(yi)
    elif sid in test_subjects:
        X_test.append(xi)
        y_test.append(yi)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")

# === Normalize ===
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === KNN Classifier ===
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# === Evaluation ===
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Test Accuracy: {acc * 100:.2f}%")
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["CN", "EMCI", "LMCI", "AD"]))

# Optional: Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["CN", "EMCI", "LMCI", "AD"])


#another script for visualization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Load data
X = np.load("vae_latents.npy")
y = np.load("vae_labels.npy")

# Basic stats
print("Shape:", X.shape)
print("Min:", np.min(X))
print("Max:", np.max(X))
print("Mean:", np.mean(X))
print("Std:", np.std(X))

# ========== 3D t-SNE Visualization ==========


# ========== 3D PCA Visualization ==========
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y, cmap='tab10', alpha=0.7)
ax.set_title("3D PCA of VAE Latent Space")
fig.colorbar(scatter, label="Class Label")
plt.show()
