import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from stage2_preprocessing import opencv_transform
from stage3_cnn import CIFAR10_CNN

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ── Load trained model ────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CIFAR10_CNN().to(device)
model.load_state_dict(torch.load("cifar10_cnn.pth", map_location=device))
model.eval()

# ── Test data (no augmentation — we want clean evaluation) ───────────────────
test_dataset = datasets.CIFAR10(root="./data", train=False, download=False,
                                 transform=lambda img: opencv_transform(img, augment=False))
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ── Collect all predictions and true labels ───────────────────────────────────
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ── Overall accuracy ──────────────────────────────────────────────────────────
overall_acc = 100 * (all_preds == all_labels).mean()
print(f"Overall Test Accuracy: {overall_acc:.2f}%\n")

# ── Per-class accuracy ────────────────────────────────────────────────────────
# Shows which classes the model handles well vs struggles with
print(f"{'Class':<12} {'Correct':>7} {'Total':>7} {'Accuracy':>9}")
print("-" * 40)
for i, name in enumerate(CLASS_NAMES):
    mask    = all_labels == i
    correct = (all_preds[mask] == i).sum()
    total   = mask.sum()
    acc     = 100 * correct / total
    print(f"{name:<12} {correct:>7} {total:>7} {acc:>8.1f}%")


num_classes = len(CLASS_NAMES)
conf = np.zeros((num_classes, num_classes), dtype=int)

for true, pred in zip(all_labels, all_preds):
    conf[true][pred] += 1


conf_norm = conf.astype(float) / conf.sum(axis=1, keepdims=True)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(conf_norm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax, label="Fraction of true class")

ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title(f"Confusion Matrix — Test Accuracy: {overall_acc:.2f}%")

for i in range(num_classes):
    for j in range(num_classes):
        color = "white" if conf_norm[i, j] > 0.5 else "black"
        ax.text(j, i, str(conf[i, j]), ha="center", va="center",
                fontsize=8, color=color)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix saved to confusion_matrix.png")
