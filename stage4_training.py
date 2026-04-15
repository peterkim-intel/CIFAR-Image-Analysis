import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

from stage2_preprocessing import opencv_transform
from stage3_cnn import CIFAR10_CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

BATCH_SIZE = 64

train_dataset = datasets.CIFAR10(root="./data", train=True,  download=False,
                                 transform=lambda img: opencv_transform(img, augment=True))
test_dataset  = datasets.CIFAR10(root="./data", train=False, download=False,
                                 transform=lambda img: opencv_transform(img, augment=False))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

model     = CIFAR10_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=75)

EPOCHS  = 75
history = {"train_loss": [], "test_acc": []}

for epoch in range(EPOCHS):

    # ── Train phase ───────────────────────────────────────────────────────────
    model.train()   # activates Dropout
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()   # PyTorch accumulates gradients; clear before each batch
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    avg_train_loss = running_loss / len(train_loader)

    # ── Evaluation phase ──────────────────────────────────────────────────────
    model.eval()    # disables Dropout
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    history["train_loss"].append(avg_train_loss)
    history["test_acc"].append(test_acc)

    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch [{epoch+1:02d}/{EPOCHS}]  "
          f"Loss: {avg_train_loss:.4f}  "
          f"Test Acc: {test_acc:.2f}%  "
          f"LR: {current_lr:.6f}")

torch.save(model.state_dict(), "cifar10_cnn.pth")
print("\nModel saved to cifar10_cnn.pth")

# ── Plot learning curves ──────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history["train_loss"], color="steelblue")
ax1.set_title("Training Loss per Epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True)

ax2.plot(history["test_acc"], color="darkorange")
ax2.axhline(y=90, color="red", linestyle="--", label="90% target")
ax2.set_title("Test Accuracy per Epoch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150)
plt.show()
print("Learning curves saved to learning_curves.png")
