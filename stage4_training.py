import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

from stage2_preprocessing import opencv_transform
from stage3_cnn import CIFAR10_CNN

# ── Device setup ──────────────────────────────────────────────────────────────
# Use GPU if available, otherwise CPU. The model and data must be on the same device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# ── Data ──────────────────────────────────────────────────────────────────────
BATCH_SIZE = 64

train_dataset = datasets.CIFAR10(root="./data", train=True,  download=False,
                                 transform=lambda img: opencv_transform(img, augment=True))
test_dataset  = datasets.CIFAR10(root="./data", train=False, download=False,
                                 transform=lambda img: opencv_transform(img, augment=False))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ── Model, loss, optimizer ────────────────────────────────────────────────────
model = CIFAR10_CNN().to(device)   # move all 4.5M weights to the device

# CrossEntropyLoss: measures how wrong the model's 10 class scores are
# vs the true label. Returns one number per batch (averaged across 64 images).
criterion = nn.CrossEntropyLoss()

# Adam optimizer with weight_decay: the 1e-4 penalty discourages large
# weights, reducing overfitting and improving generalization.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# CosineAnnealingLR: smoothly decays learning rate from 0.001 → ~0 over
# T_max epochs. Large steps early (fast learning), tiny steps late (fine-tuning).
# This is the standard technique to squeeze from ~85% → 90%+ on CIFAR-10.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=75)

# ── Training loop ─────────────────────────────────────────────────────────────
EPOCHS = 75

# We store these to plot the learning curves after training
history = {"train_loss": [], "test_acc": []}

for epoch in range(EPOCHS):

    # ── Train phase ───────────────────────────────────────────────────────────
    model.train()   # enables Dropout (disabled during evaluation)
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # clear gradients from the previous batch
                                       # (PyTorch accumulates them by default)

        outputs = model(images)        # forward pass → (64, 10) class scores
        loss = criterion(outputs, labels)  # how wrong were we this batch?

        loss.backward()                # backprop: compute gradient for every weight
        optimizer.step()               # nudge every weight by lr × its gradient

        running_loss += loss.item()

    scheduler.step()   # decay the learning rate after each epoch
    avg_train_loss = running_loss / len(train_loader)

    # ── Evaluation phase ──────────────────────────────────────────────────────
    model.eval()    # disables Dropout — we want deterministic predictions
    correct = 0
    total = 0

    with torch.no_grad():              # don't compute gradients — saves memory
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)                    # (64, 10) scores
            _, predicted = torch.max(outputs, dim=1)  # take the highest score as prediction
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total

    # Store for plotting
    history["train_loss"].append(avg_train_loss)
    history["test_acc"].append(test_acc)

    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch [{epoch+1:02d}/{EPOCHS}]  "
          f"Loss: {avg_train_loss:.4f}  "
          f"Test Acc: {test_acc:.2f}%  "
          f"LR: {current_lr:.6f}")

# ── Save the trained model weights ────────────────────────────────────────────
# We save only the weights (state_dict), not the whole model object.
# This is the standard PyTorch convention — portable and compact.
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
