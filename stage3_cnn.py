import torch
import torch.nn as nn


class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # ── Block 1: Edge and color pattern detection ─────────────────────────
        # Conv2d(in_channels, out_channels, kernel_size, padding)
        #   in_channels=3   → input has 3 channels (R, G, B)
        #   out_channels=32 → we learn 32 different filters, producing 32 feature maps
        #   kernel_size=3   → each filter is a 3×3 sliding window
        #   padding=1       → adds a 1-pixel border of zeros so the output stays 32×32
        #                     without padding, each conv would shrink the spatial size
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # (B, 3,  32, 32) → (B, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # output: (B, 32, 16, 16)
        )

        # ── Block 2: Shape and texture detection ──────────────────────────────
        # in_channels=32 → must match Block 1's out_channels (we're stacking on top)
        # out_channels=64 → we double the filters; deeper layers need more capacity
        #                    to represent combinations of the simple patterns Block 1 found
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B, 32, 16, 16) → (B, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # output: (B, 64, 8, 8)
        )

        # ── Block 3: High-level feature composition ───────────────────────────
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (B, 64, 8, 8) → (B, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(),                                    # output: (B, 128, 8, 8)
        )

        # ── Block 4: Fine-grained discrimination ──────────────────────────────
        # Added because the model plateaued at 88% — it could distinguish broad
        # categories but struggled with similar classes (cat vs dog, car vs truck).
        # An extra conv layer at the same spatial size (8×8) lets the model learn
        # more subtle combinations of Block 3's features without further shrinking.
        # We use two conv layers in sequence (no pool between) to double the
        # receptive field depth cheaply — a pattern borrowed from VGG architecture.
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# (B, 128, 8, 8) → (B, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),# (B, 256, 8, 8) → (B, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # output: (B, 256, 4, 4)
        )

        # ── Classifier: Convert feature maps → 10 class scores ───────────────
        # Block 4 output: 256 channels × 4×4 spatial = 4,096 values per image
        # Two dropout layers: one after the first linear (heavy regularization),
        # one after the second (lighter) — prevents the deeper classifier from overfitting
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # (B, 256, 4, 4) → (B, 4096)
            nn.Linear(256 * 4 * 4, 512),    # (B, 4096) → (B, 512)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),            # (B, 512) → (B, 256)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),             # (B, 256) → (B, 10)
        )

    def forward(self, x):
        x = self.block1(x)      # spatial feature extraction, level 1
        x = self.block2(x)      # spatial feature extraction, level 2
        x = self.block3(x)      # spatial feature extraction, level 3
        x = self.block4(x)      # fine-grained discrimination, level 4
        x = self.classifier(x)  # flatten → compress → 10 class scores
        return x                 # shape: (B, 10)


# ── Sanity check: trace one full batch end-to-end ────────────────────────────
if __name__ == "__main__":
    model = CIFAR10_CNN()
    dummy_batch = torch.randn(64, 3, 32, 32)   # fake batch: 64 images, 3 channels, 32×32

    output = model(dummy_batch)
    print(f"Input  shape : {dummy_batch.shape}")  # expect torch.Size([64, 3, 32, 32])
    print(f"Output shape : {output.shape}")        # expect torch.Size([64, 10])

    # Count total learnable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters : {total_params:,}")  # how many weights backprop will tune

