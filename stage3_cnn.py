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
        # in_channels=64 → matches Block 2's out_channels
        # out_channels=128 → more filters again; this layer combines textures into
        #                     object-level features (e.g. "fur + round shape = cat face")
        # No MaxPool here — at 8×8 we're already small; pooling again → 4×4 loses too much
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (B, 64, 8, 8) → (B, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(),                                    # output: (B, 128, 8, 8)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


# ── Sanity check: trace one batch through Block 1 ────────────────────────────
if __name__ == "__main__":
    model = CIFAR10_CNN()
    dummy_batch = torch.randn(64, 3, 32, 32)   # fake batch: 64 images, 3 channels, 32×32

    output = model(dummy_batch)
    print(f"Input  shape : {dummy_batch.shape}")  # expect (64, 3,  32, 32)
    print(f"Output shape : {output.shape}")        # expect (64, 32, 16, 16)
