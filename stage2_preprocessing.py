import numpy as np
import cv2
import torch

# ─────────────────────────────────────────────────────────────────────────────
# HOW CIFAR_MEAN AND CIFAR_STD WERE ORIGINALLY CALCULATED (conceptual code)
# You would run this ONCE on the training set before any training begins.
#
# all_pixels = []
# for pil_image, _ in train_dataset:
#     img = np.array(pil_image).astype(np.float32) / 255.0  # scale to 0–1
#     all_pixels.append(img)                                 # collect every image
#
# all_pixels = np.stack(all_pixels)    # shape: (50000, 32, 32, 3)
#
# CIFAR_MEAN = all_pixels.mean(axis=(0, 1, 2))  # mean per channel across ALL pixels
# CIFAR_STD  = all_pixels.std(axis=(0, 1, 2))   # std  per channel across ALL pixels
#
# axis=(0,1,2) collapses: images (50000), height (32), width (32)
# leaving only the channel axis (3) → three numbers for mean, three for std
#
# Result: CIFAR_MEAN ≈ (0.4914, 0.4822, 0.4465)
#         CIFAR_STD  ≈ (0.2470, 0.2435, 0.2616)
# ─────────────────────────────────────────────────────────────────────────────

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

# The model expects images at this fixed size.
# CIFAR-10 is always 32×32, but real images (e.g. webcam frames in Stage 6)
# can be any size — resize forces them to match what the CNN was trained on.
TARGET_SIZE = (32, 32)


def opencv_transform(pil_image, augment=False):
    """
    Converts a raw PIL image from the dataset into a normalized PyTorch tensor.
    augment=True  → apply random augmentations (use for training only)
    augment=False → deterministic, no randomness (use for test/inference)
    """

    # Step 1: PIL -> NumPy array
    # PIL stores pixels as (H, W, C) with values 0–255 — so does OpenCV
    img = np.array(pil_image)                        # shape: (H, W, 3), dtype: uint8

    # Step 2: Resize to the fixed input size the CNN expects
    img = cv2.resize(img, TARGET_SIZE)               # shape: (32, 32, 3)

    # Step 3: Data augmentation — training only
    # These transforms run randomly each epoch, so the model never sees
    # the exact same image twice. This is the primary lever for 90%+ accuracy.
    if augment:
        # RandomHorizontalFlip: flip left-right with 50% probability
        # A truck facing left is still a truck — safe for all CIFAR-10 classes
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)                   # 1 = horizontal flip

        # RandomCrop: pad 4px on each side with zeros, then crop back to 32×32
        # Forces model to learn from partial views of objects, not exact positions
        pad = 4
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)  # (40, 40, 3)
        top  = np.random.randint(0, 2 * pad)         # random crop offset (0–7)
        left = np.random.randint(0, 2 * pad)
        img = img[top:top+32, left:left+32]          # crop back to (32, 32, 3)

    # Step 4: RGB -> BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       # shape still: (32, 32, 3)

    # Step 5: Scale pixel values from 0–255 to 0.0–1.0
    img = img.astype(np.float32) / 255.0             # shape: (32, 32, 3), dtype: float32

    # Step 6: Normalize each channel using global mean and std
    img = (img - CIFAR_MEAN) / CIFAR_STD             # shape: (32, 32, 3), values now ~[-2, 2]

    # Step 7: (H, W, C) -> (C, H, W) — PyTorch expects channels first
    img = np.transpose(img, (2, 0, 1))               # shape: (3, 32, 32)

    # Step 8: NumPy -> PyTorch tensor
    return torch.tensor(img, dtype=torch.float32)
    # np.transpose(img, (2, 0, 1)) reorders the axes:
    #   axis 2 (channels=3) moves to position 0
    #   axis 0 (height=32)  moves to position 1
    #   axis 1 (width=32)   moves to position 2
    # Result: (32, 32, 3) -> (3, 32, 32)
    img = np.transpose(img, (2, 0, 1))               # shape: (3, 32, 32)

    # Step 7: NumPy -> PyTorch tensor
    return torch.tensor(img, dtype=torch.float32)    # shape: (3, 32, 32)


# --- Quick sanity check ---
if __name__ == "__main__":
    import torchvision.datasets as datasets

    raw_dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=None)
    pil_image, label = raw_dataset[0]

    tensor = opencv_transform(pil_image)

    print(f"Output shape  : {tensor.shape}")          # expect torch.Size([3, 32, 32])
    print(f"dtype         : {tensor.dtype}")          # expect torch.float32
    print(f"Value range   : [{tensor.min():.3f}, {tensor.max():.3f}]")  # expect roughly [-2, 2]
    print(f"Channel means : {tensor.mean(dim=[1,2])}")  # should be close to [0, 0, 0]
