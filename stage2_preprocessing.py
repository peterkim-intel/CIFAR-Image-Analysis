import numpy as np
import cv2
import torch

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

TARGET_SIZE = (32, 32)


def opencv_transform(pil_image, augment=False):
    """
    Convert a PIL image to a normalized PyTorch tensor.
    augment=True  → random flip + crop (training only)
    augment=False → deterministic (test/inference)
    """
    img = np.array(pil_image)
    img = cv2.resize(img, TARGET_SIZE)

    if augment:
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)

        pad = 4
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        top  = np.random.randint(0, 2 * pad)
        left = np.random.randint(0, 2 * pad)
        img = img[top:top+32, left:left+32]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0
    img = (img - CIFAR_MEAN) / CIFAR_STD
    img = np.transpose(img, (2, 0, 1))               # (H, W, C) → (C, H, W)

    return torch.tensor(img, dtype=torch.float32)


# --- Quick sanity check ---
if __name__ == "__main__":
    import torchvision.datasets as datasets

    raw_dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=None)
    pil_image, label = raw_dataset[0]

    tensor = opencv_transform(pil_image)

    print(f"Output shape  : {tensor.shape}")
    print(f"dtype         : {tensor.dtype}")
    print(f"Value range   : [{tensor.min():.3f}, {tensor.max():.3f}]")
    print(f"Channel means : {tensor.mean(dim=[1,2])}")
