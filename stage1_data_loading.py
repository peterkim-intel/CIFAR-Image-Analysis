import torchvision.datasets as datasets
from stage2_preprocessing import opencv_transform

# CIFAR-10 class names in label order (label 0 = airplane, label 9 = truck)
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Download and load the raw train/test splits
# root="./data"  -> where to store the downloaded files
# train=True/False -> which split to load
# download=True  -> download if not already present
# transform=opencv_transform -> each PIL image is now preprocessed before the DataLoader returns it
train_dataset = datasets.CIFAR10(root="./data", train=True,  download=True, transform=opencv_transform)
test_dataset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=opencv_transform)

print(f"Training samples : {len(train_dataset)}")   # expect 50,000
print(f"Test samples     : {len(test_dataset)}")    # expect 10,000

# Inspect a single sample
image, label = train_dataset[0]
print(f"\nFirst sample:")
print(f"  Image type : {type(image)}")              # PIL.Image.Image
print(f"  Image size : {image.size}")               # (32, 32) — width x height in PIL
print(f"  Label      : {label} ({CLASS_NAMES[label]})")

# --- DataLoader ---
from torch.utils.data import DataLoader

BATCH_SIZE = 64   # number of images processed together per training step
                  # larger = faster training, more memory; smaller = noisier gradient updates

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"\nBatch size       : {BATCH_SIZE}")
print(f"Training batches : {len(train_loader)}")    # 50,000 / 64 = ~781
print(f"Test batches     : {len(test_loader)}")
