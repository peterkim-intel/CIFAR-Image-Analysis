import torchvision.datasets as datasets
from stage2_preprocessing import opencv_transform

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

train_dataset = datasets.CIFAR10(root="./data", train=True,  download=True, transform=opencv_transform)
test_dataset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=opencv_transform)

print(f"Training samples : {len(train_dataset)}")
print(f"Test samples     : {len(test_dataset)}")

image, label = train_dataset[0]
print(f"\nFirst sample:")
print(f"  Image type : {type(image)}")
print(f"  Image size : {image.size}")
print(f"  Label      : {label} ({CLASS_NAMES[label]})")

from torch.utils.data import DataLoader

BATCH_SIZE = 64

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"\nBatch size       : {BATCH_SIZE}")
print(f"Training batches : {len(train_loader)}")
print(f"Test batches     : {len(test_loader)}")
