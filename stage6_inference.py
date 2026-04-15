import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from stage2_preprocessing import opencv_transform
from stage3_cnn import CIFAR10_CNN

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = CIFAR10_CNN().to(device)
model.load_state_dict(torch.load("cifar10_cnn.pth", map_location=device))
model.eval()


def predict(image_path):
    """Return top-3 (label, confidence%) predictions for an image on disk."""
    bgr_frame = cv2.imread(image_path)
    if bgr_frame is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    from PIL import Image
    pil_image = Image.fromarray(rgb_frame)
    tensor = opencv_transform(pil_image, augment=False)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)

    # Softmax not applied during training (CrossEntropyLoss handles it);
    # apply explicitly here so confidences are interpretable probabilities.
    probs = torch.softmax(outputs, dim=1).squeeze()

    top3_probs, top3_indices = torch.topk(probs, 3)
    results = [
        (CLASS_NAMES[idx.item()], prob.item() * 100)
        for idx, prob in zip(top3_indices, top3_probs)
    ]
    return bgr_frame, results


def show_result(image_path):
    bgr_frame, results = predict(image_path)
    rgb_display = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(10, 4),
                                          gridspec_kw={"width_ratios": [1, 1.5]})

    ax_img.imshow(rgb_display)
    ax_img.set_title("Input Image", fontsize=11)
    ax_img.axis("off")

    labels = [r[0] for r in results]
    scores = [r[1] for r in results]
    colors = ["#2196F3", "#90CAF9", "#BBDEFB"]   # dark → light blue for rank 1→3

    bars = ax_bar.barh(labels[::-1], scores[::-1], color=colors[::-1])
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("Confidence (%)")
    ax_bar.set_title("Top-3 Predictions", fontsize=11)
    ax_bar.axvline(x=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    for bar, score in zip(bars, scores[::-1]):
        ax_bar.text(min(score + 1, 97), bar.get_y() + bar.get_height() / 2,
                    f"{score:.1f}%", va="center", fontsize=9)

    fig.suptitle(f"Prediction: {results[0][0].upper()}  ({results[0][1]:.1f}% confident)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("inference_result.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nTop-3 predictions for: {image_path}")
    for rank, (label, prob) in enumerate(results, 1):
        print(f"  {rank}. {label:<12} {prob:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stage6_inference.py <path_to_image>")
        print("Example: python stage6_inference.py dog.jpg")
        sys.exit(1)

    show_result(sys.argv[1])
