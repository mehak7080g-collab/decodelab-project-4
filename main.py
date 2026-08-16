"""
Project 4: Image Recognition (Basic)
DecodeLabs AI Training Project

Uses a pre-trained ResNet18 model from TorchVision to classify
a sample image and display the top predictions.
"""

import argparse
import sys

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


def classify_image(image_path: str, top_k: int = 5):
    """Classify an image with a pre-trained ResNet18 model."""
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    transform = weights.transforms()
    batch = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    values, indices = torch.topk(probabilities, top_k)

    categories = weights.meta["categories"]

    print("=" * 60)
    print("        PROJECT 4: IMAGE RECOGNITION (BASIC)")
    print("=" * 60)
    print(f"Input image: {image_path}")
    print("\nTop predictions:")

    for rank, (value, index) in enumerate(zip(values, indices), start=1):
        label = categories[index.item()]
        print(f"{rank}. {label} — {value.item() * 100:.2f}%")

    print("\nRecognition completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Basic image recognition using a pre-trained ResNet18 model."
    )
    parser.add_argument(
        "image",
        help="Path to the image file to classify."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of predictions to display (default: 5)."
    )
    args = parser.parse_args()

    if args.top_k < 1:
        print("Error: --top-k must be at least 1.")
        sys.exit(1)

    if not os.path.isfile(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    try:
        classify_image(args.image, args.top_k)
    except Exception as exc:
        print(f"Recognition failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
