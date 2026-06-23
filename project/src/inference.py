"""
Инференс модели.

Запуск:

python -m src.inference \
    --image image.jpg
"""

import argparse
import json

import torch
from PIL import Image
from torchvision import transforms

from src.models import create_model


def load_model(
    model_path,
    config_path,
    device,
):

    with open(config_path) as f:
        config = json.load(f)

    model = create_model(
        num_classes=config["num_classes"],
        pretrained=False,
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
        )
    )

    model.to(device)
    model.eval()

    return model, config


def preprocess_image(path):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    image = Image.open(path).convert("RGB")

    return transform(image).unsqueeze(0)


@torch.no_grad()
def predict(
    model,
    image_tensor,
    class_names,
    device,
):

    image_tensor = image_tensor.to(device)

    logits = model(image_tensor)

    probs = torch.softmax(
        logits,
        dim=1,
    )

    pred_idx = probs.argmax().item()

    return {
        "class": class_names[str(pred_idx)],
        "confidence": float(
            probs[0][pred_idx]
        ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
    )

    parser.add_argument(
        "--weights",
        default="artifacts/best_model.pt",
    )

    parser.add_argument(
        "--config",
        default="artifacts/config.json",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, cfg = load_model(
        args.weights,
        args.config,
        device,
    )

    image = preprocess_image(
        args.image
    )

    result = predict(
        model,
        image,
        cfg["class_names"],
        device,
    )

    print(result)


if __name__ == "__main__":
    main()