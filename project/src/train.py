"""
Обучение финальной модели ResNet18.

Запуск:

python -m src.train
"""

import json
from pathlib import Path

import torch
import yaml

from src.data import create_dataloaders
from src.models import (
    create_model,
    get_optimizer,
    get_scheduler,
    get_criterion,
)
from src.utils import (
    setup_logging,
    ensure_directory,
)

logger = setup_logging("INFO")


def load_config():
    config_path = Path("config/train_config.yaml")

    with open(
        config_path,
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()
        optimizer.step()

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        preds = outputs.argmax(dim=1)

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / total,
        correct / total,
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        preds = outputs.argmax(dim=1)

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / total,
        correct / total,
    )


def main():

    cfg = load_config()

    output_dir = Path(
        cfg["output_dir"]
    )

    ensure_directory(
        output_dir
    )

    if cfg["device"] == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(
            cfg["device"]
        )

    logger.info(
        f"Device: {device}"
    )

    train_loader, val_loader, test_loader, class_names = (
        create_dataloaders(
            data_dir=cfg["data_dir"],
            batch_size=cfg["batch_size"],
            img_size=cfg["img_size"],
        )
    )

    model = create_model(
        num_classes=len(class_names),
        pretrained=cfg["pretrained"],
    ).to(device)

    criterion = get_criterion(
        label_smoothing=cfg[
            "label_smoothing"
        ]
    )

    optimizer = get_optimizer(
        model,
        lr=cfg["learning_rate"],
        weight_decay=cfg[
            "weight_decay"
        ],
    )

    scheduler = get_scheduler(
        optimizer,
        scheduler_name=cfg[
            "scheduler"
        ],
        total_epochs=cfg[
            "epochs"
        ],
    )

    best_val_acc = 0.0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(
        cfg["epochs"]
    ):

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(
            train_loss
        )
        history["train_acc"].append(
            train_acc
        )

        history["val_loss"].append(
            val_loss
        )
        history["val_acc"].append(
            val_acc
        )

        logger.info(
            f"Epoch [{epoch+1}/{cfg['epochs']}] "
            f"Train Acc={train_acc:.4f} "
            f"Val Acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                output_dir /
                cfg["model_name"],
            )

            logger.info(
                f"Best model saved "
                f"({best_val_acc:.4f})"
            )

    model.load_state_dict(
        torch.load(
            output_dir /
            cfg["model_name"],
            map_location=device,
        )
    )

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    logger.info(
        f"Test Accuracy: {test_acc:.4f}"
    )

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "num_classes": len(class_names),
                "class_names": class_names,
                "test_accuracy": test_acc,
                "history": history,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
