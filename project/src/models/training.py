import copy
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights


class CornDiseaseDataset(Dataset):
    def __init__(self, df, class_to_idx, transform=None):
        self.df = df
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            image = Image.open(row["path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[row["label"]]
        return image, label


def make_class_weights(df, class_to_idx, device):
    labels = [class_to_idx[label] for label in df["label"]]

    counts = np.bincount(labels).astype(float)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


def make_weighted_sampler(df, class_to_idx):
    labels = [class_to_idx[label] for label in df["label"]]

    counts = np.bincount(labels)
    class_weights = 1.0 / counts

    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    start_time = time.time()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        batch_size = y.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(1) == y).sum().item()
        total_seen += batch_size

    return (
        total_loss / total_seen,
        total_correct / total_seen,
        time.time() - start_time,
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
    total_correct = 0
    total_seen = 0

    all_preds = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        preds = logits.argmax(1)

        batch_size = y.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (preds == y).sum().item()
        total_seen += batch_size

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return (
        total_loss / total_seen,
        total_correct / total_seen,
        all_preds,
        all_labels,
    )


def make_resnet(
    num_classes,
    device,
    freeze_backbone=False,
):
    model = resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")

    return model.to(device)


def run_experiment(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    scheduler=None,
):
    best_val_acc = 0.0
    best_weights = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(num_epochs):
        train_loss, train_acc, _ = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss, val_acc, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if scheduler is not None:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(
                model.state_dict()
            )

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return {
        "model": model,
        "best_val_acc": best_val_acc,
        "history": history,
    }
