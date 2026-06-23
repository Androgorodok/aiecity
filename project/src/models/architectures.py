"""
Архитектуры моделей для классификации болезней кукурузы.

Лучшая модель:
- ResNet18 (ImageNet pretrained)
- Full fine-tuning
"""

import torch.nn as nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)


def create_model(
    num_classes: int = 4,
    pretrained: bool = True,
) -> nn.Module:
    """
    Создание ResNet18 для классификации болезней кукурузы.

    Args:
        num_classes: количество классов
        pretrained: использовать ли ImageNet веса

    Returns:
        nn.Module
    """

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None

    model = resnet18(weights=weights)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model