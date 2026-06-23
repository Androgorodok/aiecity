"""
Вспомогательные функции для работы с моделями.
"""

from typing import Optional

import torch
import torch.nn as nn


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """
    Заморозить (или разморозить) веса backbone модели.
    Используется при использовании предобученных моделей.
    
    Args:
        model: Модель
        freeze: True для замораживания, False для разморозки
        
    Examples:
        >>> freeze_backbone(model, freeze=True)  # Заморозить веса
        >>> freeze_backbone(model, freeze=False)  # Разморозить веса
    """
    for param in model.parameters():
        param.requires_grad = not freeze


def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    lr: float = 1e-4,
    weight_decay: float = 1e-4
) -> torch.optim.Optimizer:
    """
    Создать оптимизатор.
    
    Args:
        model: Модель
        optimizer_name: Название оптимизатора ('adam', 'sgd')
        lr: Learning rate. Для ResNet18 рекомендуется 1e-4
        weight_decay: Коэффициент регуляризации L2
        
    Returns:
        Оптимизатор
        
    Examples:
        >>> optimizer = get_optimizer(model, lr=1e-4)
    """
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Неизвестный оптимизатор: {optimizer_name}")
    
    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "cosine",
    total_epochs: int = 50
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Создать scheduler для learning rate.
    
    Args:
        optimizer: Оптимизатор
        scheduler_name: Название scheduler ('cosine', 'step', 'exponential', 'none')
        total_epochs: Количество эпох
        
    Returns:
        Scheduler или None
        
    Examples:
        >>> scheduler = get_scheduler(optimizer, scheduler_name="cosine", total_epochs=50)
    """
    if scheduler_name.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs
        )
    elif scheduler_name.lower() == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=total_epochs // 3,
            gamma=0.1
        )
    elif scheduler_name.lower() == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95
        )
    else:
        return None
    
    return scheduler
