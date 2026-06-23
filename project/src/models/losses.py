import torch.nn as nn

def get_criterion(label_smoothing: float = 0.1):
    """
    Получить функцию потерь с label smoothing.
    
    Args:
        label_smoothing: Параметр сглаживания меток
        
    Returns:
        nn.CrossEntropyLoss с label smoothing
    """
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)