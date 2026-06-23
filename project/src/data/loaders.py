"""
Загрузка данных и трансформации для изображений.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import pandas as pd

from .dataset import CornDiseaseDataset


def get_transforms(img_size: int = 224, augment: bool = True) -> Dict[str, transforms.Compose]:
    """
    Получить трансформации для изображений.
    
    Args:
        img_size: Размер изображения
        augment: Применять ли аугментацию
        
    Returns:
        Словарь с трансформациями для train/val
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        normalize,
    ]) if augment else transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    
    return {"train": train_transform, "val": val_transform}


def save_splits_to_csv(
    image_paths: Dict[str, List[str]],
    image_labels: Dict[str, List[int]],
    class_names: Dict[int, str],
    output_dir: str = "splits"
):
    """
    Сохранить разбиение в три CSV файла.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "val", "test"]:
        data = []
        for path, label in zip(image_paths[split], image_labels[split]):
            data.append({
                "image_path": path,
                "label": label,
                "class_name": class_names[label]
            })
        
        df = pd.DataFrame(data)
        csv_path = output_dir / f"{split}.csv"
        df.to_csv(csv_path, index=False)
        print(f"{split}.csv сохранён ({len(data)} записей)")


def load_images_and_labels(
    data_dir: str,
    train_size: float = 0.7,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[Dict[str, List[str]], Dict[str, List[int]], Dict[int, str]]:
    """
    Загрузить изображения и метки из структурированной директории.
        
    Args:
        data_dir: Путь к корневой директории с данными
        train_size: Доля для обучения
        val_size: Доля для валидации
        random_state: Seed для разбиения
        
    Returns:
        Кортеж (словарь путей, словарь меток, словарь названий классов)
    """
    image_paths = {"train": [], "val": [], "test": []}
    image_labels = {"train": [], "val": [], "test": []}
    class_names = {}
    
    class_id = 0
    
    # Пройти по всем классам
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.isdir(class_dir):
            continue
        
        class_names[class_id] = class_name
        
        # Получить все изображения в этом классе
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            image_files.extend(
                [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                 if f.endswith(ext)]
            )
        
        # Разбить на train/val/test
        train_val_split = train_size + val_size
        
        # Первый split: train+val vs test
        tv_paths, test_paths = train_test_split(
            image_files,
            train_size=train_val_split,
            random_state=random_state
        )
        
        # Второй split: train vs val
        train_paths, val_paths = train_test_split(
            tv_paths,
            train_size=train_size / train_val_split,
            random_state=random_state
        )
        
        # Добавить в словари
        image_paths["train"].extend(train_paths)
        image_labels["train"].extend([class_id] * len(train_paths))
        
        image_paths["val"].extend(val_paths)
        image_labels["val"].extend([class_id] * len(val_paths))
        
        image_paths["test"].extend(test_paths)
        image_labels["test"].extend([class_id] * len(test_paths))
        
        print(f"Class '{class_name}' (id={class_id}): "
              f"train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")
        
        class_id += 1
    
    return image_paths, image_labels, class_names


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    train_size: float = 0.7,
    val_size: float = 0.15,
    augment: bool = True,
    save_splits: bool = True  # ← НОВЫЙ ПАРАМЕТР
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """
    Создать DataLoaders для обучения, валидации и тестирования.
    
    Args:
        data_dir: Путь к директории с данными
        batch_size: Размер батча
        img_size: Размер изображения
        num_workers: Количество воркеров для DataLoader
        train_size: Доля для обучения
        val_size: Доля для валидации
        augment: Применять ли аугментацию
        save_splits: Сохранять ли CSV-файлы с разбиением
        
    Returns:
        Кортеж (train_loader, val_loader, test_loader, class_names)
    """
    print("Загрузка данных...")
    image_paths, image_labels, class_names = load_images_and_labels(
        data_dir,
        train_size=train_size,
        val_size=val_size
    )
    
    if save_splits:
        save_splits_to_csv(image_paths, image_labels, class_names)
    
    transforms_dict = get_transforms(img_size=img_size, augment=augment)
    
    # Создать datasets
    train_dataset = CornDiseaseDataset(
        image_paths["train"],
        image_labels["train"],
        transform=transforms_dict["train"]
    )
    
    val_dataset = CornDiseaseDataset(
        image_paths["val"],
        image_labels["val"],
        transform=transforms_dict["val"]
    )
    
    test_dataset = CornDiseaseDataset(
        image_paths["test"],
        image_labels["test"],
        transform=transforms_dict["val"]
    )
    
    # Создать dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Датасеты созданы успешно.")
    print(f"Классы: {class_names}")
    
    return train_loader, val_loader, test_loader, class_names
