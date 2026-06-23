"""
Dataset для классификации болезней кукурузы.
"""

from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class CornDiseaseDataset(Dataset):
    """
    Dataset для классификации болезней кукурузы.
    """
    
    def __init__(
        self, 
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None
    ):
        """
        Инициализировать dataset.
        
        Args:
            image_paths: Список путей к изображениям
            labels: Список меток классов
            transform: Трансформации для изображений
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        assert len(image_paths) == len(labels), \
            f"Количество изображений ({len(image_paths)}) должно совпадать " \
            f"с количеством меток ({len(labels)})"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Получить изображение и его метку.
        
        Args:
            idx: Индекс элемента
            
        Returns:
            Кортеж (тензор изображения, метка класса)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Загрузить изображение
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Ошибка загрузки {image_path}: {e}")
            image = Image.new('RGB', (224, 224))
        
        # Применить трансформации
        if self.transform:
            image = self.transform(image)
        
        return image, label
