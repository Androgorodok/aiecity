# Исходный код проекта

В этой папке размещается **основной код проекта**, организованный в подпапки для удобства масштабирования.

## 📁 Структура папок

```
src/
├── data/                 # Работа с данными
│   ├── __init__.py
│   ├── dataset.py       # CornDiseaseDataset
│   └── loaders.py       # DataLoaders и трансформации
├── models/              # Архитектуры моделей
│   ├── __init__.py
│   ├── architectures.py # Модели (ResNet18 — основная)
│   └── utils.py         # Оптимизаторы, schedulers
├── training/            # Обучение и валидация
│   ├── __init__.py
│   └── train.py         # Цикл обучения
├── __init__.py
├── train.py             # Точка входа для обучения
├── inference.py         # Инференс (предсказания)
├── utils.py             # Общие утилиты
└── README.md            # Этот файл
```

## 🎯 Рекомендуемая модель

**ResNet18** показала лучшие результаты:
- **Test Accuracy: 97.85%**
- **F1 Score: 0.9717**

Используется по умолчанию при обучении.

## 🚀 Быстрый старт

### Обучение

```bash
# С параметрами по умолчанию (ResNet18)
python -m src.train

# С конфигурацией из файла
python -m src.train --config project/configs/training_config.json

# С пользовательскими параметрами
python -m src.train --model_name resnet18 --num_epochs 15 --batch_size 32 --learning_rate 0.0001
```

### Инференс

```bash
python -m src.inference --image_path project/data/Healthy/image.jpg
```

## 📚 Модули

### `data/`
- `CornDiseaseDataset` — PyTorch Dataset для изображений
- `create_dataloaders()` — создание DataLoaders для train/val/test
- `get_transforms()` — трансформации с аугментацией

### `models/`
- `create_model()` — создание моделей (ResNet18, ResNet50, VGG16, MobileNetV2, EfficientNet-B0)
- `SimpleConvNet` — простая CNN для сравнения
- `get_optimizer()`, `get_scheduler()` — настройка обучения
- `freeze_backbone()` — замораживание весов

### `training/`
- `train_epoch()` — один эпох обучения
- `validate()` — оценка на валидационном наборе
- `train()` — полный цикл обучения

### `utils.py`
- `setup_logging()` — логирование
- `load_config()`, `save_config()` — работа с конфигами
- Прочие утилиты

## 💡 Примеры использования

### Использование в других скриптах

```python
from src.data import create_dataloaders
from src.models import create_model, get_optimizer
from src.training import train_epoch, validate

# Загрузить данные
train_loader, val_loader, test_loader, classes = create_dataloaders(
    data_dir="project/data",
    batch_size=32
)

# Создать модель
model = create_model(num_classes=4, pretrained=True)  # ResNet18 по умолчанию

# Обучение
for epoch in range(num_epochs):
    loss, acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
```

## 📝 Параметры обучения

- `--data_dir` (default: `project/data`) — путь к данным
- `--model_name` (default: `resnet18`) — архитектура модели
- `--batch_size` (default: 32) — размер батча
- `--num_epochs` (default: 50) — количество эпох
- `--learning_rate` (default: 0.0001) — learning rate
- `--output_dir` (default: `project/artifacts`) — директория для результатов

