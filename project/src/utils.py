# src/utils.py
import logging
import os
from pathlib import Path

def setup_logging(level="INFO"):
    """Настройка логирования"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def ensure_directory(path):
    """Создать папку, если её нет"""
    Path(path).mkdir(parents=True, exist_ok=True)
    