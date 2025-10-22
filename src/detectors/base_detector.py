"""
Базовый класс для всех детекторов аномалий
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any
import numpy as np

class BaseDetector(ABC):
    """Абстрактный базовый класс для детекторов"""
    
    def __init__(self, name: str = "BaseDetector"):
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, data: np.ndarray, **kwargs) -> None:
        """Обучение детектора на данных"""
        pass
    
    @abstractmethod
    def detect(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Обнаружение аномалий в данных"""
        pass
    
    @abstractmethod
    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """Вероятности принадлежности к аномалиям"""
        pass
    
    def get_params(self) -> dict:
        """Получить параметры модели"""
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_') and not callable(value)}
