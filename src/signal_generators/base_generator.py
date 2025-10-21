"""
Базовый класс для генераторов сигналов
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional

class BaseSignalGenerator(ABC):
    """Абстрактный базовый класс для генераторов сигналов"""
    
    def __init__(self, name: str = "BaseSignalGenerator"):
        self.name = name
    
    @abstractmethod
    def generate(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Генерация сигнала и временной шкалы"""
        pass
    
    def add_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Добавление гауссовского шума к сигналу"""
        signal_power = np.mean(signal**2)
        snr_linear = 10**(snr_db/10)
        noise_power = signal_power / snr_linear
        
        # Генерируем шум
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
        return signal + noise
    
    def calculate_snr(self, clean_signal: np.ndarray, noisy_signal: np.ndarray) -> float:
        """Вычисление SNR в dB"""
        signal_power = np.mean(clean_signal**2)
        noise_power = np.mean((noisy_signal - clean_signal)**2)
        
        if noise_power == 0:
            return float('inf')
        
        snr_linear = signal_power / noise_power
        return 10 * np.log10(snr_linear)
