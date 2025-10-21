"""
Генератор радиолокационных сигналов
Адаптированная версия оригинального Signal_gen.py
"""

import numpy as np
import math
from typing import Tuple, Dict, Optional
from matplotlib.pyplot import figure, plot, xlabel, ylabel, grid, show, close

from .base_generator import BaseSignalGenerator

class RadarSignalGenerator(BaseSignalGenerator):
    """
    Генератор радиоимпульсов с прямоугольной огибающей
    Для тестирования байесовских детекторов
    """
    
    def __init__(self, 
                 amplitude: float = 1.5,
                 carrier_freq: float = 1e5,
                 initial_phase: float = 0.59,
                 sampling_freq: float = None,
                 pulse_width: float = 1e-3):
        
        super().__init__("RadarSignalGenerator")
        
        self.A = amplitude          # Амплитуда сигнала, В
        self.fc = carrier_freq      # Несущая частота, Гц
        self.fi0 = initial_phase    # Начальная фаза, радиан
        self.Fd = sampling_freq or (10 * carrier_freq)  # Частота дискретизации, Гц
        self.dt = 1 / self.Fd       # Период дискретизации, с
        self.ti = pulse_width       # Длительность импульса, с
        
        # Предварительные расчеты
        self.Ps = 0.5 * (self.A ** 2)  # Мощность сигнала, Вт
        self.Es = self.Ps * self.ti    # Энергия сигнала, Дж
    
    def generate(self, 
                 num_pulses: int = 1,
                 observation_time: Optional[float] = None,
                 add_noise: bool = False,
                 noise_snr_db: float = 20.0,
                 return_metadata: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерация радиоимпульсного сигнала
        
        Parameters:
        -----------
        num_pulses : int
            Количество импульсов
        observation_time : float, optional
            Время наблюдения (если None, вычисляется автоматически)
        add_noise : bool
            Добавлять ли шум к сигналу
        noise_snr_db : float
            Уровень SNR шума в dB
        return_metadata : bool
            Возвращать ли метаданные сигнала
            
        Returns:
        --------
        signal : np.ndarray
            Сгенерированный сигнал
        time_axis : np.ndarray  
            Временная шкала
        metadata : dict, optional
            Метаданные сигнала (только если return_metadata=True)
        """
        
        # Время наблюдения
        Ts = observation_time or (num_pulses * self.ti)
        
        # Формируем дискретную шкалу времени
        t = np.arange(0.0, Ts, self.dt, dtype=float)
        
        # Количество отсчетов
        nti = round(self.ti / self.dt)  # на длительность импульса
        nts = len(t)                    # всего отсчетов
        
        # Формируем огибающую
        envelope = np.zeros(nts)
        for i in range(num_pulses):
            start_idx = i * nti
            end_idx = min((i + 1) * nti, nts)
            envelope[start_idx:end_idx] = 1.0
        
        # Формируем несущую
        carrier = self.A * np.cos(2 * np.pi * self.fc * t + self.fi0)
        
        # Полный сигнал
        clean_signal = carrier * envelope
        
        # Добавляем шум если нужно
        if add_noise:
            signal = self.add_noise(clean_signal, noise_snr_db)
        else:
            signal = clean_signal
        
        if return_metadata:
            metadata = {
                'amplitude': self.A,
                'carrier_frequency': self.fc,
                'initial_phase': self.fi0,
                'sampling_frequency': self.Fd,
                'pulse_width': self.ti,
                'num_pulses': num_pulses,
                'observation_time': Ts,
                'signal_power': self.Ps,
                'signal_energy': self.Es,
                'has_noise': add_noise,
                'snr_db': noise_snr_db if add_noise else None,
                'envelope': envelope
            }
            return signal, t, metadata
        else:
            return signal, t
    
    def plot_signal(self, 
                   signal: np.ndarray, 
                   time_axis: np.ndarray,
                   title: str = "Радиоимпульсный сигнал",
                   save_path: Optional[str] = None) -> None:
        """
        Визуализация сгенерированного сигнала
        
        Parameters:
        -----------
        signal : np.ndarray
            Сигнал для визуализации
        time_axis : np.ndarray
            Временная шкала
        title : str
            Заголовок графика
        save_path : str, optional
            Путь для сохранения графика
        """
        
        fig = figure(figsize=(12, 6))
        plot(time_axis * 1e3, signal, color='k', linewidth=0.8)
        grid(True, alpha=0.3)
        xlabel('Время, мс')
        ylabel('Напряжение, В')
        title(title)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            close(fig)
        else:
            show()
    
    def generate_for_detection_test(self, 
                                   num_normal: int = 900,
                                   num_anomaly: int = 100,
                                   anomaly_snr_db: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерация данных для тестирования детектора
        
        Parameters:
        -----------
        num_normal : int
            Количество нормальных отсчетов (шум)
        num_anomaly : int
            Количество аномальных отсчетов (сигнал+шум)
        anomaly_snr_db : float
            SNR для аномальных отсчетов
            
        Returns:
        --------
        data : np.ndarray
            Объединенные данные
        labels : np.ndarray
            Метки (0 - норма, 1 - аномалия)
        """
        
        # Генерируем нормальные данные (только шум)
        noise_only = np.random.normal(0, 1, num_normal)
        
        # Генерируем аномальные данные (сигнал + шум)
        signal, _ = self.generate(num_pulses=1, add_noise=True, noise_snr_db=anomaly_snr_db)
        
        # Если сигнал слишком длинный, обрезаем
        if len(signal) > num_anomaly:
            signal = signal[:num_anomaly]
        elif len(signal) < num_anomaly:
            # Дублируем если нужно
            repeats = (num_anomaly // len(signal)) + 1
            signal = np.tile(signal, repeats)[:num_anomaly]
        
        # Объединяем данные
        data = np.concatenate([noise_only, signal])
        labels = np.concatenate([np.zeros(num_normal), np.ones(num_anomaly)])
        
        # Перемешиваем
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]
        
        return data, labels
