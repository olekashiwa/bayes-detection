"""
Главный файл проекта Bayesian Detection
Точка входа в приложение
"""

import sys
import os
from pathlib import Path

# Добавляем путь к src для импорта модулей
sys.path.append(str(Path(__file__).parent / 'src'))

import numpy as np
import argparse
import logging

# Импортируем наши модули
from src.detectors.bayesian import BayesianDetector
from src.signal_generators.radar_signal import RadarSignalGenerator

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def demo_radar_signal():
    """Демонстрация генерации радиолокационного сигнала"""
    logger.info("Демонстрация генератора радиосигналов")
    
    # Создаем генератор
    radar_gen = RadarSignalGenerator(
        amplitude=1.5,
        carrier_freq=1e5,
        pulse_width=1e-3
    )
    
    # Генерируем чистый сигнал
    clean_signal, time_axis = radar_gen.generate(
        num_pulses=2,
        observation_time=3e-3
    )
    
    # Генерируем сигнал с шумом
    noisy_signal, _ = radar_gen.generate(
        num_pulses=2,
        observation_time=3e-3,
        add_noise=True,
        noise_snr_db=10
    )
    
    logger.info(f"Параметры сигнала:")
    logger.info(f"  - Амплитуда: {radar_gen.A} В")
    logger.info(f"  - Несущая частота: {radar_gen.fc/1e3:.1f} кГц")
    logger.info(f"  - Длительность импульса: {radar_gen.ti*1e3:.1f} мс")
    logger.info(f"  - Мощность сигнала: {radar_gen.Ps:.3f} Вт")
    
    # Визуализация
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 10))
    
    # Чистый сигнал
    plt.subplot(3, 1, 1)
    plt.plot(time_axis * 1e3, clean_signal, 'b-', linewidth=0.8)
    plt.title('Чистый радиоимпульсный сигнал')
    plt.xlabel('Время, мс')
    plt.ylabel('Напряжение, В')
    plt.grid(True, alpha=0.3)
    
    # Сигнал с шумом
    plt.subplot(3, 1, 2)
    plt.plot(time_axis * 1e3, noisy_signal, 'r-', linewidth=0.8, alpha=0.8)
    plt.title('Сигнал с шумом (SNR=10 dB)')
    plt.xlabel('Время, мс')
    plt.ylabel('Напряжение, В')
    plt.grid(True, alpha=0.3)
    
    # Сравнение
    plt.subplot(3, 1, 3)
    plt.plot(time_axis * 1e3, clean_signal, 'b-', linewidth=1, label='Чистый')
    plt.plot(time_axis * 1e3, noisy_signal, 'r-', linewidth=0.8, alpha=0.6, label='С шумом')
    plt.title('Сравнение чистого и зашумленного сигналов')
    plt.xlabel('Время, мс')
    plt.ylabel('Напряжение, В')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return radar_gen

def main():
    """Основная функция приложения"""
    parser = argparse.ArgumentParser(description='Bayesian Anomaly Detection System')
    
    parser.add_argument('--false-alarm-rate', type=float, default=0.05,
                       help='Уровень ложных тревог (по умолчанию: 0.05)')
    parser.add_argument('--visualize', action='store_true',
                       help='Визуализировать результаты')
    parser.add_argument('--demo-radar', action='store_true',
                       help='Показать демонстрацию радиосигналов')
    parser.add_argument('--use-radar-data', action='store_true',
                       help='Использовать радиолокационные данные для тестирования')
    
    args = parser.parse_args()
    
    try:
        if args.demo_radar:
            demo_radar_signal()
            return
        
        logger.info("Запуск Bayesian Detection System")
        
        if args.use_radar_data:
            # Используем радиолокационные данные
            radar_gen = RadarSignalGenerator()
            data, true_labels = radar_gen.generate_for_detection_test(
                num_normal=800,
                num_anomaly=200,
                anomaly_snr_db=5.0
            )
            logger.info(f"Сгенерировано радиолокационных данных: {len(data)} точек")
            logger.info(f"Аномалий в данных: {np.sum(true_labels)}")
        else:
            # Генерация тестовых данных (оригинальный подход)
            np.random.seed(42)
            n_samples = 1000
            normal_data = np.random.normal(0, 1, int(n_samples * 0.9))
            anomaly_data = np.random.normal(3, 1, int(n_samples * 0.1))
            data = np.concatenate([normal_data, anomaly_data])
            true_labels = np.concatenate([np.zeros(len(normal_data)), np.ones(len(anomaly_data))])
            np.random.shuffle(data)
            np.random.shuffle(true_labels)
            
            logger.info(f"Сгенерировано синтетических данных: {len(data)} точек")
        
        # Инициализация и настройка детектора
        detector = BayesianDetector(
            false_alarm_rate=args.false_alarm_rate,
            mu0=0, sigma0=1,    # Параметры нормального шума
            mu1=2, sigma1=1     # Параметры аномалий
        )
        
        # Обнаружение аномалий
        logger.info("Запуск обнаружения аномалий...")
        anomalies, scores = detector.detect(data)
        
        # Статистика результатов
        n_anomalies = np.sum(anomalies)
        anomaly_ratio = n_anomalies / len(data)
        
        # Точность обнаружения (если есть true_labels)
        if 'true_labels' in locals():
            accuracy = np.mean(anomalies == true_labels)
            logger.info(f"Точность обнаружения: {accuracy:.3f}")
        
        logger.info(f"Обнаружено аномалий: {n_anomalies} ({anomaly_ratio:.2%})")
        logger.info(f"Порог обнаружения: {detector.get_detection_threshold():.3f}")
        
        # Визуализация
        if args.visualize:
            logger.info("Построение графиков...")
            import matplotlib.pyplot as plt
            
            # График результатов
            plt.figure(figsize=(12, 8))
            
            # Данные с выделенными аномалиями
            time_axis = np.arange(len(data))
            normal_mask = ~anomalies
            anomaly_mask = anomalies
            
            plt.subplot(2, 2, 1)
            plt.plot(time_axis[normal_mask], data[normal_mask], 
                    'b.', alpha=0.6, label='Нормальные точки')
            plt.plot(time_axis[anomaly_mask], data[anomaly_mask], 
                    'ro', alpha=0.8, label='Аномалии')
            plt.xlabel('Индекс')
            plt.ylabel('Значение')
            plt.title('Обнаружение аномалий')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Отношение правдоподобий
            plt.subplot(2, 2, 2)
            x = np.linspace(-1, 5, 1000)
            l_ratio = detector.likelihood_ratio(x)
            plt.plot(x, l_ratio, 'purple', linewidth=2, label='L(x) = f₁(x)/f₀(x)')
            plt.axvline(x=detector.threshold, color='red', linestyle='--',
                       label=f'Порог = {detector.threshold:.2f}')
            plt.xlabel('x')
            plt.ylabel('L(x)')
            plt.title('Отношение правдоподобий')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Распределения
            plt.subplot(2, 2, 3)
            plt.hist(data[normal_mask], bins=30, alpha=0.7, density=True, 
                    label='Нормальные', color='blue')
            plt.hist(data[anomaly_mask], bins=30, alpha=0.7, density=True, 
                    label='Аномалии', color='red')
            plt.xlabel('Значение')
            plt.ylabel('Плотность')
            plt.title('Распределения данных')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        logger.info("Программа завершена успешно!")
        
    except Exception as e:
        logger.error(f"Ошибка выполнения: {e}")
        raise

if __name__ == "__main__":
    main()
