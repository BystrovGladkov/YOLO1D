"""Контейнер сигнала дефектограммы. Перенесено из `EC_preprocess/signal.py` без изменений."""
import numpy as np

class Signal:
    """
        Класс, хранящий основную информацию о сигнале:
        Чистые данные (data), 
        данные о частотах принимаемых значений сигнала (amplitudes),
        данные о средем значении (mean) и среднеквадратическом отклонении (sigma)
    """
    
    def __init__(self, data, name="signal") -> None:
        self.data = data
        self.shape = data.shape
        self.name = name
        (self.channels, self.length) = self.shape
        self._calc_mean()

    def _calc_mean(self) -> None:
        self.mean = np.mean(self.data, axis=1)