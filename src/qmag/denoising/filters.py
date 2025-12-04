from abc import ABC, abstractmethod
import numpy as np
import scipy
from scipy import signal

# 1. Define the Interface (The Contract)
class BaseFilter(ABC):
    """Abstract base class to ensure all filters look the same to the pipeline."""
    
    @abstractmethod
    def filter(self, data: np.ndarray) -> np.ndarray:
        """Apply the filter to the data."""
        pass

class NotchFilter(BaseFilter):
    def __init__(self, freq: float, fs: float, q: float = 30.0):
        self.freq = freq
        self.fs = fs
        self.q = q
        nyquist = 0.5 * fs
        normal_freq = freq / nyquist
        self.b, self.a = signal.iirnotch(normal_freq, q)

    def filter(self, data: np.ndarray) -> np.ndarray:
        return signal.filtfilt(self.b, self.a, data)
