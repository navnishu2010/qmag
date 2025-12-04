import math 
import numpy as np
import scipy
from scipy import signal

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
