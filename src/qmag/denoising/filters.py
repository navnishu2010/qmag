import numpy as np
import scipy
from scipy import signal

# 1. Define the Interface (The Contract)
class BaseFilter(ABC):
    """Abstract base class to ensure all filters look the same to the pipeline."""
    
    @abstractmethod
    def apply_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply the filter to the data."""
        pass

class NotchFilter(BaseFilter):
    """
    Applies a digital notch filter to remove a specific frequency.
    Useful to remove power line hum (50Hz & 60Hz)
    Uses zero-phase filtering (filtfilt) to prevent phase distortion.

    Args:
        freq (float): Frequency to remove (Hz).
        fs (float): Sampling rate (Hz).
        q (float): Quality factor. Higher values = narrower notch. Defaults to 30.0.
    """
    def __init__(self, freq: float, fs: float, q: float = 30.0):
        self.freq = freq
        self.fs = fs
        self.q = q
        nyquist = 0.5 * fs
        normal_freq = freq / nyquist
        self.b, self.a = signal.iirnotch(normal_freq, q)

    def apply_filter(self, data: np.ndarray) -> np.ndarray:
        return signal.filtfilt(self.b, self.a, data)

class Butterworth(BaseFilter):
    """
    A class to create and apply a Butterworth digital filter.
    
    Attributes:
        cutoff (float): The cutoff frequency of the filter.
        fs (float): The sampling frequency.
        order (int): The order of the filter.
        btype (str): The type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop').
        _sos (np.ndarray): The Second-Order Sections coefficients.    
    """
    
    def __init__(self, cutoff: float, fs: float, order: int = 5, btype: str = 'lowpass'):
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.btype = btype
        self._sos = self._design_filter()
        
    def _design_filter(self) -> np.ndarray:
        """Computes the SOS coefficients for the filter."""
        nyquist = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyquist
        
        # Determine output format 'sos' for numerical stability
        sos = signal.butter(self.order, normal_cutoff, btype=self.btype, analog=False, output='sos')
        return sos

    def apply_filter(self, data: np.ndarray) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
            
        return signal.filtfilt(self._sos, data)
    
class MovingAverageFilter(BaseFilter):
    """
    Applies a Moving Average (Boxcar) filter to smooth the signal.
    
    This implementation uses convolution with 'mode=same', which centers the 
    window on the current point. This prevents the phase shift (delay) 
    associated with standard causal moving averages.
    
    Attributes:
        window_size (int): The number of samples to average over.
    """
    
    def __init__(self, window_size: int):
        if window_size < 1:
            raise ValueError("Window size must be at least 1.")
            
        self.window_size = int(window_size)
        self.kernel = np.ones(self.window_size) / self.window_size

    def apply(self, data: np.ndarray) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
            
        # mode='same' returns output of length max(M, N) - boundary effects are centered
        return np.convolve(data, self.kernel, mode='same')