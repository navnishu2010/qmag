import numpy as np
import pywt
import scipy
from scipy import signal
from abc import ABC, abstractmethod

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

class ButterworthFilter(BaseFilter):
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
            
        return signal.sosfiltfilt(self._sos, data)
    
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

    def apply_filter(self, data: np.ndarray) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
            
        # mode='same' returns output of length max(M, N) - boundary effects are centered
        return np.convolve(data, self.kernel, mode='same')
    
class WaveletDetrender:
    def __init__(self, wavelet='db4', level=9):
        """
        Removes low-frequency non-linear drift using Wavelet Decomposition.
        
        Args:
            wavelet (str): The wavelet shape (e.g., 'db4', 'sym8', 'haar').
            level (int): Decomposition level. Higher level = removes slower drifts.
                         If drift is very slow, use a high level (e.g., 8-10).
        """
        self.wavelet = wavelet
        self.level = level
        
    def apply_filter(self, data):
        # 1. Decompose signal
        coeffs = pywt.wavedec(data, self.wavelet, level=self.level)
        
        # 2. Isolate the Trend (Approximation Coefficients)
        # We keep coeffs[0] (the trend) and zero out all detail coefficients
        trend_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        
        # 3. Reconstruct the Trend
        trend = pywt.waverec(trend_coeffs, self.wavelet)
        
        # 4. Fix length mismatch (Wavelets sometimes add padding bytes)
        if len(trend) > len(data):
            trend = trend[:len(data)]
            
        # 5. Subtract Trend from Data
        return data - trend
    
class PolynomialDetrender:
    def __init__(self, order=3):
        """
        Removes drift by fitting a polynomial curve.
        
        Args:
            order (int): 1=Linear, 2=Quadratic, 3=Cubic.
                         Order 3 is usually perfect for temperature drift.
        """
        self.order = order
        self.trend = None
        
    def apply_filter(self, data, t=None):
        n = len(data)
        if t is None:
            t = np.arange(n)
            
        # 1. Fit the polynomial
        coeffs = np.polyfit(t, data, self.order)
        
        # 2. Calculate the trend line
        self.trend = np.polyval(coeffs, t)
        
        # 3. Subtract trend
        return data - self.trend