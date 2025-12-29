import numpy as np
import allantools

def compute_allan_deviation(data, fs):
    """
    Computes the Overlapping Allan Deviation using the allantools library.
    
    Args:
        data (np.ndarray): Signal array (magnetic field or frequency).
        fs (float): Sampling frequency in Hz.
        
    Returns:
        taus (np.ndarray): Averaging times (s).
        adev (np.ndarray): Allan Deviation values.
    """
    # "freq" mode is for sensor data (amplitude vs time).
    # "phase" mode is for timing clocks.
    # oadev (Overlapping ADEV) is preferred for better stats than standard adev.
    taus, adev, errors, ns = allantools.oadev(
        data, 
        rate=fs, 
        data_type="freq", 
        taus="octave"  # 'octave' generates nice log-spaced points automatically
    )
    
    return taus, adev
