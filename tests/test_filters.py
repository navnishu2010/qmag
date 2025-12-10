import numpy as np
import pytest
from qmag.denoising import NotchFilter, ButterworthFilter, MovingAverageFilter

def test_notch_filter_runs():
    """Check if NotchFilter runs without crashing."""
    fs = 100.0
    data = np.random.random(100)
    
    nf = NotchFilter(freq=50, fs=fs)
    clean_data = nf.apply_filter(data)
    
    # Assertions
    assert clean_data.shape == data.shape
    assert isinstance(clean_data, np.ndarray)

def test_butterworth_energy_reduction():
    """Physics check: Lowpass filter should remove high-freq energy."""
    fs = 1000
    t = np.linspace(0, 1, fs)
    # Generate high frequency noise (200 Hz)
    noise = np.sin(2 * np.pi * 200 * t)
    
    # Filter at 50 Hz
    bf = ButterworthFilter(cutoff=50, fs=fs, btype='lowpass')
    filtered = bf.apply_filter(noise)
    
    # The output energy should be much lower than input
    assert np.std(filtered) < 0.1 * np.std(noise)

def test_moving_average_smoothing():
    """
    Test that the Moving Average filter preserves shape and reduces noise.
    """
    fs = 100.0
    # Create a flat signal with random noise
    # Signal = 10 (constant) + Noise
    clean_signal = np.full(100, 10.0)
    noise = np.random.normal(0, 2.0, 100)
    noisy_signal = clean_signal + noise
    
    # Initialize Moving Average (Window size 5)
    ma = MovingAverageFilter(window_size=5)
    smoothed = ma.apply_filter(noisy_signal)
    
    # Assertion 1: Shape must be preserved (mode='nearest' ensures this)
    assert len(smoothed) == len(noisy_signal)
    
    # Assertion 2: The noise (standard deviation) should be lower after filtering
    # The spread of the data should be tighter around the mean
    original_std = np.std(noisy_signal)
    smoothed_std = np.std(smoothed)
    
    # Ideally, noise should reduce significantly
    assert smoothed_std < original_std
    
    # Assertion 3: Edge handling check
    # Ensure the first and last values are not zero (common bug with zero-padding)
    assert smoothed[0] != 0
    assert smoothed[-1] != 0
