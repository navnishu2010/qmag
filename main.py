import numpy as np
import matplotlib.pyplot as plt
from qmag.denoising import NotchFilter, ButterworthFilter, MovingAverageFilter

def main():
    # --- 1. Load or Simulate Data ---
    print("Loading data...")
    fs = 1000.0
    t = np.linspace(0, 2, int(fs * 2))
    
    # Create a clean signal + 50Hz hum + random noise
    clean_signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz signal
    hum = 0.5 * np.sin(2 * np.pi * 50 * t)    # 50 Hz power line hum
    noise = 0.2 * np.random.normal(size=len(t))
    
    raw_data = clean_signal + hum + noise

    # --- 2. Initialize the Pipeline ---
    print("Initializing filters...")
    # A. Remove 50Hz hum
    notch = NotchFilter(freq=50, fs=fs)
    
    # B. Remove high frequency noise (above 30Hz)
    lowpass = ButterworthFilter(cutoff=30, fs=fs, order=4)
    
    # C. Smooth the result visually
    smoother = MovingAverageFilter(window_size=10)

    # --- 3. Run the Pipeline ---
    print("Processing signal...")
    step1 = notch.apply_filter(raw_data)
    step2 = lowpass.apply_filter(step1)
    final_output = smoother.apply_filter(step2)

    # --- 4. Visualize Results ---
    print("Plotting...")
    plt.figure(figsize=(12, 6))
    
    plt.plot(t[:500], raw_data[:500], color='lightgray', label='Raw Data')
    plt.plot(t[:500], final_output[:500], color='red', linewidth=2, label='Cleaned Output')
    plt.plot(t[:500], clean_signal[:500], 'k--', alpha=0.5, label='True Signal')
    
    plt.title("QMag Denoising Pipeline Results")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()