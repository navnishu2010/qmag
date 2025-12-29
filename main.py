import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from qmag.io import DataLoader
from qmag.denoising import ButterworthFilter, MovingAverageFilter

def compute_spectrum(data, fs):
    """
    Computes the Power Spectral Density (PSD) using Welch's method.
    """
    freqs, psd = signal.welch(data, fs, nperseg=1024)
    return freqs, psd

def main():
    # --- 1. Load Real Data ---
    file_path = "data/meas_plotter_20251216_160513.txt"
    print(f"Loading data from: {file_path}")
    
    try:
        loader = DataLoader(file_path)
        df = loader.load(names=['time', 'frequency'])
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)

    t = df['time'].values
    raw_signal = df['frequency'].values
    
    # Calculate Sampling Rate (fs)
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt
    print(f"Detected Sampling Rate: {fs:.2f} Hz")

    # --- 2. Initialize Pipeline ---
    print("Processing signal...")
    
    # Detrend (center at zero) for better filtering/spectral analysis
    mean_val = np.mean(raw_signal)
    centered_signal = raw_signal - mean_val

    # A. Lowpass Filter (Step 1)
    lowpass = ButterworthFilter(cutoff=100, fs=fs, order=4, btype='lowpass')
    sig_step1 = lowpass.apply_filter(centered_signal)
    
    # B. Moving Average (Step 2)
    smoother = MovingAverageFilter(window_size=50)
    sig_step2 = smoother.apply_filter(sig_step1)

    # Shift back for time-domain plot
    final_output_shifted = sig_step2 + mean_val

    # --- 3. Compute Spectra for EACH Step ---
    print("Computing power spectra...")
    f_raw, psd_raw = compute_spectrum(centered_signal, fs)
    f_step1, psd_step1 = compute_spectrum(sig_step1, fs)
    f_step2, psd_step2 = compute_spectrum(sig_step2, fs)

    # --- 4. Visualize Results ---
    print("Plotting...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # --- Plot 1: Progressive Spectral Analysis ---
    # We use semilog-y to see noise reduction clearly in dB
    ax1.semilogy(f_raw, psd_raw, color='lightgray', label='1. Raw Input', linewidth=1.5)
    ax1.semilogy(f_step1, psd_step1, color='orange', label='2. After Lowpass (100Hz)', linewidth=1.5, alpha=0.8)
    ax1.semilogy(f_step2, psd_step2, color='blue', label='3. After Smoothing (Window=50)', linewidth=2)
    
    ax1.set_title("Spectral Cleaning Steps (PSD)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Power Density (V²/Hz)")
    ax1.legend(loc="upper right")
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    # Optional: limit x-axis to zoom in on relevant frequencies if fs is huge
    # ax1.set_xlim(0, 500) 

    # --- Plot 2: Time Domain Result ---
    ax2.plot(t, raw_signal, color='lightgray', label='Raw Data')
    ax2.plot(t, final_output_shifted, color='red', linewidth=2, label='Final Filtered Output')
    
    ax2.set_title(f"Time Domain: {file_path}")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()