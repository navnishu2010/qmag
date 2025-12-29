import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- IMPORTS ---
from qmag.io import DataLoader
from qmag.denoising import NotchFilter, ButterworthFilter, MovingAverageFilter
from qmag.allan import compute_allan_deviation

def compute_spectrum(data, fs):
    """Computes Power Spectral Density (PSD)"""
    freqs, psd = signal.welch(data, fs, nperseg=1024)
    return freqs, psd

def fit_allan_slope_manual(taus, adev, idx_start, idx_end):
    """
    Fits a line to a specific slice of the Allan Deviation curve.
    """
    # 1. Slice the data manually
    if idx_start < 0: idx_start = 0
    if idx_end > len(taus): idx_end = len(taus)
    
    t_fit = taus[idx_start:idx_end]
    a_fit = adev[idx_start:idx_end]
    
    if len(t_fit) < 2:
        return 0, np.array([]), np.array([])

    # 2. Linear fit in Log-Log space
    log_t = np.log10(t_fit)
    log_a = np.log10(a_fit)
    
    slope, intercept = np.polyfit(log_t, log_a, 1)
    
    # 3. Generate line for plotting
    fitted_line = (10**intercept) * (t_fit**slope)
    
    return slope, fitted_line, t_fit

def main():
    # --- 1. Load Real Data ---
    file_path = "data/meas_plotter_20251216_160513.txt"
    try:
        loader = DataLoader(file_path)
        df = loader.load(names=['time', 'frequency'])
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)

    t = df['time'].values
    raw_signal = df['frequency'].values
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt
    print(f"Detected Sampling Rate: {fs:.2f} Hz")

    # --- 2. Pipeline ---
    mean_val = np.mean(raw_signal)
    centered_signal = raw_signal - mean_val
    
    # A. Notch (50Hz)
    notch = NotchFilter(freq=50, fs=fs, q=30)
    step1 = notch.apply_filter(centered_signal)
    
    # B. Lowpass (100Hz)
    lowpass = ButterworthFilter(cutoff=100, fs=fs, order=4)
    step2 = lowpass.apply_filter(step1)
    
    # C. Smooth
    smoother = MovingAverageFilter(window_size=1000)
    step3 = smoother.apply_filter(step2)
    final_output_shifted = step3 + mean_val

    # --- 3. Compute Metrics ---
    print("Computing metrics...")
    
    # Spectra (for Plot 1)
    f_raw, psd_raw = compute_spectrum(centered_signal, fs)
    f_step1, psd_step1 = compute_spectrum(step1, fs) # After Notch
    f_step2, psd_step2 = compute_spectrum(step2, fs) # After Lowpass
    f_step3, psd_step3 = compute_spectrum(step3, fs) # After Smoothing

    # Allan Deviation (for Plot 3)
    taus_raw, adev_raw = compute_allan_deviation(raw_signal, fs)
    taus_clean, adev_clean = compute_allan_deviation(final_output_shifted, fs)

    # --- 4. MANUAL FITTING CONFIG ---
    # Change these to fit different parts of the curve
    FIT_A = 2   
    FIT_B = 6  
    
    beta_raw1, line_raw1, t_raw_fit1 = fit_allan_slope_manual(taus_raw, adev_raw, FIT_A, FIT_B)
    beta_raw2, line_raw2, t_raw_fit2 = fit_allan_slope_manual(taus_raw, adev_raw, 10, 15)
    beta_raw3, line_raw3, t_raw_fit3 = fit_allan_slope_manual(taus_raw, adev_raw, 16, 19)
    #beta_clean, line_clean, t_clean_fit = fit_allan_slope_manual(taus_clean, adev_clean, FIT_A, FIT_B)

    # --- 5. Visualize (3 Subplots) ---
    print("Plotting dashboard...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # PLOT 1: Frequency Domain
    ax1.semilogy(f_raw, psd_raw, color='lightgray', label='Raw Input')
    ax1.semilogy(f_step1, psd_step1, color='purple', label='After Notch (50Hz)', alpha=0.6)
    ax1.semilogy(f_step2, psd_step2, color='orange', label='After Lowpass', alpha=0.6)
    ax1.semilogy(f_step3, psd_step3, color='blue', label='Final Output', linewidth=1.5)
    ax1.set_title("1. Frequency Domain Analysis")
    ax1.set_ylabel("PSD")
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid(True, alpha=0.3)

    # PLOT 2: Time Domain
    ax2.plot(t, raw_signal, color='lightgray', label='Raw Signal')
    ax2.plot(t, final_output_shifted, color='red', label='Final Filtered')
    ax2.set_title("2. Time Domain Trace")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # PLOT 3: Allan Deviation (with Fit)
    ax3.loglog(taus_raw, adev_raw, 'o-', color='gray', label='Raw Stability', alpha=0.4)
    ax3.loglog(taus_clean, adev_clean, 's-', color='green', label='Filtered Stability', alpha=0.4)
    
    # Add Fit Lines
    if len(t_raw_fit1) > 0:
        ax3.loglog(t_raw_fit1, line_raw1, '--', color='black', linewidth=2, 
                   label=f'Raw Fit: $\\beta$={beta_raw1:.2f}')
    if len(t_raw_fit2) > 0:
        ax3.loglog(t_raw_fit2, line_raw2, '--', color='darkgreen', linewidth=2, 
                   label=f'Raw Fit: $\\beta$={beta_raw2:.2f}')
    if len(t_raw_fit3) > 0:
        ax3.loglog(t_raw_fit3, line_raw3, '--', color='brown', linewidth=2, 
                   label=f'Raw Fit: $\\beta$={beta_raw3:.2f}')
    #if len(t_clean_fit) > 0:
    #    ax3.loglog(t_clean_fit, line_clean, '--', color='darkgreen', linewidth=2, 
    #               label=f'Clean Fit: $\\beta$={beta_clean:.2f}')

    ax3.set_title(f"3. Stability Analysis (Slope Fit indices {FIT_A}-{FIT_B})")
    ax3.set_ylabel("Allan Deviation")
    ax3.set_xlabel("Tau (s)")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()