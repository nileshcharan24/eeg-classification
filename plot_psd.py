import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Ensure we can import from src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data_loader import load_dat_file

def plot_psd():
    # Path to a sample data file
    file_path = os.path.join('data', 'raw', 'data_preprocessed_python', 's01.dat')
    
    if not os.path.exists(file_path):
        print(f"Error: Data file {file_path} not found.")
        return

    print(f"Loading data from {file_path}...")
    data, labels = load_dat_file(file_path)
    
    # Select Trial 0, Channel 0 (e.g., Fp1 or similar depending on DEAP montage)
    trial_idx = 0
    channel_idx = 0
    eeg_signal = data[trial_idx, channel_idx, :]
    
    fs = 128.0  # DEAP dataset downsampled frequency
    nperseg = min(int(2 * fs), len(eeg_signal))
    
    # Compute Power Spectral Density using Welch's method
    freqs, psd = welch(eeg_signal, fs=fs, nperseg=nperseg)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, psd, color='blue', lw=2, label='PSD')
    
    # Define frequency bands
    bands = {
        'Theta (4-8 Hz)': (4, 8, 'orange'),
        'Alpha (8-13 Hz)': (8, 13, 'green'),
        'Beta (13-30 Hz)': (13, 30, 'red'),
        'Gamma (30-45 Hz)': (30, 45, 'purple')
    }
    
    # Shade the frequency bands
    for band_name, (low, high, color) in bands.items():
        # Find indices within the band
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        plt.fill_between(freqs, psd, where=idx_band, color=color, alpha=0.3, label=band_name)

    plt.title(f"EEG Power Spectral Density (Subject 1, Trial {trial_idx+1}, Channel {channel_idx+1})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (V^2/Hz)")
    plt.xlim(0, 50)  # Focus on standard EEG frequency range
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Save the plot
    output_filename = "psd_spectrum_report.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_filename}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    plot_psd()
