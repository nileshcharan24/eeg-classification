import numpy as np
from scipy.signal import butter, lfilter, welch

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=4.0, highcut=45.0, fs=128.0, order=5):
    """
    Applies a band-pass filter to the EEG signals.
    data shape: (channels, samples)
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=-1)
    return y

def extract_psd_features(eeg_data, fs=128.0):
    """
    Extracts Power Spectral Density (PSD) for standard frequency bands using FFT (Welch's method).
    eeg_data shape: (channels, samples)
    Bands: Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-45 Hz)
    Returns: Features array of shape (channels * 4,)
    """
    bands = {
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45)
    }
    
    num_channels = eeg_data.shape[0]
    features = np.zeros(num_channels * len(bands))
    
    nperseg = min(int(2 * fs), eeg_data.shape[-1])
    freqs, psd = welch(eeg_data, fs=fs, nperseg=nperseg, axis=-1)
    
    feature_idx = 0
    for ch in range(num_channels):
        channel_psd = psd[ch, :]
        for band_name, (low, high) in bands.items():
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            band_power = np.mean(channel_psd[idx_band])
            features[feature_idx] = band_power
            feature_idx += 1
            
    return features

def extract_dataset_features(data_array, y=None, subject_ids=None, trial_ids=None, window_size=4, overlap=0.5, fs=128.0):
    """
    Applies the feature extraction pipeline to the entire dataset using a sliding window.
    data_array shape: (Trials, 32_Channels, Samples)
    """
    num_trials = data_array.shape[0]
    samples_per_trial = data_array.shape[2]
    
    window_samples = int(window_size * fs)
    stride_samples = int(window_samples * (1 - overlap))
    
    all_features = []
    all_y = []
    all_subject_ids = []
    all_trial_ids = []
    
    for i in range(num_trials):
        if (i+1) % 100 == 0:
            print(f"Processing trial {i+1}/{num_trials}...")
            
        trial_data = data_array[i]
        
        # Apply filter to the whole trial first to avoid edge effects in each window
        filtered_data = apply_bandpass_filter(trial_data, lowcut=4.0, highcut=45.0, fs=fs)
        
        # Sliding window
        for start in range(0, samples_per_trial - window_samples + 1, stride_samples):
            window_data = filtered_data[:, start:start + window_samples]
            features = extract_psd_features(window_data, fs=fs)
            
            all_features.append(features)
            
            if y is not None:
                all_y.append(y[i])
            if subject_ids is not None:
                all_subject_ids.append(subject_ids[i])
            if trial_ids is not None:
                all_trial_ids.append(trial_ids[i])
                
    if y is not None:
        return np.array(all_features), np.array(all_y), np.array(all_subject_ids), np.array(all_trial_ids)
    return np.array(all_features)
