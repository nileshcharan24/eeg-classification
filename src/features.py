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

def extract_rf_features(eeg_data, fs=128.0):
    """
    Extracts statistical features for Random Forest based on the paper:
    Mean, Standard Deviation, Maximum - Minimum Amplitude.
    Calculates these for Theta, Alpha, Beta, Gamma bands.
    eeg_data shape: (channels, samples)
    Returns: Features array of shape (channels * 4 * 3,)
    """
    bands = {
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45)
    }
    
    num_channels = eeg_data.shape[0]
    features = np.zeros(num_channels * len(bands) * 3) # 3 statistical features
    
    feature_idx = 0
    for ch in range(num_channels):
        channel_data = eeg_data[ch, :]
        for band_name, (low, high) in bands.items():
            # Bandpass filter the channel data for the specific band
            b, a = butter_bandpass(low, high, fs, order=5)
            band_data = lfilter(b, a, channel_data)
            
            # Statistical features
            features[feature_idx] = np.mean(band_data)
            feature_idx += 1
            
            features[feature_idx] = np.std(band_data)
            feature_idx += 1
            
            features[feature_idx] = np.max(band_data) - np.min(band_data)
            feature_idx += 1
            
    return features

def extract_psd_features(eeg_data, fs=128.0):
    """
    Extracts Power Spectral Density (PSD) and Differential Entropy (DE) for standard frequency bands.
    eeg_data shape: (channels, samples)
    Bands: Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-45 Hz)
    Returns: Features array of shape (channels * 4 * 2,)
    """
    bands = {
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45)
    }
    
    num_channels = eeg_data.shape[0]
    features = np.zeros(num_channels * len(bands) * 2) # *2 for PSD and DE
    
    nperseg = min(int(2 * fs), eeg_data.shape[-1])
    freqs, psd = welch(eeg_data, fs=fs, nperseg=nperseg, axis=-1)
    
    feature_idx = 0
    for ch in range(num_channels):
        channel_psd = psd[ch, :]
        for band_name, (low, high) in bands.items():
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            band_power = np.mean(channel_psd[idx_band])
            
            # 1. Power Spectral Density (PSD)
            features[feature_idx] = band_power
            feature_idx += 1
            
            # 2. Differential Entropy (DE)
            # DE for a Gaussian distribution can be estimated as 0.5 * log(2 * pi * e * variance)
            # We approximate variance within the band using the band power. Add epsilon to avoid log(0)
            de = 0.5 * np.log(2 * np.pi * np.e * (band_power + 1e-9))
            features[feature_idx] = de
            feature_idx += 1
            
    return features

def extract_dataset_features(data_array, y=None, subject_ids=None, trial_ids=None, window_size=4, overlap=0.5, fs=128.0, feature_type='psd'):
    """
    Applies the feature extraction pipeline to the entire dataset using a sliding window.
    data_array shape: (Trials, 32_Channels, Samples)
    feature_type: 'psd' for CNN features (PSD & DE), 'rf' for Random Forest features (Statistical)
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
        # We apply broadband filter here
        filtered_data = apply_bandpass_filter(trial_data, lowcut=4.0, highcut=45.0, fs=fs)
        
        # Sliding window
        for start in range(0, samples_per_trial - window_samples + 1, stride_samples):
            window_data = filtered_data[:, start:start + window_samples]
            
            if feature_type == 'psd':
                features = extract_psd_features(window_data, fs=fs)
            elif feature_type == 'rf':
                features = extract_rf_features(window_data, fs=fs)
            else:
                raise ValueError("Invalid feature_type. Choose 'psd' or 'rf'.")
            
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
