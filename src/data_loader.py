import os
import pickle
import numpy as np

def load_dat_file(file_path):
    """Loads a single DEAP .dat file using pickle."""
    with open(file_path, 'rb') as f:
        content = pickle.load(f, encoding='latin1')
    return content['data'], content['labels']

def verify_data_shape(file_path):
    """Dynamically inspects the first .dat file to print and verify shapes."""
    try:
        data, labels = load_dat_file(file_path)
        print(f"--- Verification for {os.path.basename(file_path)} ---")
        print(f"Data shape: {data.shape} (Expected: (40, 40, 8064) -> (Trials, Channels, Samples))")
        print(f"Labels shape: {labels.shape} (Expected: (40, 4) -> (Trials, Labels))")
        return data, labels
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def engineer_labels(labels):
    """
    Maps continuous Valence and Arousal scores into 3 discrete classes:
    - 0: 'Stress' (Low Valence < 5, High Arousal > 5)
    - 1: 'Relaxation' (High Valence >= 5, Low Arousal <= 5)
    - 2: 'Attention/Normal' (Moderate levels or everything else)
    """
    mapped_labels = []
    for label in labels:
        valence = label[0]
        arousal = label[1]
        
        if valence < 5 and arousal > 5:
            mapped_labels.append(0) # Stress
        elif valence >= 5 and arousal <= 5:
            mapped_labels.append(1) # Relaxation
        else:
            mapped_labels.append(2) # Attention/Normal
            
    return np.array(mapped_labels)

def load_all_data(data_dir):
    """Loads all subject files and generates targets."""
    all_data = []
    all_labels = []
    subject_ids = []
    trial_ids = []
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
    files.sort()
    
    for subject_idx, file in enumerate(files):
        file_path = os.path.join(data_dir, file)
        print(f"Loading {file_path}...")
        data, labels = load_dat_file(file_path)
        
        # Only keep the 32 EEG channels (first 32 out of 40)
        eeg_data = data[:, :32, :]
        discrete_labels = engineer_labels(labels)
        
        for trial_idx in range(eeg_data.shape[0]):
            subject_ids.append(subject_idx + 1)
            trial_ids.append(trial_idx + 1)
            all_data.append(eeg_data[trial_idx])
            all_labels.append(discrete_labels[trial_idx])
            
    return np.array(all_data), np.array(all_labels), np.array(subject_ids), np.array(trial_ids)

if __name__ == "__main__":
    # Test verification function
    test_file = os.path.join('data', 'raw', 'data_preprocessed_python', 's01.dat')
    if os.path.exists(test_file):
        verify_data_shape(test_file)
    else:
        print(f"Please ensure {test_file} exists to run the verification.")
