import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_all_data, verify_data_shape
from src.features import extract_dataset_features
from src.model import EEGClassifier, get_device

# Define classes mapping
CLASS_MAP = {
    0: 'Stress',
    1: 'Relaxation',
    2: 'Attention/Normal'
}

def main():
    print("--- Setting up Environment ---")
    device = get_device()
    print(f"Using device natively bound to: {device}\n")

    data_dir = os.path.join('data', 'raw', 'data_preprocessed_python')
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found.")
        print("Please ensure you have downloaded and placed the DEAP .dat files there.")
        return

    # 1. Load Data
    print("--- Step 1: Loading Raw Data ---")
    X_raw, y, subject_ids, trial_ids = load_all_data(data_dir)
    print(f"Loaded total {X_raw.shape[0]} trials.\n")

    # 2. Feature Extraction
    print("--- Step 2: Feature Extraction (PSD & DE) ---")
    print("This might take a moment. Extracting features with sliding window...")
    X_features, y, subject_ids, trial_ids = extract_dataset_features(X_raw, y, subject_ids, trial_ids)
    print(f"Feature extraction completed. Features shape: {X_features.shape}\n")

    # 3. Subject-Wise Normalization
    print("--- Step 3: Subject-Wise Normalization ---")
    print("Applying Standard Scaling independently to each subject...")
    unique_subjects = np.unique(subject_ids)
    for subj in unique_subjects:
        subj_mask = (subject_ids == subj)
        scaler = StandardScaler()
        X_features[subj_mask] = scaler.fit_transform(X_features[subj_mask])
    print("Normalization complete.\n")

    # 4. Train-Test Split (Re-creating exact split)
    print("--- Step 4: Re-creating Train/Test Split ---")
    indices = np.arange(len(X_features))
    _, X_test_scaled, _, y_test, _, idx_test = train_test_split(
        X_features, y, indices, test_size=0.2, random_state=42, stratify=y
    )
    
    test_subject_ids = subject_ids[idx_test]
    test_trial_ids = trial_ids[idx_test]
    print(f"Re-created test set with {X_test_scaled.shape[0]} samples.\n")

    # 5. Load Model
    print("--- Step 5: Loading Trained Model ---")
    input_size = X_test_scaled.shape[1]
    model_path = os.path.join('models', 'eeg_classifier.pth')
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Please run train.py first.")
        return
        
    model = EEGClassifier(input_size=input_size, num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.\n")

    # 6. Run Inference
    print("--- Step 6: Running Inference on Test Set ---")
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), 
                                 torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 7. Export to CSV
    print("\n--- Step 7: Exporting Results ---")
    df_results = pd.DataFrame({
        'True_Class': [CLASS_MAP[label] for label in all_labels],
        'Predicted_Class': [CLASS_MAP[pred] for pred in all_preds],
        'Subject_ID': test_subject_ids,
        'Trial_ID': test_trial_ids
    })
    
    output_dir = os.path.join('data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'full_test_predictions.csv')
    
    df_results.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print("🎉 SUCCESS: Inference Complete!")
    print(f"💾 File Saved to: {output_path}")
    print(f"📊 Total Rows Saved: {len(df_results)}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
