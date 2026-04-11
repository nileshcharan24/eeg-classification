import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_all_data, verify_data_shape
from src.features import extract_dataset_features
from src.model import EEGClassifier, CNNLSTMClassifier, EEGNetClassifier, get_device

# Simple Early Stopping mechanism based on validation loss
class EarlyStopping:
    def __init__(self, patience=10, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def main(model_type):
    data_dir = os.path.join('data', 'raw', 'data_preprocessed_python')
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found.")
        print("Please ensure you have downloaded and placed the DEAP .dat files there.")
        return

    # 1. Verify and Load Data
    print("--- Step 1: Loading Data ---")
    first_file = os.path.join(data_dir, 's01.dat')
    if os.path.exists(first_file):
        verify_data_shape(first_file)
    
    X_raw, y, subject_ids, trial_ids = load_all_data(data_dir)
    print(f"Loaded total {X_raw.shape[0]} trials.")

    # 2. Feature Extraction
    feature_desc = 'PSD & DE' if model_type in ['cnn', 'cnn_lstm', 'eegnet'] else 'Statistical Features'
    print(f"\n--- Step 2: Feature Extraction ({feature_desc}) ---")
    feature_type_arg = 'psd' if model_type in ['cnn', 'cnn_lstm', 'eegnet'] else 'rf'
    
    # The caching logic is handled internally within extract_dataset_features
    X_features, y, subject_ids, trial_ids = extract_dataset_features(
        X_raw, y, subject_ids, trial_ids, feature_type=feature_type_arg
    )
    print(f"\nFeature pipeline completed. Final Features shape: {X_features.shape}")

    # 2.5. Subject-Wise Normalization
    print("\n--- Step 2.5: Subject-Wise Normalization ---")
    print("Applying Standard Scaling independently to each subject...")
    unique_subjects = np.unique(subject_ids)
    for subj in unique_subjects:
        subj_mask = (subject_ids == subj)
        scaler = StandardScaler()
        # Scale the features for this specific subject
        X_features[subj_mask] = scaler.fit_transform(X_features[subj_mask])

    # 3. Train-Test Split
    print("\n--- Step 3: Train/Test Split ---")
    indices = np.arange(len(X_features))
    X_train_scaled, X_test_scaled, y_train, y_test, idx_train, idx_test = train_test_split(
        X_features, y, indices, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train_scaled.shape[0]}, Test size: {X_test_scaled.shape[0]}")

    # 4. Prepare UI Data
    print("\n--- Step 4: Preparing UI Test Data ---")
    ui_test_dir = os.path.join('data', 'processed')
    os.makedirs(ui_test_dir, exist_ok=True)
    if model_type == 'eegnet':
        ui_data_filename = 'ui_test_data_eegnet.csv'
    elif model_type == 'cnn_lstm':
        ui_data_filename = 'ui_test_data_cnn_lstm.csv'
    elif model_type == 'cnn':
        ui_data_filename = 'ui_test_data.csv'
    elif model_type == 'knn':
        ui_data_filename = 'ui_test_data_knn.csv'
    else:
        ui_data_filename = 'ui_test_data_rf.csv'
    ui_data_path = os.path.join(ui_test_dir, ui_data_filename)
    
    df_ui = pd.DataFrame(X_test_scaled, columns=[f"feat_{i}" for i in range(X_test_scaled.shape[1])])
    df_ui['label'] = y_test
    df_ui['subject_id'] = subject_ids[idx_test]
    df_ui['trial_id'] = trial_ids[idx_test]
    
    df_ui.to_csv(ui_data_path, index=False)
    print(f"Saved UI test data to: {ui_data_path}")

    # 5. Model Setup & Class Weights
    print(f"\n--- Step 5: Model Training Setup ({model_type.upper()}) ---")
    
    # Calculate Class Weights to handle severe class imbalance
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    print(f"Computed Class Weights: {class_weights}")
    
    target_names = ['Stress', 'Relaxation', 'Attention/Normal']
    os.makedirs('models', exist_ok=True)
    
    if model_type in ['cnn', 'cnn_lstm', 'eegnet']:
        device = get_device()
        print(f"Using device natively bound to: {device}")
        
        # Reshape data for CNN-LSTM or EEGNet
        if model_type == 'cnn_lstm':
            # Ensure inputs are shaped as (batch_size, sequence_length, features)
            # where sequence_length=1 as requested
            X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
            X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
        elif model_type == 'eegnet':
            # Ensure inputs are shaped for EEGNet: (batch, 1, channels, time_samples/features)
            # Assume 32 channels based on standard DEAP dataset output
            num_channels = 32
            features_per_channel = X_train_scaled.shape[-1] // num_channels
            X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, num_channels, features_per_channel)
            X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, num_channels, features_per_channel)
        
        # Send model and weights to the designated GPU/CPU device
        input_size = X_train_scaled.shape[-1]
        if model_type == 'eegnet':
            # For EEGNet input_size passed here will be features_per_channel since we already reshaped it,
            # or we could pass the flat size. The class handles both.
            model = EEGNetClassifier(input_size=input_size, num_classes=3, num_channels=32).to(device)
        elif model_type == 'cnn_lstm':
            model = CNNLSTMClassifier(input_size=input_size, num_classes=3).to(device)
        else:
            model = EEGClassifier(input_size=input_size, num_classes=3).to(device)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        # Pass calculated weights into CrossEntropyLoss to penalize ignoring minority classes
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        # Create DataLoaders
        train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), 
                                      torch.tensor(y_train, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), 
                                     torch.tensor(y_test, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        # 6. Training Loop with Early Stopping & LR Scheduler
        if model_type == 'eegnet':
            model_name = "EEGNet"
        elif model_type == 'cnn_lstm':
            model_name = "CNN-LSTM"
        else:
            model_name = "Deep 1D-CNN"
        print(f"\n--- Step 6: Training Model ({model_name}) ---")
        epochs = 150
        early_stopper = EarlyStopping(patience=15, delta=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device) # Force to GPU
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            train_loss = running_loss / len(train_loader)
            
            # Validation evaluation for Early Stopping
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device) # Force to GPU
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            val_loss /= len(test_loader)
            
            # Step the scheduler
            scheduler.step(val_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - LR: {current_lr:.6f}")
            
            early_stopper(val_loss)
            if early_stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}. Model validation loss stopped improving.")
                break

        # 7. Evaluation
        print("\n--- Step 7: Evaluation ---")
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        print(f"\nClassification Report ({model_name}):")
        print(classification_report(all_labels, all_preds, labels=[0, 1, 2], target_names=target_names, zero_division=0))
        
        final_accuracy = accuracy_score(all_labels, all_preds)
        print(f"\n--- FINAL TEST ACCURACY ({model_name}): {final_accuracy * 100:.2f}% ---")

        # 8. Save Model
        print("\n--- Step 8: Saving Model ---")
        if model_type == 'eegnet':
            model_save_path = os.path.join('models', 'eegnet_classifier.pth')
        elif model_type == 'cnn_lstm':
            model_save_path = os.path.join('models', 'cnn_lstm_classifier.pth')
        else:
            model_save_path = os.path.join('models', 'eeg_classifier.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")
        
    elif model_type == 'rf':
        print("\n--- Step 6: Training Model (Random Forest) ---")
        # Initialize Random Forest
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        rf_model = RandomForestClassifier(
            n_estimators=100, 
            class_weight=class_weight_dict, 
            random_state=42,
            n_jobs=-1 # Use all available cores
        )
        
        # Train
        rf_model.fit(X_train_scaled, y_train)
        print("Training completed.")
        
        # 7. Evaluation
        print("\n--- Step 7: Evaluation ---")
        all_preds = rf_model.predict(X_test_scaled)
        all_labels = y_test
        
        print("\nClassification Report (Random Forest):")
        print(classification_report(all_labels, all_preds, labels=[0, 1, 2], target_names=target_names, zero_division=0))
        
        final_accuracy = accuracy_score(all_labels, all_preds)
        print(f"\n--- FINAL TEST ACCURACY (RF): {final_accuracy * 100:.2f}% ---")
        
        # 8. Save Model
        print("\n--- Step 8: Saving Model ---")
        model_save_path = os.path.join('models', 'rf_classifier.joblib')
        joblib.dump(rf_model, model_save_path)
        print(f"Model saved to: {model_save_path}")
        
    elif model_type == 'knn':
        print("\n--- Step 6: Training Model (K-Nearest Neighbors) ---")
        # Initialize KNN (ensure 2D shape n_samples x features)
        # X_train_scaled is already 2D from feature extraction
        knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
        
        # Train
        knn_model.fit(X_train_scaled, y_train)
        print("Training completed.")
        
        # 7. Evaluation
        print("\n--- Step 7: Evaluation ---")
        all_preds = knn_model.predict(X_test_scaled)
        all_labels = y_test
        
        print("\nClassification Report (KNN):")
        print(classification_report(all_labels, all_preds, labels=[0, 1, 2], target_names=target_names, zero_division=0))
        
        final_accuracy = accuracy_score(all_labels, all_preds)
        print(f"\n--- FINAL TEST ACCURACY (KNN): {final_accuracy * 100:.2f}% ---")
        
        # 8. Save Model
        print("\n--- Step 8: Saving Model ---")
        model_save_path = os.path.join('models', 'knn_classifier.joblib')
        joblib.dump(knn_model, model_save_path)
        print(f"Model saved to: {model_save_path}")

    # 9. Terminal Predictions Demo
    print("\n--- Step 9: Terminal Predictions Demo ---")
    print("Testing 5 random samples from the unseen Test Set...\n")
    
    demo_indices = np.random.choice(len(X_test_scaled), 5, replace=False)
    class_map = {0: 'Stress', 1: 'Relaxation', 2: 'Attention/Normal'}
    
    if model_type in ['cnn', 'cnn_lstm', 'eegnet']:
        model.eval()
        with torch.no_grad():
            for i, idx in enumerate(demo_indices):
                sample = X_test_scaled[idx]
                # For all PyTorch models, sample is currently without batch dim
                # e.g., eegnet: (1, channels, features), cnn_lstm: (1, features), cnn: (features,)
                sample_features = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
                true_label_idx = y_test[idx]
                
                output = model(sample_features)
                predicted_idx = torch.argmax(output, dim=1).item()
                
                actual_class = class_map[true_label_idx]
                pred_class = class_map[predicted_idx]
                
                print(f"Sample {i+1} | Actual: [{actual_class:^16}] | Predicted: [{pred_class:^16}]")
    else:
        for i, idx in enumerate(demo_indices):
            sample_features = X_test_scaled[idx].reshape(1, -1)
            true_label_idx = y_test[idx]
            
            if model_type == 'rf':
                predicted_idx = rf_model.predict(sample_features)[0]
            elif model_type == 'knn':
                predicted_idx = knn_model.predict(sample_features)[0]
            
            actual_class = class_map[true_label_idx]
            pred_class = class_map[predicted_idx]
            
            print(f"Sample {i+1} | Actual: [{actual_class:^16}] | Predicted: [{pred_class:^16}]")

    print(f"\nPipeline completed successfully for {model_type.upper()} model!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EEG Classification Model")
    parser.add_argument(
        '--model',
        type=str,
        choices=['cnn', 'rf', 'cnn_lstm', 'eegnet', 'knn'],
        default='cnn',
        help="Select the model to train: 'cnn' (Deep 1D-CNN), 'rf' (Random Forest), 'cnn_lstm' (Hybrid CNN-LSTM), 'eegnet' (EEGNet), or 'knn' (K-Nearest Neighbors)"
    )
    args = parser.parse_args()
    
    main(model_type=args.model)
