import os
import streamlit as st
import pandas as pd
import torch
import numpy as np
import joblib

from src.model import EEGClassifier, CNNLSTMClassifier, EEGNetClassifier

# Define classes mapping
CLASS_MAP = {
    0: 'Stress',
    1: 'Relaxation',
    2: 'Attention/Normal'
}

# --- Cache Data and Model Loading ---
@st.cache_data
def load_data(model_type):
    """Load the preprocessed UI test data."""
    if model_type == 'EEGNet':
        filename = 'ui_test_data_eegnet.csv'
    elif model_type == 'CNN-LSTM':
        filename = 'ui_test_data_cnn_lstm.csv'
    elif model_type == 'CNN':
        filename = 'ui_test_data.csv'
    elif model_type == 'K-Nearest Neighbors':
        filename = 'ui_test_data_knn.csv'
    else:
        filename = 'ui_test_data_rf.csv'
    data_path = os.path.join('data', 'processed', filename)
    if not os.path.exists(data_path):
        return None
    return pd.read_csv(data_path)

@st.cache_resource
def load_model(model_type, input_size):
    """Load the trained model."""
    if model_type in ['CNN', 'CNN-LSTM', 'EEGNet']:
        if model_type == 'EEGNet':
            model_path = os.path.join('models', 'eegnet_classifier.pth')
            model_class = EEGNetClassifier
        elif model_type == 'CNN':
            model_path = os.path.join('models', 'eeg_classifier.pth')
            model_class = EEGClassifier
        else:
            model_path = os.path.join('models', 'cnn_lstm_classifier.pth')
            model_class = CNNLSTMClassifier

        if not os.path.exists(model_path):
            return None
        
        # Initialize the model
        model = model_class(input_size=input_size, num_classes=3)
        
        # Load weights
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval() # Set to evaluation mode
            return model
        except Exception as e:
            st.error(f"Error loading {model_type} model: {e}")
            return None
    else:
        if model_type == 'K-Nearest Neighbors':
            model_path = os.path.join('models', 'knn_classifier.joblib')
        else:
            model_path = os.path.join('models', 'rf_classifier.joblib')
            
        if not os.path.exists(model_path):
            return None
        
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading {model_type} model: {e}")
            return None

# --- Main App ---
def main():
    st.set_page_config(page_title="EEG Mental State Classifier", page_icon="🧠", layout="wide")
    
    st.title("🧠 EEG Mental State Classification")
    st.markdown("""
    This application predicts mental states (**Stress**, **Relaxation**, **Attention/Normal**)
    from EEG signal features extracted from the DEAP dataset.
    You can choose between a Deep 1D-CNN model, a Hybrid CNN-LSTM model, an EEGNet model (using PSD/DE features), a Random Forest model, and a K-Nearest Neighbors model (using Statistical features).
    """)

    # Model Selection
    st.sidebar.header("Model Selection")
    model_type = st.sidebar.radio("Choose Model", ('CNN', 'Random Forest', 'CNN-LSTM', 'EEGNet', 'K-Nearest Neighbors'))

    # Load resources
    df = load_data(model_type)
    
    if df is None:
        if model_type == 'EEGNet':
            filename = 'ui_test_data_eegnet.csv'
        elif model_type == 'CNN-LSTM':
            filename = 'ui_test_data_cnn_lstm.csv'
        elif model_type == 'CNN':
            filename = 'ui_test_data.csv'
        elif model_type == 'K-Nearest Neighbors':
            filename = 'ui_test_data_knn.csv'
        else:
            filename = 'ui_test_data_rf.csv'
        st.warning(f"⚠️ Data file not found. Please run the training pipeline with the appropriate model flag to generate `data/processed/{filename}`.")
        return
        
    # Infer input size dynamically from the features in the ui test data
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    input_size = len(feature_cols)

    model = load_model(model_type, input_size)
    
    if model is None:
        if model_type == 'EEGNet':
            model_filename = 'eegnet_classifier.pth'
        elif model_type == 'CNN-LSTM':
            model_filename = 'cnn_lstm_classifier.pth'
        elif model_type == 'CNN':
            model_filename = 'eeg_classifier.pth'
        elif model_type == 'K-Nearest Neighbors':
            model_filename = 'knn_classifier.joblib'
        else:
            model_filename = 'rf_classifier.joblib'
        st.warning(f"⚠️ Model weights not found. Please run the training pipeline first to save `models/{model_filename}`.")
        return

    st.sidebar.header("Select Instance")
    
    # Subject Selection
    subjects = sorted(df['subject_id'].unique())
    selected_subject = st.sidebar.selectbox("Select Subject ID", subjects)
    
    # Trial Selection based on Subject
    subject_df = df[df['subject_id'] == selected_subject]
    trials = sorted(subject_df['trial_id'].unique())
    selected_trial = st.sidebar.selectbox("Select Trial ID", trials)

    # Filter data to the selected instance
    instance = subject_df[subject_df['trial_id'] == selected_trial].iloc[0]
    
    # Extract features and true label
    true_label_idx = int(instance['label'])
    true_label_name = CLASS_MAP[true_label_idx]
    
    features = instance[feature_cols].values.astype(np.float32)

    # --- Prediction ---
    if model_type in ['CNN', 'CNN-LSTM', 'EEGNet']:
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            if model_type == 'EEGNet':
                # EEGNet expects (batch, 1, channels, time_samples/features)
                num_channels = 32
                features_per_channel = len(features) // num_channels
                input_tensor = torch.tensor(features).reshape(1, 1, num_channels, features_per_channel)
            elif model_type == 'CNN-LSTM':
                # CNN-LSTM expects (batch_size, sequence_length=1, features)
                input_tensor = torch.tensor(features).unsqueeze(0).unsqueeze(0)
            else:
                input_tensor = torch.tensor(features).unsqueeze(0)
                
            output = model(input_tensor)
            
            # Get probabilities and prediction
            probabilities = torch.nn.functional.softmax(output, dim=1).numpy()[0]
            predicted_idx = np.argmax(probabilities)
            predicted_name = CLASS_MAP[predicted_idx]
    else:
        # ML model prediction (Random Forest or KNN)
        # Flatten into 2D array: (1, total_features)
        input_features = features.reshape(1, -1)
        predicted_idx = model.predict(input_features)[0]
        predicted_name = CLASS_MAP[predicted_idx]
        probabilities = model.predict_proba(input_features)[0]

    # --- UI Layout ---
    st.write(f"### Prediction Results ({model_type})")
    
    col1, col2 = st.columns(2)
    
    # Ground Truth Card
    with col1:
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h4>Ground Truth (Actual)</h4>
            <h2 style="color: #4a4a4a;">{true_label_name}</h2>
        </div>
        """, unsafe_allow_html=True)
        
    # Prediction Card
    with col2:
        # Change color based on match
        color = "#28a745" if predicted_idx == true_label_idx else "#dc3545"
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h4>Model Prediction</h4>
            <h2 style="color: {color};">{predicted_name}</h2>
        </div>
        """, unsafe_allow_html=True)

    # Prediction Probabilities Bar Chart
    st.write("### Prediction Confidence")
    prob_df = pd.DataFrame({
        'Mental State': list(CLASS_MAP.values()),
        'Confidence': probabilities * 100
    })
    
    st.bar_chart(prob_df.set_index('Mental State'), height=300)
    
    # Show Raw Features (Optional Toggle)
    feature_desc = 'PSD & DE' if model_type in ['CNN', 'CNN-LSTM', 'EEGNet'] else 'Statistical'
    with st.expander(f"View Input Features (Normalized) - {feature_desc}"):
        st.line_chart(features)

if __name__ == '__main__':
    main()
