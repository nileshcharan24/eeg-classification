import os
import streamlit as st
import pandas as pd
import torch
import numpy as np

from src.model import EEGClassifier

# Define classes mapping
CLASS_MAP = {
    0: 'Stress',
    1: 'Relaxation',
    2: 'Attention/Normal'
}

# --- Cache Data and Model Loading ---
@st.cache_data
def load_data():
    """Load the preprocessed UI test data."""
    data_path = os.path.join('data', 'processed', 'ui_test_data.csv')
    if not os.path.exists(data_path):
        return None
    return pd.read_csv(data_path)

@st.cache_resource
def load_model(input_size):
    """Load the trained PyTorch model."""
    model_path = os.path.join('models', 'eeg_classifier.pth')
    if not os.path.exists(model_path):
        return None
    
    # Initialize the model
    model = EEGClassifier(input_size=input_size, num_classes=3)
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval() # Set to evaluation mode
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Main App ---
def main():
    st.set_page_config(page_title="EEG Mental State Classifier", page_icon="🧠", layout="wide")
    
    st.title("🧠 EEG Mental State Classification")
    st.markdown("""
    This application predicts mental states (**Stress**, **Relaxation**, **Attention/Normal**) 
    from EEG signal features (Power Spectral Density and Differential Entropy) extracted from the DEAP dataset.
    """)

    # Load resources
    df = load_data()
    
    if df is None:
        st.warning("⚠️ Data file not found. Please run the training pipeline first to generate `data/processed/ui_test_data.csv`.")
        return
        
    # Infer input size dynamically from the features in the ui test data
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    input_size = len(feature_cols)

    model = load_model(input_size)
    
    if model is None:
        st.warning("⚠️ Model weights not found. Please run the training pipeline first to save `models/eeg_classifier.pth`.")
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
    with torch.no_grad():
        # Convert to tensor and add batch dimension
        input_tensor = torch.tensor(features).unsqueeze(0)
        output = model(input_tensor)
        
        # Get probabilities and prediction
        probabilities = torch.nn.functional.softmax(output, dim=1).numpy()[0]
        predicted_idx = np.argmax(probabilities)
        predicted_name = CLASS_MAP[predicted_idx]

    # --- UI Layout ---
    st.write("### Prediction Results")
    
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
    with st.expander("View Input PSD & DE Features (Normalized)"):
        st.line_chart(features)

if __name__ == '__main__':
    main()
