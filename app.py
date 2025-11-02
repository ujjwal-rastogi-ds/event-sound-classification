"""
Audio Event Classification Web App
Upload audio and get predictions with confidence scores
"""

import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import tempfile
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Audio Event Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stAlert {
        background-color: #e3f2fd;
    }
    h1 {
        color: #1976d2;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class AudioFeatureExtractor:
    """Extract features from audio files"""
    
    def __init__(self, sr=22050, n_mfcc=40, max_len=174):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        
    def extract_mfcc_sequence(self, audio_path):
        """Extract MFCC features as sequence (T, 40)"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sr, duration=4)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            mfccs = mfccs.T
            
            if mfccs.shape[0] < self.max_len:
                pad_width = self.max_len - mfccs.shape[0]
                mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
            else:
                mfccs = mfccs[:self.max_len, :]
                
            return mfccs
        except Exception as e:
            st.error(f"Error extracting MFCC: {e}")
            return None
    
    def extract_mel_spectrogram(self, audio_path, n_mels=128):
        """Extract Mel Spectrogram (Freq, T)"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sr, duration=4)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            if mel_spec_db.shape[1] < self.max_len:
                pad_width = self.max_len - mel_spec_db.shape[1]
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mel_spec_db = mel_spec_db[:, :self.max_len]
                
            return mel_spec_db
        except Exception as e:
            st.error(f"Error extracting Mel Spectrogram: {e}")
            return None

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_waveform(audio_path):
    """Plot audio waveform"""
    audio, sr = librosa.load(audio_path, sr=22050)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sr, ax=ax, color='#1976d2')
    ax.set_title('Audio Waveform', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_mel_spectrogram(audio_path):
    """Plot Mel Spectrogram"""
    audio, sr = librosa.load(audio_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
    ax.set_title('Mel Spectrogram', fontsize=14, fontweight='bold')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig

def plot_mfcc(audio_path):
    """Plot MFCC features"""
    audio, sr = librosa.load(audio_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax, cmap='coolwarm')
    ax.set_title('MFCC Features', fontsize=14, fontweight='bold')
    ax.set_ylabel('MFCC Coefficients')
    fig.colorbar(img, ax=ax)
    plt.tight_layout()
    return fig

def plot_predictions_bar(predictions, class_names):
    """Create interactive bar chart for predictions"""
    df = pd.DataFrame({
        'Class': class_names,
        'Probability': predictions * 100
    })
    df = df.sort_values('Probability', ascending=True)
    
    fig = px.bar(df, x='Probability', y='Class', orientation='h',
                 title='Class Probabilities',
                 labels={'Probability': 'Confidence (%)'},
                 color='Probability',
                 color_continuous_scale='Blues')
    
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_range=[0, 100]
    )
    
    return fig

def plot_predictions_gauge(confidence):
    """Create gauge chart for confidence"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Prediction Confidence", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 75], 'color': '#fff9c4'},
                {'range': [75, 100], 'color': '#c8e6c9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

# ============================================================================
# MODEL LOADING AND PREDICTION
# ============================================================================

@st.cache_resource
def load_models():
    """Load all trained models"""
    models_dict = {}
    model_files = {
        'LSTM': 'models/best_lstm.h5',
        'GRU': 'models/best_gru.h5',
        '1D CNN': 'models/best_1d_cnn.h5',
        '2D CNN': 'models/best_2d_cnn.h5',
        'CRNN': 'models/best_crnn.h5'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                models_dict[name] = keras.models.load_model(path)
            except Exception as e:
                st.warning(f"Could not load {name} model: {e}")
    
    return models_dict

@st.cache_resource
def load_label_encoder():
    """Load label encoder"""
    with open('models/label_encoder.pkl', 'rb') as f:
        return pickle.load(f)

def predict_audio(audio_path, model, model_name, extractor, le):
    """Make prediction on audio file"""
    
    # Extract features based on model type
    if model_name in ['LSTM', 'GRU', '1D CNN']:
        features = extractor.extract_mfcc_sequence(audio_path)
        if features is None:
            return None, None
        features = features[np.newaxis, ...]  # Add batch dimension
    else:  # 2D CNN or CRNN
        features = extractor.extract_mel_spectrogram(audio_path)
        if features is None:
            return None, None
        features = features[..., np.newaxis]  # Add channel dimension
        features = features[np.newaxis, ...]  # Add batch dimension
    
    # Predict
    predictions = model.predict(features, verbose=0)[0]
    predicted_class_idx = np.argmax(predictions)
    predicted_class = le.classes_[predicted_class_idx]
    confidence = predictions[predicted_class_idx]
    
    return predicted_class, predictions, confidence

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Title and description
    st.title("🎵 Audio Event Classification System")
    st.markdown("""
    Upload an audio file to classify environmental sounds using advanced deep learning models.
    Supports multiple architectures: LSTM, GRU, 1D CNN, 2D CNN, and CRNN.
    """)
    
    # Sidebar
    st.sidebar.title("📊 Model Selection")
    st.sidebar.markdown("---")
    
    # Load models and label encoder
    try:
        models = load_models()
        le = load_label_encoder()
        extractor = AudioFeatureExtractor()
        
        if not models:
            st.error("No models found! Please train models first using the training script.")
            st.write("📂 Current working directory:", os.getcwd())
            st.write("📁 Contents of models/:", os.listdir("models") if os.path.exists("models") else "No models folder found")
            return
        
        st.sidebar.success(f"✅ {len(models)} models loaded")
        
        # Model selection
        selected_model_name = st.sidebar.selectbox(
            "Choose a model:",
            list(models.keys()),
            help="Select which model architecture to use for prediction"
        )
        
        # Model info
        st.sidebar.markdown("### Model Information")
        model_info = {
            'LSTM': "Long Short-Term Memory network for sequential patterns",
            'GRU': "Gated Recurrent Unit for temporal dependencies",
            '1D CNN': "1D Convolutional network for temporal features",
            '2D CNN': "2D Convolutional network for spectrogram images",
            'CRNN': "Hybrid CNN+LSTM for spatial and temporal features"
        }
        st.sidebar.info(model_info.get(selected_model_name, ""))
        
        # Class names
        st.sidebar.markdown("### Supported Classes")
        class_names = le.classes_
        for i, cls in enumerate(class_names, 1):
            st.sidebar.write(f"{i}. {cls}")
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return
    
    # Main content
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an audio file (WAV, MP3, OGG)",
        type=['wav', 'mp3', 'ogg', 'flac'],
        help="Upload an audio file to classify"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📈 Visualizations", "ℹ️ Details"])
        
        with tab1:
            # Prediction section
            st.markdown("### Prediction Results")
            
            with st.spinner("Analyzing audio..."):
                predicted_class, predictions, confidence = predict_audio(
                    tmp_path, 
                    models[selected_model_name], 
                    selected_model_name,
                    extractor,
                    le
                )
            
            if predicted_class is not None:
                # Main prediction display
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("#### Predicted Class")
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style="color: #1976d2; text-align: center;">{predicted_class}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence gauge
                    st.plotly_chart(plot_predictions_gauge(confidence), use_container_width=True)
                
                with col2:
                    # Probability distribution
                    st.plotly_chart(plot_predictions_bar(predictions, class_names), use_container_width=True)
                
                # Detailed probabilities
                st.markdown("### Detailed Class Probabilities")
                prob_df = pd.DataFrame({
                    'Class': class_names,
                    'Probability (%)': predictions * 100,
                    'Confidence': ['⭐' * int(p * 5) for p in predictions]
                })
                prob_df = prob_df.sort_values('Probability (%)', ascending=False)
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
            else:
                st.error("Error making prediction. Please try another file.")
        
        with tab2:
            # Visualization section
            st.markdown("### Audio Feature Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Waveform")
                fig = plot_waveform(tmp_path)
                st.pyplot(fig)
                plt.close()
                
                st.markdown("#### MFCC Features")
                fig = plot_mfcc(tmp_path)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.markdown("#### Mel Spectrogram")
                fig = plot_mel_spectrogram(tmp_path)
                st.pyplot(fig)
                plt.close()
                
                # Audio statistics
                audio, sr = librosa.load(tmp_path, sr=22050)
                st.markdown("#### Audio Statistics")
                stats_df = pd.DataFrame({
                    'Metric': ['Duration', 'Sample Rate', 'RMS Energy', 'Zero Crossing Rate'],
                    'Value': [
                        f"{len(audio)/sr:.2f} seconds",
                        f"{sr} Hz",
                        f"{np.mean(librosa.feature.rms(y=audio)):.4f}",
                        f"{np.mean(librosa.feature.zero_crossing_rate(audio)):.4f}"
                    ]
                })
                st.dataframe(stats_df, hide_index=True, use_container_width=True)
        
        with tab3:
            # Details section
            st.markdown("### Model & Processing Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Model Architecture")
                st.info(f"""
                **Model:** {selected_model_name}
                
                **Input Features:** {model_info.get(selected_model_name, "")}
                
                **Classes:** {len(class_names)}
                
                **Preprocessing:** 
                - Sample Rate: 22050 Hz
                - Duration: 4 seconds
                - MFCC Coefficients: 40
                - Mel Bins: 128
                """)
            
            with col2:
                st.markdown("#### Processing Pipeline")
                st.success("""
                1. **Audio Loading**: Load and resample to 22050 Hz
                2. **Feature Extraction**: Extract MFCCs or Mel Spectrogram
                3. **Normalization**: Standardize features
                4. **Prediction**: Feed to neural network
                5. **Softmax**: Convert to probabilities
                """)
                
                st.markdown("#### File Information")
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
                st.write(f"**Type:** {uploaded_file.type}")
        
        # Clean up temp file
        os.unlink(tmp_path)
    
    else:
        # Instructions when no file uploaded
        st.info("""
        👆 **Upload an audio file to get started!**
        
        Supported formats: WAV, MP3, OGG, FLAC
        
        The model can classify the following sound events:
        - Air Conditioner
        - Car Horn
        - Children Playing
        - Dog Bark
        - Drilling
        - Engine Idling
        - Gun Shot
        - Jackhammer
        - Siren
        - Street Music
        """)

if __name__ == "__main__":

    main()
