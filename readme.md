# 🎵 Advanced Audio Event Classification System

A comprehensive deep learning system for classifying environmental sounds from the UrbanSound8K dataset, featuring multiple state-of-the-art architectures and an interactive web interface.

## 🌟 Features

### Models Implemented
1. **LSTM** - Long Short-Term Memory networks for sequential pattern recognition
2. **GRU** - Gated Recurrent Units for efficient temporal modeling
3. **1D CNN** - Temporal convolutional networks for MFCC sequences
4. **2D CNN** - Image-like processing of Mel spectrograms
5. **CRNN** - Hybrid CNN+LSTM for combined spatial-temporal features

### Interactive Web App
- 🎯 Real-time audio classification
- 📊 Confidence scores and probability distributions
- 📈 Visual feature analysis (waveform, MFCC, Mel spectrogram)
- 🔄 Support for multiple audio formats (WAV, MP3, OGG, FLAC)
- 📱 Responsive design with interactive charts

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.13.0
- See `requirements.txt` for complete dependencies

## 🚀 Installation

### 1. Clone or Download the Project

```bash
mkdir audio_classification_project
cd audio_classification_project
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download UrbanSound8K Dataset

1. Visit: https://urbansounddataset.weebly.com/urbansound8k.html
2. Request and download the dataset
3. Extract to your project folder

**Expected folder structure:**
```
audio_classification_project/
├── UrbanSound8K/
│   ├── audio/
│   │   ├── fold1/
│   │   ├── fold2/
│   │   └── ...
│   └── metadata/
│       └── UrbanSound8K.csv
├── models/  (will be created during training)
├── train_models.py
├── app.py
└── requirements.txt
```

## 🎓 Training Models

### Step 1: Update Paths in Training Script

Open `train_models.py` and update these paths:

```python
METADATA_PATH = 'UrbanSound8K/metadata/UrbanSound8K.csv'
AUDIO_DIR = 'UrbanSound8K/audio'
```

### Step 2: Run Training

```bash
python train_models.py
```

This will:
- Extract features from all audio files
- Train 5 different models (LSTM, GRU, 1D CNN, 2D CNN, CRNN)
- Save best models to `models/` directory
- Generate confusion matrices
- Display evaluation metrics

**Training time:** Approximately 2-4 hours depending on your hardware

**Expected outputs:**
- `models/best_lstm.h5`
- `models/best_gru.h5`
- `models/best_1d_cnn.h5`
- `models/best_2d_cnn.h5`
- `models/best_crnn.h5`
- `models/label_encoder.pkl`
- Confusion matrix images

## 🌐 Running the Web App

### Start the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the App

1. **Select Model**: Choose from dropdown in sidebar (LSTM, GRU, 1D CNN, 2D CNN, CRNN)
2. **Upload Audio**: Click "Browse files" and select an audio file
3. **View Results**:
   - **Prediction Tab**: See predicted class and confidence scores
   - **Visualizations Tab**: Explore audio features (waveform, MFCC, Mel spectrogram)
   - **Details Tab**: View model and processing information

## 📊 Model Architectures

### LSTM Model
```
Input: (174, 40) - Sequence of 40 MFCC features
→ LSTM(128) with return_sequences
→ Dropout(0.3)
→ LSTM(64)
→ Dropout(0.3)
→ Dense(64, relu)
→ Dense(10, softmax)
```

### GRU Model
```
Input: (174, 40) - Sequence of 40 MFCC features
→ GRU(128) with return_sequences
→ Dropout(0.3)
→ GRU(64)
→ Dropout(0.3)
→ Dense(64, relu)
→ Dense(10, softmax)
```

### 1D CNN Model
```
Input: (174, 40) - Sequence of 40 MFCC features
→ Conv1D(64, 3) → BatchNorm → MaxPool → Dropout
→ Conv1D(128, 3) → BatchNorm → MaxPool → Dropout
→ Conv1D(256, 3) → BatchNorm → MaxPool → Dropout
→ GlobalAveragePooling1D
→ Dense(128, relu)
→ Dense(10, softmax)
```

### 2D CNN Model
```
Input: (128, 174, 1) - Mel Spectrogram
→ Conv2D(32, 3×3) → BatchNorm → MaxPool → Dropout
→ Conv2D(64, 3×3) → BatchNorm → MaxPool → Dropout
→ Conv2D(128, 3×3) → BatchNorm → MaxPool → Dropout
→ Flatten
→ Dense(128, relu)
→ Dense(10, softmax)
```

### CRNN Model
```
Input: (128, 174, 1) - Mel Spectrogram
→ Conv2D(32, 3×3) → BatchNorm → MaxPool → Dropout
→ Conv2D(64, 3×3) → BatchNorm → MaxPool → Dropout
→ Conv2D(128, 3×3) → BatchNorm → MaxPool → Dropout
→ Reshape for LSTM
→ LSTM(64) with return_sequences
→ LSTM(32)
→ Dense(64, relu)
→ Dense(10, softmax)
```

## 🎯 Supported Classes

The models classify 10 environmental sound events:
1. Air Conditioner
2. Car Horn
3. Children Playing
4. Dog Bark
5. Drilling
6. Engine Idling
7. Gun Shot
8. Jackhammer
9. Siren
10. Street Music

## 📈 Feature Extraction

### MFCC (Mel-Frequency Cepstral Coefficients)
- Used for: LSTM, GRU, 1D CNN
- Dimensions: (174, 40)
- Represents: Temporal sequence of spectral features

### Mel Spectrogram
- Used for: 2D CNN, CRNN
- Dimensions: (128, 174, 1)
- Represents: Time-frequency representation

## 🔧 Customization

### Adjust Model Parameters

Edit hyperparameters in `train_models.py`:

```python
# Change number of MFCC coefficients
extractor = AudioFeatureExtractor(n_mfcc=40)  # Change to 20, 30, etc.

# Adjust training epochs
train_model(..., epochs=50)  # Change to desired number

# Modify batch size
history = model.fit(..., batch_size=32)  # Change to 16, 64, etc.
```

### Add Custom Models

Add your own architecture in `train_models.py`:

```python
def build_custom_model(input_shape, num_classes):
    model = models.Sequential([
        # Your layers here
    ])
    return model
```

## 📝 File Descriptions

- **`train_models.py`**: Complete training pipeline for all models
- **`app.py`**: Streamlit web application for inference
- **`requirements.txt`**: Python package dependencies
- **`README.md`**: This documentation

## 🐛 Troubleshooting

### Model Loading Error
**Problem:** "No models found"
**Solution:** Ensure you've run `train_models.py` first and models are saved in `models/` directory

### Audio Processing Error
**Problem:** "Error extracting features"
**Solution:** Check audio file format and ensure it's readable by librosa

### Memory Error During Training
**Problem:** Out of memory
**Solution:** Reduce batch size or use a subset of data

### Import Error
**Problem:** Module not found
**Solution:** Ensure all dependencies are installed: `pip install -r requirements.txt`

## 📊 Expected Performance

Typical accuracy ranges (may vary based on training):
- **LSTM**: 75-85%
- **GRU**: 75-85%
- **1D CNN**: 80-88%
- **2D CNN**: 82-90%
- **CRNN**: 85-92% (usually best performing)

## 🎓 Tips for Better Results

1. **Data Augmentation**: Add pitch shifting, time stretching
2. **Ensemble Methods**: Combine predictions from multiple models
3. **Fine-tuning**: Adjust learning rate and optimizer
4. **Feature Engineering**: Experiment with different audio features
5. **Model Architecture**: Add more layers or adjust units

## 📚 References

- UrbanSound8K Dataset: https://urbansounddataset.weebly.com/
- Librosa Documentation: https://librosa.org/
- TensorFlow Documentation: https://www.tensorflow.org/
- Streamlit Documentation: https://docs.streamlit.io/

## 🤝 Contributing

Feel free to:
- Add new model architectures
- Improve preprocessing techniques
- Enhance the web interface
- Add more visualizations

## 📄 License

This project uses the UrbanSound8K dataset. Please cite the original paper:
```
J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 
22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.
```

## 🎉 Acknowledgments

- UrbanSound8K dataset creators
- TensorFlow and Keras teams
- Librosa developers
- Streamlit community

---

**Happy Classifying! 🎵🤖**