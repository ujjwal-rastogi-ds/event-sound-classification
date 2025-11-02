"""
Advanced Audio Event Classification with Time-Series Models
UrbanSound8K Dataset

This notebook implements:
- RNN (LSTM/GRU)
- 1D CNN
- 2D CNN
- CRNN (CNN + LSTM)
"""

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

class AudioFeatureExtractor:
    """Extract various features from audio files"""
    
    def __init__(self, sr=22050, n_mfcc=40, max_len=174):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_len = max_len  # For padding/truncating sequences
        
    def extract_mfcc_sequence(self, audio_path):
        """Extract MFCC features as sequence (T, 40)"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sr, duration=4)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            # Transpose to (T, n_mfcc)
            mfccs = mfccs.T
            
            # Pad or truncate to fixed length
            if mfccs.shape[0] < self.max_len:
                pad_width = self.max_len - mfccs.shape[0]
                mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
            else:
                mfccs = mfccs[:self.max_len, :]
                
            return mfccs
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return np.zeros((self.max_len, self.n_mfcc))
    
    def extract_mel_spectrogram(self, audio_path, n_mels=128):
        """Extract Mel Spectrogram (Freq, T) for 2D CNN"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sr, duration=4)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Pad or truncate time dimension
            if mel_spec_db.shape[1] < self.max_len:
                pad_width = self.max_len - mel_spec_db.shape[1]
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mel_spec_db = mel_spec_db[:, :self.max_len]
                
            return mel_spec_db
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return np.zeros((n_mels, self.max_len))

def load_urbansound8k(metadata_path, audio_dir):
    """Load UrbanSound8K dataset"""
    metadata = pd.read_csv(metadata_path)
    
    # Build full paths
    metadata['file_path'] = metadata.apply(
        lambda row: os.path.join(audio_dir, f"fold{row['fold']}", row['slice_file_name']),
        axis=1
    )
    
    return metadata

# ============================================================================
# 2. MODEL ARCHITECTURES
# ============================================================================

def build_lstm_model(input_shape, num_classes):
    """LSTM model for sequence classification"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name='LSTM_Model')
    
    return model

def build_gru_model(input_shape, num_classes):
    """GRU model for sequence classification"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.GRU(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name='GRU_Model')
    
    return model

def build_1d_cnn_model(input_shape, num_classes):
    """1D CNN for temporal patterns in MFCCs"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        
        layers.Conv1D(128, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        
        layers.Conv1D(256, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name='1D_CNN_Model')
    
    return model

def build_2d_cnn_model(input_shape, num_classes):
    """2D CNN for Mel Spectrogram (image-like)"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name='2D_CNN_Model')
    
    return model

def build_crnn_model(input_shape, num_classes):
    """CRNN: CNN for spatial features + LSTM for temporal patterns"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # CNN layers for spatial features
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Reshape for LSTM (time_steps, features)
        layers.Reshape((-1, 128)),
        
        # LSTM layers for temporal patterns
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name='CRNN_Model')
    
    return model

# ============================================================================
# 3. TRAINING PIPELINE
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val, model_name, epochs=25):
    """Train a model with callbacks"""
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    checkpoint = callbacks.ModelCheckpoint(
        f'models/best_{model_name}.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate model and print metrics"""
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes, target_names=class_names))
    
    print(f"\nAccuracy: {accuracy_score(y_test_classes, y_pred_classes):.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model.name}.png')
    plt.close()

# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Paths (UPDATE THESE)
    # METADATA_PATH = 'UrbanSound8K/metadata/UrbanSound8K.csv'
    # AUDIO_DIR = 'UrbanSound8K/audio'
    AUDIO_DIR=r'C:\Users\ujjwa\Desktop\ATSA_PROJECT\UrbanSound8K\audio'
    METADATA_PATH=r'C:\Users\ujjwa\Desktop\ATSA_PROJECT\UrbanSound8K\metadata\UrbanSound8K.csv'
    
    print("Loading dataset...")
    metadata = load_urbansound8k(METADATA_PATH, AUDIO_DIR)
    print(f"Total samples: {len(metadata)}")
    print(f"Classes: {metadata['class'].unique()}")
    
    # Encode labels
    le = LabelEncoder()
    metadata['label'] = le.fit_transform(metadata['class'])
    class_names = le.classes_
    num_classes = len(class_names)
    
    # Save label encoder
    import pickle
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    # Split data
    train_df, test_df = train_test_split(metadata, test_size=0.2, random_state=42, stratify=metadata['label'])
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])
    
    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize feature extractor
    extractor = AudioFeatureExtractor()
    
    # ========================================================================
    # TRAIN LSTM/GRU and 1D CNN (using MFCC sequences)
    # ========================================================================
    print("\n" + "="*70)
    print("Extracting MFCC sequences for LSTM/GRU/1D CNN...")
    print("="*70)
    
    X_train_mfcc = np.array([extractor.extract_mfcc_sequence(path) for path in train_df['file_path']])
    X_val_mfcc = np.array([extractor.extract_mfcc_sequence(path) for path in val_df['file_path']])
    X_test_mfcc = np.array([extractor.extract_mfcc_sequence(path) for path in test_df['file_path']])
    
    y_train = to_categorical(train_df['label'].values, num_classes)
    y_val = to_categorical(val_df['label'].values, num_classes)
    y_test = to_categorical(test_df['label'].values, num_classes)
    
    print(f"MFCC shape: {X_train_mfcc.shape}")
    
    # Train LSTM
    print("\n" + "="*70)
    print("Training LSTM Model...")
    print("="*70)
    lstm_model = build_lstm_model((X_train_mfcc.shape[1], X_train_mfcc.shape[2]), num_classes)
    lstm_history = train_model(lstm_model, X_train_mfcc, y_train, X_val_mfcc, y_val, 'lstm')
    evaluate_model(lstm_model, X_test_mfcc, y_test, class_names)
    
    # Train GRU
    print("\n" + "="*70)
    print("Training GRU Model...")
    print("="*70)
    gru_model = build_gru_model((X_train_mfcc.shape[1], X_train_mfcc.shape[2]), num_classes)
    gru_history = train_model(gru_model, X_train_mfcc, y_train, X_val_mfcc, y_val, 'gru')
    evaluate_model(gru_model, X_test_mfcc, y_test, class_names)
    
    # Train 1D CNN
    print("\n" + "="*70)
    print("Training 1D CNN Model...")
    print("="*70)
    cnn1d_model = build_1d_cnn_model((X_train_mfcc.shape[1], X_train_mfcc.shape[2]), num_classes)
    cnn1d_history = train_model(cnn1d_model, X_train_mfcc, y_train, X_val_mfcc, y_val, '1d_cnn')
    evaluate_model(cnn1d_model, X_test_mfcc, y_test, class_names)
    
    # ========================================================================
    # TRAIN 2D CNN and CRNN (using Mel Spectrograms)
    # ========================================================================
    print("\n" + "="*70)
    print("Extracting Mel Spectrograms for 2D CNN and CRNN...")
    print("="*70)
    
    X_train_mel = np.array([extractor.extract_mel_spectrogram(path) for path in train_df['file_path']])
    X_val_mel = np.array([extractor.extract_mel_spectrogram(path) for path in val_df['file_path']])
    X_test_mel = np.array([extractor.extract_mel_spectrogram(path) for path in test_df['file_path']])
    
    # Add channel dimension for 2D CNN
    X_train_mel = X_train_mel[..., np.newaxis]
    X_val_mel = X_val_mel[..., np.newaxis]
    X_test_mel = X_test_mel[..., np.newaxis]
    
    print(f"Mel Spectrogram shape: {X_train_mel.shape}")
    
    # Train 2D CNN
    print("\n" + "="*70)
    print("Training 2D CNN Model...")
    print("="*70)
    cnn2d_model = build_2d_cnn_model((X_train_mel.shape[1], X_train_mel.shape[2], 1), num_classes)
    cnn2d_history = train_model(cnn2d_model, X_train_mel, y_train, X_val_mel, y_val, '2d_cnn')
    evaluate_model(cnn2d_model, X_test_mel, y_test, class_names)
    
    # Train CRNN
    print("\n" + "="*70)
    print("Training CRNN Model...")
    print("="*70)
    crnn_model = build_crnn_model((X_train_mel.shape[1], X_train_mel.shape[2], 1), num_classes)
    crnn_history = train_model(crnn_model, X_train_mel, y_train, X_val_mel, y_val, 'crnn')
    evaluate_model(crnn_model, X_test_mel, y_test, class_names)
    
    print("\n" + "="*70)
    print("Training Complete! All models saved in 'models/' directory")
    print("="*70)