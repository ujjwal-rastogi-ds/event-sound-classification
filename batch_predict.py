"""
Batch Audio Prediction Script
Process multiple audio files and generate predictions CSV
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
from tensorflow import keras
import pickle
from tqdm import tqdm
import argparse
from pathlib import Path

class AudioPredictor:
    """Batch audio file predictor"""
    
    def __init__(self, model_path, label_encoder_path, model_type='lstm'):
        """
        Args:
            model_path: Path to trained model
            label_encoder_path: Path to label encoder pickle
            model_type: Type of model ('lstm', 'gru', '1d_cnn', '2d_cnn', 'crnn')
        """
        self.model = keras.models.load_model(model_path)
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        self.model_type = model_type.lower()
        self.sr = 22050
        self.n_mfcc = 40
        self.max_len = 174
        
        print(f"✅ Loaded {model_type.upper()} model")
        print(f"✅ Classes: {', '.join(self.label_encoder.classes_)}")
    
    def extract_mfcc_sequence(self, audio_path):
        """Extract MFCC features"""
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
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def extract_mel_spectrogram(self, audio_path, n_mels=128):
        """Extract Mel Spectrogram"""
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
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def predict_single(self, audio_path):
        """Predict single audio file"""
        # Extract features based on model type
        if self.model_type in ['lstm', 'gru', '1d_cnn']:
            features = self.extract_mfcc_sequence(audio_path)
            if features is None:
                return None
            features = features[np.newaxis, ...]
        else:  # 2D CNN or CRNN
            features = self.extract_mel_spectrogram(audio_path)
            if features is None:
                return None
            features = features[..., np.newaxis]
            features = features[np.newaxis, ...]
        
        # Predict
        predictions = self.model.predict(features, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        predicted_class = self.label_encoder.classes_[predicted_idx]
        confidence = predictions[predicted_idx]
        
        # Get top 3 predictions
        top3_idx = np.argsort(predictions)[-3:][::-1]
        top3_predictions = [(self.label_encoder.classes_[i], predictions[i]) 
                           for i in top3_idx]
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': predictions,
            'top3': top3_predictions
        }
    
    def predict_batch(self, audio_dir, output_csv='predictions.csv'):
        """Predict all audio files in a directory"""
        # Find all audio files
        audio_extensions = ['.wav', '.mp3', '.ogg', '.flac']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(Path(audio_dir).rglob(f'*{ext}'))
        
        if not audio_files:
            print(f"❌ No audio files found in {audio_dir}")
            return
        
        print(f"📁 Found {len(audio_files)} audio files")
        print(f"🎵 Processing with {self.model_type.upper()} model...")
        
        # Process each file
        results = []
        for audio_path in tqdm(audio_files, desc="Processing"):
            prediction = self.predict_single(str(audio_path))
            
            if prediction:
                result = {
                    'filename': audio_path.name,
                    'filepath': str(audio_path),
                    'predicted_class': prediction['predicted_class'],
                    'confidence': f"{prediction['confidence']:.4f}"
                }
                
                # Add all class probabilities
                for i, class_name in enumerate(self.label_encoder.classes_):
                    result[f'prob_{class_name}'] = f"{prediction['all_probabilities'][i]:.4f}"
                
                # Add top 3 predictions
                for i, (cls, prob) in enumerate(prediction['top3'], 1):
                    result[f'top{i}_class'] = cls
                    result[f'top{i}_prob'] = f"{prob:.4f}"
                
                results.append(result)
        
        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        
        print(f"\n✅ Predictions saved to {output_csv}")
        print(f"📊 Processed {len(results)} files successfully")
        
        # Show summary statistics
        print("\n📈 Prediction Summary:")
        print(df['predicted_class'].value_counts())
        
        return df

def main():
    parser = argparse.ArgumentParser(description='Batch Audio Prediction')
    parser.add_argument('--audio_dir', type=str, required=True,
                       help='Directory containing audio files')
    parser.add_argument('--model', type=str, default='models/best_crnn.h5',
                       help='Path to trained model')
    parser.add_argument('--model_type', type=str, default='crnn',
                       choices=['lstm', 'gru', '1d_cnn', '2d_cnn', 'crnn'],
                       help='Type of model')
    parser.add_argument('--label_encoder', type=str, default='models/label_encoder.pkl',
                       help='Path to label encoder')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.label_encoder):
        print(f"❌ Label encoder not found: {args.label_encoder}")
        sys.exit(1)
    
    if not os.path.exists(args.audio_dir):
        print(f"❌ Audio directory not found: {args.audio_dir}")
        sys.exit(1)
    
    # Create predictor and run
    predictor = AudioPredictor(args.model, args.label_encoder, args.model_type)
    predictor.predict_batch(args.audio_dir, args.output)

if __name__ == "__main__":
    # Example usage without command line args
    if len(sys.argv) == 1:
        print("="*70)
        print("🎵 Batch Audio Prediction Script")
        print("="*70)
        print("\nUsage:")
        print("python batch_predict.py --audio_dir path/to/audio/files")
        print("\nOptional arguments:")
        print("--model          Path to model (default: models/best_crnn.h5)")
        print("--model_type     Model type: lstm, gru, 1d_cnn, 2d_cnn, crnn")
        print("--label_encoder  Path to label encoder (default: models/label_encoder.pkl)")
        print("--output         Output CSV file (default: predictions.csv)")
        print("\nExample:")
        print("python batch_predict.py --audio_dir test_audio/ --model_type crnn --output results.csv")
        print("\n" + "="*70)
    else:
        main()