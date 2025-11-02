"""
Quick Setup Script for Audio Classification Project
This script helps verify your setup and provides guidance
"""

import os
import sys

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Please upgrade to Python 3.8+")
        return False

def check_directories():
    """Check and create necessary directories"""
    print("\n🔍 Checking directory structure...")
    
    directories = {
        'models': 'For storing trained models',
        'UrbanSound8K': 'For dataset (you need to download this)',
        'UrbanSound8K/audio': 'For audio files',
        'UrbanSound8K/metadata': 'For metadata CSV'
    }
    
    for dir_path, description in directories.items():
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/ exists - {description}")
        else:
            if dir_path == 'models':
                os.makedirs(dir_path, exist_ok=True)
                print(f"✅ Created {dir_path}/ - {description}")
            else:
                print(f"⚠️  {dir_path}/ missing - {description}")

def check_dataset():
    """Check if UrbanSound8K dataset is present"""
    print("\n🔍 Checking dataset...")
    
    metadata_path = 'UrbanSound8K/metadata/UrbanSound8K.csv'
    audio_dir = 'UrbanSound8K/audio'
    
    if os.path.exists(metadata_path):
        print("✅ Metadata file found")
        
        # Count folds
        fold_count = 0
        if os.path.exists(audio_dir):
            folds = [f for f in os.listdir(audio_dir) if f.startswith('fold')]
            fold_count = len(folds)
            print(f"✅ Found {fold_count} audio folds")
        
        if fold_count == 10:
            print("✅ Complete dataset detected!")
            return True
        else:
            print("⚠️  Incomplete dataset - should have 10 folds")
            return False
    else:
        print("❌ Dataset not found!")
        print("\n📥 How to get the dataset:")
        print("1. Visit: https://urbansounddataset.weebly.com/urbansound8k.html")
        print("2. Fill out the form to request download")
        print("3. Extract to: UrbanSound8K/")
        return False

def check_packages():
    """Check if required packages can be imported"""
    print("\n🔍 Checking required packages...")
    
    packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'librosa': 'librosa',
        'tensorflow': 'tensorflow',
        'streamlit': 'streamlit',
        'matplotlib': 'matplotlib',
        'sklearn': 'scikit-learn'
    }
    
    missing = []
    for pkg_import, pkg_install in packages.items():
        try:
            __import__(pkg_import)
            print(f"✅ {pkg_install}")
        except ImportError:
            print(f"❌ {pkg_install} - Not installed")
            missing.append(pkg_install)
    
    if missing:
        print(f"\n⚠️  Missing packages detected!")
        print(f"Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages installed!")
        return True

def check_trained_models():
    """Check if models are trained"""
    print("\n🔍 Checking trained models...")
    
    model_files = [
        'models/best_lstm.h5',
        'models/best_gru.h5',
        'models/best_1d_cnn.h5',
        'models/best_2d_cnn.h5',
        'models/best_crnn.h5',
        'models/label_encoder.pkl'
    ]
    
    found_models = []
    missing_models = []
    
    for model_path in model_files:
        if os.path.exists(model_path):
            found_models.append(model_path)
            print(f"✅ {os.path.basename(model_path)}")
        else:
            missing_models.append(model_path)
            print(f"❌ {os.path.basename(model_path)} - Not found")
    
    if missing_models:
        print(f"\n⚠️  {len(missing_models)} models not found")
        print("You need to train the models first!")
        print("Run: python train_models.py")
        return False
    else:
        print(f"\n✅ All {len(found_models)} models ready!")
        return True

def main():
    """Main setup check"""
    print("="*70)
    print("🎵 Audio Classification Project - Setup Verification")
    print("="*70)
    
    # Run all checks
    python_ok = check_python_version()
    check_directories()
    dataset_ok = check_dataset()
    packages_ok = check_packages()
    models_ok = check_trained_models()
    
    # Summary
    print("\n" + "="*70)
    print("📋 SETUP SUMMARY")
    print("="*70)
    
    if python_ok and packages_ok and dataset_ok:
        print("\n✅ READY TO TRAIN!")
        print("\nNext steps:")
        print("1. Run: python train_models.py  (This will take 2-4 hours)")
        print("2. After training, run: streamlit run app.py")
    elif python_ok and packages_ok and dataset_ok and models_ok:
        print("\n✅ FULLY READY!")
        print("\nYou can start the web app:")
        print("Run: streamlit run app.py")
    else:
        print("\n⚠️  SETUP INCOMPLETE")
        print("\nPlease fix the issues above:")
        if not python_ok:
            print("- Upgrade Python to 3.8+")
        if not packages_ok:
            print("- Install packages: pip install -r requirements.txt")
        if not dataset_ok:
            print("- Download and extract UrbanSound8K dataset")
        if not models_ok and dataset_ok and packages_ok:
            print("- Train models: python train_models.py")
    
    print("\n" + "="*70)
    print("\n📚 For detailed instructions, see README.md")
    print("❓ For issues, check the Troubleshooting section in README.md")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()