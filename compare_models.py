"""
Model Comparison Script
Compare performance of all trained models and generate comprehensive reports
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from tensorflow import keras
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_recall_fscore_support,
                            roc_curve, auc)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ModelComparator:
    """Compare multiple trained models"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.results = {}
        
    def load_models(self):
        """Load all trained models"""
        model_files = {
            'LSTM': 'best_lstm.h5',
            'GRU': 'best_gru.h5',
            '1D CNN': 'best_1d_cnn.h5',
            '2D CNN': 'best_2d_cnn.h5',
            'CRNN': 'best_crnn.h5'
        }
        
        print("Loading models...")
        for name, filename in model_files.items():
            filepath = os.path.join(self.models_dir, filename)
            if os.path.exists(filepath):
                self.models[name] = keras.models.load_model(filepath)
                print(f"✅ Loaded {name}")
            else:
                print(f"⚠️  {name} not found")
        
        # Load label encoder
        with open(os.path.join(self.models_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print(f"\nTotal models loaded: {len(self.models)}")
        
    def evaluate_all_models(self, X_test_dict, y_test):
        """Evaluate all models on test data
        
        X_test_dict: Dictionary with keys like 'mfcc' and 'mel' for different feature types
        y_test: True labels
        """
        print("\n" + "="*70)
        print("EVALUATING ALL MODELS")
        print("="*70)
        
        for model_name, model in self.models.items():
            print(f"\n📊 Evaluating {model_name}...")
            
            # Select appropriate test data
            if model_name in ['LSTM', 'GRU', '1D CNN']:
                X_test = X_test_dict['mfcc']
            else:  # 2D CNN, CRNN
                X_test = X_test_dict['mel']
            
            # Predictions
            y_pred = model.predict(X_test, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_true_classes, y_pred_classes)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_classes, y_pred_classes, average='weighted'
            )
            
            self.results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'y_true': y_true_classes,
                'y_pred': y_pred_classes,
                'y_pred_proba': y_pred
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
    
    def plot_comparison_metrics(self):
        """Plot comparison of metrics across models"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in self.models.keys()]
            models = list(self.models.keys())
            
            ax = axes[idx]
            bars = ax.bar(models, values, color=sns.color_palette("husl", len(models)))
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison', 
                        fontsize=14, fontweight='bold')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_comparison_metrics.png', dpi=300, bbox_inches='tight')
        print("\n✅ Saved: model_comparison_metrics.png")
        plt.close()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 3, figsize=(20, 13))
        axes = axes.ravel()
        
        class_names = self.label_encoder.classes_
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            cm = confusion_matrix(
                self.results[model_name]['y_true'],
                self.results[model_name]['y_pred']
            )
            
            # Normalize
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            ax = axes[idx]
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       ax=ax, cbar_kws={'label': 'Proportion'})
            ax.set_title(f'{model_name} - Accuracy: {self.results[model_name]["accuracy"]:.3f}',
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
        
        # Hide extra subplot
        if n_models < 6:
            for idx in range(n_models, 6):
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: confusion_matrices_comparison.png")
        plt.close()
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        n_classes = len(self.label_encoder.classes_)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 13))
        axes = axes.ravel()
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            ax = axes[idx]
            
            # Binarize labels
            y_true_bin = label_binarize(
                self.results[model_name]['y_true'],
                classes=range(n_classes)
            )
            y_pred_proba = self.results[model_name]['y_pred_proba']
            
            # Compute ROC curve and AUC for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Plot all ROC curves
            colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
            for i, color in enumerate(colors):
                ax.plot(fpr[i], tpr[i], color=color, lw=1.5,
                       label=f'{self.label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')
            
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.50)')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{model_name} - ROC Curves', fontsize=12, fontweight='bold')
            ax.legend(loc="lower right", fontsize=7)
            ax.grid(alpha=0.3)
        
        # Hide extra subplot
        if len(self.models) < 6:
            for idx in range(len(self.models), 6):
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: roc_curves_comparison.png")
        plt.close()
    
    def generate_detailed_report(self):
        """Generate detailed classification reports"""
        print("\n" + "="*70)
        print("DETAILED CLASSIFICATION REPORTS")
        print("="*70)
        
        class_names = self.label_encoder.classes_
        
        with open('classification_reports.txt', 'w') as f:
            for model_name in self.models.keys():
                print(f"\n{'='*70}", file=f)
                print(f"{model_name} MODEL", file=f)
                print(f"{'='*70}", file=f)
                
                report = classification_report(
                    self.results[model_name]['y_true'],
                    self.results[model_name]['y_pred'],
                    target_names=class_names,
                    digits=4
                )
                print(report, file=f)
        
        print("\n✅ Saved: classification_reports.txt")
    
    def create_summary_table(self):
        """Create summary table of all models"""
        summary_data = []
        
        for model_name in self.models.keys():
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{self.results[model_name]['accuracy']:.4f}",
                'Precision': f"{self.results[model_name]['precision']:.4f}",
                'Recall': f"{self.results[model_name]['recall']:.4f}",
                'F1-Score': f"{self.results[model_name]['f1_score']:.4f}"
            })
        
        df = pd.DataFrame(summary_data)
        
        # Sort by accuracy
        df['Accuracy_numeric'] = df['Accuracy'].astype(float)
        df = df.sort_values('Accuracy_numeric', ascending=False)
        df = df.drop('Accuracy_numeric', axis=1)
        
        print("\n" + "="*70)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*70)
        print(df.to_string(index=False))
        
        # Save to CSV
        df.to_csv('model_comparison_summary.csv', index=False)
        print("\n✅ Saved: model_comparison_summary.csv")
        
        # Create visual table
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center',
                        colWidths=[0.15, 0.15, 0.15, 0.15, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows alternately
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.savefig('model_comparison_table.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: model_comparison_table.png")
        plt.close()
    
    def plot_per_class_performance(self):
        """Plot per-class F1 scores for each model"""
        class_names = self.label_encoder.classes_
        n_classes = len(class_names)
        
        # Calculate per-class F1 scores
        model_class_f1 = {}
        for model_name in self.models.keys():
            _, _, f1_scores, _ = precision_recall_fscore_support(
                self.results[model_name]['y_true'],
                self.results[model_name]['y_pred'],
                average=None
            )
            model_class_f1[model_name] = f1_scores
        
        # Plot
        fig, ax = plt.subplots(figsize=(16, 8))
        
        x = np.arange(n_classes)
        width = 0.15
        multiplier = 0
        
        for model_name, f1_scores in model_class_f1.items():
            offset = width * multiplier
            ax.bar(x + offset, f1_scores, width, label=model_name)
            multiplier += 1
        
        ax.set_xlabel('Sound Classes', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class F1-Score Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig('per_class_performance.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: per_class_performance.png")
        plt.close()
    
    def generate_full_report(self, X_test_dict, y_test):
        """Generate complete comparison report"""
        print("\n" + "="*70)
        print("🎵 AUDIO CLASSIFICATION MODEL COMPARISON REPORT")
        print("="*70)
        
        self.load_models()
        
        if not self.models:
            print("❌ No models found! Please train models first.")
            return
        
        self.evaluate_all_models(X_test_dict, y_test)
        
        print("\n📊 Generating visualizations...")
        self.plot_comparison_metrics()
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        self.plot_per_class_performance()
        
        print("\n📝 Generating reports...")
        self.create_summary_table()
        self.generate_detailed_report()
        
        print("\n" + "="*70)
        print("✅ COMPARISON COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("1. model_comparison_metrics.png")
        print("2. confusion_matrices_comparison.png")
        print("3. roc_curves_comparison.png")
        print("4. per_class_performance.png")
        print("5. model_comparison_table.png")
        print("6. model_comparison_summary.csv")
        print("7. classification_reports.txt")
        print("\n" + "="*70)

# Example usage
if __name__ == "__main__":
    # You need to provide test data
    # This is just a template - modify based on your data loading
    
    print("⚠️  This script requires test data to be loaded.")
    print("Please use this after running train_models.py")
    print("\nExample usage:")
    print("""
    from compare_models import ModelComparator
    
    # Load your test data
    X_test_mfcc = ...  # Your MFCC test data
    X_test_mel = ...   # Your Mel spectrogram test data
    y_test = ...       # Your test labels
    
    # Create comparator
    comparator = ModelComparator()
    
    # Generate full report
    X_test_dict = {'mfcc': X_test_mfcc, 'mel': X_test_mel}
    comparator.generate_full_report(X_test_dict, y_test)
    """)