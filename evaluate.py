import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Ensure output directory exists
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_learning_curves(history, model_name: str, mode: str):
    """
    Plots training and validation loss and accuracy curves and saves the plot as an image.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='#1a73e8', linewidth=2)
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='#e8710a', linewidth=2)
    plt.title(f'{model_name} ({mode.capitalize()}) - Accuracy', fontsize=14, fontweight='bold', pad=10)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='#d93025', linewidth=2)
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss', color='#f29900', linewidth=2)
    plt.title(f'{model_name} ({mode.capitalize()}) - Loss', fontsize=14, fontweight='bold', pad=10)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, f"{model_name.lower().replace(' ', '_')}_{mode}_curves.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[EVALUATOR] Saved learning curves to: {filepath}")

def plot_confusion_matrix_heatmap(y_true, y_pred_classes, model_name: str, mode: str, class_names=None):
    """
    Generates and saves an annotated confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    
    # Style the heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=class_names if class_names else True, 
        yticklabels=class_names if class_names else True,
        cbar=True,
        annot_kws={"size": 12, "weight": "bold"}
    )
    
    plt.title(f'{model_name} ({mode.capitalize()}) - Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Class', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    filepath = os.path.join(OUTPUT_DIR, f"{model_name.lower().replace(' ', '_')}_{mode}_confusion_matrix.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[EVALUATOR] Saved confusion matrix heatmap to: {filepath}")

def evaluate_performance(y_true, y_pred, model_name: str, mode: str, class_names=None):
    """
    Calculates performance metrics (Accuracy, Precision, Recall, F1)
    and prints a classification report.
    Returns: accuracy, precision, recall, f1, predictions_classes
    """
    # y_pred will contain probabilities. Convert to class labels.
    if mode == "binary":
        y_pred_classes = (y_pred > 0.5).astype(int).flatten()
        average_metric = 'binary'
    else:
        y_pred_classes = np.argmax(y_pred, axis=1)
        average_metric = 'weighted'
        
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_classes, average=average_metric, zero_division=0
    )
    
    print("\n" + "="*50)
    print(f" PERFORMANCE REPORT: {model_name} ({mode.upper()}) ")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("-"*50)
    print("Detailed Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names, zero_division=0))
    print("="*50 + "\n")
    
    # Generate Heatmap
    plot_confusion_matrix_heatmap(y_true, y_pred_classes, model_name, mode, class_names)
    
    return accuracy, precision, recall, f1, y_pred_classes
