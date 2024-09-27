from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_anomaly_detection(y_true, y_pred, method_name):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_pred)
    
    print(f"Results for {method_name}:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print()

def plot_confusion_matrix(y_true, y_pred, method_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {method_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_comparison(y_true, z_score_pred, autoencoder_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='True Labels', alpha=0.7)
    plt.plot(z_score_pred, label='Z-Score Predictions', alpha=0.7)
    plt.plot(autoencoder_pred, label='Autoencoder Predictions', alpha=0.7)
    plt.title('Anomaly Detection Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Anomaly (1) / Normal (0)')
    plt.legend()
    plt.show()
