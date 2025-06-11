import numpy as np
import torch
from torch.utils.data import DataLoader
from real_sequence_dataset import RealSequenceDataset
from model import CrowdTCN
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from synthetic_data import detect_peaks

def estimate_baseline(x):
    peaks_per_frame = [detect_peaks(frame.cpu().numpy()) for frame in x[0]]
    return torch.tensor([np.mean(peaks_per_frame)], dtype=torch.float32)

def evaluate():
    # Load best model
    model = CrowdTCN()
    model.load_state_dict(torch.load("best_model_tuned.pth"))
    model.eval()
    print("Model Loaded")

    data_path = "C:/Users/smith/Projects/people-gait/room1"  # Update to your people-gait processed data path
    real_data = RealSequenceDataset(data_path, sequence_len=10, limit_per_label=50)
    print("Got Real Data")
    # Group indices by label
    loader = torch.utils.data.DataLoader(real_data, batch_size=1)
    true_counts, pred_counts = [], []
    # Metrics storage
    all_true = []
    all_pred = []
    per_label_mae = {i: [] for i in range(1, 6)}
    
    with torch.no_grad():
        for x, true_count in loader:
            # Preprocess
            x = x / (x.max(dim=-1, keepdim=True).values.clamp_min(1e-6))
            
            # Predict residual
            pred_residual = model(x)
            
            # Calculate baseline (visible count)
            #baseline = x.sum(dim=-1).mean(dim=1, keepdim=True)
            baseline = estimate_baseline(x).to(x.device).unsqueeze(1)
            baseline = baseline * 1.8
            
            # Final count estimation
            pred_count = baseline + 1.2 * pred_residual.mean(dim=1, keepdim=True)
                # Debug: print baseline and true label
            print(f"True label: {true_count.item()} | Baseline estimate: {baseline.item():.2f}  | Residual: {pred_residual.mean().item():.2f}")
            pred_count = pred_count.round().clamp(min=1, max=5).int()
            print(f"Pred count: {pred_count.item():.2f} | Residual: {pred_residual.mean().item():.2f}")
            # Store results
            all_true.extend(true_count.cpu().numpy())
            all_pred.extend(pred_count.cpu().numpy())
            
            # Store per-label errors
            for t, p in zip(true_count, pred_count):
                per_label_mae[t.item()].append(abs(t.item() - p.item()))
    
    # Calculate overall MAE
    mae = mean_absolute_error(all_true, all_pred)
    print(f"Overall MAE: {mae:.2f}")
    
    # Calculate per-label MAE
    print("\nPer-label MAE:")
    for label in sorted(per_label_mae):
        if per_label_mae[label]:
            label_mae = np.mean(per_label_mae[label])
            print(f"Label {label}: MAE = {label_mae:.2f}")
        else:
            print(f"Label {label}: No samples")
    
    # Save predictions for analysis
    np.savez("evaluation_results.npz", true=np.array(all_true), pred=np.array(all_pred))
    # Confusion Matrix
    cm = confusion_matrix(all_true, all_pred, labels=[1, 2, 3, 4, 5])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4, 5])
    correct = np.trace(cm)
    total = np.sum(cm)
    accuracy = correct / total
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Crowd Count Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    evaluate()