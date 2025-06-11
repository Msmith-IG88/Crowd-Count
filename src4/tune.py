import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from model import CrowdTCN
from real_sequence_dataset import RealSequenceDataset
from synthetic_data import detect_peaks
import matplotlib.pyplot as plt
import numpy as np
import os

def estimate_baseline(x):
    peaks_per_frame = [detect_peaks(frame.cpu().numpy()) for frame in x[0]]
    return torch.tensor([np.mean(peaks_per_frame)], dtype=torch.float32)

def sample_real_data_per_label(dataset, num_samples_per_label):
    label_buckets = {label: [] for label in range(1, 6)}
    for idx in range(len(dataset)):
        x, label = dataset[idx]
        label = int(label)
        if len(label_buckets[label]) < num_samples_per_label.get(label, 0):
            label_buckets[label].append((x, label))
    # Flatten into one list
    all_samples = [item for bucket in label_buckets.values() for item in bucket]
    return all_samples

class SubsetDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, label = self.samples[idx]
        return x, torch.tensor(label, dtype=torch.float32)

def tune(label_sample_counts, epochs=20, batch_size=2, model_path="best_model.pth", save_path="best_model_tuned.pth"):
    model = CrowdTCN()
    model.load_state_dict(torch.load(model_path))
    model.train()
    print(f"Loaded model from {model_path}")

    data_path = "C:/Users/smith/Projects/people-gait/room1"  # Update if needed
    full_dataset = RealSequenceDataset(data_path, sequence_len=10, limit_per_label=50)

    samples = sample_real_data_per_label(full_dataset, label_sample_counts)
    dataset = SubsetDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for x, true_count in loader:
            x = x / (x.max(dim=-1, keepdim=True).values.clamp_min(1e-6))

            pred_residual = model(x)
            baseline = estimate_baseline(x).to(x.device).unsqueeze(1)
            baseline = baseline * 1.8  # match original scale

            target_residual = (true_count.unsqueeze(1) - baseline).float()

            loss = criterion(pred_residual.mean(dim=1, keepdim=True), target_residual)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Epoch Loss: {epoch_loss:.4f}")
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved fine-tuned model to {save_path}")
    plt.figure()
    plt.plot(range(1, epochs+1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Tuning Loss Curve")
    plt.grid()
    plt.savefig("tuning_loss.png")
    plt.show()

if __name__ == "__main__":
    # Control the number of samples per label here:
    sample_counts = {
        1: 20,
        2: 30,  # more samples from label 2
        3: 30,
        4: 30,
        5: 30  # more samples from label 5
    }
    tune(label_sample_counts=sample_counts)
