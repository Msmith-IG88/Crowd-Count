from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from synthetic_data import ResidualDataset, ResidualDataset_old
from model import CrowdTCN
import matplotlib.pyplot as plt
import numpy as np

def weighted_mse(pred, target, weights):
    """Weighted mean squared error"""
    per_sample_loss = (pred - target).pow(2)
    return (per_sample_loss * weights).mean()

def train():
    # Create datasets
    train_dataset = ResidualDataset_old(num_per_label=2000)
    val_dataset = ResidualDataset_old(num_per_label=400)
    #train_dataset = ResidualDataset(num_per_label={1: 1000, 2: 2000, 3: 1000, 4: 1000, 5: 1000})
    #val_dataset = ResidualDataset(num_per_label={1: 200, 2: 400, 3: 200, 4: 200, 5: 200})

    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Model and optimizer
    model = CrowdTCN()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Loss function with class weighting
    class_weights = torch.tensor([1.0, 3.0, 2.0, 1.8, 3.0])  # Higher weight for medium crowds
    min_val_loss = 0
    train_losses = []
    val_losses = []
    for epoch in range(20):
        # Training
        model.train()
        train_loss = 0
        for x, true_count, residual in train_loader:
            optimizer.zero_grad()
            
            # Normalize input
            x = x / (x.max(dim=-1, keepdim=True).values.clamp_min(1e-6))
            
            # Forward pass
            pred_residual = model(x).mean(dim=1)
            
            # Compute loss with class weighting
            weights = class_weights[true_count.long() - 1]
            loss = weighted_mse(pred_residual, residual, weights)
            
            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, true_count, residual in val_loader:
                x = x / (x.max(dim=-1, keepdim=True).values.clamp_min(1e-6))
                pred_residual = model(x).mean(dim=1)
                loss = weighted_mse(pred_residual, residual, torch.ones_like(true_count))
                val_loss += loss.item()
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f} | Val Loss {val_loss/len(val_loader):.4f}")
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        # Save best model
        torch.save(model.state_dict(), "best_model.pth")
        min_val_loss = val_loss

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train()