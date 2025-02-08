import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Detect device: Use CUDA (NVIDIA GPU), MPS (Mac), or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print(f"🚀 Using device: {device}")

# Load dataset
def load_data():
    print("\nLoading dataset...")
    data = np.load("data/hnefatafl_dataset.npz")

    # Extract training data
    X_boards = torch.tensor(data["X_boards"], dtype=torch.float32)
    y_start = torch.tensor(data["y_start"], dtype=torch.long)  # Ensure categorical
    y_end = torch.tensor(data["y_end"], dtype=torch.long)  # Ensure categorical
    y_winner = torch.tensor(data["y_winner"], dtype=torch.float32).unsqueeze(1)  # Shape (num_samples, 1)

    # Shuffle before splitting (avoids bias from sequential games)
    indices = torch.randperm(len(X_boards))  
    split = int(0.8 * len(X_boards))

    train_indices, val_indices = indices[:split], indices[split:]

    X_train, X_val = X_boards[train_indices], X_boards[val_indices]
    y_start_train, y_start_val = y_start[train_indices], y_start[val_indices]
    y_end_train, y_end_val = y_end[train_indices], y_end[val_indices]
    y_winner_train, y_winner_val = y_winner[train_indices], y_winner[val_indices]

    print(f"✅ Training set: {X_train.shape[0]} samples")
    print(f"✅ Validation set: {X_val.shape[0]} samples")

    return (X_train, y_start_train, y_end_train, y_winner_train), (X_val, y_start_val, y_end_val, y_winner_val)

# Define PyTorch Dataset class
class HnefataflDataset(Dataset):
    def __init__(self, X, y_start, y_end, y_winner):
        self.X = X
        self.y_start = y_start
        self.y_end = y_end
        self.y_winner = y_winner  # ✅ Include winner label

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_start[idx], self.y_end[idx], self.y_winner[idx]  # ✅ Return winner label

# Function to create DataLoaders
def get_dataloaders(batch_size=32):
    (X_train, y_start_train, y_end_train, y_winner_train), (X_val, y_start_val, y_end_val, y_winner_val) = load_data()

    train_dataset = HnefataflDataset(X_train, y_start_train, y_end_train, y_winner_train)
    val_dataset = HnefataflDataset(X_val, y_start_val, y_end_val, y_winner_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
