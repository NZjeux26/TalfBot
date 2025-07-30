import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Detect device: Use CUDA (NVIDIA GPU), MPS (Mac), or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print(f"🚀 Using device: {device}")

# Load dataset
def load_data():
    print("\nLoading dataset...")
    data = np.load("data/hnefatafl_dataset_small.npz")

    # Extract raw NumPy data before converting to tensors
    X_boards = data["X_boards"]
    y_start = data["y_start"]
    y_end = data["y_end"]
    y_winner = data["y_winner"]

    # Apply data augmentation
    X_boards, y_start, y_end, y_winner = augment_data(X_boards, y_start, y_end, y_winner)

    # Convert augmented data to PyTorch tensors
    X_boards = torch.tensor(X_boards, dtype=torch.float32)
    y_start = torch.tensor(y_start, dtype=torch.long)
    y_end = torch.tensor(y_end, dtype=torch.long)
    y_winner = torch.tensor(y_winner, dtype=torch.float32).unsqueeze(1)

    # Let's create a three-way split: 70% train, 15% validation, 15% test
    total_samples = len(X_boards)
    train_size = int(0.75 * total_samples)
    val_size = int(0.10 * total_samples)
    test_size = int(0.15 * total_samples)

    # Create random indices for the entire dataset
    indices = torch.randperm(total_samples)

    # Split indices into three parts
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create the splits using these indices
    X_train, X_val, X_test = X_boards[train_indices], X_boards[val_indices], X_boards[test_indices]
    y_start_train, y_start_val, y_start_test = y_start[train_indices], y_start[val_indices], y_start[test_indices]
    y_end_train, y_end_val, y_end_test = y_end[train_indices], y_end[val_indices], y_end[test_indices]
    y_winner_train, y_winner_val, y_winner_test = y_winner[train_indices], y_winner[val_indices], y_winner[test_indices]

    print(f"✅ Training set: {X_train.shape[0]} samples")
    print(f"✅ Validation set: {X_val.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")

    # Return three tuples, one for each split
    return (
        (X_train, y_start_train, y_end_train, y_winner_train),  # Training data
        (X_val, y_start_val, y_end_val, y_winner_val),          # Validation data
        (X_test, y_start_test, y_end_test, y_winner_test)       # Test data
    )
def rotate_board(board, k):
    """Rotate the board 90° k times (k=1 → 90°, k=2 → 180°, k=3 → 270°)."""
    return np.rot90(board, k, axes=(1, 2))  # Rotate along spatial axes

def flip_board(board, axis):
    """Flip the board horizontally (axis=1) or vertically (axis=2)."""
    return np.flip(board, axis=axis)

def transform_move(move_idx, transform_type, board_size=11):
    """Transform move indices based on board rotation or flipping."""
    row, col = divmod(move_idx, board_size)

    if transform_type == "rotate_90":
        new_row, new_col = col, board_size - 1 - row
    elif transform_type == "rotate_180":
        new_row, new_col = board_size - 1 - row, board_size - 1 - col
    elif transform_type == "rotate_270":
        new_row, new_col = board_size - 1 - col, row
    elif transform_type == "flip_horizontal":
        new_row, new_col = row, board_size - 1 - col
    elif transform_type == "flip_vertical":
        new_row, new_col = board_size - 1 - row, col
    else:
        return move_idx  # No transformation

    return new_row * board_size + new_col  # Convert back to flattened index

def augment_data(X_boards, y_start, y_end, y_winner):
    """Apply rotations and flips to augment training data."""
    augmented_X, augmented_y_start, augmented_y_end, augmented_y_winner = [], [], [], []

    for i in range(len(X_boards)):
        board = X_boards[i]
        start_pos = y_start[i]
        end_pos = y_end[i]
        winner = y_winner[i]

        # Original board (unchanged)
        augmented_X.append(board)
        augmented_y_start.append(start_pos)
        augmented_y_end.append(end_pos)
        augmented_y_winner.append(winner)

        # Apply rotations
        for k, transform_name in enumerate(["rotate_90", "rotate_180", "rotate_270"], 1):
            rotated_board = rotate_board(board, k)
            new_start = transform_move(start_pos, transform_name)
            new_end = transform_move(end_pos, transform_name)

            augmented_X.append(rotated_board)
            augmented_y_start.append(new_start)
            augmented_y_end.append(new_end)
            augmented_y_winner.append(winner)  # Outcome is unchanged

        # Apply flips
        for axis, transform_name in zip([2, 1], ["flip_horizontal", "flip_vertical"]):
            flipped_board = flip_board(board, axis)
            new_start = transform_move(start_pos, transform_name)
            new_end = transform_move(end_pos, transform_name)

            augmented_X.append(flipped_board)
            augmented_y_start.append(new_start)
            augmented_y_end.append(new_end)
            augmented_y_winner.append(winner)

    return np.array(augmented_X), np.array(augmented_y_start), np.array(augmented_y_end), np.array(augmented_y_winner)

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
    """
    Creates DataLoader objects for training, validation, and testing.
    
    Args:
        batch_size: Number of samples per batch
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load the split data
    train_data, val_data, test_data = load_data()
    
    # Create dataset objects
    train_dataset = TensorDataset(*train_data)
    val_dataset = TensorDataset(*val_data)
    test_dataset = TensorDataset(*test_data)
    
    # Create the dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True  # Shuffle training data
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False  # No need to shuffle validation data
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False  # No need to shuffle test data
    )
    
    return train_loader, val_loader, test_loader
