import numpy as np

# Load dataset
data = np.load("data/hnefatafl_dataset.npz")

X_boards = data["X_boards"]  # (num_samples, 6, 11, 11)

# Print first board state raw values
print("Raw board tensor values:")
print(X_boards[2])  # First board in dataset
