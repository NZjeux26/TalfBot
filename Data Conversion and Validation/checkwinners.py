import numpy as np

# Load dataset
data = np.load("hnefatafl_dataset.npz")

# Extract winner labels
y_winner = data["y_winner"]

# Count how many times each winner appears
unique_labels, counts = np.unique(y_winner, return_counts=True)

# Print results
print(f"Unique winner labels: {unique_labels}")
print(f"Counts per label: {counts}")