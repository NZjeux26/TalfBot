import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import PolicyValueNetwork
from utils import get_device, Timer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from metrics import evaluate_model

# Get device
device = get_device()
print(f"🚀 Using device: {device}")

# Load Data
train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

# Initialize Model
model = PolicyValueNetwork().to(device)

# Loss functions
policy_loss_fn = nn.CrossEntropyLoss()
end_loss_fn = nn.CrossEntropyLoss()  # For y_end
value_loss_fn = nn.MSELoss()  # For y_winner (regression)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

# Training Stats
EPOCHS = 10
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

print("\n🚀 Starting Training...")
timer = Timer()

for epoch in range(EPOCHS):
    model.train()
    epoch_timer = Timer()

    total_train_loss, correct_train, total_train_samples = 0, 0, 0

    for X_batch, y_start_batch, y_end_batch, y_winner_batch in train_loader:
        X_batch, y_start_batch, y_end_batch, y_winner_batch = (
            X_batch.to(device),
            y_start_batch.to(device),
            y_end_batch.to(device),
            y_winner_batch.to(device),
        )

        # Forward pass
        y_start_pred, y_end_pred, value_pred = model(X_batch)

        # Convert one-hot labels to class indices
        y_start_indices = y_start_batch.view(y_start_batch.size(0), -1).argmax(dim=1)
        y_end_indices = y_end_batch.view(y_end_batch.size(0), -1).argmax(dim=1)

        # Compute losses
        start_loss = policy_loss_fn(y_start_pred, y_start_indices)
        end_loss = end_loss_fn(y_end_pred, y_end_indices)
        value_loss = value_loss_fn(value_pred, y_winner_batch)

        # Total loss
        total_loss = start_loss + end_loss + value_loss

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Track training loss
        total_train_loss += total_loss.item()

        # Track training accuracy (correct start and end predictions)
        predicted_start = y_start_pred.argmax(dim=1)
        predicted_end = y_end_pred.argmax(dim=1)

        correct_train += ((predicted_start == y_start_indices) & (predicted_end == y_end_indices)).sum().item()
        total_train_samples += y_start_indices.size(0)


    # Compute average training loss & accuracy
    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = correct_train / total_train_samples

    # === Validation Phase ===
    model.eval()
    total_val_loss, correct_val, total_val_samples = 0, 0, 0

    with torch.no_grad():
        for X_batch, y_start_batch, y_end_batch, y_winner_batch in val_loader:
            X_batch, y_start_batch, y_end_batch, y_winner_batch = (
                X_batch.to(device),
                y_start_batch.to(device),
                y_end_batch.to(device),
                y_winner_batch.to(device),
            )

            y_start_pred, y_end_pred, value_pred = model(X_batch)

            y_start_indices = y_start_batch.view(y_start_batch.size(0), -1).argmax(dim=1)
            y_end_indices = y_end_batch.view(y_end_batch.size(0), -1).argmax(dim=1)

            start_loss = policy_loss_fn(y_start_pred, y_start_indices)
            end_loss = end_loss_fn(y_end_pred, y_end_indices)
            value_loss = value_loss_fn(value_pred, y_winner_batch)

            total_loss = start_loss + end_loss + (0.01 * value_loss)  # Reduce weight on value loss

            total_val_loss += total_loss.item()

            predicted_start = y_start_pred.argmax(dim=1)
            predicted_end = y_end_pred.argmax(dim=1)

            correct_val += ((predicted_start == y_start_indices) & (predicted_end == y_end_indices)).sum().item()
            total_val_samples += y_start_indices.size(0)

    # Compute average validation loss & accuracy
    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct_val / total_val_samples

    # Store results for graphing
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accuracy = correct_train / total_train_samples
    val_accuracy = correct_val / total_val_samples

    print(f"⏳ Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f} - Time: {epoch_timer.elapsed():.2f}s")

torch.save(model.state_dict(), "hnefatafl_policy_value_model.pth")
print(f"\n✅ Training completed in {timer.elapsed():.2f} seconds")
print("\n📊 Evaluating final model performance...")
# After training your model
metrics = evaluate_model(
    model_path="hnefatafl_policy_value_model.pth",
    test_loader=test_loader,
    device=device
)
# # === Plot Training Progress ===
# plt.figure(figsize=(12, 5))

# # Loss Plot
# plt.subplot(1, 2, 1)
# plt.plot(range(1, EPOCHS+1), train_losses, label="Train Loss")
# plt.plot(range(1, EPOCHS+1), val_losses, label="Val Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training & Validation Loss")
# plt.legend()

# # Accuracy Plot
# plt.subplot(1, 2, 2)
# plt.plot(range(1, EPOCHS+1), train_accuracies, label="Train Acc")
# plt.plot(range(1, EPOCHS+1), val_accuracies, label="Val Acc")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("Training & Validation Accuracy")
# plt.legend()

# # Show the graph
# plt.tight_layout()
# plt.show()
