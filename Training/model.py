import torch.nn as nn
import torch

class PolicyValueNetwork(nn.Module):
    def __init__(self):
        super(PolicyValueNetwork, self).__init__()

        # === Convolutional Feature Extractor ===
        self.conv_layers = nn.Sequential(
            # First conv block: (6 x 11 x 11) -> (32 x 11 x 11)
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Second conv block: (32 x 11 x 11) -> (64 x 11 x 11)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
             
            # Third conv block: (64 x 11 x 11) -> (128 x 11 x 11)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),
             
            # Fourth conv block: (128 x 11 x 11) -> (256 x 11 x 11)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.2),
        )

        # === Policy Head (Predicts Start & End Position Separately) ===
        self.policy_start_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),  # (256 x 11 x 11) -> (1 x 11 x 11)
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),  # (1 x 11 x 11) -> (121)
            nn.Linear(121, 121)  # Output: Start position probabilities (121 classes)
        )

        self.policy_end_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),  # (256 x 11 x 11) -> (1 x 11 x 11)
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),  # (1 x 11 x 11) -> (121)
            nn.Linear(121, 121)  # Output: End position probabilities (121 classes)
        )

        # === Value Head (Position Evaluation) ===
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),  # (256 x 11 x 11) -> (1 x 11 x 11)
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),  # (1 x 11 x 11) -> (121)
            nn.Linear(121, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()  # Outputs value between -1 (Black wins) and 1 (White wins)
        )

    def forward(self, x):
       # print(f"Input to conv_layers: {x.shape}")
        x = self.conv_layers(x)  # Extract features
        #print(f"After conv_layers: {x.shape}")

        # Predict start and end positions separately
        y_start_pred = self.policy_start_head(x)
        y_end_pred = self.policy_end_head(x)
       # print(f"y_start_pred shape: {y_start_pred.shape}")
        #print(f"y_end_pred shape: {y_end_pred.shape}")

        # Normalize outputs with softmax
        y_start_pred = torch.softmax(y_start_pred, dim=1)
        y_end_pred = torch.softmax(y_end_pred, dim=1)
        #print(f"Softmax applied: y_start_pred shape: {y_start_pred.shape}, y_end_pred shape: {y_end_pred.shape}")

        # Predict value (winning probability)
        value = self.value_head(x)
        #print(f"value shape: {value.shape}")

        return y_start_pred, y_end_pred, value
