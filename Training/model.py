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
            nn.Dropout2d(0.2),
             
            # Third conv block: (64 x 11 x 11) -> (128 x 11 x 11)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),
             
            # Fourth conv block: (128 x 11 x 11) -> (256 x 11 x 11)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3),
        )

        # === Policy Head (Move Selection) ===
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1),  # (256 x 11 x 11) -> (2 x 11 x 11)
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),  # (2 x 11 x 11) -> (242)
            nn.Linear(2 * 11 * 11, 242)  # Predicts a probability distribution over 242 possible moves
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
        x = self.conv_layers(x)  # Extract features
        policy = self.policy_head(x)  # Move probabilities
        value = self.value_head(x)  # Board evaluation
        return policy, value