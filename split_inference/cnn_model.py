import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Simple CNN (Agent 1)
    Inspired by LeNet. Fast and light.
    Matches profiles in rl_pdnn/utils.py (5 layers)
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Profiles: (5.0, 20.0, 15.0), (8.0, 40.0, 8.0), (0.5, 0.5, 0.5), (5.0, 150.0, 0.2), (0.1, 1.0, 0.01)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ), # Layer 0: Conv1
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ), # Layer 1: Conv2
            nn.Flatten(), # Layer 2: Flatten
            nn.Sequential(
                nn.Linear(64 * 5 * 5, 128),
                nn.ReLU()
            ), # Layer 3: FC1
            nn.Linear(128, 10) # Layer 4: Output
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DeepCNN(nn.Module):
    """
    Deeper CNN (Agent 2)
    Better generalization.
    Matches profiles in rl_pdnn/utils.py (5 layers)
    """
    def __init__(self):
        super(DeepCNN, self). __init__()
        # Profiles: (10.0, 40.0, 20.0), (15.0, 80.0, 10.0), (0.5, 0.5, 0.5), (8.0, 200.0, 0.5), (0.2, 2.0, 0.01)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ), # Layer 0: Block 1
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ), # Layer 1: Block 2
            nn.Flatten(), # Layer 2: Flatten
            nn.Sequential(
                nn.Linear(64 * 7 * 7, 256),
                nn.ReLU()
            ), # Layer 3: FC1
            nn.Linear(256, 10) # Layer 4: Output
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MiniResNet(nn.Module):
    """
    Mini ResNet (Agent 3)
    Modern architecture with skip connection.
    Matches profiles in rl_pdnn/utils.py (5 layers)
    """
    def __init__(self):
        super(MiniResNet, self).__init__()
        # Profiles: (12.0, 50.0, 25.0), (18.0, 60.0, 25.0), (1.0, 1.0, 1.0), (6.0, 150.0, 0.2), (0.1, 1.0, 0.01)
        
        self.conv1_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # We define the ResBlock separately but we need to wrap it for split inference
        class ResBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            def forward(self, x):
                skip = x
                x = F.relu(self.conv(x))
                return x + skip

        self.layers = nn.ModuleList([
            self.conv1_block, # Layer 0
            ResBlock(),       # Layer 1
            nn.Sequential(
                nn.MaxPool2d(2),
                nn.Flatten()
            ),                # Layer 2
            nn.Sequential(
                nn.Linear(32 * 14 * 14, 128),
                nn.ReLU()
            ),                # Layer 3
            nn.Linear(128, 10) # Layer 4
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
