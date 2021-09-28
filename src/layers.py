import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.BatchNorm2d(num_features=6),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, t):
        return self.block(t)


class LinearBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.block = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(1176, 500),
            nn.ReLU(),

            nn.Dropout(p=0.25),
            nn.Linear(500, 150),
            nn.ReLU(),

            nn.Linear(150, 10)
        )

    def forward(self, t):
        return self.block(t)
