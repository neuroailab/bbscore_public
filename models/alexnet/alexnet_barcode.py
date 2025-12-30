import torch
import torch.nn as nn


class AlexNetBarcode(nn.Module):
    def __init__(self, num_classes: int = 32, dropout: float = 0.5):
        super(AlexNetBarcode, self).__init__()
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv2
            nn.Conv2d(in_channels=64, out_channels=192,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv3
            nn.Conv2d(in_channels=192, out_channels=384,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Conv4
            nn.Conv2d(in_channels=384, out_channels=384,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Conv5
            nn.Conv2d(in_channels=384, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            # FC6
            nn.Linear(in_features=256 * 5 * 5, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            # FC7
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            # FC8
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
