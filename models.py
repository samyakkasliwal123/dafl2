import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

class ResCNNWithAuxiliaries(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualBlock(3, 64),
            ResidualBlock(64, 128),
            ResidualBlock(128, 256)
        ])
        
        self.aux_classifiers = nn.ModuleList([
            self._make_aux_classifier(64, num_classes),
            self._make_aux_classifier(128, num_classes),
            nn.Linear(256, num_classes)
        ])
        
    def _make_aux_classifier(self, in_channels, num_classes):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = torch.flatten(x, 1)
        return self.aux_classifiers[-1](x)