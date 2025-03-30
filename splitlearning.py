import torch.nn as nn

class ClientModel(nn.Module):
    def __init__(self, full_model, split_idx):
        super().__init__()
        self.model = nn.Sequential(*list(full_model.children())[:split_idx])

    def forward(self, x):
        return self.model(x)

class ServerModel(nn.Module):
    def __init__(self, full_model, split_idx):
        super().__init__()
        self.model = nn.Sequential(*list(full_model.children())[split_idx:])

    def forward(self, x):
        return self.model(x)