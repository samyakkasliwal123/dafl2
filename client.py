import torch
import torch.nn as nn
from models import ResCNNWithAuxiliaries

class Client:
    def __init__(self, client_id, train_loader):
        self.id = client_id
        self.train_loader = train_loader
        self.model = None
        self.current_layer = 0
        
    def initialize_model(self, global_model):
        """Initialize client model with global weights"""
        self.model = ResCNNWithAuxiliaries()
        self.model.load_state_dict(global_model.state_dict())
        self.freeze_previous_layers()
        
    def freeze_previous_layers(self):
        """Freeze layers before current training layer"""
        for idx, layer in enumerate(self.model.layers):
            for param in layer.parameters():
                param.requires_grad = (idx == self.current_layer)
                
    def train_layer(self, layer_idx, epochs=1, lr=0.01):
        """Train specific layer with auxiliary classifier"""
        self.current_layer = layer_idx
        self.freeze_previous_layers()
        
        optimizer = torch.optim.SGD(
            list(self.model.layers[layer_idx].parameters()) + 
            list(self.model.aux_classifiers[layer_idx].parameters()),
            lr=lr
        )
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(epochs):
            for data, labels in self.train_loader:
                optimizer.zero_grad()
                
                # Forward pass through current layer and auxiliary classifier
                x = data
                for i in range(layer_idx):
                    x = self.model.layers[i](x).detach()
                    
                x = self.model.layers[layer_idx](x)
                aux_output = self.model.aux_classifiers[layer_idx](x)
                
                loss = criterion(aux_output, labels)
                loss.backward()
                optimizer.step()
                
        return {
            'layer_weights': self.model.layers[layer_idx].state_dict(),
            'aux_weights': self.model.aux_classifiers[layer_idx].state_dict()
        }