import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitModel(nn.Module):
    """Base class for split learning models."""
    def __init__(self):
        super(SplitModel, self).__init__()
        
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")

class ClientModel(SplitModel):
    """Client-side part of the split model."""
    def __init__(self, original_model, split_layer):
        super(ClientModel, self).__init__()
        self.layers = nn.ModuleList()
        
        # Extract layers up to the split point
        for i, layer in enumerate(original_model.children()):
            if i < split_layer:
                self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ServerModel(SplitModel):
    """Server-side part of the split model."""
    def __init__(self, original_model, split_layer):
        super(ServerModel, self).__init__()
        self.layers = nn.ModuleList()
        
        # Extract layers after the split point
        for i, layer in enumerate(original_model.children()):
            if i >= split_layer:
                self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SplitLearningManager:
    def __init__(self, full_model, split_layer, device):
        self.device = device
        self.client_model = ClientModel(full_model, split_layer).to(device)
        self.server_model = ServerModel(full_model, split_layer).to(device)
        self.split_layer = split_layer
        
    def train_step(self, inputs, targets, optimizer, criterion):
        """Perform one training step in the split learning setting."""
        # Client forward pass
        client_output = self.client_model(inputs)
        
        # Send to server (in real scenarios, this would be a network transfer)
        server_input = client_output.detach().requires_grad_()
        
        # Server forward pass
        server_output = self.server_model(server_input)
        
        # Compute loss on server
        loss = criterion(server_output, targets)
        
        # Server backward pass
        loss.backward()
        
        # Send gradients back to client (in real scenarios, this would be a network transfer)
        client_grads = server_input.grad
        
        # Client backward pass with received gradients
        client_output.backward(client_grads)
        
        # Update parameters
        optimizer.step()
        optimizer.zero_grad()
        
        return loss.item(), server_output
    
    def merge_models(self):
        """Merge client and server models back into a single model."""
        full_model = nn.Sequential()
        
        # Add client layers
        for layer in self.client_model.layers:
            full_model.add_module(str(len(full_model)), layer)
            
        # Add server layers
        for layer in self.server_model.layers:
            full_model.add_module(str(len(full_model)), layer)
            
        return full_model

# Integration with DAFL
def setup_split_learning_for_dafl(student_model, split_layer, device):
    return SplitLearningManager(student_model, split_layer, device)