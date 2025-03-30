import torch
import torch.nn as nn
import torch.optim as optim

class LayerwiseTrainer:
    def __init__(self, model, device, learning_rate=0.01):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        
    def train_layer(self, layer_idx, data_generator, epochs=5):
        """Train a specific layer while freezing others."""
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze only the target layer
        target_layer = None
        current_idx = 0
        
        for name, module in self.model.named_children():
            if current_idx == layer_idx:
                target_layer = module
                for param in module.parameters():
                    param.requires_grad = True
                break
            current_idx += 1
            
        if target_layer is None:
            raise ValueError(f"Layer index {layer_idx} is out of range")
            
        # Create optimizer for just this layer
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), 
                             lr=self.learning_rate)
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, targets) in enumerate(data_generator):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
            print(f'Layer {layer_idx}, Epoch {epoch+1}: Loss: {running_loss/len(data_generator):.3f} | '
                  f'Acc: {100.*correct/total:.3f}%')
    
    def train_model_layerwise(self, data_generator, epochs_per_layer=5):
        """Train the model layer by layer."""
        num_layers = len(list(self.model.children()))
        
        for layer_idx in range(num_layers):
            print(f"Training layer {layer_idx}/{num_layers-1}")
            self.train_layer(layer_idx, data_generator, epochs=epochs_per_layer)
            
        # Final fine-tuning with all layers
        print("Fine-tuning all layers")
        for param in self.model.parameters():
            param.requires_grad = True
            
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate/10)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs_per_layer):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, targets) in enumerate(data_generator):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
            print(f'Fine-tuning, Epoch {epoch+1}: Loss: {running_loss/len(data_generator):.3f} | '
                  f'Acc: {100.*correct/total:.3f}%')

# Usage with DAFL
def get_layerwise_dafl_trainer(student_model, device):
    return LayerwiseTrainer(student_model, device)