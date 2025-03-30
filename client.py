import torch
import torch.nn as nn
from models import ResCNNWithAuxiliaries
import time

class Client:
    def __init__(self, client_id, train_loader):
        self.id = client_id
        self.train_loader = train_loader
        self.model = None
        self.current_layer = 0
        self.logs = []

    def initialize_model(self, global_model):
        self.model = ResCNNWithAuxiliaries()
        self.model.load_state_dict(global_model.state_dict())
        self.freeze_previous_layers()

    def freeze_previous_layers(self):
        for idx, layer in enumerate(self.model.layers):
            for param in layer.parameters():
                param.requires_grad = (idx == self.current_layer)

    def train_layer(self, layer_idx, epochs=1, lr=0.01):
        self.current_layer = layer_idx
        self.freeze_previous_layers()

        optimizer = torch.optim.SGD(
            list(self.model.layers[layer_idx].parameters()) +
            list(self.model.aux_classifiers[layer_idx].parameters()),
            lr=lr
        )
        criterion = nn.CrossEntropyLoss()

        epoch_logs = []
        for epoch in range(epochs):
            correct = 0
            total = 0
            epoch_loss = 0.0
            start = time.time()

            for data, labels in self.train_loader:
                optimizer.zero_grad()
                x = data
                for i in range(layer_idx):
                    x = self.model.layers[i](x).detach()
                x = self.model.layers[layer_idx](x)
                aux_output = self.model.aux_classifiers[layer_idx](x)
                loss = criterion(aux_output, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = aux_output.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            accuracy = 100. * correct / total
            duration = time.time() - start
            print(f"[Client {self.id}] Layer {layer_idx}, Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={accuracy:.2f}%, Time={duration:.2f}s")
            epoch_logs.append({
                "layer": layer_idx,
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "accuracy": accuracy,
                "time": duration
            })

        self.logs.extend(epoch_logs)
        return {
            'layer_weights': self.model.layers[layer_idx].state_dict(),
            'aux_weights': self.model.aux_classifiers[layer_idx].state_dict()
        }