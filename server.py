import copy
from collections import defaultdict
from models import ResCNNWithAuxiliaries

class Server:
    def __init__(self, num_layers):
        self.global_model = ResCNNWithAuxiliaries()
        self.num_layers = num_layers
        self.layer_updates = defaultdict(list)

    def aggregate_updates(self, client_updates):
        for update in client_updates:
            layer_idx = update['layer_idx']
            self.layer_updates[layer_idx].append(update['weights'])

        for layer_idx, weights_list in self.layer_updates.items():
            avg_weights = self._average_weights(weights_list)
            self._update_global_model(layer_idx, avg_weights)

        self.layer_updates.clear()

    def _average_weights(self, weights_list):
        avg_weights = {}
        for key in weights_list[0].keys():
            avg_weights[key] = sum([w[key] for w in weights_list]) / len(weights_list)
        return avg_weights

    def _update_global_model(self, layer_idx, new_weights):
        self.global_model.layers[layer_idx].load_state_dict(new_weights['layer_weights'])
        if layer_idx < self.num_layers - 1:
            del self.global_model.aux_classifiers[layer_idx]