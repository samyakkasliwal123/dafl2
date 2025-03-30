from client import Client

class LayerwiseManager:
    def __init__(self, clients, server, num_layers):
        self.clients = clients
        self.server = server
        self.num_layers = num_layers

    def train(self):
        for layer_idx in range(self.num_layers):
            client_updates = []
            for client in self.clients:
                update = send_updates(client, layer_idx)
                client_updates.append(update)
            self.server.aggregate_updates(client_updates)
            broadcast_model(self.server, self.clients)
