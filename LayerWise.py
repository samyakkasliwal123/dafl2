from communication import send_updates, broadcast_model
import pandas as pd

class LayerwiseManager:
    def __init__(self, clients, server, num_layers):
        self.clients = clients
        self.server = server
        self.num_layers = num_layers
        self.metrics = []

    def train(self):
        for layer_idx in range(self.num_layers):
            client_updates = []
            print(f"\n=== Training Layer {layer_idx} ===")
            for client in self.clients:
                update = send_updates(client, layer_idx)
                client_updates.append(update)
            self.server.aggregate_updates(client_updates)
            broadcast_model(self.server, self.clients)

        # Collect logs
        all_logs = []
        for client in self.clients:
            for log in client.logs:
                log['client_id'] = client.id
                all_logs.append(log)

        df = pd.DataFrame(all_logs)
        df.to_csv("layerwise_training_logs.csv", index=False)
        print("\n[✓] Logs saved to layerwise_training_logs.csv")
