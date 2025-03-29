def send_updates(client, layer_idx):
    """Simulate client-to-server communication"""
    return {
        'client_id': client.id,
        'layer_idx': layer_idx,
        'weights': client.train_layer(layer_idx)
    }

def broadcast_model(server, clients):
    """Update all clients with global model"""
    global_weights = server.global_model.state_dict()
    for client in clients:
        client.model.load_state_dict(global_weights)