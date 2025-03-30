def send_updates(client, layer_idx):
    return {
        'client_id': client.id,
        'layer_idx': layer_idx,
        'weights': client.train_layer(layer_idx)
    }

def broadcast_model(server, clients):
    global_weights = server.global_model.state_dict()
    for client in clients:
        client.model.load_state_dict(global_weights)