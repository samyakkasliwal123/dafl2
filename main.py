if __name__ == "__main__":
    # Example usage
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    from client import Client
    from server import Server
    from communication import send_updates, broadcast_model
    
    # Initialize datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Create 3 clients with non-IID data
    clients = [
        Client(0, DataLoader(Subset(train_data, range(0, 10000)), batch_size=32)),
        Client(1, DataLoader(Subset(train_data, range(10000, 20000)), batch_size=32)),
        Client(2, DataLoader(Subset(train_data, range(20000, 30000)), batch_size=32))
    ]
    
    server = Server(num_layers=3)
    
    # Federated training loop
    num_rounds = 5
    for round in range(num_rounds):
        print(f"Round {round+1}/{num_rounds}")
        
        # Layer-wise training from bottom to top
        for layer_idx in range(3):
            # Select clients for current layer
            selected_clients = clients  # Can implement selection strategy
            
            # Train and collect updates
            updates = []
            for client in selected_clients:
                client.initialize_model(server.global_model)
                updates.append(send_updates(client, layer_idx))
            
            # Aggregate updates
            server.aggregate_updates(updates)
            
            # Broadcast updated model
            broadcast_model(server, selected_clients)