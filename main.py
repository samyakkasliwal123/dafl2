if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    from client import Client
    from server import Server
    from communication import send_updates, broadcast_model
    from layerwise import LayerwiseManager

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    clients = [
        Client(0, DataLoader(Subset(train_data, range(0, 10000)), batch_size=32)),
        Client(1, DataLoader(Subset(train_data, range(10000, 20000)), batch_size=32)),
        Client(2, DataLoader(Subset(train_data, range(20000, 30000)), batch_size=32))
    ]

    server = Server(num_layers=3)
    for client in clients:
        client.initialize_model(server.global_model)

    broadcast_model(server, clients)

    layerwise = LayerwiseManager(clients, server, num_layers=3)
    layerwise.train()

    print("\nFederated training complete.")
