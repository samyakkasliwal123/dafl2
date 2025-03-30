if __name__ == "__main__":
    import os
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    from client import Client
    from server import Server
    from communication import send_updates, broadcast_model
    from layerwise import LayerwiseManager

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_path = os.path.join("data", "VisDrone2019-DET-train", "images")
    if not os.path.exists(dataset_path):
        raise RuntimeError("Dataset not found. Run download_data.py first.")

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # Simulate 3 non-IID clients
    total = len(dataset)
    third = total // 3
    clients = [
        Client(0, DataLoader(Subset(dataset, list(range(0, third))), batch_size=32)),
        Client(1, DataLoader(Subset(dataset, list(range(third, 2*third))), batch_size=32)),
        Client(2, DataLoader(Subset(dataset, list(range(2*third, total))), batch_size=32))
    ]

    server = Server(num_layers=3)
    for client in clients:
        client.initialize_model(server.global_model)

    broadcast_model(server, clients)

    layerwise = LayerwiseManager(clients, server, num_layers=3)
    layerwise.train()

    print("\nFederated training complete. See 'layerwise_training_logs.csv' for metrics.")
