# DAFL2: Federated Learning with VisDrone Dataset

This project implements a federated learning framework with support for layer-wise training and auxiliary classifiers using the VisDrone 2019 drone image classification dataset.

## Project Structure
```
├── client.py               # Client logic for local training
├── server.py               # Server logic for model aggregation
├── models.py               # Residual CNN model with auxiliary classifiers
├── communication.py        # Utilities for broadcasting and collecting updates
├── layerwise.py            # Manages training each layer across clients
├── splitlearning.py        # Optional module for model splitting
├── download_data.py        # Downloads and extracts VisDrone 2019 dataset
├── main.py                 # Entry point for training
├── test_main.py            # Unit tests
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
```

## Installation
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

## Download Dataset
Run the following to automatically download and extract the VisDrone 2019 dataset:
```bash
python download_data.py
```

## Run Federated Training
```bash
python main.py
```
This will launch layer-wise federated training across simulated clients using the drone images.

Training logs including per-client layer loss, accuracy, and time will be saved to:
```
layerwise_training_logs.csv
```

## Run Tests
Run unit tests to verify model structure and output:
```bash
python test_main.py
```

## Notes
- The dataset is split across 3 clients for non-IID training.
- Each client trains one layer at a time using auxiliary heads.
- The server aggregates weights layer-wise and updates the global model.

This setup demonstrates scalable and modular federated learning on real-world drone imagery without needing centralized data.

You can integrate split learning using the `splitlearning.py` module. See the `ClientModel` and `ServerModel` classes for splitting the architecture at any layer.

---

Made to demonstrate modular and flexible federated learning pipelines.
