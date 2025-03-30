# DAFL2: Decentralized Federated Learning with Layer-wise and Split Learning Integration

This project implements a federated learning framework with support for layer-wise training, auxiliary classifiers, and optional split learning components. It uses CIFAR-10 for demonstration.

##  Project Structure
```
├── client.py               # Client-side logic
├── server.py               # Server-side logic
├── models.py               # CNN model with residual blocks + auxiliary heads
├── communication.py        # Sync utilities for model weights
├── layerwise.py            # Layer-by-layer federated manager
├── splitlearning.py        # Optional model split utilities
├── main.py                 # Training entry point
├── test_main.py            # Unit tests
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
```

##  Installation
```bash
pip install -r requirements.txt
```

##  Running the Code
```bash
python main.py
```
This launches a federated learning loop with 3 clients. Each client trains one layer at a time. The server aggregates updates using simple averaging.

##  Running Tests
```bash
python test_main.py
```
This checks the output shape of the model and validates the forward pass.

##  Optional: Using Split Learning
You can integrate split learning using the `splitlearning.py` module. See the `ClientModel` and `ServerModel` classes for splitting the architecture at any layer.

---

Made to demonstrate modular and flexible federated learning pipelines.
